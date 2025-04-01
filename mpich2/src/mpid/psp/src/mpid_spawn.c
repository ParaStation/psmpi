/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"
#include <unistd.h>
#include <sys/types.h>
#include "mpid_debug.h"

/*
Strategy of a MPID_Open_port, Accept, Connect
================================================

MPID_Open_port
    inter_socket = pscom_open_socket()
    pscom_listen() // Restricted to TCP connections
    pscom_socket_get_ep_str(&ep_str) // Socket named "int%05u".
    inter_sockets_add(inter_socket);

MPID_Comm_accept(root_ep_str)
    // root_ep_str undefined at ranks != root !!!
    open_all_sockets

    @root:
	// ep_strs are collected in open_all_sockets
	remote_root = pscom_accept(inter_socket)

---------------------------------------------------------------------------------------------------
	pscom_send(inter_socket, ep_strs, remote_context_id, remote_size) (send_ep_strs_remote)
	pscom_recv(inter_socket, ep_strs, remote_context_id, remote_size) (recv_ep_strs_remote)

    connect_ep_strs
=========================== REPLACED BY:

    forward_pg_info(con, comm, root, ep_strs, intercomm) --> MPID_PG_ForwardPGInfo(...)
---------------------------------------------------------------------------------------------------

    @root: barrier with remote root

    MPIR_Barrier_impl(comm) // assure all remote ranks are connected

MPID_Comm_connect()
    open_all_sockets

    @root:
	// ep_strs are collected in open_all_sockets
	remote_root = pscom_connect(inter_socket, ep_str, PSCOM_RANK_UNDEFINED, flags)
---------------------------------------------------------------------------------------------------
	pscom_send(inter_socket, ep_strs, remote_context_id, remote_size) (send_ep_strs_remote)
	pscom_recv(inter_socket, ep_strs, remote_context_id, remote_size) (recv_ep_strs_remote)

    connect_ep_strs
=========================== REPLACED BY:

    forward_pg_info(con, comm, root, ep_strs, intercomm) --> MPID_PG_ForwardPGInfo(...)
---------------------------------------------------------------------------------------------------

    @root: barrier with remote root

    MPIR_Barrier_impl(comm) // assure all remote ranks are connected


Strategy of a spawn:
=========================

Parent
---------------------
Spawn
  MPID_Open_port
  @root:
    MPIR_pmi_Spawn_multiple(ep_str@root)

  MPID_Comm_accept
  MPID_Close_port


Child
-------------------
MPI_Init
  MPID_PSP_Get_parent_ep_str
  MPID_Comm_connect


Helper
---------------
open_all_sockets(root)
    pscom_socket_get_ep_str(&ep_str);
    MPI_Gather ep_str -> ep_strs@root

connect_ep_strs(root)
    MPI_Bcast ep_strs from root
    for p in ep_strs:
       pscom_connect(p)
    MPIR_Barrier_impl(comm) // assure all ranks are connected
*/

#define WARN_NOT_IMPLEMENTED						\
do {									\
	static int warned = 0;						\
	if (!warned) {							\
		warned = 1;						\
		fprintf(stderr, "Warning: %s() not implemented\n", __func__); \
	}								\
} while (0)


#define INTER_SOCKETS_MAX 1024

/*
 * Inter sockets
 *
 * Mapping between a ep_str from MPID_Open_port and a pscom_socket.
 */

static
pscom_socket_t *inter_sockets[INTER_SOCKETS_MAX];

static
void inter_sockets_add(pscom_socket_t * socket)
{
    int i;
    for (i = 0; i < INTER_SOCKETS_MAX; i++) {
        if (inter_sockets[i] == NULL) {
            inter_sockets[i] = socket;
            return;
        }
    }
    fprintf(stderr, "Too many open ports (More than %d calls to MPI_Open_port())\n",
            INTER_SOCKETS_MAX);
    _exit(1);   /* ToDo: Graceful shutdown */
}


static
int inter_sockets_get_by_ep_str(const char *ep_str, pscom_socket_t ** socket)
{
    int mpi_errno = MPI_SUCCESS;
    int i;
    int found = 0;

    MPIR_Assert(socket);

    for (i = 0; i < INTER_SOCKETS_MAX; i++) {
        pscom_socket_t *sock_i = inter_sockets[i];
        if (!sock_i) {
            continue;
        }
#if MPID_PSP_HAVE_PSCOM_ABI_5
        pscom_err_t rc = PSCOM_SUCCESS;
        char *ep_str_i = NULL;
        rc = pscom_socket_get_ep_str(sock_i, &ep_str_i);
        MPIR_ERR_CHKANDJUMP1(rc != PSCOM_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                             "**psp|getepstr", "**psp|getepstr %s", pscom_err_str(rc));
        MPIR_ERR_CHKANDJUMP1(!ep_str_i, mpi_errno, MPI_ERR_OTHER, "**psp|nullendpoint",
                             "**psp|nullendpoint %s", sock_i->local_con_info.name);
        found = !strcmp(ep_str_i, ep_str);
        pscom_socket_free_ep_str(ep_str_i);
#else
        const char *ep_str_i = NULL;
        ep_str_i = pscom_listen_socket_ondemand_str(sock_i);
        MPIR_ERR_CHKANDJUMP1(!ep_str_i, mpi_errno, MPI_ERR_OTHER, "**psp|nullendpoint",
                             "**psp|nullendpoint %s", sock_i->local_con_info.name);
        found = !strcmp(ep_str_i, ep_str);
        if (!found) {
            /* Try with direct string */
            ep_str_i = pscom_listen_socket_str(sock_i);
            found = !strcmp(ep_str_i, ep_str);
        }
#endif
        if (found) {
            *socket = sock_i;
            break;
        }
    }

    if (!found) {
        *socket = NULL;
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


static
void inter_sockets_del_by_socket(pscom_socket_t * socket)
{
    int i;
    for (i = 0; i < INTER_SOCKETS_MAX; i++) {
        if (inter_sockets[i] == socket) {
            inter_sockets[i] = NULL;
            return;
        }
    }
}


/*
 * Communicator helpers
 */
static
void init_intercomm(MPIR_Comm * comm, MPIR_Context_id_t remote_context_id,
                    unsigned remote_comm_size, MPIR_Comm * intercomm, int create_vcrt_flag)
{
    /* compare with SetupNewIntercomm() in /src/mpid/ch3/src/ch3u_port.c:1143 */
    int mpi_errno;
    MPIDI_VCRT_t *vcrt;

    intercomm->context_id = remote_context_id;
    /* intercomm->recvcontext_id already set in create_intercomm */
    intercomm->is_low_group = 1;


    /* init sizes */
    intercomm->attributes = NULL;
    intercomm->remote_size = remote_comm_size;
    intercomm->local_size = comm->local_size;
    intercomm->rank = comm->rank;
    intercomm->local_group = NULL;
    intercomm->remote_group = NULL;
    intercomm->comm_kind = MPIR_COMM_KIND__INTERCOMM;
    intercomm->local_comm = NULL;

    /* Point local vcr at those of incoming intracommunicator */
    vcrt = MPIDI_VCRT_Dup(comm->vcrt);
    MPIR_Assert(vcrt);
    MPID_PSP_comm_set_local_vcrt(intercomm, vcrt);

    if (create_vcrt_flag) {
        vcrt = MPIDI_VCRT_Create(intercomm->remote_size);
        MPIR_Assert(vcrt);
        MPID_PSP_comm_set_vcrt(intercomm, vcrt);
    }

    /* MPIDI_VCR_Initialize() will be called later for every intercomm->remote_size rank */

    mpi_errno = MPIR_Comm_commit(intercomm);
    MPIR_Assert(mpi_errno == MPI_SUCCESS);
}


/*
 * helper
 */

static
int iam_root(int root, MPIR_Comm * comm)
{
    return comm->rank == root;
}

/*
 *  Tell the remote side about the local size plus the set of local GPIDs and receive the remote information in return (only root).
 *  Then distribute the new information among all local processes.
 *  Go with this information and the set of all local endpoint strings into the central MPID_PG_ForwardPGInfo() function for establishing
 *  all still missing connections while complementing the current view onto the PG topology.
 *  Finally, exchange the remote conext_id and build the new inter-communicator.
 *  (forward_pg_info() plus parts of MPID_PG_ForwardPGInfo() thus replace the former send_/recv_/connect_ep_strs() functions...)
 */
static
int forward_pg_info(pscom_connection_t * con, MPIR_Comm * comm, int root,
                    char *ep_strs, MPI_Aint * ep_strs_sizes, MPI_Aint ep_strs_total_size,
                    MPIR_Comm * intercomm)
{
    pscom_err_t rc;
    MPIR_Errflag_t errflag = FALSE;
    int mpi_errno = MPI_SUCCESS;

    int local_size = comm->local_size;
    int remote_size = 0;
    MPIDI_Gpid *local_gpids;
    MPIDI_Gpid *remote_gpids;
    uint64_t *remote_lpids;
    int local_context_id;
    int remote_context_id;

    pscom_socket_t *socket = intercomm->pscom_socket;

    if (iam_root(root, comm) && con) {
        pscom_send(con, NULL, 0, &local_size, sizeof(int));
        rc = pscom_recv_from(con, NULL, 0, &remote_size, sizeof(int));
        MPIR_Assert(rc == PSCOM_SUCCESS);
    }

    mpi_errno = MPIR_Bcast(&remote_size, 1, MPI_INT, root, comm, errflag);
    MPIR_Assert(mpi_errno == MPI_SUCCESS);

    if (remote_size == 0)
        goto err_failed;        /* this happens if root has no valid 'con' (see above!) */

    local_gpids = (MPIDI_Gpid *) MPL_malloc(local_size * sizeof(MPIDI_Gpid), MPL_MEM_OBJECT);
    remote_gpids = (MPIDI_Gpid *) MPL_malloc(remote_size * sizeof(MPIDI_Gpid), MPL_MEM_OBJECT);

    MPIDI_GPID_GetAllInComm(comm, local_size, local_gpids, NULL);

    if (iam_root(root, comm)) {
        pscom_send(con, NULL, 0, local_gpids, local_size * sizeof(MPIDI_Gpid));
        rc = pscom_recv_from(con, NULL, 0, remote_gpids, remote_size * sizeof(MPIDI_Gpid));
        MPIR_Assert(rc == PSCOM_SUCCESS);
    }

    mpi_errno =
        MPIR_Bcast(remote_gpids, remote_size * sizeof(MPIDI_Gpid), MPI_CHAR, root, comm, errflag);
    MPIR_Assert(mpi_errno == MPI_SUCCESS);


    /* Call the central routine for establishing all missing connections: */
    MPIDI_PG_ForwardPGInfo(NULL, comm, remote_size, remote_gpids, root, -1, -1, con,
                           ep_strs, ep_strs_sizes, ep_strs_total_size, socket);


    /* distribute remote values */
    if (iam_root(root, comm)) {
        local_context_id = intercomm->recvcontext_id;
        pscom_send(con, NULL, 0, &local_context_id, sizeof(int));
        rc = pscom_recv_from(con, NULL, 0, &remote_context_id, sizeof(int));
        MPIR_Assert(rc == PSCOM_SUCCESS);
    }
    local_context_id = intercomm->context_id;
    mpi_errno = MPIR_Bcast(&local_context_id, 1, MPI_INT, root, comm, errflag);
    MPIR_Assert(mpi_errno == MPI_SUCCESS);
    mpi_errno = MPIR_Bcast(&remote_context_id, 1, MPI_INT, root, comm, errflag);
    MPIR_Assert(mpi_errno == MPI_SUCCESS);

    if (!iam_root(root, comm)) {
        /* assure equal local context_id on all ranks */
        MPIR_Context_id_t context_id = local_context_id;
        MPIR_Assert(context_id == intercomm->context_id);
    }

    /* Update intercom (without creating a VCRT because it will be created in the MPID_Create_intercomm_from_lpids() call below) */
    init_intercomm(comm, remote_context_id, remote_size, intercomm, 0 /*create_vcrt_flag */);

    remote_lpids = (uint64_t *) MPL_malloc(remote_size * sizeof(uint64_t), MPL_MEM_OTHER);
    MPIDI_GPID_ToLpidArray(remote_size, remote_gpids, remote_lpids);
    MPID_Create_intercomm_from_lpids(intercomm, remote_size, remote_lpids);

    MPL_free(local_gpids);
    MPL_free(remote_gpids);
    MPL_free(remote_lpids);

    return MPI_SUCCESS;
  err_failed:
    init_intercomm(comm, MPIR_INVALID_CONTEXT_ID, 0 /* remote_size */ , intercomm,
                   1 /* create_vcrt_flag */);
    return MPI_ERR_COMM;
}


static
void inter_barrier(pscom_connection_t * con)
{
    int dummy = 0;
    int rc;

    /* Workaround for timing of pscom ondemand connections. Be
     * sure both sides have called pscom_connect before
     * using the connections. step 2 of 3 */
    pscom_send(con, NULL, 0, &dummy, sizeof(dummy));

    rc = pscom_recv_from(con, NULL, 0, &dummy, sizeof(dummy));
    MPIR_Assert(rc == PSCOM_SUCCESS);
}


int MPID_PSP_open_all_sockets(int root, MPIR_Comm * comm, MPIR_Comm * intercomm,
                              char **ep_strs, MPI_Aint ** ep_strs_sizes,
                              MPI_Aint * ep_strs_total_size)
{
    pscom_socket_t *socket_new = NULL;
    int local_size = comm->local_size;
    char *_ep_strs = NULL;      // only at root
    char *ep_str = NULL;
    MPI_Aint ep_strlen = 0;
    MPI_Aint *_ep_strs_sizes = NULL;    // only at root
    MPI_Aint _ep_strs_total_size = 0;   // only at root
    MPI_Aint *displs = NULL;    // only at root
    int mpi_error = MPI_SUCCESS;
    MPIR_Errflag_t errflag = FALSE;

    /* Create the new socket for the intercom and listen on it */
    {
        pscom_err_t rc;
#if MPID_PSP_HAVE_PSCOM_ABI_5
        uint64_t flags = PSCOM_SOCK_FLAG_INTER_JOB;
        socket_new = pscom_open_socket(0, 0, MPIDI_Process.my_pg_rank, flags);
#else
        socket_new = pscom_open_socket(0, 0);
#endif
        MPIR_ERR_CHKANDJUMP(!socket_new, mpi_error, MPI_ERR_OTHER, "**psp|opensocket");
        {
            char name[10];
            /* We have to provide a socket name that is locally unique
             * (e.g. for retrieving the right connection via pscom_ondemand_find_con)
             * and that in addition is distinct with respect to remote socket names in other PGs
             * (e.g. for distinguishing between direct/indirect connect in pscom_ondemand_write_start).
             * Local PG id plus local PG rank would be applicable here, however, we are limited in the number of chars.
             * So, for the debug case, we want some kind of readable format whereas for the non-debug case, we adjust the digits used:
             */
            if (MPIDI_Process.env.debug_level) {
                snprintf(name, sizeof(name), "i%03ur%03u", MPIDI_Process.my_pg->id_num % 1000,
                         MPIDI_Process.my_pg_rank % 1000);
            } else {
                int rank_range = 1;
                int pg_id_mod = 1;
                int pg_size = MPIDI_Process.my_pg_size;
                while (pg_size >>= 4)
                    rank_range++;
                pg_id_mod = 1 << (8 - rank_range) * 4;
                snprintf(name, sizeof(name), "%0*x%0*x", 8 - rank_range,
                         MPIDI_Process.my_pg->id_num % pg_id_mod, rank_range,
                         MPIDI_Process.my_pg_rank);
            }
            pscom_socket_set_name(socket_new, name);
        }

        rc = pscom_listen(socket_new, PSCOM_ANYPORT);
        /* ToDo: Graceful shutdown in case of error */
        MPIR_ERR_CHKANDSTMT1((rc != PSCOM_SUCCESS), mpi_error, MPI_ERR_OTHER, _exit(1),
                             "**psp|listen_anyport", "**psp|listen_anyport %s", pscom_err_str(rc));

#if MPID_PSP_HAVE_PSCOM_ABI_5
        char *_ep_str = NULL;
        rc = pscom_socket_get_ep_str(socket_new, &_ep_str);
        /* ToDo: Graceful shutdown in case of error */
        MPIR_ERR_CHKANDSTMT1(rc != PSCOM_SUCCESS, mpi_error, MPI_ERR_OTHER, _exit(1),
                             "**psp|getepstr", "**psp|getepstr %s", pscom_err_str(rc));
        ep_str = MPL_strdup(_ep_str);
        pscom_socket_free_ep_str(_ep_str);
#else
        ep_str = MPL_strdup(pscom_listen_socket_ondemand_str(socket_new));
#endif
        MPIR_ERR_CHKANDJUMP1(!ep_str, mpi_error, MPI_ERR_OTHER, "**psp|nullendpoint",
                             "**psp|nullendpoint %s", socket_new->local_con_info.name);
        ep_strlen = strlen(ep_str) + 1; /* +1 to account for NULL terminator */

        intercomm->pscom_socket = socket_new;
    }

    if (iam_root(root, comm)) {
        _ep_strs_sizes = (MPI_Aint *) MPL_calloc(local_size, sizeof(MPI_Aint), MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!_ep_strs_sizes, mpi_error, MPI_ERR_OTHER, "**nomem");
        displs = (MPI_Aint *) MPL_calloc(local_size, sizeof(MPI_Aint), MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!displs, mpi_error, MPI_ERR_OTHER, "**nomem");
    }

    /* Gather size of all ep strings from ranks in comm */
    mpi_error = MPID_Gather((void *) &ep_strlen, 1, MPI_AINT, (void *) _ep_strs_sizes, 1,
                            MPI_AINT, root, comm, errflag);
    MPIR_ERR_CHECK(mpi_error);
    MPIR_Assert(errflag == FALSE);

    if (iam_root(root, comm)) {
        /* Calculate displacement vector and allocate contiguous memory block for ep strings */
        for (int i = 0; i < local_size; i++) {
            if (i == 0) {
                displs[i] = 0;
            } else {
                displs[i] = _ep_strs_sizes[i - 1] + displs[i - 1];
            }
            _ep_strs_total_size += _ep_strs_sizes[i];
        }

        MPIR_Assert(_ep_strs_total_size > 0);
        _ep_strs = (char *) MPL_calloc(_ep_strs_total_size, sizeof(char), MPL_MEM_STRINGS);
        MPIR_ERR_CHKANDJUMP(!_ep_strs, mpi_error, MPI_ERR_OTHER, "**nomem");
    }

    /* Gather all ep strings from ranks in comm */
    mpi_error = MPID_Gatherv((void *) ep_str, ep_strlen, MPI_CHAR, (void *) _ep_strs,
                             _ep_strs_sizes, displs, MPI_CHAR, root, comm, errflag);
    MPIR_ERR_CHECK(mpi_error);
    MPIR_Assert(errflag == FALSE);

#if 0
    /* ToDo: Debug */
    if (iam_root(root, comm)) {
        int i;
        for (i = 0; i < local_size; i++) {
            printf("#%03u connect: %s\n", i, _ep_strs_sizes[i]);
        }
    }
#endif

    *ep_strs = _ep_strs;
    *ep_strs_sizes = _ep_strs_sizes;
    *ep_strs_total_size = _ep_strs_total_size;

  fn_exit:
    MPL_free(ep_str);
    MPL_free(displs);
    return mpi_error;
  fn_fail:
    goto fn_exit;
}


/*@
   MPID_Open_port - Open an MPI Port

   Input Arguments:
.  MPI_Info info - info

   Output Arguments:
.  char port_name[MPI_MAX_PORT_NAME] - port name

   Notes:

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_OTHER
@*/
int MPID_Open_port(MPIR_Info * info_ptr, char *port_name)
{
    int mpi_error = MPI_SUCCESS;
    static unsigned portnum = 0;
    int rc;
    int tcp_enabled = 1;
    pscom_socket_t *socket = NULL;

#if MPID_PSP_HAVE_PSCOM_ABI_5
    uint64_t flags = PSCOM_SOCK_FLAG_INTER_JOB;
    socket = pscom_open_socket(0, 0, MPIDI_Process.my_pg_rank, flags);
#else
    socket = pscom_open_socket(0, 0);
#endif
    MPIR_ERR_CHKANDJUMP(!socket, mpi_error, MPI_ERR_OTHER, "**psp|opensocket");

    {
        char name[10];
        snprintf(name, sizeof(name), "int%05u", (unsigned) portnum);
        pscom_socket_set_name(socket, name);
        portnum++;
    }

    /* Allow TCP only. ToDo: Allow RDP connects when they are implemented */
    /* If TCP plugin is disabled (no pscom payload via TCP), we cannot enforce TCP... */
    tcp_enabled = MPIDI_PSP_env_get_int("PSP_TCP", 1);
    if (tcp_enabled)
        pscom_con_type_mask_only(socket, PSCOM_CON_TYPE_TCP);

    rc = pscom_listen(socket, PSCOM_ANYPORT);
    /* ToDo: Graceful shutdown in case of error */
    MPIR_ERR_CHKANDSTMT1((rc != PSCOM_SUCCESS), mpi_error, MPI_ERR_OTHER, _exit(1),
                         "**psp|listen_anyport", "**psp|listen_anyport %s", pscom_err_str(rc));

#if MPID_PSP_HAVE_PSCOM_ABI_5
    char *ep_str = NULL;
    rc = pscom_socket_get_ep_str(socket, &ep_str);
    /* ToDo: Graceful shutdown in case of error */
    MPIR_ERR_CHKANDSTMT1(rc != PSCOM_SUCCESS, mpi_error, MPI_ERR_OTHER, _exit(1),
                         "**psp|getepstr", "**psp|getepstr %s", pscom_err_str(rc));
#else
    const char *ep_str = NULL;
    ep_str = pscom_listen_socket_str(socket);
#endif
    MPIR_ERR_CHKANDJUMP1(!ep_str, mpi_error, MPI_ERR_OTHER, "**psp|nullendpoint",
                         "**psp|nullendpoint %s", socket->local_con_info.name);

    /* Check if endpoint string exceeds length MPI_MAX_PORT_NAME */
    MPIR_ERR_CHKANDJUMP1(strlen(ep_str) > MPI_MAX_PORT_NAME, mpi_error, MPI_ERR_OTHER,
                         "**psp|endpointlength", "**psp|endpointlength %s", ep_str);

    inter_sockets_add(socket);

    strcpy(port_name, ep_str);
    /* Typical ch3 {port_name}s: */
    /* First  MPI_Open_port: "<tag#0$description#phoenix$port#55364$ifname#192.168.254.21$>" */
    /* Second MPI_Open_port: "<tag#1$description#phoenix$port#55364$ifname#192.168.254.21$>" */

#if MPID_PSP_HAVE_PSCOM_ABI_5
    pscom_socket_free_ep_str(ep_str);
#endif

  fn_exit:
    return mpi_error;
  fn_fail:
    goto fn_exit;
}


/*@
   MPID_Close_port - Close port

   Input Parameter:
.  port_name - Name of MPI port to close

   Notes:

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_OTHER

@*/
int MPID_Close_port(const char *port_name)
{
    int mpi_errno = MPI_SUCCESS;
    pscom_socket_t *socket = NULL;

    /* printf("%s(port_name:\"%s\")\n", __func__, port_name); */
    mpi_errno = inter_sockets_get_by_ep_str(port_name, &socket);
    MPIR_ERR_CHECK(mpi_errno);

    if (socket) {
        inter_sockets_del_by_socket(socket);
        pscom_close_socket(socket);
    }
  fn_exit:
    return MPI_SUCCESS;
  fn_fail:
    goto fn_exit;
}


static
MPIR_Comm *create_intercomm(MPIR_Comm * comm)
{
    MPIR_Comm *intercomm;
    MPIR_Context_id_t recvcontext_id = MPIR_INVALID_CONTEXT_ID;

    int mpi_errno = MPIR_Comm_create(&intercomm);
    MPIR_Assert(mpi_errno == MPI_SUCCESS);

    mpi_errno = MPIR_Get_contextid_sparse(comm, &recvcontext_id, FALSE);
    MPIR_Assert(mpi_errno == MPI_SUCCESS);

    intercomm->context_id = MPIR_INVALID_CONTEXT_ID;    /* finally set in init_intercomm() to recvcontext_id of the remote */
    intercomm->recvcontext_id = recvcontext_id;

    MPIR_Comm_set_session_ptr(intercomm, comm->session_ptr);

    return intercomm;
}


static
void warmup_intercomm_send(MPIR_Comm * comm)
{
    int i;
    if (!MPIDI_Process.env.enable_direct_connect_spawn)
        return;

    for (i = 0; i < comm->remote_size; i++) {
        int rank = (i + comm->rank) % comm->remote_size;        /* destination rank */
        /* printf("#S%d: Send #%d to #%d ctx:%u rctx:%u\n",
         * comm->rank, comm->rank, rank, comm->context_id, comm->recvcontext_id); */
        pscom_connection_t *con = MPID_PSCOM_rank2connection(comm, rank);
        MPIDI_PSP_SendCtrl(MPIDI_PSP_CTRL_TAG__WARMUP__PING /* tag */ , comm->context_id,
                           comm->rank /* src_rank */ ,
                           con, MPID_PSP_MSGTYPE_DATA_ACK);
        MPIDI_PSP_RecvCtrl(MPIDI_PSP_CTRL_TAG__WARMUP__PONG /* tag */ , comm->recvcontext_id,
                           rank /* src_rank */ ,
                           con, MPID_PSP_MSGTYPE_DATA_ACK);
    }
}


static
void warmup_intercomm_recv(MPIR_Comm * comm)
{
    int i;
    if (!MPIDI_Process.env.enable_direct_connect_spawn)
        return;

    for (i = 0; i < comm->remote_size; i++) {
        int rank = (comm->remote_size - i + comm->rank) % comm->remote_size;    /* source rank */
        /* printf("#R%d: Recv #%d to #%d ctx:%u rctx:%u\n",
         * comm->rank, rank, comm->rank, comm->context_id, comm->recvcontext_id); */
        pscom_connection_t *con = MPID_PSCOM_rank2connection(comm, rank);
        MPIDI_PSP_RecvCtrl(MPIDI_PSP_CTRL_TAG__WARMUP__PING /* tag */ , comm->recvcontext_id,
                           rank /* src_rank */ ,
                           con, MPID_PSP_MSGTYPE_DATA_ACK);
        MPIDI_PSP_SendCtrl(MPIDI_PSP_CTRL_TAG__WARMUP__PONG /* tag */ , comm->context_id,
                           comm->rank /* src_rank */ ,
                           con, MPID_PSP_MSGTYPE_DATA_ACK);
    }
}


/*@
   MPID_Comm_accept - MPID entry point for MPI_Comm_accept

   Input Parameters:
+  port_name - port name
.  info - info
.  root - root
-  comm - communicator

   Output Parameters:
.  MPI_Comm *_intercomm - new communicator

  Return Value:
  'MPI_SUCCESS' or a valid MPI error code.
@*/
int MPID_Comm_accept(const char *port_name, MPIR_Info * info, int root,
                     MPIR_Comm * comm, MPIR_Comm ** _intercomm)
{
    int mpi_error = MPI_SUCCESS;
    MPIR_Comm *intercomm = create_intercomm(comm);
    char *ep_strs = NULL;
    MPI_Aint *ep_strs_sizes = NULL;
    MPI_Aint ep_strs_total_size = 0;
    MPIR_Errflag_t errflag = FALSE;

    mpi_error = MPID_PSP_open_all_sockets(root, comm, intercomm, &ep_strs, &ep_strs_sizes,
                                          &ep_strs_total_size);
    MPIR_ERR_CHECK(mpi_error);

    if (iam_root(root, comm)) {
        pscom_socket_t *socket = NULL;
        pscom_connection_t *con;
        mpi_error = inter_sockets_get_by_ep_str(port_name, &socket);
        MPIR_ERR_CHECK(mpi_error);
        MPIR_ERR_CHKANDJUMP(!socket, mpi_error, MPI_ERR_OTHER, "**psp|opensocket");

        /* Wait for a connection on this socket */
        while (1) {
            con = pscom_get_next_connection(socket, NULL);
            if (con)
                break;

            pscom_wait_any();
        }

        mpi_error =
            forward_pg_info(con, comm, root, ep_strs, ep_strs_sizes, ep_strs_total_size, intercomm);

        inter_barrier(con);
        pscom_flush(con);
        pscom_close_connection(con);

    } else {
        mpi_error =
            forward_pg_info(NULL, comm, root, ep_strs, ep_strs_sizes, ep_strs_total_size,
                            intercomm);
    }

    MPL_free(ep_strs);
    MPL_free(ep_strs_sizes);

    MPIR_ERR_CHECK(mpi_error);

    /* Workaround for timing of pscom ondemand connections. Be
     * sure both sides have called pscom_connect before
     * using the connections. step 3 of 3 */
    mpi_error = MPIR_Barrier_impl(comm, errflag);
    MPIR_ERR_CHECK(mpi_error);

    *_intercomm = intercomm;
    warmup_intercomm_recv(intercomm);

    /* the accepting rank is in the high group */
    intercomm->is_low_group = 0;
  fn_exit:
    return mpi_error;
  fn_fail:
    goto fn_exit;
}


/*@
   MPID_Comm_connect - MPID entry point for MPI_Comm_connect

   Input Parameters:
+  port_name - port name
.  info - info
.  root - root
-  comm - communicator

   Output Parameters:
.  newcomm_ptr - new intercommunicator

  Return Value:
  'MPI_SUCCESS' or a valid MPI error code.
@*/
int MPID_Comm_connect(const char *port_name, MPIR_Info * info, int root,
                      MPIR_Comm * comm, MPIR_Comm ** _intercomm)
{
    int mpi_error = MPI_SUCCESS;
    MPIR_Comm *intercomm = create_intercomm(comm);
    char *ep_strs = NULL;
    MPI_Aint *ep_strs_sizes = NULL;
    MPI_Aint ep_strs_total_size = 0;
    MPIR_Errflag_t errflag = FALSE;

    mpi_error = MPID_PSP_open_all_sockets(root, comm, intercomm, &ep_strs, &ep_strs_sizes,
                                          &ep_strs_total_size);
    MPIR_ERR_CHECK(mpi_error);

    if (iam_root(root, comm)) {
        pscom_socket_t *socket = NULL;
        pscom_connection_t *con = NULL;
        pscom_err_t rc;
        int con_failed;
#if MPID_PSP_HAVE_PSCOM_ABI_5
        uint64_t socket_flags = PSCOM_SOCK_FLAG_INTER_JOB;
        socket = pscom_open_socket(0, 0, MPIDI_Process.my_pg_rank, socket_flags);
#else
        socket = pscom_open_socket(0, 0);
#endif
        MPIR_ERR_CHKANDJUMP(!socket, mpi_error, MPI_ERR_OTHER, "**psp|opensocket");
        con = pscom_open_connection(socket);
        MPIR_ERR_CHKANDJUMP(!con, mpi_error, MPI_ERR_OTHER, "**psp|openconn");

#if MPID_PSP_HAVE_PSCOM_ABI_5
        uint64_t conn_flags = PSCOM_CON_FLAG_DIRECT;
        rc = pscom_connect(con, port_name, PSCOM_RANK_UNDEFINED, conn_flags);
#else
        rc = pscom_connect_socket_str(con, port_name);
#endif
        con_failed = (rc != PSCOM_SUCCESS);

        if (!con_failed) {
            mpi_error = forward_pg_info(con, comm, root, ep_strs, ep_strs_sizes,
                                        ep_strs_total_size, intercomm);
            inter_barrier(con);
            pscom_flush(con);
        } else {
            mpi_error = forward_pg_info(NULL, comm, root, ep_strs, ep_strs_sizes,
                                        ep_strs_total_size, intercomm);
        }
        pscom_close_connection(con);
        pscom_close_socket(socket);

    } else {
        mpi_error = forward_pg_info(NULL, comm, root, ep_strs, ep_strs_sizes,
                                    ep_strs_total_size, intercomm);
    }

    MPL_free(ep_strs);
    MPL_free(ep_strs_sizes);

    /* Workaround for timing of pscom ondemand connections. Be
     * sure both sides have called pscom_connect before
     * using the connections. step 3 of 3 */

    if (mpi_error == MPI_SUCCESS) {
        MPIR_Barrier_impl(comm, errflag);
        *_intercomm = intercomm;
        warmup_intercomm_send(intercomm);

        /* the connecting ranks are in the low group */
        intercomm->is_low_group = 1;
    } else {
        /* error. Release intercomm */
        MPID_Comm_disconnect(intercomm);
    }

  fn_exit:
    return mpi_error;
  fn_fail:
    goto fn_exit;
}


int MPID_Comm_disconnect(MPIR_Comm * comm_ptr)
{
    int mpi_errno;

    MPIR_Assert(comm_ptr);
    comm_ptr->is_disconnected = 1;
    mpi_errno = MPIR_Comm_release(comm_ptr);

    return mpi_errno;
}

#define PARENT_EP_STR_KVSKEY "PARENT_ROOT_EP_STR_NAME"
#define MPIDI_MAX_KVS_VALUE_LEN    4096

/* Name of parent endpoint string if this process was spawned (and is root of comm world) or null */
static char parent_ep_str[MPIDI_MAX_KVS_VALUE_LEN] = { 0 };


int MPID_PSP_Get_parent_ep_str(char **ep_str)
{
    if (!parent_ep_str[0]) {
        MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
        MPIR_pmi_kvs_parent_get(PARENT_EP_STR_KVSKEY, parent_ep_str, sizeof(parent_ep_str));
        MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    }

    MPIR_Assert(ep_str != NULL);
    if (parent_ep_str[0]) {
        *ep_str = parent_ep_str;
    } else {
        *ep_str = NULL;
    }

    return MPI_SUCCESS;
}

static
int count_total_processes(int count, const int maxprocs[])
{
    int total_num_processes = 0;
    int i;
    for (i = 0; i < count; i++) {
        total_num_processes += maxprocs[i];
    }
    return total_num_processes;
}



/* FIXME: Correct description of function */
/*@
   MPID_Comm_spawn_multiple -

   Input Arguments:
+  int count - count
.  char *array_of_commands[] - commands
.  char* *array_of_argv[] - arguments
.  int array_of_maxprocs[] - maxprocs
.  MPI_Info array_of_info[] - infos
.  int root - root
-  MPI_Comm comm - communicator

   Output Arguments:
+  MPI_Comm *intercomm - intercommunicator
-  int array_of_errcodes[] - error codes

   Notes:

.N Errors
.N MPI_SUCCESS
@*/
int MPID_Comm_spawn_multiple(int count, char *array_of_commands[],
                             char **array_of_argv[], const int array_of_maxprocs[],
                             MPIR_Info * array_of_info_ptrs[], int root,
                             MPIR_Comm * comm_ptr, MPIR_Comm ** intercomm, int array_of_errcodes[])
{
    int mpi_errno = MPI_SUCCESS;
    int *pmi_errcodes = NULL;
    char ep_str[MPI_MAX_PORT_NAME];
    int total_num_processes = 0;
    int should_accept = 1;

    /*
     * printf("%s:%u:%s Spawn from context_id: %u\n", __FILE__, __LINE__, __func__, comm_ptr->context_id);
     */
    /* Open a socket for spawned processes to connect to */
    mpi_errno = MPID_Open_port(NULL, ep_str);
    MPIR_ERR_CHECK(mpi_errno);

    if (comm_ptr->rank == root) {
        int i;
        total_num_processes = count_total_processes(count, array_of_maxprocs);
        struct MPIR_PMI_KEYVAL preput_keyval_vector;
        preput_keyval_vector.key = PARENT_EP_STR_KVSKEY;
        preput_keyval_vector.val = ep_str;

        /* create an array for the pmi error codes */
        pmi_errcodes = (int *) MPL_malloc(sizeof(int) * total_num_processes, MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!pmi_errcodes, mpi_errno, MPI_ERR_OTHER, "**nomem");

        mpi_errno = MPIR_pmi_spawn_multiple(count,
                                            array_of_commands,
                                            array_of_argv,
                                            array_of_maxprocs,
                                            array_of_info_ptrs, 1, &preput_keyval_vector,
                                            pmi_errcodes, NULL);
        if (mpi_errno != MPI_SUCCESS) {
            char errstr[MPI_MAX_ERROR_STRING];
            int len = 0;
            /* We should not accept if MPIR_pmi_spawn_multiple returns an error.
             * Do not jump to fn_fail here, but inform all other processes that
             * something went wrong via bcast of should_accept (see below).
             * Print an error message here because mpi_errno gets overwritten by
             * the bcast below. */
            should_accept = 0;
            MPIR_Error_string_impl(mpi_errno, errstr, &len);
            fprintf(stderr, "Error: Spawn failed.\n%s\n", errstr);
        }

        /* FIXME: translate the pmi error codes here */
        if (array_of_errcodes != MPI_ERRCODES_IGNORE) {
            memcpy(array_of_errcodes, pmi_errcodes, sizeof(int) * total_num_processes);
        }

        /* Only if no general spawn error occurred, check pmi_errcodes */
        if (should_accept) {
            for (i = 0; i < total_num_processes; i++) {
                /* We want to accept if any of the spawns succeeded.
                 * Alternatively, this is the same as we want to NOT accept if
                 * all of them failed. should_accept = NAND(e_0, ..., e_n)
                 * Remember, success equals false (e_x == 0). */
                should_accept = should_accept && pmi_errcodes[i];
            }
            should_accept = !should_accept;     /* the `N' in NAND */
        }
        /*
         * printf("%s:%u:%s Spawn done\n", __FILE__, __LINE__, __func__);
         */
    }
    /* root */

    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    mpi_errno = MPIR_Bcast(&should_accept, 1, MPI_INT, root, comm_ptr, errflag);
    MPIR_ERR_CHECK(mpi_errno);
    MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    if (array_of_errcodes != MPI_ERRCODES_IGNORE) {
        mpi_errno = MPIR_Bcast(&total_num_processes, 1, MPI_INT, root, comm_ptr, errflag);
        MPIR_ERR_CHECK(mpi_errno);
        MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

        mpi_errno =
            MPIR_Bcast(array_of_errcodes, total_num_processes, MPI_INT, root, comm_ptr, errflag);
        MPIR_ERR_CHECK(mpi_errno);
        MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");
    }

    if (should_accept) {
        mpi_errno = MPID_Comm_accept(ep_str, NULL, root, comm_ptr, intercomm);
        MPIR_ERR_CHECK(mpi_errno);
        MPIR_Assert(*intercomm != NULL);
    } else {
        /* spawn failed, return error */
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**spawn");
    }

    mpi_errno = MPID_Close_port(ep_str);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    if (pmi_errcodes) {
        MPL_free(pmi_errcodes);
    }
    return mpi_errno;

  fn_fail:
    if (*intercomm != NULL) {
        MPIR_Comm_free_impl(*intercomm);
    }
    goto fn_exit;
}
