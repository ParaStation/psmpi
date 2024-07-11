/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2024 ParTec AG, Munich
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
    port = pscom_listen_socket_str() // Socket named "int%05u".
    pscom_inter_sockets_add(port, inter_socket);

MPID_Comm_accept(root_port)
    // root_port undefined at ranks != root !!!
    open_all_ports

    @root:
	// all_ports are collected in open_all_ports
	remote_root = pscom_accept(inter_socket)

---------------------------------------------------------------------------------------------------
	pscom_send(inter_socket, all_ports, remote_context_id, remote_size) (send_all_ports_remote)
	pscom_recv(inter_socket, all_ports, remote_context_id, remote_size) (recv_all_ports_remote)

    connect_all_ports
=========================== REPLACED BY:

    forward_pg_info(con, comm, root, all_ports, intercomm) --> MPID_PG_ForwardPGInfo(...)
---------------------------------------------------------------------------------------------------

    @root: barrier with remote root

    MPIR_Barrier_impl(comm) // assure all remote ranks are connected

MPID_Comm_connect()
    open_all_ports

    @root:
	// all_ports are collected in open_all_ports
	remote_root = pscom_connect(inter_socket, port)
---------------------------------------------------------------------------------------------------
	pscom_send(inter_socket, all_ports, remote_context_id, remote_size) (send_all_ports_remote)
	pscom_recv(inter_socket, all_ports, remote_context_id, remote_size) (recv_all_ports_remote)

    connect_all_ports
=========================== REPLACED BY:

    forward_pg_info(con, comm, root, all_ports, intercomm) --> MPID_PG_ForwardPGInfo(...)
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
    MPIR_pmi_Spawn_multiple(port@root)

  MPID_Comm_accept
  MPID_Close_port


Child
-------------------
MPI_Init
  MPID_PSP_GetParentPort
  MPID_Comm_connect


Helper
---------------
open_all_ports(root)
    port = pscom_listen_socket_ondemand_str;
    MPI_Gather port -> all_ports@root

connect_all_ports(root)
    MPI_Bcast all_ports from root
    for p in all_ports:
       pscom_connect_socket_str(p)
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


#define PSCOM_INTER_SOCKETS_MAX 1024

/*
 * Inter sockets
 *
 * Mapping between a port_str from MPID_Open_port and a pscom_socket.
 */
typedef struct pscom_inter_socket {
    pscom_port_str_t port_str;
    pscom_socket_t *pscom_socket;
} pscom_inter_socket_t;

static
pscom_inter_socket_t pscom_inter_sockets[PSCOM_INTER_SOCKETS_MAX];

static
void pscom_inter_sockets_add(const pscom_port_str_t port_str, pscom_socket_t * pscom_socket)
{
    int i;
    for (i = 0; i < PSCOM_INTER_SOCKETS_MAX; i++) {
        if (pscom_inter_sockets[i].pscom_socket == NULL) {
            strcpy(pscom_inter_sockets[i].port_str, port_str);
            pscom_inter_sockets[i].pscom_socket = pscom_socket;
            return;
        }
    }
    fprintf(stderr, "To many open ports (More than %d calls to MPI_Open_port())\n",
            PSCOM_INTER_SOCKETS_MAX);
    _exit(1);   /* ToDo: Graceful shutdown */
}


static
pscom_socket_t *pscom_inter_sockets_get_by_port_str(const pscom_port_str_t port_str)
{
    int i;
    for (i = 0; i < PSCOM_INTER_SOCKETS_MAX; i++) {
        if (!strcmp(pscom_inter_sockets[i].port_str, port_str)) {
            return pscom_inter_sockets[i].pscom_socket;
        }
    }
    return NULL;
}


static
void pscom_inter_sockets_del_by_pscom_socket(pscom_socket_t * pscom_socket)
{
    int i;
    for (i = 0; i < PSCOM_INTER_SOCKETS_MAX; i++) {
        if (pscom_inter_sockets[i].pscom_socket == pscom_socket) {
            pscom_inter_sockets[i].pscom_socket = NULL;
            pscom_inter_sockets[i].port_str[0] = 0;
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

    MPIR_Comm_set_session_ptr(intercomm, comm->session_ptr);

    /* Point local vcr at those of incoming intracommunicator */
    vcrt = MPIDI_VCRT_Dup(comm->vcrt);
    assert(vcrt);
    MPID_PSP_comm_set_local_vcrt(intercomm, vcrt);

    if (create_vcrt_flag) {
        vcrt = MPIDI_VCRT_Create(intercomm->remote_size);
        assert(vcrt);
        MPID_PSP_comm_set_vcrt(intercomm, vcrt);
    }

    /* MPIDI_VCR_Initialize() will be called later for every intercomm->remote_size rank */

    mpi_errno = MPIR_Comm_commit(intercomm);
    assert(mpi_errno == MPI_SUCCESS);
}


/*
 * helper
 */

static
int iam_root(int root, MPIR_Comm * comm)
{
    return comm->rank == root;
}


static
pscom_port_str_t *alloc_all_ports(unsigned comm_size)
{
    return (pscom_port_str_t *) MPL_malloc(comm_size * sizeof(pscom_port_str_t), MPL_MEM_STRINGS);
}


static
void free_all_ports(pscom_port_str_t * all_ports)
{
    MPL_free(all_ports);
}

/*
 *  Tell the remote side about the local size plus the set of local GPIDs and receive the remote information in return (only root).
 *  Then distribute the new information among all local processes.
 *  Go with this information and the set of all local ports into the central MPID_PG_ForwardPGInfo() function for establishing
 *  all still missing connections while complementing the current view onto the PG topology.
 *  Finally, exchange the remote conext_id and build the new inter-communicator.
 *  (forward_pg_info() plus parts of MPID_PG_ForwardPGInfo() thus replace the former send_/recv_/connect_all_ports() functions...)
 */
static
int forward_pg_info(pscom_connection_t * con, MPIR_Comm * comm, int root,
                    pscom_port_str_t * all_ports, MPIR_Comm * intercomm)
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

    pscom_socket_t *pscom_socket = intercomm->pscom_socket;

    if (iam_root(root, comm) && con) {
        pscom_send(con, NULL, 0, &local_size, sizeof(int));
        rc = pscom_recv_from(con, NULL, 0, &remote_size, sizeof(int));
        assert(rc == PSCOM_SUCCESS);
    }

    mpi_errno = MPIR_Bcast(&remote_size, 1, MPI_INT, root, comm, &errflag);
    assert(mpi_errno == MPI_SUCCESS);

    if (remote_size == 0)
        goto err_failed;        /* this happens if root has no valid 'con' (see above!) */

    local_gpids = (MPIDI_Gpid *) MPL_malloc(local_size * sizeof(MPIDI_Gpid), MPL_MEM_OBJECT);
    remote_gpids = (MPIDI_Gpid *) MPL_malloc(remote_size * sizeof(MPIDI_Gpid), MPL_MEM_OBJECT);

    MPIDI_GPID_GetAllInComm(comm, local_size, local_gpids, NULL);

    if (iam_root(root, comm)) {
        pscom_send(con, NULL, 0, local_gpids, local_size * sizeof(MPIDI_Gpid));
        rc = pscom_recv_from(con, NULL, 0, remote_gpids, remote_size * sizeof(MPIDI_Gpid));
        assert(rc == PSCOM_SUCCESS);
    }

    mpi_errno =
        MPIR_Bcast(remote_gpids, remote_size * sizeof(MPIDI_Gpid), MPI_CHAR, root, comm, &errflag);
    assert(mpi_errno == MPI_SUCCESS);


    /* Call the central routine for establishing all missing connections: */
    MPIDI_PG_ForwardPGInfo(NULL, comm, remote_size, remote_gpids, root, -1, -1, con,
                           (char *) all_ports, pscom_socket);


    /* distribute remote values */
    if (iam_root(root, comm)) {
        local_context_id = intercomm->recvcontext_id;
        pscom_send(con, NULL, 0, &local_context_id, sizeof(int));
        rc = pscom_recv_from(con, NULL, 0, &remote_context_id, sizeof(int));
        assert(rc == PSCOM_SUCCESS);
    }
    local_context_id = intercomm->context_id;
    mpi_errno = MPIR_Bcast(&local_context_id, 1, MPI_INT, root, comm, &errflag);
    assert(mpi_errno == MPI_SUCCESS);
    mpi_errno = MPIR_Bcast(&remote_context_id, 1, MPI_INT, root, comm, &errflag);
    assert(mpi_errno == MPI_SUCCESS);

    if (!iam_root(root, comm)) {
        /* assure equal local context_id on all ranks */
        MPIR_Context_id_t context_id = local_context_id;
        assert(context_id == intercomm->context_id);
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
     * sure both sides have called pscom_connect_socket_str before
     * using the connections. step 2 of 3 */
    pscom_send(con, NULL, 0, &dummy, sizeof(dummy));

    rc = pscom_recv_from(con, NULL, 0, &dummy, sizeof(dummy));
    assert(rc == PSCOM_SUCCESS);
}


pscom_port_str_t *MPID_PSP_open_all_ports(int root, MPIR_Comm * comm, MPIR_Comm * intercomm)
{
    pscom_socket_t *socket_new;
    int local_size = comm->local_size;
    pscom_port_str_t *all_ports = NULL;
    pscom_port_str_t my_port;
    MPIR_Errflag_t err;
    int mpi_error = MPI_SUCCESS;

    /* Create the new socket for the intercom and listen on it */
    {
        pscom_err_t rc;
        socket_new = pscom_open_socket(0, 0);
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

        memset(my_port, 0, sizeof(pscom_port_str_t));
        strcpy(my_port, pscom_listen_socket_ondemand_str(socket_new));

        intercomm->pscom_socket = socket_new;
    }

    if (iam_root(root, comm)) {
        all_ports = alloc_all_ports(local_size);
    }

    err = FALSE;
    mpi_error = MPIR_Gather_allcomm_auto(my_port, sizeof(pscom_port_str_t), MPI_CHAR,
                                         all_ports, sizeof(pscom_port_str_t), MPI_CHAR,
                                         root, comm, &err);

    assert(mpi_error == MPI_SUCCESS);
    assert(err == MPI_SUCCESS);

#if 0
    /* ToDo: Debug */
    if (iam_root(root, comm)) {
        int i;
        for (i = 0; i < local_size; i++) {
            printf("#%03u connect: %s\n", i, all_ports[i]);
        }
    }
#endif

    return all_ports;
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
    const char *port_str;
    int tcp_enabled = 1;

    pscom_socket_t *socket = pscom_open_socket(0, 0);
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

    port_str = pscom_listen_socket_str(socket);
    pscom_inter_sockets_add(port_str, socket);

    strcpy(port_name, port_str);
    /* Typical ch3 {port_name}s: */
    /* First  MPI_Open_port: "<tag#0$description#phoenix$port#55364$ifname#192.168.254.21$>" */
    /* Second MPI_Open_port: "<tag#1$description#phoenix$port#55364$ifname#192.168.254.21$>" */

    return mpi_error;
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
    /* printf("%s(port_name:\"%s\")\n", __func__, port_name); */
    pscom_socket_t *socket = pscom_inter_sockets_get_by_port_str(port_name);
    if (socket) {
        pscom_inter_sockets_del_by_pscom_socket(socket);
        pscom_close_socket(socket);
    }
    return MPI_SUCCESS;
}


static
MPIR_Comm *create_intercomm(MPIR_Comm * comm)
{
    MPIR_Comm *intercomm;
    MPIR_Context_id_t recvcontext_id = MPIR_INVALID_CONTEXT_ID;

    int mpi_errno = MPIR_Comm_create(&intercomm);
    assert(mpi_errno == MPI_SUCCESS);

    mpi_errno = MPIR_Get_contextid_sparse(comm, &recvcontext_id, FALSE);
    assert(mpi_errno == MPI_SUCCESS);

    intercomm->context_id = MPIR_INVALID_CONTEXT_ID;    /* finally set in init_intercomm() to recvcontext_id of the remote */
    intercomm->recvcontext_id = recvcontext_id;

    MPIR_Comm_set_session_ptr(intercomm, comm->session_ptr);

    return intercomm;
}


static
void warmup_intercomm_send(MPIR_Comm * comm)
{
    int i;
    if (MPIDI_Process.env.enable_ondemand_spawn)
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
    if (MPIDI_Process.env.enable_ondemand_spawn)
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
    MPIR_Comm *intercomm = create_intercomm(comm);
    pscom_port_str_t *all_ports = MPID_PSP_open_all_ports(root, comm, intercomm);
    MPIR_Errflag_t errflag = FALSE;

    if (iam_root(root, comm)) {
        pscom_socket_t *socket = pscom_inter_sockets_get_by_port_str(port_name);
        pscom_connection_t *con;

        /* Wait for a connection on this socket */
        while (1) {
            con = pscom_get_next_connection(socket, NULL);
            if (con)
                break;

            pscom_wait_any();
        }

        forward_pg_info(con, comm, root, all_ports, intercomm);

        inter_barrier(con);
        pscom_flush(con);
        pscom_close_connection(con);

    } else {
        forward_pg_info(NULL, comm, root, all_ports, intercomm);
    }

    free_all_ports(all_ports);
    all_ports = NULL;

    /* Workaround for timing of pscom ondemand connections. Be
     * sure both sides have called pscom_connect_socket_str before
     * using the connections. step 3 of 3 */
    MPIR_Barrier_impl(comm, &errflag);
    *_intercomm = intercomm;
    warmup_intercomm_recv(intercomm);

    /* the accepting rank is in the high group */
    intercomm->is_low_group = 0;
    return MPI_SUCCESS;
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
    MPIR_Comm *intercomm = create_intercomm(comm);
    pscom_port_str_t *all_ports = MPID_PSP_open_all_ports(root, comm, intercomm);
    MPIR_Errflag_t errflag = FALSE;
    int mpi_error;

    if (iam_root(root, comm)) {
        pscom_socket_t *socket = pscom_open_socket(0, 0);
        pscom_connection_t *con = pscom_open_connection(socket);
        pscom_err_t rc;
        int con_failed;

        rc = pscom_connect_socket_str(con, port_name);
        con_failed = (rc != PSCOM_SUCCESS);

        if (!con_failed) {
            mpi_error = forward_pg_info(con, comm, root, all_ports, intercomm);
            inter_barrier(con);
            pscom_flush(con);
        } else {
            mpi_error = forward_pg_info(NULL, comm, root, all_ports, intercomm);
        }
        pscom_close_connection(con);
        pscom_close_socket(socket);

    } else {
        mpi_error = forward_pg_info(NULL, comm, root, all_ports, intercomm);
    }

    free_all_ports(all_ports);
    all_ports = NULL;

    /* Workaround for timing of pscom ondemand connections. Be
     * sure both sides have called pscom_connect_socket_str before
     * using the connections. step 3 of 3 */

    if (mpi_error == MPI_SUCCESS) {
        MPIR_Barrier_impl(comm, &errflag);
        *_intercomm = intercomm;
        warmup_intercomm_send(intercomm);

        /* the connecting ranks are in the low group */
        intercomm->is_low_group = 1;
    } else {
        /* error. Release intercomm */
        MPID_Comm_disconnect(intercomm);
    }

    return mpi_error;
}


int MPID_Comm_disconnect(MPIR_Comm * comm_ptr)
{
    int mpi_errno;

    assert(comm_ptr);
    comm_ptr->is_disconnected = 1;
    mpi_errno = MPIR_Comm_release(comm_ptr);

    return mpi_errno;
}

#define PARENT_PORT_KVSKEY "PARENT_ROOT_PORT_NAME"
#define MPIDI_MAX_KVS_VALUE_LEN    4096

/* Name of parent port if this process was spawned (and is root of comm world) or null */
static char parent_port_name[MPIDI_MAX_KVS_VALUE_LEN] = { 0 };


int MPID_PSP_GetParentPort(char **parent_port)
{
    int mpi_errno = MPI_SUCCESS;

    if (!parent_port_name[0]) {
        MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
        mpi_errno =
            MPIR_pmi_kvs_parent_get(PARENT_PORT_KVSKEY, parent_port_name, sizeof(parent_port_name));
        MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
        MPIR_ERR_CHECK(mpi_errno);
    }

    *parent_port = parent_port_name;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
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
    char port_name[MPI_MAX_PORT_NAME];
    int total_num_processes = 0;
    int should_accept = 1;

    /*
     * printf("%s:%u:%s Spawn from context_id: %u\n", __FILE__, __LINE__, __func__, comm_ptr->context_id);
     */
    /* Open a port for spawned processes to connect to */
    mpi_errno = MPID_Open_port(NULL, port_name);
    MPIR_ERR_CHECK(mpi_errno);

    if (comm_ptr->rank == root) {
        int i;
        total_num_processes = count_total_processes(count, array_of_maxprocs);
        struct MPIR_PMI_KEYVAL preput_keyval_vector;
        preput_keyval_vector.key = PARENT_PORT_KVSKEY;
        preput_keyval_vector.val = port_name;

        /* create an array for the pmi error codes */
        pmi_errcodes = (int *) MPL_malloc(sizeof(int) * total_num_processes, MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!pmi_errcodes, mpi_errno, MPI_ERR_OTHER, "**nomem");

        mpi_errno = MPIR_pmi_spawn_multiple(count,
                                            array_of_commands,
                                            array_of_argv,
                                            array_of_maxprocs,
                                            array_of_info_ptrs, 1, &preput_keyval_vector,
                                            pmi_errcodes);
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
    mpi_errno = MPIR_Bcast(&should_accept, 1, MPI_INT, root, comm_ptr, &errflag);
    MPIR_ERR_CHECK(mpi_errno);
    MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    if (array_of_errcodes != MPI_ERRCODES_IGNORE) {
        mpi_errno = MPIR_Bcast(&total_num_processes, 1, MPI_INT, root, comm_ptr, &errflag);
        MPIR_ERR_CHECK(mpi_errno);
        MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

        mpi_errno =
            MPIR_Bcast(array_of_errcodes, total_num_processes, MPI_INT, root, comm_ptr, &errflag);
        MPIR_ERR_CHECK(mpi_errno);
        MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");
    }

    if (should_accept) {
        mpi_errno = MPID_Comm_accept(port_name, NULL, root, comm_ptr, intercomm);
        MPIR_ERR_CHECK(mpi_errno);
        MPIR_Assert(*intercomm != NULL);
    } else {
        /* spawn failed, return error */
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**spawn");
    }

    mpi_errno = MPID_Close_port(port_name);
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
