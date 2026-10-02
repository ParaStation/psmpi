/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"
#include "uthash.h"     /* for hash function */
#include <unistd.h>
#include <sys/types.h>
#include "mpid_debug.h"


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


int MPID_PSP_open_all_sockets(char **ep_str_out, pscom_socket_t ** inter_job_socket_out)
{
    pscom_socket_t *socket_new = NULL;
    char *ep_str = NULL;
    int mpi_error = MPI_SUCCESS;

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
    }

    *ep_str_out = ep_str;
    *inter_job_socket_out = socket_new;

  fn_exit:
    return mpi_error;
  fn_fail:
    goto fn_exit;
}

/* Allow only TCP connection on socket by masking pscom connection types.
 *
 * This prevents that more than a plain TCP connection is established when the two
 * root processes of a spawn get in contact in order to exchange the endpoint
 * information of the other processes.
 *
 * Reason: There is no high-speed connection required for this exchange and it
 * makes no sense to first negotiate another/higher plug-in connection via TCP
 * and set it up only to exchange a small set of user data (i.e., the endpoint
 * information) and to terminate the connection directly afterwards.
 *
 * ToDo: Allow RDP connects when they are implemented
 */
static
int use_tcp_connection(pscom_socket_t * socket)
{
    int mpi_errno = MPI_SUCCESS;
    int tcp_enabled = 1;
    char *ep_str = NULL;
    int use_tcp_precon = 1;

    /* If TCP plugin is disabled (no pscom payload via TCP), we cannot enforce TCP... */
    tcp_enabled = MPIDI_PSP_env_get_int("PSP_TCP", 1);

    /* Check if we are using TCP as precon in pscom. We should not enforce using TCP for
     * payload communication if we are using RRComm because pscom's RRComm precon
     * and TCP plugin are mututal exclusive for now.
     * ToDo: Remove this constraint once both are supported simultaneously. */
    mpi_errno = MPIDI_PSP_socket_get_ep_str(MPIDI_Process.socket, &ep_str);
    MPIR_ERR_CHECK(mpi_errno);
    if (!ep_str) {
        /* RRComm provides no ep str */
        use_tcp_precon = 0;
    }

    if (tcp_enabled && use_tcp_precon) {
        pscom_con_type_mask_only(socket, PSCOM_CON_TYPE_TCP);
    }
  fn_exit:
    MPL_free(ep_str);
    return mpi_errno;
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
int MPID_Open_port(MPIR_Info * info_ptr, char *port_name, int len)
{
    int mpi_error = MPI_SUCCESS;
    static unsigned portnum = 0;
    int rc;
    pscom_socket_t *socket = NULL;

    MPIR_Assert(len >= MPI_MAX_PORT_NAME);

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

    mpi_error = use_tcp_connection(socket);
    MPIR_ERR_CHECK(mpi_error);

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
void warmup_intercomm_send(MPIR_Comm * comm)
{
    int i;
    if (!MPIDI_Process.env.enable_direct_connect_spawn)
        return;

    for (i = 0; i < comm->remote_size; i++) {
        int rank = (i + comm->rank) % comm->remote_size;        /* destination rank */
        /* printf("#S%d: Send #%d to #%d ctx:%u rctx:%u\n",
         * comm->rank, comm->rank, rank, comm->context_id, comm->recvcontext_id); */
        pscom_connection_t *con = NULL;
        MPIDI_PSP_comm_get_con(comm, rank, &con);
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
        pscom_connection_t *con = NULL;
        MPIDI_PSP_comm_get_con(comm, rank, &con);
        MPIDI_PSP_RecvCtrl(MPIDI_PSP_CTRL_TAG__WARMUP__PING /* tag */ , comm->recvcontext_id,
                           rank /* src_rank */ ,
                           con, MPID_PSP_MSGTYPE_DATA_ACK);
        MPIDI_PSP_SendCtrl(MPIDI_PSP_CTRL_TAG__WARMUP__PONG /* tag */ , comm->context_id,
                           comm->rank /* src_rank */ ,
                           con, MPID_PSP_MSGTYPE_DATA_ACK);
    }
}

static
int create_tag_from_port(const char *port_name, int *tag_out)
{
    int tag = 0;
    /* Use a hash function to create the tag */
    HASH_FNV(port_name, strlen(port_name), tag);

    *tag_out = tag;

    return MPI_SUCCESS;
}

/* Dynamic peer lpids are used for building inter communicators, such as MPID_Comm_connect/accept,
 * when we need temoprarily establish communication betweer peer group leaders.
 * The dynamic peer lpids are only used by a peer comm until the intercomm is committed.
 * */
static
MPIR_Lpid get_dyn_peer_lpid(void)
{
    MPIR_Lpid peer_lpid = MPIR_LPID_DYNAMIC_MASK | MPIDI_Process.next_dyn_peer_lpid;
    MPIDI_Process.next_dyn_peer_lpid++;

    return peer_lpid;
}

/* Establish a pscom connection to a peer process using the endpoint string port_name.
 * The resulting socket and connection can be used in a temporary peer comm to create
 * an intercomm */
static
int establish_peer_conn(const char *port_name, int is_sender, int timeout,
                        pscom_socket_t ** peer_socket, pscom_connection_t ** peer_con)
{
    int mpi_errno = MPI_SUCCESS;
    pscom_socket_t *new_socket = NULL;
    pscom_connection_t *new_con = NULL;
    pscom_err_t rc;
    int con_failed;

    if (is_sender) {    /* Comm connect - need a new inter-job socket and connection */
#if MPID_PSP_HAVE_PSCOM_ABI_5
        uint64_t socket_flags = PSCOM_SOCK_FLAG_INTER_JOB;
        new_socket = pscom_open_socket(0, 0, MPIDI_Process.my_pg_rank, socket_flags);
#else
        new_socket = pscom_open_socket(0, 0);
#endif
        MPIR_ERR_CHKANDJUMP(!new_socket, mpi_errno, MPI_ERR_OTHER, "**psp|opensocket");
        new_con = pscom_open_connection(new_socket);
        MPIR_ERR_CHKANDJUMP(!new_con, mpi_errno, MPI_ERR_OTHER, "**psp|openconn");

#if MPID_PSP_HAVE_PSCOM_ABI_5
        uint64_t conn_flags = PSCOM_CON_FLAG_DIRECT;
        rc = pscom_connect(new_con, port_name, PSCOM_RANK_UNDEFINED, conn_flags);
#else
        rc = pscom_connect_socket_str(new_con, port_name);
#endif
        con_failed = (rc != PSCOM_SUCCESS);
        MPIR_ERR_CHKANDJUMP(con_failed, mpi_errno, MPI_ERR_PORT, "**comm_connect_fail");
    } else {    /* Comm accept */
        mpi_errno = inter_sockets_get_by_ep_str(port_name, &new_socket);
        MPIR_ERR_CHECK(mpi_errno);
        MPIR_ERR_CHKANDJUMP(!new_socket, mpi_errno, MPI_ERR_OTHER, "**psp|opensocket");

        /* Wait for a connection on this socket */
        while (1) {
            new_con = pscom_get_next_connection(new_socket, NULL);
            if (new_con)
                break;

            pscom_wait_any();
        }

        con_failed = 0; /* TODO: Error handling on con accept side via timeout */
    }

    if (!con_failed) {
        /* Workaround for timing of pscom ondemand connections. */
        inter_barrier(new_con);
        pscom_flush(new_con);
    }


    *peer_socket = new_socket;
    *peer_con = new_con;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static
int dynamic_intercomm_create(const char *port_name, MPIR_Info * info, int root,
                             MPIR_Comm * comm_ptr, int timeout,
                             bool is_sender, MPIR_Comm ** intercomm)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Lpid remote_lpid = MPIR_LPID_INVALID;  /* root only */
    MPIR_Comm *peer_comm = NULL;        /* root only */
    pscom_socket_t *peer_socket = NULL; /* root only */
    pscom_connection_t *peer_conn = NULL;       /* root only */
    int peer_tag = 0; /* root only */ ;

    if (comm_ptr->rank == root) {
        /* create a tag from the provided port name */
        mpi_errno = create_tag_from_port(port_name, &peer_tag);
        MPIR_ERR_CHECK(mpi_errno);

        /* Create a pscom connection to the peer process */
        mpi_errno = establish_peer_conn(port_name, is_sender, timeout, &peer_socket, &peer_conn);
        MPIR_ERR_CHECK(mpi_errno);

        /* create peer intercomm - see ch4 dynamic_intercomm_create(...)
         * Since we will only use peer intercomm to call back MPID_Intercomm_exchange, which
         * just need to extract remote_lpid from the peer_comm, we can cheat a bit here - just
         * fill peer_comm->remote_group.
         */
        peer_comm = (MPIR_Comm *) MPIR_Handle_obj_alloc(&MPIR_Comm_mem);
        MPIR_ERR_CHKANDJUMP(!peer_comm, mpi_errno, MPI_ERR_OTHER, "**nomem");

        peer_comm->comm_kind = MPIR_COMM_KIND__INTERCOMM;
        peer_comm->remote_size = 1;
        peer_comm->local_size = 1;
        peer_comm->rank = 0;
        peer_comm->local_group = NULL;
        /* We have not exchanged context_id yet, set them to 0. This is okay since
         * the dynamic exchange is established between a pair of addresses (lpids) that
         * no other communications can happen yet. */
        peer_comm->context_id = 0;
        peer_comm->recvcontext_id = 0;
        peer_comm->ref_count = 1;       /* fake ref counter */

        /* set the peer socket and connection reference table in the peer comm */
        peer_comm->pscom_socket = peer_socket;

        MPIDI_VCRT_t *vcrt = MPIDI_VCRT_Create(1);
        MPIR_ERR_CHKANDJUMP(!vcrt, mpi_errno, MPI_ERR_OTHER, "**nomem");

        /* Set remote vcrt of peer comm */
        peer_comm->remote_vcrt = vcrt;
        peer_comm->remote_vcr = vcrt->vcr;

        /* Create a preliminary remote lpid that can be used to create a peer comm */
        remote_lpid = get_dyn_peer_lpid();

        peer_comm->remote_vcr[0] = MPIDI_VC_Create(NULL, MPIR_LPID_WORLD_RANK(remote_lpid),
                                                   peer_conn, remote_lpid);
        MPIR_ERR_CHKANDJUMP(!(peer_comm->remote_vcr[0]), mpi_errno, MPI_ERR_OTHER, "**nomem");

        /* Create the remote group of the peer comm */
        mpi_errno = MPIR_Group_create_stride(1, 0, NULL, remote_lpid, 1, &peer_comm->remote_group);
        MPIR_ERR_CHECK(mpi_errno);

      fn_fail:
        /* In case root fails, we bcast mpi_errno so other ranks will abort too */
        MPIR_Bcast_impl(&mpi_errno, 1, MPIR_INT_INTERNAL, root, comm_ptr, MPIR_COLL_ATTR_SYNC);
    } else {
        int root_errno;
        MPIR_Bcast_impl(&root_errno, 1, MPIR_INT_INTERNAL, root, comm_ptr, MPIR_COLL_ATTR_SYNC);
        if (root_errno) {
            MPIR_ERR_SET(mpi_errno, MPI_ERR_PORT, "**comm_connect_fail");
        }
    }

    if (mpi_errno == MPI_SUCCESS) {
        mpi_errno = MPIR_Intercomm_create_timeout(comm_ptr, root, peer_comm, 0, peer_tag, timeout,
                                                  intercomm);
    }

    if (comm_ptr->rank == root && peer_comm) {
        /* Destroy remote group */
        MPIR_Group_release(peer_comm->remote_group);

        if (peer_conn) {
            /* Close peer connection - needs to be done for both comm accept and connect */
            pscom_close_connection(peer_conn);
        }

        if (is_sender && peer_socket) {
            /* Close peer socket - needs to be done ony for comm connect.
             * For comm accept the socket was opened via MPID_Open_port and hence needs
             * to be closed via MPID_Close_port(). */
            pscom_close_socket(peer_socket);
        }

        /* Clean-up connection table in peer comm */
        if (peer_comm->remote_vcrt) {
            MPIDI_VC_t *peer_vcr = peer_comm->remote_vcr[0];
            MPIDI_VCRT_Release(peer_comm->remote_vcrt, 1);
            if (peer_vcr) {
                MPL_free(peer_vcr);
            }
        }

        /* destroy peer_comm */
        MPIR_Handle_obj_free(&MPIR_Comm_mem, peer_comm);
    }
    return mpi_errno;
}


/*@
   MPID_Comm_accept - MPID entry point for MPI_Comm_accept

   Input Parameters:
+  port_name - port name
.  info - info
.  root - root
-  comm - communicator

   Output Parameters:
.  MPI_Comm *newcomm_ptr - new inter-communicator

  Return Value:
  'MPI_SUCCESS' or a valid MPI error code.
@*/
int MPID_Comm_accept(const char *port_name, MPIR_Info * info, int root,
                     MPIR_Comm * comm, MPIR_Comm ** newcomm_ptr)
{
    int mpi_error = MPI_SUCCESS;

    MPIR_FUNC_ENTER;

    int timeout = 0;            /* TODO allow setting timeout via info and/or CVAR */
    bool is_sender = false;
    mpi_error = dynamic_intercomm_create(port_name, info, root, comm,
                                         timeout, is_sender, newcomm_ptr);
    MPIR_ERR_CHECK(mpi_error);

    /* Workaround for timing of pscom ondemand connections. Be
     * sure both sides have called pscom_connect before
     * using the connections. */
    warmup_intercomm_recv(*newcomm_ptr);

  fn_exit:
    MPIR_FUNC_EXIT;
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
                      MPIR_Comm * comm, MPIR_Comm ** newcomm_ptr)
{
    int mpi_error = MPI_SUCCESS;

    MPIR_FUNC_ENTER;

    bool is_sender = true;
    int timeout = 0;            /* TODO allow setting timeout via info and/or CVAR */

    mpi_error = dynamic_intercomm_create(port_name, info, root, comm,
                                         timeout, is_sender, newcomm_ptr);
    MPIR_ERR_CHECK(mpi_error);

    /* Workaround for timing of pscom ondemand connections. Be
     * sure both sides have called pscom_connect before
     * using the connections. */
    warmup_intercomm_send(*newcomm_ptr);

  fn_exit:
    MPIR_FUNC_EXIT;
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

#define MPIDI_MAX_KVS_VALUE_LEN    4096

/* Name of parent endpoint string if this process was spawned (and is root of comm world) or null */
static char parent_ep_str[MPIDI_MAX_KVS_VALUE_LEN] = { 0 };


int MPID_PSP_Get_parent_ep_str(char **ep_str)
{
    if (!parent_ep_str[0]) {
        MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
        MPIR_pmi_get_parent_port(parent_ep_str, sizeof(parent_ep_str));
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
    /* Open a port for spawned processes to connect to */
    mpi_errno = MPID_Open_port(NULL, ep_str, MPI_MAX_PORT_NAME);
    MPIR_ERR_CHECK(mpi_errno);

    if (comm_ptr->rank == root) {
        int i;
        total_num_processes = count_total_processes(count, array_of_maxprocs);

        /* create an array for the pmi error codes */
        pmi_errcodes = (int *) MPL_malloc(sizeof(int) * total_num_processes, MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!pmi_errcodes, mpi_errno, MPI_ERR_OTHER, "**nomem");

        mpi_errno = MPIR_pmi_spawn_multiple(count,
                                            array_of_commands,
                                            array_of_argv,
                                            array_of_maxprocs,
                                            array_of_info_ptrs, ep_str, pmi_errcodes, NULL);
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

    int coll_attr = MPIR_COLL_ATTR_SYNC;
    mpi_errno = MPIR_Bcast(&should_accept, 1, MPIR_INT_INTERNAL, root, comm_ptr, coll_attr);
    MPIR_ERR_CHECK(mpi_errno);
    MPIR_ERR_CHKANDJUMP(MPIR_COLL_ATTR_HAS_ERR(coll_attr), mpi_errno, MPI_ERR_OTHER, "**coll_fail");

    if (array_of_errcodes != MPI_ERRCODES_IGNORE) {
        mpi_errno =
            MPIR_Bcast(&total_num_processes, 1, MPIR_INT_INTERNAL, root, comm_ptr, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);
        MPIR_ERR_CHKANDJUMP(MPIR_COLL_ATTR_HAS_ERR(coll_attr), mpi_errno, MPI_ERR_OTHER,
                            "**coll_fail");

        mpi_errno =
            MPIR_Bcast(array_of_errcodes, total_num_processes, MPIR_INT_INTERNAL, root, comm_ptr,
                       coll_attr);
        MPIR_ERR_CHECK(mpi_errno);
        MPIR_ERR_CHKANDJUMP(MPIR_COLL_ATTR_HAS_ERR(coll_attr), mpi_errno, MPI_ERR_OTHER,
                            "**coll_fail");
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
