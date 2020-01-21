/*
 * ParaStation
 *
 * Copyright (C) 2006-2019 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include "mpidimpl.h"
#include "pmi.h"
#include <unistd.h>
#include <sys/types.h>
#include "pmi.h"
#include "mpid_debug.h"

// This must be the last include before sysmbols are defined:
#include "mpid_visibility.h"


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
    PMI_Spawn_multiple(port@root)

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
	pscom_port_str_t	port_str;
	pscom_socket_t		*pscom_socket;
} pscom_inter_socket_t;

static
pscom_inter_socket_t pscom_inter_sockets[PSCOM_INTER_SOCKETS_MAX];

static
void pscom_inter_sockets_add(const pscom_port_str_t port_str, pscom_socket_t *pscom_socket)
{
	int i;
	for (i = 0; i < PSCOM_INTER_SOCKETS_MAX; i++) {
		if (pscom_inter_sockets[i].pscom_socket == NULL) {
			strcpy(pscom_inter_sockets[i].port_str, port_str);
			pscom_inter_sockets[i].pscom_socket = pscom_socket;
			return;
		}
	}
	fprintf(stderr, "To many open ports (More than %d calls to MPI_Open_port())\n", PSCOM_INTER_SOCKETS_MAX);
	_exit(1); /* ToDo: Graceful shutdown */
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
void pscom_inter_sockets_del_by_pscom_socket(pscom_socket_t *pscom_socket)
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
void init_intercomm(MPIR_Comm *comm, MPIR_Context_id_t remote_context_id, unsigned remote_comm_size, MPIR_Comm *intercomm, int create_vcrt_flag)
{
	/* compare with SetupNewIntercomm() in /src/mpid/ch3/src/ch3u_port.c:1143*/
	int mpi_errno;
	MPIDI_VCRT_t *vcrt;

	intercomm->context_id     = remote_context_id;
	/* intercomm->recvcontext_id already set in create_intercomm */
	intercomm->is_low_group   = 1;


	/* init sizes */
	intercomm->attributes   = NULL;
	intercomm->remote_size = remote_comm_size;
	intercomm->local_size  = comm->local_size;
	intercomm->rank = comm->rank;
	intercomm->local_group  = NULL;
	intercomm->remote_group = NULL;
	intercomm->comm_kind = MPIR_COMM_KIND__INTERCOMM;
	intercomm->local_comm   = NULL;

	/* Point local vcr at those of incoming intracommunicator */
	vcrt = MPIDI_VCRT_Dup(comm->vcrt);
	assert(vcrt);
	MPID_PSP_comm_set_local_vcrt(intercomm, vcrt);

	if(create_vcrt_flag) {
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
int iam_root(int root, MPIR_Comm *comm)
{
	return comm->rank == root;
}


static
pscom_port_str_t *alloc_all_ports(unsigned comm_size)
{
	return (pscom_port_str_t *)MPL_malloc(comm_size * sizeof(pscom_port_str_t), MPL_MEM_STRINGS);
}


static
void free_all_ports(pscom_port_str_t *all_ports)
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
int forward_pg_info(pscom_connection_t *con,  MPIR_Comm *comm, int root, pscom_port_str_t *all_ports, MPIR_Comm *intercomm)
{
	pscom_err_t rc;
	MPIR_Errflag_t errflag = FALSE;
	int mpi_errno = MPI_SUCCESS;

	int local_size = comm->local_size;
	int remote_size = 0;
	MPIDI_Gpid *local_gpids;
	MPIDI_Gpid *remote_gpids;
	int *remote_lpids;
	int local_context_id;
	int remote_context_id;

	pscom_socket_t *pscom_socket = intercomm->pscom_socket;

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
	/* Disable SMP-awareness as soon as dynamic process spawning comes into play. */
	if(MPIDI_Process.node_id_table) {
		MPL_free(MPIDI_Process.node_id_table);
		MPIDI_Process.node_id_table = NULL;
	}
#endif

	if(iam_root(root, comm) && con) {
		pscom_send(con, NULL, 0, &local_size, sizeof(int));
		rc = pscom_recv_from(con, NULL, 0, &remote_size, sizeof(int));
		assert(rc == PSCOM_SUCCESS);
	}

	mpi_errno = MPIR_Bcast(&remote_size, 1, MPI_INT, root, comm, &errflag);
	assert(mpi_errno == MPI_SUCCESS);

	if (remote_size == 0) goto err_failed; /* this happens if root has no valid 'con' (see above!) */

	local_gpids = (MPIDI_Gpid*)MPL_malloc(local_size * sizeof(MPIDI_Gpid), MPL_MEM_OBJECT);
	remote_gpids = (MPIDI_Gpid*)MPL_malloc(remote_size * sizeof(MPIDI_Gpid), MPL_MEM_OBJECT);

	MPIDI_GPID_GetAllInComm(comm, local_size, local_gpids, NULL);

	if(iam_root(root, comm)) {
		pscom_send(con, NULL, 0, local_gpids, local_size * sizeof(MPIDI_Gpid));
		rc = pscom_recv_from(con, NULL, 0, remote_gpids, remote_size * sizeof(MPIDI_Gpid));
		assert(rc == PSCOM_SUCCESS);
	}

	mpi_errno = MPIR_Bcast(remote_gpids, remote_size * sizeof(MPIDI_Gpid), MPI_CHAR, root, comm, &errflag);
	assert(mpi_errno == MPI_SUCCESS);


	/* Call the central routine for establishing all missing connections: */
	MPIDI_PG_ForwardPGInfo(NULL, comm, remote_size, remote_gpids, root, -1, -1, con, (char*)all_ports, pscom_socket);


	/* distribute remote values */
	if(iam_root(root, comm)) {
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
	init_intercomm(comm, remote_context_id, remote_size, intercomm, 0 /*create_vcrt_flag*/);

	remote_lpids = (int*)MPL_malloc(remote_size*sizeof(int), MPL_MEM_OTHER);
	MPIDI_GPID_ToLpidArray(remote_size, remote_gpids, remote_lpids);
	MPID_Create_intercomm_from_lpids(intercomm, remote_size, remote_lpids);

	MPL_free(local_gpids);
	MPL_free(remote_gpids);
	MPL_free(remote_lpids);

	return MPI_SUCCESS;
err_failed:
	init_intercomm(comm, MPIR_INVALID_CONTEXT_ID, 0 /* remote_size*/, intercomm, 1 /* create_vcrt_flag */);
	return MPI_ERR_COMM;
}


static
void inter_barrier(pscom_connection_t *con)
{
	int dummy;
	int rc;

	/* Workaround for timing of pscom ondemand connections. Be
	   sure both sides have called pscom_connect_socket_str before
	   using the connections. step 2 of 3 */
	pscom_send(con, NULL, 0, &dummy, sizeof(dummy));

	rc = pscom_recv_from(con, NULL, 0, &dummy, sizeof(dummy));
	assert(rc == PSCOM_SUCCESS);
}


pscom_port_str_t *MPID_PSP_open_all_ports(int root, MPIR_Comm *comm, MPIR_Comm *intercomm)
{
	pscom_socket_t *socket_new;
	int local_size = comm->local_size;
	pscom_port_str_t *all_ports = NULL;
	pscom_port_str_t my_port;
	MPIR_Errflag_t err;
	int mpi_error;

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
			if (mpid_psp_debug_level) {
				snprintf(name, sizeof(name), "i%03ur%03u", MPIDI_Process.my_pg->id_num % 1000, MPIDI_Process.my_pg_rank % 1000);
			} else {
				int rank_range = 1;
				int pg_id_mod = 1;
				int pg_size = MPIDI_Process.my_pg_size;
				while (pg_size >>= 4) rank_range++;
				pg_id_mod = 1 << (8-rank_range)*4;
				snprintf(name, sizeof(name), "%0*x%0*x", 8-rank_range, MPIDI_Process.my_pg->id_num % pg_id_mod, rank_range, MPIDI_Process.my_pg_rank);
			}
			pscom_socket_set_name(socket_new, name);
		}

		rc = pscom_listen(socket_new, PSCOM_ANYPORT);
		if (rc != PSCOM_SUCCESS) {
			PRINTERROR("pscom_listen(PSCOM_ANYPORT)");
			_exit(1); /* ToDo: Graceful shutdown */
		}

		strcpy(my_port, pscom_listen_socket_ondemand_str(socket_new));

		intercomm->pscom_socket = socket_new;
	}

	if (iam_root(root, comm)) {
		all_ports = alloc_all_ports(local_size);
	}

	err = FALSE;
	mpi_error = MPIR_Gather_intra_auto(my_port, sizeof(pscom_port_str_t), MPI_CHAR,
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
int MPID_Open_port(MPIR_Info *info_ptr, char *port_name)
{
	static unsigned portnum = 0;
	int rc;
	const char *port_str;
	int tcp_enabled = 1;

	pscom_socket_t *socket = pscom_open_socket(0, 0);
	{
		char name[10];
		snprintf(name, sizeof(name), "int%05u", (unsigned)portnum);
		pscom_socket_set_name(socket, name);
		portnum++;
	}

	/* Allow TCP only. ToDo: Allow RDP connects when they are implemented */
	/* If TCP plugin is disabled (no pscom payload via TCP), we cannot enforce TCP... */
	pscom_env_get_uint(&tcp_enabled, "PSP_TCP");
	if(tcp_enabled) pscom_con_type_mask_only(socket, PSCOM_CON_TYPE_TCP);

	rc = pscom_listen(socket, PSCOM_ANYPORT);
	if (rc != PSCOM_SUCCESS) {
		PRINTERROR("pscom_listen(PSCOM_ANYPORT)");
		_exit(1); /* ToDo: Graceful shutdown */
	}

	port_str = pscom_listen_socket_str(socket);
	pscom_inter_sockets_add(port_str, socket);

	strcpy(port_name, port_str);
	/* Typical ch3 {port_name}s: */
	/* First  MPI_Open_port: "<tag#0$description#phoenix$port#55364$ifname#192.168.254.21$>" */
	/* Second MPI_Open_port: "<tag#1$description#phoenix$port#55364$ifname#192.168.254.21$>" */

	return MPI_SUCCESS;
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
	/* printf("%s(port_name:\"%s\")\n", __func__, port_name);*/
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

	intercomm->context_id     = MPIR_INVALID_CONTEXT_ID; /* finally set in init_intercomm() to recvcontext_id of the remote */
	intercomm->recvcontext_id = recvcontext_id;

	return intercomm;
}


static
void warmup_intercomm_send(MPIR_Comm *comm)
{
	int i;
	if (MPIDI_Process.env.enable_ondemand_spawn) return;

	for (i = 0; i < comm->remote_size; i++) {
		int rank = (i + comm->rank) % comm->remote_size; /* destination rank */
		/* printf("#S%d: Send #%d to #%d ctx:%u rctx:%u\n",
		   comm->rank, comm->rank, rank, comm->context_id, comm->recvcontext_id); */
		pscom_connection_t *con = MPID_PSCOM_rank2connection(comm, rank);
		MPID_PSP_SendCtrl(17 /* tag */, comm->context_id, comm->rank /* src_rank */,
				  con, MPID_PSP_MSGTYPE_DATA_ACK);
		MPID_PSP_RecvCtrl(15 /* tag */, comm->recvcontext_id, rank /* src_rank */,
				  con, MPID_PSP_MSGTYPE_DATA_ACK);
	}
}


static
void warmup_intercomm_recv(MPIR_Comm *comm)
{
	int i;
	if (MPIDI_Process.env.enable_ondemand_spawn) return;

	for (i = 0; i < comm->remote_size; i++) {
		int rank = (comm->remote_size - i + comm->rank) % comm->remote_size; /* source rank */
		/* printf("#R%d: Recv #%d to #%d ctx:%u rctx:%u\n",
		   comm->rank, rank, comm->rank, comm->context_id, comm->recvcontext_id); */
		pscom_connection_t *con = MPID_PSCOM_rank2connection(comm, rank);
		MPID_PSP_RecvCtrl(17 /* tag */, comm->recvcontext_id, rank /* src_rank */,
				  con, MPID_PSP_MSGTYPE_DATA_ACK);
		MPID_PSP_SendCtrl(15 /* tag */, comm->context_id, comm->rank /* src_rank */,
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
int MPID_Comm_accept(const char * port_name, MPIR_Info * info, int root,
		     MPIR_Comm * comm, MPIR_Comm **_intercomm)
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
			if (con) break;

			pscom_wait_any();
		}

		forward_pg_info(con, comm, root, all_ports, intercomm);

		inter_barrier(con);
		pscom_flush(con);
		pscom_close_connection(con);

	} else {
		forward_pg_info(NULL, comm, root, all_ports, intercomm);
	}

	free_all_ports(all_ports); all_ports = NULL;

	/* Workaround for timing of pscom ondemand connections. Be
	   sure both sides have called pscom_connect_socket_str before
	   using the connections. step 3 of 3 */
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
int MPID_Comm_connect(const char * port_name, MPIR_Info * info, int root,
		      MPIR_Comm * comm, MPIR_Comm **_intercomm)
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

	free_all_ports(all_ports); all_ports = NULL;

	/* Workaround for timing of pscom ondemand connections. Be
	   sure both sides have called pscom_connect_socket_str before
	   using the connections. step 3 of 3 */

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


int MPID_Comm_disconnect(MPIR_Comm *comm_ptr)
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
	int pmi_errno;

	if (!parent_port_name[0]) {
		char *pg_id = MPIDI_Process.pg_id_name;

		MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
		pmi_errno = PMI_KVS_Get(pg_id, PARENT_PORT_KVSKEY, parent_port_name, sizeof(parent_port_name));
		MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
		if (pmi_errno) {
			mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL, "MPID_PSP_GetParentPort",
							 __LINE__, MPI_ERR_OTHER,
							 "**pmi_kvsget", "**pmi_kvsget %d", pmi_errno);
			goto fn_exit;
		}
	}

	*parent_port = parent_port_name;

 fn_exit:
	return mpi_errno;
 fn_fail:
	goto fn_exit;
}


static
int  mpi_to_pmi_keyvals(MPIR_Info *info_ptr, const PMI_keyval_t **kv_ptr,
			       int *nkeys_ptr )
{
	char key[MPI_MAX_INFO_KEY];
	PMI_keyval_t *kv = 0;
	int          i, nkeys = 0, vallen, flag, mpi_errno=MPI_SUCCESS;

	if (!info_ptr || info_ptr->handle == MPI_INFO_NULL) {
		goto fn_exit;
	}

	MPIR_Info_get_nkeys_impl( info_ptr, &nkeys );
	if (nkeys == 0) {
		goto fn_exit;
	}
	kv = (PMI_keyval_t *)MPL_malloc( nkeys * sizeof(PMI_keyval_t) , MPL_MEM_PM);
	assert(kv);

	for (i=0; i<nkeys; i++) {
		mpi_errno = MPIR_Info_get_nthkey_impl( info_ptr, i, key );
		assert(mpi_errno == MPI_SUCCESS);
		MPIR_Info_get_valuelen_impl( info_ptr, key, &vallen, &flag );

		kv[i].key = MPL_strdup(key);
		kv[i].val = MPL_malloc( vallen + 1 , MPL_MEM_PM);
		assert(kv[i].key);
		assert(kv[i].val);

		MPIR_Info_get_impl( info_ptr, key, vallen+1, kv[i].val, &flag );
		/* MPIU_DBG_PRINTF(("key: <%s>, value: <%s>\n", kv[i].key, kv[i].val)); */
	}

 fn_exit:
	*kv_ptr    = kv;
	*nkeys_ptr = nkeys;
	return mpi_errno;
}


static
void pmi_keyvals_free(const PMI_keyval_t *kv, int nkeys)
{
	int i;
	if (!kv) return;

	for (i = 0; i < nkeys; i++) {
		MPL_free((char *)kv[i].key);
		MPL_free(kv[i].val);
	}
	MPL_free((void*)kv);
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
#undef FCNAME
#define FCNAME "MPID_Comm_spawn_multiple"
#undef FUNCNAME
#define FUNCNAME MPID_Comm_spawn_multiple
int MPID_Comm_spawn_multiple(int count, char *array_of_commands[],
			     char ** array_of_argv[], const int array_of_maxprocs[],
			     MPIR_Info * array_of_info_ptrs[], int root,
			     MPIR_Comm * comm_ptr, MPIR_Comm ** intercomm,
			     int array_of_errcodes[])
{
	int rc;

	char port_name[MPI_MAX_PORT_NAME];

	/*
	printf("%s:%u:%s Spawn from context_id: %u\n", __FILE__, __LINE__, __func__, comm_ptr->context_id);
	*/

	rc = MPID_Open_port(NULL, port_name);
	assert(rc == MPI_SUCCESS);

	if (comm_ptr->rank == root) {
		int ret, i;
		const PMI_keyval_t **info_keyval_vectors = 0;
		PMI_keyval_t preput_keyval_vector;
		int *info_keyval_sizes = 0;
		int *pmi_errcodes;
		int total_num_processes = count_total_processes(count, array_of_maxprocs);

		info_keyval_sizes   = (int *)MPL_malloc(count * sizeof(int), MPL_MEM_OTHER);
		assert(info_keyval_sizes);

		info_keyval_vectors =
			(const PMI_keyval_t**) MPL_malloc(count * sizeof(PMI_keyval_t*), MPL_MEM_OTHER);
		assert(info_keyval_vectors);

		if (!array_of_info_ptrs) {
			for (i = 0; i < count; i++) {
				info_keyval_vectors[i] = 0;
				info_keyval_sizes[i]   = 0;
			}
		} else {
			for (i = 0; i < count; i++) {
				rc = mpi_to_pmi_keyvals(array_of_info_ptrs[i],
							&info_keyval_vectors[i],
							&info_keyval_sizes[i]);
				assert(rc == MPI_SUCCESS);
			}
		}

		preput_keyval_vector.key = PARENT_PORT_KVSKEY;
		preput_keyval_vector.val = port_name;

		/* create an array for the pmi error codes */
		pmi_errcodes = (int*)MPL_malloc(sizeof(int) * total_num_processes, MPL_MEM_OTHER);
		assert(pmi_errcodes);

		/* initialize them to 0 */
		for (i = 0; i < total_num_processes; i++) pmi_errcodes[i] = 0;

		ret = PMI_Spawn_multiple(count,
					 (const char **)array_of_commands,
					 (const char ***)array_of_argv,
					 array_of_maxprocs,

					 info_keyval_sizes,
					 info_keyval_vectors,

					 1,
					 &preput_keyval_vector,
					 pmi_errcodes);
		assert(ret == PMI_SUCCESS);

		if (array_of_errcodes != MPI_ERRCODES_IGNORE) {
			for (i = 0; i < total_num_processes; i++) {
				/* FIXME: translate the pmi error codes here */
				array_of_errcodes[i] = pmi_errcodes[i];
				/* We want to accept if any of the spawns succeeded.
				   Alternatively, this is the same as we want to NOT accept if
				   all of them failed.  should_accept = NAND(e_0, ..., e_n)
				   Remember, success equals false (0). */
				/*should_accept = should_accept && errcodes[i];*/
			}
			/* should_accept = !should_accept; *//* the `N' in NAND */
		}

		MPL_free(pmi_errcodes);
		for (i = 0; i < count; i++) {
			pmi_keyvals_free(info_keyval_vectors[i],
					 info_keyval_sizes[i]);
		}
		MPL_free(info_keyval_vectors);
		MPL_free(info_keyval_sizes);

		/*
		printf("%s:%u:%s Spawn done\n", __FILE__, __LINE__, __func__);
		*/
	} /* root */

	rc = MPID_Comm_accept(port_name, NULL, root, comm_ptr, intercomm);
	assert(rc == MPI_SUCCESS);

	rc = MPID_Close_port(port_name);
	assert(rc == MPI_SUCCESS);

	return 0;
}
#undef FUNCNAME
#undef FCNAME
