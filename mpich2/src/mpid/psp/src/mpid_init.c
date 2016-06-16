/*
 * ParaStation
 *
 * Copyright (C) 2006-2010 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include <assert.h>
#include <unistd.h>
#include "mpidimpl.h"
#include "pmi.h"
#include "errno.h"
#include "mpid_debug.h"
#include "mpid_collective.h"

#if defined(__GNUC__)
#define dinit(name) .name =
#else
#define dinit(name)
#endif
MPIDI_Process_t MPIDI_Process = {
	dinit(grank2con)	NULL,
	dinit(my_pg_rank)	-1,
	dinit(my_pg_size)	0,
	dinit(pg_id_name)	NULL,
	dinit(env)		{
		dinit(enable_collectives)	0,
		dinit(enable_ondemand)		0,
		dinit(enable_ondemand_spawn)	0,
	},
};

#define PMICALL(func) do {										\
	int pmi_errno = (func);										\
	if (pmi_errno != PMI_SUCCESS) {									\
		PRINTERROR("PMI: " #func " = %d", pmi_errno);						\
		exit(1);										\
	}												\
} while (0)

static void checked_PMI_KVS_Get(const char kvsname[], const char key[], char value[], int length)
{
	int pmi_errno = PMI_KVS_Get(kvsname, key, value, length);
	if (pmi_errno != PMI_SUCCESS) {
		PRINTERROR("PMI: PMI_KVS_Get(kvsname=\"%s\", key=\"%s\", val, sizeof(val)) : failed",
			   kvsname, key);
		exit(1);
	}
}


static
void grank2con_set(int dest_grank, pscom_connection_t *con)
{
	unsigned int pg_size = MPIDI_Process.my_pg_size;
/*	int pg_rank = MPIDI_Process.my_pg_rank; */

	assert((unsigned int)dest_grank < pg_size);

	MPIDI_Process.grank2con[dest_grank] = con;
}

/* return connection */
static
pscom_connection_t *grank2con_get(int dest_grank)
{
	unsigned int pg_size = MPIDI_Process.my_pg_size;

	assert((unsigned int)dest_grank < pg_size);

	return MPIDI_Process.grank2con[dest_grank];
}

static
void init_grank_port_mapping(void)
{
	static int initialized = 0;
	unsigned int pg_size = MPIDI_Process.my_pg_size;
	unsigned int i;

	if (initialized) {
		PRINTERROR("Multiple calls of init_grank_port_mapping()\n");
		exit(1);
	}

	MPIDI_Process.grank2con = MPIU_Malloc(sizeof(MPIDI_Process.grank2con[0]) * pg_size);
	assert(MPIDI_Process.grank2con);

	for (i = 0; i < pg_size; i++) {
		grank2con_set(i, NULL);
	}

	initialized = 1;
}


struct InitMsg {
	unsigned int from_rank;
};



static
void cb_io_done_init_msg(pscom_request_t *req)
{
	if (pscom_req_successful(req)) {
		pscom_connection_t *old_connection;

		struct InitMsg *init_msg = (struct InitMsg *)req->data;

		old_connection = grank2con_get(init_msg->from_rank);
		if (old_connection) {
			if (old_connection == req->connection) {
				/* Loopback connection */
				;
			} else {
				/* Already connected??? */
				PRINTERROR("Second connection from %s as rank %u. Closing second.",
					   pscom_con_info_str(&old_connection->remote_con_info),
					   init_msg->from_rank);

				PRINTERROR("Old    connection from %s.",
					   pscom_con_info_str(&req->connection->remote_con_info));
				pscom_close_connection(req->connection);
			}
		} else {
			/* register connection */
			grank2con_set(init_msg->from_rank, req->connection);
		}
	} else {
		pscom_close_connection(req->connection);
	}
	pscom_request_free(req);
}


static
void mpid_con_accept(pscom_connection_t *new_connection)
{
	pscom_request_t *req;
	req = pscom_request_create(0, sizeof(struct InitMsg));

	req->xheader_len = 0;
	req->data_len = sizeof(struct InitMsg);
	req->data = req->user;
	req->connection = new_connection;
	req->ops.io_done = cb_io_done_init_msg;

	pscom_post_recv(req);
}

static
void do_wait(int pg_rank, int src) {
	/* printf("Accepting (rank %d to %d).\n", src, pg_rank); */
	while (!grank2con_get(src)) {
		pscom_wait_any();
	}
}


static
int do_connect(pscom_socket_t *socket, int pg_rank, int dest, char *dest_addr)
{
	pscom_connection_t *con;
	pscom_err_t rc;
	struct InitMsg init_msg;

	/* printf("Connecting (rank %d to %d) (%s)\n", pg_rank, dest, dest_addr); */
	con = pscom_open_connection(socket);
	rc = pscom_connect_socket_str(con, dest_addr);

	if (rc != PSCOM_SUCCESS) {
		PRINTERROR("Connecting %s to %s (rank %d to %d) failed : %s",
			   pscom_listen_socket_str(socket),
			   dest_addr, pg_rank, dest, pscom_err_str(rc));
		return -1; /* error */
	}
	grank2con_set(dest, con);

	init_msg.from_rank = pg_rank;
	pscom_send(con, NULL, 0, &init_msg, sizeof(init_msg));
	return 0;
}


static
void i_version_set(char *pg_id, int pg_rank, const char *ver)
{
	if (pg_rank == 0) {
		PMICALL(PMI_KVS_Put(pg_id, "i_version", ver));
	}
}


static
void i_version_check(char *pg_id, int pg_rank, const char *ver)
{
	if (pg_rank != 0) {
		char val[100] = "unknown";
		int pmi_errno = PMI_KVS_Get(pg_id, "i_version", val, sizeof(val));

		assert(pmi_errno == PMI_SUCCESS);

		if (strcmp(val, ver)) {
			fprintf(stderr,
				"MPI: warning: different mpi init versions (rank 0:'%s' != rank %d:'%s')\n",
				val, pg_rank, ver);
		}
	}
}


#define MAGIC_PMI_KEY	0x49aef1a2
#define MAGIC_PMI_VALUE 0x29a5f212

#define FCNAME "InitPortConnections"
#define FUNCNAME InitPortConnections
static
int InitPortConnections(pscom_socket_t *socket) {
	char key[50];
	unsigned long guard_pmi_key = MAGIC_PMI_KEY;
	int i;
	int mpi_errno = MPI_SUCCESS;

	int pg_rank = MPIDI_Process.my_pg_rank;
	int pg_size = MPIDI_Process.my_pg_size;
	char *pg_id = MPIDI_Process.pg_id_name;
	char *listen_socket;
	char **psp_port = NULL;

	/* Distribute my contact information */
	snprintf(key, sizeof(key), "psp%d", pg_rank);

	listen_socket = MPIU_Strdup(pscom_listen_socket_str(socket));
	PMICALL(PMI_KVS_Put(pg_id, key, listen_socket));

#define INIT_VERSION "ps_v5.0"
	i_version_set(pg_id, pg_rank, INIT_VERSION);
	PMICALL(PMI_KVS_Commit(pg_id));

	PMICALL(PMI_Barrier());

	i_version_check(pg_id, pg_rank, INIT_VERSION);

	init_grank_port_mapping();

	/* Get portlist */
	psp_port = MPIU_Malloc(pg_size * sizeof(*psp_port));
	assert(psp_port);

	for (i = 0; i < pg_size; i++) {
		char val[100];
		unsigned long guard_pmi_value = MAGIC_PMI_VALUE;

		if (i != pg_rank) {
			snprintf(key, sizeof(key), "psp%d", i);
			checked_PMI_KVS_Get(pg_id, key, val, sizeof(val));
			/* simple_pmi.c has a bug.(fixed in mpich2-1.0.5)
			   Test for the bugfix: */
			assert(guard_pmi_value == MAGIC_PMI_VALUE);
			assert(guard_pmi_key == MAGIC_PMI_KEY);
		} else {
			/* myself: Dont use PMI_KVS_Get, because this fail
			   in the case of no pm (SINGLETON_INIT_BUT_NO_PM) */
			strcpy(val, listen_socket);
		}

		psp_port[i] = MPIU_Strdup(val);
	}

	/* connect ranks pg_rank..(pg_rank + pg_size/2) */
	for (i = 0; i <= pg_size / 2; i++) {
		int dest = (pg_rank + i) % pg_size;
		int src = (pg_rank + pg_size - i) % pg_size;

		if (!i || (pg_rank / i) % 2) {
			/* connect, accept */
			if (do_connect(socket, pg_rank, dest, psp_port[dest])) goto fn_fail;
			if (!i || src != dest) {
				do_wait(pg_rank, src);
			}
		} else {
			/* accept, connect */
			do_wait(pg_rank, src);
			if (src != dest) {
				if (do_connect(socket, pg_rank, dest, psp_port[dest])) goto fn_fail;
			}
		}

	}

	/* Wait for all connections: (already done?) */
	for (i = 0; i < pg_size; i++) {
		while (!grank2con_get(i)) {
			pscom_wait_any();
		}
	}

	/* ToDo: */
	pscom_stop_listen(socket);

 fn_exit:
	if (psp_port) {
		for (i = 0; i < pg_size; i++) {
			MPIU_Free(psp_port[i]);
			psp_port[i] = NULL;
		}
		MPIU_Free(psp_port);
	}

	MPIU_Free(listen_socket);
	return mpi_errno;
	/* --- */
 fn_fail:
	mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL,
					 "InitPortConnections", __LINE__, MPI_ERR_OTHER, "**connfailed", 0);
	goto fn_exit;
}
#undef FUNCNAME
#undef FCNAME

#ifdef PSCOM_HAS_ON_DEMAND_CONNECTIONS
#define FCNAME "InitPscomConnections"
#define FUNCNAME InitPscomConnections
static
int InitPscomConnections(pscom_socket_t *socket) {
	char key[50];
	unsigned long guard_pmi_key = MAGIC_PMI_KEY;
	int i;
	int mpi_errno = MPI_SUCCESS;

	int pg_rank = MPIDI_Process.my_pg_rank;
	int pg_size = MPIDI_Process.my_pg_size;
	char *pg_id = MPIDI_Process.pg_id_name;
	char *listen_socket;
	char **psp_port = NULL;

	/* Distribute my contact information */
	snprintf(key, sizeof(key), "pscom%d", pg_rank);

	listen_socket = MPIU_Strdup(pscom_listen_socket_ondemand_str(socket));
	PMICALL(PMI_KVS_Put(pg_id, key, listen_socket));

#define IPSCOM_VERSION "pscom_v5.0"
	i_version_set(pg_id, pg_rank, IPSCOM_VERSION);

	PMICALL(PMI_KVS_Commit(pg_id));

	PMICALL(PMI_Barrier());

	i_version_check(pg_id, pg_rank, IPSCOM_VERSION);

	init_grank_port_mapping();

	/* Get portlist */
	psp_port = MPIU_Malloc(pg_size * sizeof(*psp_port));
	assert(psp_port);

	for (i = 0; i < pg_size; i++) {
		char val[100];
		unsigned long guard_pmi_value = MAGIC_PMI_VALUE;

		if (i != pg_rank) {
			snprintf(key, sizeof(key), "pscom%d", i);
			checked_PMI_KVS_Get(pg_id, key, val, sizeof(val));
			/* simple_pmi.c has a bug.(fixed in mpich2-1.0.5)
			   Test for the bugfix: */
			assert(guard_pmi_value == MAGIC_PMI_VALUE);
			assert(guard_pmi_key == MAGIC_PMI_KEY);
		} else {
			/* myself: Dont use PMI_KVS_Get, because this fail
			   in the case of no pm (SINGLETON_INIT_BUT_NO_PM) */
			strcpy(val, listen_socket);
		}

		psp_port[i] = MPIU_Strdup(val);
	}

	/* Create all connections */
	for (i = 0; i < pg_size; i++) {
		pscom_connection_t *con;
		pscom_err_t rc;
		const char *dest;

		dest = psp_port[i];

		con = pscom_open_connection(socket);
		rc = pscom_connect_socket_str(con, dest);

		if (rc != PSCOM_SUCCESS) {
			PRINTERROR("Connecting %s to %s (rank %d to %d) failed : %s",
				   listen_socket, dest, pg_rank, i, pscom_err_str(rc));
			goto fn_fail;
		}

		grank2con_set(i, con);
	}

	pscom_stop_listen(socket);
 fn_exit:
	if (psp_port) {
		for (i = 0; i < pg_size; i++) {
			MPIU_Free(psp_port[i]);
			psp_port[i] = NULL;
		}
		MPIU_Free(psp_port);
	}

	MPIU_Free(listen_socket);
	return mpi_errno;
	/* --- */
 fn_fail:
	mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL,
					 "InitPscomConnections", __LINE__, MPI_ERR_OTHER, "**connfailed", 0);
	goto fn_exit;
}
#undef FUNCNAME
#undef FCNAME
#else /* !PSCOM_HAS_ON_DEMAND_CONNECTIONS */
#warning "Pscom without on demand connections! You should update to pscom >= 5.0.24."
static
int InitPscomConnections(void) {
	fprintf(stderr, "Please recompile psmpi with pscom \"on demand connections\"!\n");
	exit(1);
}
#endif

#define FCNAME "MPID_Init"
#define FUNCNAME MPID_Init
int MPID_Init(int *argc, char ***argv,
	      int threadlevel_requested, int *threadlevel_provided,
	      int *has_args, int *has_env)
{
	int mpi_errno = MPI_SUCCESS;
	int pg_rank, pg_size, pg_id_sz;
	int appnum = -1;
	/* int universe_size; */
	int has_parent;
	pscom_socket_t *socket;
	pscom_err_t rc;
	char *pg_id_name;
	char *parent_port;

	mpid_debug_init();

	assert(PSCOM_ANYPORT == -1); /* all codeplaces which depends on it are marked with: "assert(PSP_ANYPORT == -1);"  */

	MPIDI_STATE_DECL(MPID_STATE_MPID_INIT);
	MPIDI_FUNC_ENTER(MPID_STATE_MPID_INIT);

	PMICALL(PMI_Init(&has_parent));
	PMICALL(PMI_Get_rank(&pg_rank));
	PMICALL(PMI_Get_size(&pg_size));
	PMICALL(PMI_Get_appnum(&appnum));

	*has_args = 1;
	*has_env  = 1;

	/* without PMI_Get_universe_size() we see pmi error:
	   '[unset]: write_line error; fd=-1' in PMI_KVS_Get()! */
	/* PMICALL(PMI_Get_universe_size(&universe_size)); */

	if (pg_rank < 0) pg_rank = 0;
	if (pg_size <= 0) pg_size = 1;

	if (
#ifndef MPICH_IS_THREADED
		1
#else
		threadlevel_requested < MPI_THREAD_MULTIPLE
#endif
	) {
		rc = pscom_init(PSCOM_VERSION);
		if (rc != PSCOM_SUCCESS) {
			fprintf(stderr, "pscom_init(0x%04x) failed : %s\n",
				PSCOM_VERSION,
				pscom_err_str(rc));
			exit(1);
		}
	} else {
		rc = pscom_init_thread(PSCOM_VERSION);
		if (rc != PSCOM_SUCCESS) {
			fprintf(stderr, "pscom_init_thread(0x%04x) failed : %s\n",
				PSCOM_VERSION,
				pscom_err_str(rc));
			exit(1);
		}
	}

	/* Initialize the switches */
	pscom_env_get_uint(&MPIDI_Process.env.enable_collectives, "PSP_COLLECTIVES");
#ifdef PSCOM_HAS_ON_DEMAND_CONNECTIONS
	/* if (pg_size > 32) MPIDI_Process.env.enable_ondemand = 1; */
	pscom_env_get_uint(&MPIDI_Process.env.enable_ondemand, "PSP_ONDEMAND");
#else
	MPIDI_Process.env.enable_ondemand = 0;
#endif
	/* enable_ondemand_spawn defaults to enable_ondemand */
	MPIDI_Process.env.enable_ondemand_spawn = MPIDI_Process.env.enable_ondemand;
	pscom_env_get_uint(&MPIDI_Process.env.enable_ondemand_spawn, "PSP_ONDEMAND_SPAWN");
	/*
	pscom_env_get_uint(&mpir_allgather_short_msg,	"PSP_ALLGATHER_SHORT_MSG");
	pscom_env_get_uint(&mpir_allgather_long_msg,	"PSP_ALLGATHER_LONG_MSG");
	pscom_env_get_uint(&mpir_allreduce_short_msg,	"PSP_ALLREDUCE_SHORT_MSG");
	pscom_env_get_uint(&mpir_alltoall_short_msg,	"PSP_ALLTOALL_SHORT_MSG");
	pscom_env_get_uint(&mpir_alltoall_medium_msg,	"PSP_ALLTOALL_MEDIUM_MSG");
	pscom_env_get_uint(&mpir_alltoall_throttle,     "PSP_ALLTOALL_THROTTLE");
	pscom_env_get_uint(&mpir_bcast_short_msg,	"PSP_BCAST_SHORT_MSG");
	pscom_env_get_uint(&mpir_bcast_long_msg,	"PSP_BCAST_LONG_MSG");
	pscom_env_get_uint(&mpir_bcast_min_procs,	"PSP_BCAST_MIN_PROCS");
	pscom_env_get_uint(&mpir_gather_short_msg,	"PSP_GATHER_SHORT_MSG");
	pscom_env_get_uint(&mpir_gather_vsmall_msg,	"PSP_GATHER_VSMALL_MSG");
	pscom_env_get_uint(&mpir_redscat_commutative_long_msg,	"PSP_REDSCAT_COMMUTATIVE_LONG_MSG");
	pscom_env_get_uint(&mpir_redscat_noncommutative_short_msg,	"PSP_REDSCAT_NONCOMMUTATIVE_SHORT_MSG");
	pscom_env_get_uint(&mpir_reduce_short_msg,	"PSP_REDUCE_SHORT_MSG");
	pscom_env_get_uint(&mpir_scatter_short_msg,	"PSP_SCATTER_SHORT_MSG");
	*/
	socket = pscom_open_socket(0, 0);

	if (!MPIDI_Process.env.enable_ondemand) {
		socket->ops.con_accept = mpid_con_accept;
	}

	{
		char name[10];
		snprintf(name, sizeof(name), "r%07u", (unsigned)pg_rank);
		pscom_socket_set_name(socket, name);
	}

	rc = pscom_listen(socket, PSCOM_ANYPORT);
	if (rc != PSCOM_SUCCESS) { PRINTERROR("pscom_listen(PSCOM_ANYPORT)"); goto fn_fail; }

	/* Note that if pmi is not availble, the value of MPI_APPNUM is not set */
/*	if (appnum != -1) {*/
	MPIR_Process.attrs.appnum = appnum;
/*	}*/
#if 0
//	see mpiimpl.h:
//	typedef struct PreDefined_attrs {
//		int appnum;          /* Application number provided by mpiexec (MPI-2) */
//		int host;            /* host */
//		int io;              /* standard io allowed */
//		int lastusedcode;    /* last used error code (MPI-2) */
//		int tag_ub;          /* Maximum message tag */
//		int universe;        /* Universe size from mpiexec (MPI-2) */
//		int wtime_is_global; /* Wtime is global over processes in COMM_WORLD */
//	} PreDefined_attrs;
#endif
	MPIR_Process.attrs.tag_ub = MPIDI_TAG_UB;

	/* obtain the id of the process group */

	PMICALL(PMI_KVS_Get_name_length_max(&pg_id_sz));

	pg_id_name = MPIU_Malloc(pg_id_sz + 1);
	if (!pg_id_name) { PRINTERROR("MPIU_Malloc()"); goto fn_fail; }

	PMICALL(PMI_KVS_Get_my_name(pg_id_name, pg_id_sz));

	/* safe */
	/* MPIDI_Process.socket = socket; */
	MPIDI_Process.my_pg_rank = pg_rank;
	MPIDI_Process.my_pg_size = pg_size;
	MPIDI_Process.pg_id_name = pg_id_name;

	if (!MPIDI_Process.env.enable_ondemand) {
		/* Create and establish all connections */
		if (InitPortConnections(socket) != MPI_SUCCESS) goto fn_fail;
	} else {
		/* Create all connections as "on demand" connections. */
		if (InitPscomConnections(socket) != MPI_SUCCESS) goto fn_fail;
	}

	/*
	 * Initialize the MPI_COMM_WORLD object
	 */
	{
		MPID_Comm * comm;
		int grank;
		MPIDI_PG_t * pg_ptr;
		int pg_id_num;

		comm = MPIR_Process.comm_world;

		comm->rank        = pg_rank;
		comm->remote_size = pg_size;
		comm->local_size  = pg_size;
		comm->pscom_socket = socket;

		mpi_errno = MPID_VCRT_Create(comm->remote_size, &comm->vcrt);
		assert(mpi_errno == MPI_SUCCESS);

		mpi_errno = MPID_VCRT_Get_ptr(comm->vcrt, &comm->vcr);
		assert(mpi_errno == MPI_SUCCESS);

		MPIDI_PG_Convert_id(pg_id_name, &pg_id_num);
		MPIDI_PG_Create(pg_size, pg_id_num, &pg_ptr);
		assert(pg_ptr == MPIDI_Process.my_pg);

		for (grank = 0; grank < pg_size; grank++) {
			/* MPIR_CheckDisjointLpids() in mpi/comm/intercomm_create.c expect
			   lpid to be smaller than 4096!!!
			   Else you will see an "Fatal error in MPI_Intercomm_create"
			*/

			pscom_connection_t *con = grank2con_get(grank);

			MPID_VCR_Initialize(pg_ptr->vcr + grank, pg_ptr, grank, con, grank);
			MPID_VCR_Dup(pg_ptr->vcr[grank], comm->vcr + grank);
		}

		mpi_errno = MPIR_Comm_commit(comm);
		assert(mpi_errno == MPI_SUCCESS);
	}

	/*
	 * Initialize the MPI_COMM_SELF object
	 */
	{
		MPID_Comm * comm;

		comm = MPIR_Process.comm_self;
		comm->rank        = 0;
		comm->remote_size = 1;
		comm->local_size  = 1;
		comm->pscom_socket = socket;

		mpi_errno = MPID_VCRT_Create(comm->remote_size, &comm->vcrt);
		assert(mpi_errno == MPI_SUCCESS);

		mpi_errno = MPID_VCRT_Get_ptr(comm->vcrt, &comm->vcr);
		assert(mpi_errno == MPI_SUCCESS);

		MPID_VCR_Dup(MPIR_Process.comm_world->vcr[pg_rank], &comm->vcr[0]);

		mpi_errno = MPIR_Comm_commit(comm);
		assert(mpi_errno == MPI_SUCCESS);
	}

	/* ToDo: move MPID_enable_receive_dispach to bg thread */
	MPID_enable_receive_dispach(socket);

	if (threadlevel_provided) {
		*threadlevel_provided = (MPICH_THREAD_LEVEL < threadlevel_requested) ?
			MPICH_THREAD_LEVEL : threadlevel_requested;
	}



	if (has_parent) {
		MPID_Comm * comm;

		mpi_errno = MPID_PSP_GetParentPort(&parent_port);
		assert(mpi_errno == MPI_SUCCESS);

		/*
		printf("%s:%u:%s Child with Parent: %s\n", __FILE__, __LINE__, __func__, parent_port);
		*/

		mpi_errno = MPID_Comm_connect(parent_port, NULL, 0,
					      MPIR_Process.comm_world, &comm);
		if (mpi_errno != MPI_SUCCESS) {
			fprintf(stderr, "MPI_Comm_connect(parent) failed!\n");
			goto fn_fail;
		}

		assert(comm != NULL);
		MPIU_Strncpy(comm->name, "MPI_COMM_PARENT", MPI_MAX_OBJECT_NAME);
		MPIR_Process.comm_parent = comm;
	}

	MPID_PSP_shm_rma_init();

 fn_exit:
	MPIDI_FUNC_EXIT(MPID_STATE_MPID_INIT);
	return mpi_errno;
	/* --- */
 fn_fail:
	/* A failing MPI_Init() did'nt call the MPI error handler, which
	   mostly calls abort(). This cause MPI_Init() to return the mpi_errno,
	   which nobody check, causing segfaultm double frees and so on. To
	   prevent strange error messages, we now call _exit(1) here.
	*/
	_exit(1);
}
#undef FUNCNAME
#undef FCNAME


/* return connection_t for rank, NULL on error */
pscom_connection_t *MPID_PSCOM_rank2connection(MPID_Comm *comm, int rank)
{
	if ((rank >= 0) && (rank < comm->remote_size)) {
		return comm->vcr[rank]->con;
	} else {
		return NULL;
	}
}


/*
 * MPID_Get_universe_size - Get the universe size from the process manager
 */
#define FCNAME "MPID_Get_universe_size"
#define FUNCNAME MPID_Get_universe_size
int MPID_Get_universe_size(int *universe_size)
{
	int mpi_errno = MPI_SUCCESS;

	PMICALL(PMI_Get_universe_size(universe_size));

 fn_exit:
	return mpi_errno;
	/* --- */
 fn_fail:
	goto fn_exit;
}
#undef FUNCNAME
#undef FCNAME
