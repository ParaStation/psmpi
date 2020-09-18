/*
 * ParaStation
 *
 * Copyright (C) 2006-2020 ParTec Cluster Competence Center GmbH, Munich
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
#include "mpi-ext.h"
#include "mpl.h"
#include "pmi.h"
#include "errno.h"
#include "mpid_debug.h"
#include "mpid_coll.h"
#include "datatype.h"

/*
 * MPIX_Query_cuda_support - Query CUDA support of the MPI library
 */
#define FCNAME "MPIX_Query_cuda_support"
#define FUNCNAME MPIX_Query_cuda_support
int __attribute__((visibility("default")))
MPIX_Query_cuda_support(void)
{
	return MPID_Query_cuda_support();
}
#undef FUNCNAME
#undef FCNAME

#if defined(__GNUC__) || defined (__PGI)
#define dinit(name) .name =
#else
#define dinit(name)
#endif
MPIDI_Process_t MPIDI_Process = {
	dinit(grank2con)	NULL,
	dinit(my_pg_rank)	-1,
	dinit(my_pg_size)	0,
	dinit(singleton_but_no_pm)	0,
	dinit(pg_id_name)	NULL,
	dinit(next_lpid)	0,
	dinit(my_pg)		NULL,
	dinit(shm_attr_key)	0,
	dinit(smp_node_id)      -1,
	dinit(msa_module_id)    -1,
	dinit(env)		{
		dinit(enable_collectives)	0,
		dinit(enable_ondemand)		0,
		dinit(enable_ondemand_spawn)	0,
		dinit(enable_smp_awareness)	1,
		dinit(enable_msa_awareness)	0,
#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
		dinit(enable_smp_aware_collops)	0,
		dinit(enable_msa_aware_collops)	1,
#ifdef HAVE_LIBHCOLL
		dinit(enable_hcoll)	        0,
#endif
#endif
#ifdef MPID_PSP_HISTOGRAM
		dinit(enable_histogram)		0,
#endif
		dinit(enable_lazy_disconnect)	1,
	},
#ifdef MPID_PSP_HISTOGRAM
	dinit(histo)		{
		dinit(con_type_str)          NULL,
		dinit(con_type_int)            -1,
		dinit(max_size)      64*1024*1024,
		dinit(min_size)                64,
		dinit(step_width)               1,
		dinit(points)                   0,
		dinit(limit)                 NULL,
		dinit(count)                 NULL,
	},
#endif
};

/*
 * Check the success/failure of PMI calls if and only if we are not a singleton
 * without a process manager
 */
#define PMICALL(func) do {								\
	int pmi_errno = MPIDI_Process.singleton_but_no_pm? PMI_SUCCESS : (func);	\
	if (pmi_errno != PMI_SUCCESS) {							\
		PRINTERROR("PMI: " #func " = %d", pmi_errno);				\
		exit(1);								\
	}										\
} while (0)

static
void grank2con_set(int dest_grank, pscom_connection_t *con)
{
	unsigned int pg_size = MPIDI_Process.my_pg_size;

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

	MPIDI_Process.grank2con = MPL_malloc(sizeof(MPIDI_Process.grank2con[0]) * pg_size, MPL_MEM_OBJECT);
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

	listen_socket = MPL_strdup(pscom_listen_socket_str(socket));
	PMICALL(PMI_KVS_Put(pg_id, key, listen_socket));

#define INIT_VERSION "ps_v5.0"
	i_version_set(pg_id, pg_rank, INIT_VERSION);
	PMICALL(PMI_KVS_Commit(pg_id));

	PMICALL(PMI_Barrier());

	i_version_check(pg_id, pg_rank, INIT_VERSION);

	init_grank_port_mapping();

	/* Get portlist */
	psp_port = MPL_malloc(pg_size * sizeof(*psp_port), MPL_MEM_OBJECT);
	assert(psp_port);

	for (i = 0; i < pg_size; i++) {
		char val[100];
		unsigned long guard_pmi_value = MAGIC_PMI_VALUE;

		if (i != pg_rank) {
			snprintf(key, sizeof(key), "psp%d", i);
			PMICALL(PMI_KVS_Get(pg_id, key, val, sizeof(val)));
			/* simple_pmi.c has a bug.(fixed in mpich2-1.0.5)
			   Test for the bugfix: */
			assert(guard_pmi_value == MAGIC_PMI_VALUE);
			assert(guard_pmi_key == MAGIC_PMI_KEY);
		} else {
			/* myself: Dont use PMI_KVS_Get, because this fail
			   in the case of no pm (SINGLETON_INIT_BUT_NO_PM) */
			strcpy(val, listen_socket);
		}

		psp_port[i] = MPL_strdup(val);
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
			MPL_free(psp_port[i]);
			psp_port[i] = NULL;
		}
		MPL_free(psp_port);
	}

	MPL_free(listen_socket);
	return mpi_errno;
	/* --- */
 fn_fail:
	mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL,
					 "InitPortConnections", __LINE__, MPI_ERR_OTHER, "**sock|connfailed", 0);
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

	listen_socket = MPL_strdup(pscom_listen_socket_ondemand_str(socket));
	PMICALL(PMI_KVS_Put(pg_id, key, listen_socket));

#define IPSCOM_VERSION "pscom_v5.0"
	i_version_set(pg_id, pg_rank, IPSCOM_VERSION);

	PMICALL(PMI_KVS_Commit(pg_id));

	PMICALL(PMI_Barrier());

	i_version_check(pg_id, pg_rank, IPSCOM_VERSION);

	init_grank_port_mapping();

	/* Get portlist */
	psp_port = MPL_malloc(pg_size * sizeof(*psp_port), MPL_MEM_OBJECT);
	assert(psp_port);

	for (i = 0; i < pg_size; i++) {
		char val[100];
		unsigned long guard_pmi_value = MAGIC_PMI_VALUE;

		if (i != pg_rank) {
			snprintf(key, sizeof(key), "pscom%d", i);
			PMICALL(PMI_KVS_Get(pg_id, key, val, sizeof(val)));
			/* simple_pmi.c has a bug.(fixed in mpich2-1.0.5)
			   Test for the bugfix: */
			assert(guard_pmi_value == MAGIC_PMI_VALUE);
			assert(guard_pmi_key == MAGIC_PMI_KEY);
		} else {
			/* myself: Dont use PMI_KVS_Get, because this fail
			   in the case of no pm (SINGLETON_INIT_BUT_NO_PM) */
			strcpy(val, listen_socket);
		}

		psp_port[i] = MPL_strdup(val);
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
			MPL_free(psp_port[i]);
			psp_port[i] = NULL;
		}
		MPL_free(psp_port);
	}

	MPL_free(listen_socket);
	return mpi_errno;
	/* --- */
 fn_fail:
	mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL,
					 "InitPscomConnections", __LINE__, MPI_ERR_OTHER, "**sock|connfailed", 0);
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
	int pg_id_sz;
	int pg_rank = 0;
	int pg_size = -1;
	int appnum = -1;
	/* int universe_size; */
	int has_parent;
	pscom_socket_t *socket;
	pscom_err_t rc;
	char *pg_id_name;

	/* Call any and all MPID_Init type functions */
	MPIR_Err_init();
	MPIR_Datatype_init();
	MPIR_Group_init();

	mpid_debug_init();

	assert(PSCOM_ANYPORT == -1); /* all codeplaces which depends on it are marked with: "assert(PSP_ANYPORT == -1);"  */

	MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPID_INIT);
	MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPID_INIT);

	/*
	 * PMI_Init() and PMI_Get_appnum() have to be called in any case to
	 * determine if we are a singleton without a process manager
	 */
	PMICALL(PMI_Init(&has_parent));
	PMICALL(PMI_Get_appnum(&appnum));

	/* keep track if we are a singleton without process manager */
	MPIDI_Process.singleton_but_no_pm = (appnum == -1)? 1 : 0;

	PMICALL(PMI_Get_rank(&pg_rank));
	PMICALL(PMI_Get_size(&pg_size));

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

	/* take SMP-related locality information into account (e.g., for MPI_Win_allocate_shared) */
	pscom_env_get_uint(&MPIDI_Process.env.enable_smp_awareness, "PSP_SMP_AWARENESS");
	if(MPIDI_Process.env.enable_smp_awareness) {
		pscom_env_get_uint(&MPIDI_Process.smp_node_id, "PSP_SMP_NODE_ID");
#ifdef MPID_PSP_MSA_AWARENESS
		pscom_env_get_uint(&MPIDI_Process.smp_node_id, "PSP_MSA_NODE_ID");
#endif
	}

#ifdef MPID_PSP_MSA_AWARENESS
	/* take MSA-related topology information into account */
	pscom_env_get_uint(&MPIDI_Process.env.enable_msa_awareness, "PSP_MSA_AWARENESS");
	if(MPIDI_Process.env.enable_msa_awareness) {
		pscom_env_get_uint(&MPIDI_Process.msa_module_id, "PSP_MSA_MODULE_ID");
	}
#endif

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
	/* use hierarchy-aware collectives on SMP level */
	pscom_env_get_uint(&MPIDI_Process.env.enable_smp_aware_collops, "PSP_SMP_AWARE_COLLOPS");

#ifdef HAVE_LIBHCOLL
	pscom_env_get_uint(&MPIDI_Process.env.enable_hcoll, "PSP_HCOLL");
	if(MPIDI_Process.env.enable_hcoll) {
		if(1) { /* HCOLL with SHARP support? Just map the envars... */
			int hcoll_enable_sharp = 0;
			pscom_env_get_uint(&hcoll_enable_sharp, "PSP_HCOLL_ENABLE_SHARP");
			if(hcoll_enable_sharp) {
				setenv("HCOLL_ENABLE_SHARP", "1", 0);
			}
		}
		MPIR_CVAR_ENABLE_HCOLL = 1;
		/* HCOLL demands for MPICH's SMP awareness: */
		MPIDI_Process.env.enable_smp_awareness     = 1;
		MPIDI_Process.env.enable_smp_aware_collops = 1;
		/* ...but if SMP awareness for collectives is explicitly disabled... */
		pscom_env_get_uint(&MPIDI_Process.env.enable_smp_awareness, "PSP_SMP_AWARENESS");
		pscom_env_get_uint(&MPIDI_Process.env.enable_smp_aware_collops, "PSP_SMP_AWARE_COLLOPS");
		if (!MPIDI_Process.env.enable_smp_awareness || !MPIDI_Process.env.enable_smp_aware_collops) {
			/* ... we can at least fake the node affiliation: */
			MPIDI_Process.smp_node_id = pg_rank;
			MPIDI_Process.env.enable_smp_awareness     = 1;
			MPIDI_Process.env.enable_smp_aware_collops = 1;
		}
	} else {
		/* Avoid possible conflicts with externally set variables: */
		unsetenv("MPIR_CVAR_ENABLE_HCOLL");
		unsetenv("MPIR_CVAR_CH3_ENABLE_HCOLL");
	}
	/* (For now, the usage of HCOLL and MSA aware collops are mutually exclusive / FIX ME!) */
#else
	/* use hierarchy-aware collectives on MSA level */
	pscom_env_get_uint(&MPIDI_Process.env.enable_msa_aware_collops, "PSP_MSA_AWARE_COLLOPS");
#endif
#endif

#ifdef MPID_PSP_HISTOGRAM
	/* collect statistics information and print them at the end of a run */
	pscom_env_get_uint(&MPIDI_Process.env.enable_histogram, "PSP_HISTOGRAM");
	pscom_env_get_uint(&MPIDI_Process.histo.max_size,   "PSP_HISTOGRAM_MAX");
	pscom_env_get_uint(&MPIDI_Process.histo.min_size,   "PSP_HISTOGRAM_MIN");
	pscom_env_get_uint(&MPIDI_Process.histo.step_width, "PSP_HISTOGRAM_SHIFT");
	MPIDI_Process.histo.con_type_str = getenv("PSP_HISTOGRAM_CONTYPE");
	if (MPIDI_Process.histo.con_type_str) {
		for (MPIDI_Process.histo.con_type_int = PSCOM_CON_TYPE_GW; MPIDI_Process.histo.con_type_int >  PSCOM_CON_TYPE_NONE; MPIDI_Process.histo.con_type_int--) {
			if (strcmp(MPIDI_Process.histo.con_type_str, pscom_con_type_str(MPIDI_Process.histo.con_type_int)) == 0) break;
		}
	}
#endif
	pscom_env_get_uint(&MPIDI_Process.env.enable_lazy_disconnect, "PSP_LAZY_DISCONNECT");

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

	pg_id_name = MPL_malloc(pg_id_sz + 1, MPL_MEM_STRINGS);
	if (!pg_id_name) { PRINTERROR("MPL_malloc()"); goto fn_fail; }

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

	MPID_enable_receive_dispach(socket); /* ToDo: move MPID_enable_receive_dispach to bg thread */
	MPIR_Process.comm_world->pscom_socket = socket;
	MPIR_Process.comm_self->pscom_socket = socket;

	/* Call the other init routines */
	mpi_errno = MPID_PSP_comm_init(has_parent);
	if (MPI_SUCCESS != mpi_errno) {
		MPIR_ERR_POP(mpi_errno);
	}
	MPID_PSP_shm_rma_init();

	/*
	 * Setup the MPI_INFO_ENV object
	 */
	{
		MPIR_Info *info_ptr = NULL;
		MPIR_Info_get_ptr(MPI_INFO_ENV, info_ptr);
		if (MPID_Query_cuda_support()) {
			mpi_errno = MPIR_Info_set_impl(info_ptr, "cuda_aware", "true");
		} else {
			mpi_errno = MPIR_Info_set_impl(info_ptr, "cuda_aware", "false");
		}
		if (MPI_SUCCESS != mpi_errno) {
			MPIR_ERR_POP(mpi_errno);
		}
#ifdef MPID_PSP_MSA_AWARENESS
		char id_str[64];
		if(MPIDI_Process.msa_module_id >= 0) {
			snprintf(id_str, 63, "%d", MPIDI_Process.msa_module_id);
			mpi_errno = MPIR_Info_set_impl(info_ptr, "msa_module_id", id_str);
			if (MPI_SUCCESS != mpi_errno) {
				MPIR_ERR_POP(mpi_errno);
			}
		}
		if(MPIDI_Process.smp_node_id >= 0 && MPIDI_Process.env.enable_msa_awareness) {
			snprintf(id_str, 63, "%d", MPIDI_Process.smp_node_id);
			mpi_errno = MPIR_Info_set_impl(info_ptr, "msa_node_id", id_str);
			if (MPI_SUCCESS != mpi_errno) {
				MPIR_ERR_POP(mpi_errno);
			}
		}
#endif
	}


	if (threadlevel_provided) {
		*threadlevel_provided = (MPICH_THREAD_LEVEL < threadlevel_requested) ?
			MPICH_THREAD_LEVEL : threadlevel_requested;
	}


fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPID_INIT);
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
pscom_connection_t *MPID_PSCOM_rank2connection(MPIR_Comm *comm, int rank)
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

/*
 * MPID_Query_cuda_support - Query CUDA support of the device
 */
#define FCNAME "MPID_Query_cuda_support"
#define FUNCNAME MPID_Query_cuda_support
int
MPID_Query_cuda_support(void)
{
#if MPIX_CUDA_AWARE_SUPPORT
	return pscom_is_cuda_enabled();
#else
	return 0;
#endif
}
#undef FUNCNAME
#undef FCNAME
