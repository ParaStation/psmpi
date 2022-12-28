/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2023 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <assert.h>
#include <unistd.h>
#include "mpidimpl.h"
#include "mpi-ext.h"
#include "mpl.h"
#include "errno.h"
#include "mpid_debug.h"
#include "mpid_coll.h"
#include "datatype.h"

/*
 * MPIX_Query_cuda_support - Query CUDA support of the MPI library
 */
int __attribute__((visibility("default")))
MPIX_Query_cuda_support(void)
{
	return MPID_Query_cuda_support();
}

#if defined(__GNUC__) || defined (__PGI)
#define dinit(name) .name =
#else
#define dinit(name)
#endif
MPIDI_Process_t MPIDI_Process = {
	dinit(socket)		NULL,
	dinit(grank2con)	NULL,
	dinit(my_pg_rank)	-1,
	dinit(my_pg_size)	0,
	dinit(singleton_but_no_pm)	0,
	dinit(pg_id_name)	NULL,
	dinit(next_lpid)	0,
	dinit(my_pg)		NULL,
#ifdef MPID_PSP_MSA_AWARE_COLLOPS
	dinit(topo_levels)      NULL,
#endif
	dinit(shm_attr_key)	0,
	dinit(smp_node_id)      -1,
	dinit(msa_module_id)    -1,
	dinit(use_world_model)  0,
	dinit(env)		{
		dinit(debug_level)		0,
		dinit(debug_version)		0,
		dinit(enable_collectives)	0,
		dinit(enable_ondemand)		0,
		dinit(enable_ondemand_spawn)	0,
		dinit(enable_smp_awareness)	1,
		dinit(enable_msa_awareness)	0,
		dinit(enable_smp_aware_collops)	0,
#ifdef MPID_PSP_MSA_AWARE_COLLOPS
		dinit(enable_msa_aware_collops)	1,
#endif
#ifdef HAVE_HCOLL
		dinit(enable_hcoll)	        0,
#endif
#ifdef MPID_PSP_HISTOGRAM
		dinit(enable_histogram)		0,
#endif
#ifdef MPID_PSP_HCOLL_STATS
		dinit(enable_hcoll_stats)       0,
#endif
		dinit(enable_lazy_disconnect)	1,
	},
#ifdef MPIDI_PSP_WITH_SESSION_STATISTICS
	dinit(stats)            {
#ifdef MPID_PSP_HISTOGRAM
		dinit(histo)                    {
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
#ifdef MPID_PSP_HCOLL_STATS
		dinit(hcoll)           {
			dinit(counter)                {0},
		},
#endif
	},
#endif /* MPIDI_PSP_WITH_SESSION_STATISTICS */
};

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
void init_send_done(pscom_req_state_t state, void *priv)
{
	int *send_done = (int *)priv;
	*send_done = 1;
}


static
int do_connect(pscom_socket_t *socket, int pg_rank, int dest, char *dest_addr)
{
	pscom_connection_t *con;
	pscom_err_t rc;
	struct InitMsg init_msg;
	int init_msg_sent = 0;

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

	/* send the initialization message and wait for its completion */
	init_msg.from_rank = pg_rank;
	pscom_send_inplace(con, NULL, 0, &init_msg, sizeof(init_msg),
			   init_send_done, &init_msg_sent);

	while (!init_msg_sent) {
		pscom_wait_any();
	}

	return 0;
}


static
int i_version_set(int pg_rank, const char *ver)
{
	int mpi_errno = MPI_SUCCESS;

	/* There is no need to check for version in the singleton case and we moreover must
	   not use MPIR_pmi_kvs_put() in this case either since there is no process manager. */
	if (MPIDI_Process.singleton_but_no_pm) goto fn_exit;

	if (pg_rank == 0) {
		mpi_errno = MPIR_pmi_kvs_put("i_version", ver);
		MPIR_ERR_CHECK(mpi_errno);
	}

 fn_exit:
	return mpi_errno;
 fn_fail:
	PRINTERROR("MPI errno:  MPIR_pmi_kvs_put  = %d in i_version_set", mpi_errno);
	goto fn_exit;
}


static
int i_version_check(int pg_rank, const char *ver)
{
        int mpi_errno = MPI_SUCCESS;

	/* There is no need to check for version in the singleton case and we moreover must
	   not use MPIR_pmi_kvs_get() in this case either since there is no process manager. */
	if (MPIDI_Process.singleton_but_no_pm) goto fn_exit;

	if (pg_rank != 0) {
		char val[100] = "unknown";
		mpi_errno = MPIR_pmi_kvs_get(0, "i_version", val, sizeof(val));
		MPIR_ERR_CHECK(mpi_errno);

		if (strcmp(val, ver)) {
			fprintf(stderr,
				"MPI: warning: different mpi init versions (rank 0:'%s' != rank %d:'%s')\n",
				val, pg_rank, ver);
		}
	}

 fn_exit:
	return mpi_errno;
 fn_fail:
	PRINTERROR("MPI errno:  MPIR_pmi_kvs_get  = %d in i_version_check", mpi_errno);
	goto fn_exit;
}


#define MAGIC_PMI_KEY	0x49aef1a2
#define MAGIC_PMI_VALUE 0x29a5f212

static
int InitPortConnections(pscom_socket_t *socket) {
	char key[50];
	unsigned long guard_pmi_key = MAGIC_PMI_KEY;
	int i;
	int mpi_errno = MPI_SUCCESS;

	int pg_rank = MPIDI_Process.my_pg_rank;
	int pg_size = MPIDI_Process.my_pg_size;
	char *listen_socket;
	char **psp_port = NULL;

	/* Distribute my contact information */
	snprintf(key, sizeof(key), "psp%d", pg_rank);
	listen_socket = MPL_strdup(pscom_listen_socket_str(socket));

	/* PMI(x)_put and PMI(x)_commit() */
	mpi_errno = MPIR_pmi_kvs_put(key,listen_socket);
	MPIR_ERR_CHECK(mpi_errno);

#define INIT_VERSION "ps_v5.0"
	mpi_errno = i_version_set(pg_rank, INIT_VERSION);
	MPIR_ERR_CHECK(mpi_errno);

	mpi_errno = MPIR_pmi_barrier();
	MPIR_ERR_CHECK(mpi_errno);

	mpi_errno = i_version_check(pg_rank, INIT_VERSION);
	MPIR_ERR_CHECK(mpi_errno);

	init_grank_port_mapping();

	/* Get portlist */
	psp_port = MPL_malloc(pg_size * sizeof(*psp_port), MPL_MEM_OBJECT);
	assert(psp_port);

	for (i = 0; i < pg_size; i++) {
		char val[100];
		unsigned long guard_pmi_value = MAGIC_PMI_VALUE;

		if (i != pg_rank) {
			snprintf(key, sizeof(key), "psp%d", i);
			/*"i" is the source who published the information */
			mpi_errno = MPIR_pmi_kvs_get(i, key, val, sizeof(val));
			MPIR_ERR_CHECK(mpi_errno);

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
			if (do_connect(socket, pg_rank, dest, psp_port[dest])) goto fn_fail_connect;
			if (!i || src != dest) {
				do_wait(pg_rank, src);
			}
		} else {
			/* accept, connect */
			do_wait(pg_rank, src);
			if (src != dest) {
				if (do_connect(socket, pg_rank, dest, psp_port[dest])) goto fn_fail_connect;
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
 fn_fail_connect:
	mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL,
					 "InitPortConnections", __LINE__, MPI_ERR_OTHER, "**sock|connfailed", 0);
	goto fn_exit;
 fn_fail:
	PRINTERROR("MPI errno %d, PMI func failed at %d in InitPortConnections", mpi_errno, __LINE__);
	goto fn_exit;
}

#ifdef PSCOM_HAS_ON_DEMAND_CONNECTIONS
static
int InitPscomConnections(pscom_socket_t *socket) {
	char key[50];
	unsigned long guard_pmi_key = MAGIC_PMI_KEY;
	int i;
	int mpi_errno = MPI_SUCCESS;

	int pg_rank = MPIDI_Process.my_pg_rank;
	int pg_size = MPIDI_Process.my_pg_size;
	char *listen_socket;
	char **psp_port = NULL;

	/* Distribute my contact information */
	snprintf(key, sizeof(key), "pscom%d", pg_rank);
	listen_socket = MPL_strdup(pscom_listen_socket_ondemand_str(socket));

	/* PMI(x)_put and PMI(x)_commit() */
	mpi_errno = MPIR_pmi_kvs_put(key,listen_socket);
	MPIR_ERR_CHECK(mpi_errno);

#define IPSCOM_VERSION "pscom_v5.0"
	mpi_errno = i_version_set(pg_rank, IPSCOM_VERSION);
	MPIR_ERR_CHECK(mpi_errno);

	mpi_errno = MPIR_pmi_barrier();
	MPIR_ERR_CHECK(mpi_errno);

	mpi_errno = i_version_check(pg_rank, IPSCOM_VERSION);
	MPIR_ERR_CHECK(mpi_errno);

	init_grank_port_mapping();

	/* Get portlist */
	psp_port = MPL_malloc(pg_size * sizeof(*psp_port), MPL_MEM_OBJECT);
	assert(psp_port);

	for (i = 0; i < pg_size; i++) {
		char val[100];
		unsigned long guard_pmi_value = MAGIC_PMI_VALUE;

		if (i != pg_rank) {
			snprintf(key, sizeof(key), "pscom%d", i);
			/*"i" is the source who published the information */
			mpi_errno = MPIR_pmi_kvs_get(i, key, val, sizeof(val));
			MPIR_ERR_CHECK(mpi_errno);

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
			goto fn_fail_connect;
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
 fn_fail_connect:
	mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL,
					 "InitPscomConnections", __LINE__, MPI_ERR_OTHER, "**sock|connfailed", 0);
	goto fn_exit;
 fn_fail:
	PRINTERROR("MPI errno %d, PMI func failed at %d in InitPscomConnections", mpi_errno, __LINE__);
	goto fn_exit;
}
#else /* !PSCOM_HAS_ON_DEMAND_CONNECTIONS */
#warning "Pscom without on demand connections! You should update to pscom >= 5.0.24."
static
int InitPscomConnections(void) {
	fprintf(stderr, "Please recompile psmpi with pscom \"on demand connections\"!\n");
	exit(1);
}
#endif

int MPID_Init(int requested, int *provided)
{
	int mpi_errno = MPI_SUCCESS;
	int pg_rank = 0;
	int pg_size = -1;
	int appnum = -1;
	/* int universe_size; */
	pscom_socket_t *socket;
	pscom_err_t rc;

	mpid_debug_init();

	assert(PSCOM_ANYPORT == -1); /* all codeplaces which depends on it are marked with: "assert(PSP_ANYPORT == -1);"  */

	MPIR_FUNC_ENTER;

	// PMI or PMIx init
	mpi_errno = MPIR_pmi_init();
	MPIR_ERR_CHECK(mpi_errno);

	pg_rank = MPIR_Process.rank;
	pg_size = MPIR_Process.size;

	// appnum is set to 0 with PMIx
	appnum = MPIR_Process.appnum;

	/* keep track if we are a singleton without process manager */
	MPIDI_Process.singleton_but_no_pm = (appnum == -1)? 1 : 0;

	if (pg_rank < 0) pg_rank = 0;
	if (pg_size <= 0) pg_size = 1;

	if (
#ifndef MPICH_IS_THREADED
		1
#else
		requested < MPI_THREAD_MULTIPLE
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

	/* add the callback for applying a Barrier (if enabled) at the very beginninf of Finalize */
	MPIR_Add_finalize(MPIDI_PSP_finalize_add_barrier_cb, NULL, MPIR_FINALIZE_CALLBACK_MAX_PRIO);

	/* take SMP-related locality information into account (e.g., for MPI_Win_allocate_shared) */
	pscom_env_get_uint(&MPIDI_Process.env.enable_smp_awareness, "PSP_SMP_AWARENESS");
	if(MPIDI_Process.env.enable_smp_awareness) {
		pscom_env_get_int(&MPIDI_Process.smp_node_id, "PSP_SMP_NODE_ID");
#ifdef MPID_PSP_MSA_AWARENESS
		pscom_env_get_int(&MPIDI_Process.smp_node_id, "PSP_MSA_NODE_ID");
#endif
	}

#ifdef MPID_PSP_MSA_AWARENESS
	/* take MSA-related topology information into account */
	pscom_env_get_uint(&MPIDI_Process.env.enable_msa_awareness, "PSP_MSA_AWARENESS");
	if(MPIDI_Process.env.enable_msa_awareness) {
		pscom_env_get_int(&MPIDI_Process.msa_module_id, "PSP_MSA_MODULE_ID");
	}
#endif

	/* use hierarchy-aware collectives on SMP level */
	pscom_env_get_uint(&MPIDI_Process.env.enable_smp_aware_collops, "PSP_SMP_AWARE_COLLOPS");

#ifdef HAVE_HCOLL
	MPIDI_Process.env.enable_hcoll = MPIR_CVAR_ENABLE_HCOLL;
	if (MPIDI_Process.env.enable_hcoll) {
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
	}
	/* (For now, the usage of HCOLL and MSA aware collops are mutually exclusive / FIX ME!) */
#else
#ifdef MPID_PSP_MSA_AWARE_COLLOPS
	/* use hierarchy-aware collectives on MSA level */
	pscom_env_get_uint(&MPIDI_Process.env.enable_msa_aware_collops, "PSP_MSA_AWARE_COLLOPS");
#endif
#endif

#ifdef MPID_PSP_HISTOGRAM
	/* collect statistics information and print them at the end of a run */
	pscom_env_get_uint(&MPIDI_Process.env.enable_histogram, "PSP_HISTOGRAM");
	pscom_env_get_int(&MPIDI_Process.stats.histo.max_size,   "PSP_HISTOGRAM_MAX");
	pscom_env_get_int(&MPIDI_Process.stats.histo.min_size,   "PSP_HISTOGRAM_MIN");
	pscom_env_get_int(&MPIDI_Process.stats.histo.step_width, "PSP_HISTOGRAM_SHIFT");
	MPIDI_Process.stats.histo.con_type_str = getenv("PSP_HISTOGRAM_CONTYPE");
	if (MPIDI_Process.stats.histo.con_type_str) {
		for (MPIDI_Process.stats.histo.con_type_int = PSCOM_CON_TYPE_GW; MPIDI_Process.stats.histo.con_type_int >  PSCOM_CON_TYPE_NONE; MPIDI_Process.stats.histo.con_type_int--) {
			if (strcmp(MPIDI_Process.stats.histo.con_type_str, pscom_con_type_str(MPIDI_Process.stats.histo.con_type_int)) == 0) break;
		}
	}
#endif
#ifdef MPID_PSP_HCOLL_STATS
	/* collect usage information of hcoll collectives and print them at the end of a run */
	pscom_env_get_uint(&MPIDI_Process.env.enable_hcoll_stats, "PSP_HCOLL_STATS");
#endif
#ifdef MPIDI_PSP_WITH_SESSION_STATISTICS
	/* add a callback for printing statistical information (if enabled) during Finalize */
	MPIR_Add_finalize(MPIDI_PSP_finalize_print_stats_cb, NULL, MPIR_FINALIZE_CALLBACK_PRIO + 1);
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
		snprintf(name, sizeof(name), "r%07u", (unsigned)pg_rank % 100000000);
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

	/* safe */
	/* MPIDI_Process.socket = socket; */
	MPIDI_Process.my_pg_rank = pg_rank;
	MPIDI_Process.my_pg_size = pg_size;
        //now pg_id can be obtained directly from MPIR layer where pg_id is stored
	MPIDI_Process.pg_id_name = MPL_strdup(MPIR_pmi_job_id()); //pg_id_name;


	if (!MPIDI_Process.env.enable_ondemand) {
		/* Create and establish all connections */
		mpi_errno = InitPortConnections(socket);
		MPIR_ERR_CHECK(mpi_errno);

	} else {
		/* Create all connections as "on demand" connections. */
		mpi_errno = InitPscomConnections(socket);
		MPIR_ERR_CHECK(mpi_errno);
	}

	MPID_enable_receive_dispach(socket); /* ToDo: move MPID_enable_receive_dispach to bg thread */
	MPIDI_Process.socket = socket;

#ifdef MPID_PSP_MSA_AWARENESS
	/* Initialize the hierarchical topology information as used for MSA-aware collectives. */
	MPIDI_PSP_topo_init();
#endif
	MPID_PSP_shm_rma_init();

	if (provided) {
		*provided = (MPICH_THREAD_LEVEL < requested) ?
			MPICH_THREAD_LEVEL : requested;
	}

	/* init lists for partitioned communication operations (used on receiver side) */
	INIT_LIST_HEAD(&(MPIDI_Process.part_posted_list));
	INIT_LIST_HEAD(&(MPIDI_Process.part_unexp_list));

fn_fail:
fn_exit:
    MPIR_FUNC_EXIT;
	return mpi_errno;
}


int MPID_InitCompleted( void )
{
	int mpi_errno = MPI_SUCCESS;

	/* Do the world model related device initialization here. */
	MPIDI_Process.use_world_model = 1;

	MPIR_Process.comm_world->pscom_socket = MPIDI_Process.socket;
	MPIR_Process.comm_self->pscom_socket = MPIDI_Process.socket;

	/* Call the other init routines */
	mpi_errno = MPID_PSP_comm_init(MPIR_Process.has_parent);
	if (MPI_SUCCESS != mpi_errno) {
		MPIR_ERR_POP(mpi_errno);
	}

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

	return MPI_SUCCESS;

fn_fail:
fn_exit:
    MPIR_FUNC_EXIT;
	return mpi_errno;
}




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
int MPID_Get_universe_size(int *universe_size)
{
	int mpi_errno = MPI_SUCCESS;

	mpi_errno = MPIR_pmi_get_universe_size(universe_size);
	MPIR_ERR_CHECK(mpi_errno);

 fn_exit:
	return mpi_errno;
 fn_fail:
	PRINTERROR("MPI errno: MPIR_pmi_get_universe_size = %d", mpi_errno);
	goto fn_exit;
}

/*
 * MPID_Query_cuda_support - Query CUDA support of the device
 */
int
MPID_Query_cuda_support(void)
{
#if MPIX_CUDA_AWARE_SUPPORT
	return pscom_is_cuda_enabled();
#else
	return 0;
#endif
}
