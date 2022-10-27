/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2022 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <unistd.h>
#include <signal.h>
#include "mpidimpl.h"

static
int _getenv_i(const char *env_name, int _default)
{
	char *val = getenv(env_name);
	return val ? atoi(val) : _default;
}

int MPIDI_PSP_print_psp_stats(void *param ATTRIBUTE((unused)))
{
#ifdef MPID_PSP_HISTOGRAM
	if (MPIDI_Process.env.enable_histogram && MPIDI_Process.stats.histo.points > 0) {

		int idx;
		MPIR_Errflag_t errflag = MPIR_ERR_NONE;

		if (MPIR_Process.comm_world->rank != 0) {
			MPIR_Reduce_impl(MPIDI_Process.stats.histo.count, NULL, MPIDI_Process.stats.histo.points, MPI_LONG_LONG_INT, MPI_SUM, 0, MPIR_Process.comm_world, &errflag);
		} else {
			MPIR_Reduce_impl(MPI_IN_PLACE, MPIDI_Process.stats.histo.count, MPIDI_Process.stats.histo.points, MPI_LONG_LONG_INT, MPI_SUM, 0, MPIR_Process.comm_world, &errflag);

			/* determine digits for formated printing */
			int max_limit = MPIDI_Process.stats.histo.limit[MPIDI_Process.stats.histo.points-2];
			int max_digits;

			for (max_digits = 0; max_limit > 0; ++max_digits) {
				max_limit /= 10;
			}

			/* print the histogram */
			if (!MPIDI_Process.stats.histo.con_type_str)
				printf(" %*s  freq\n", max_digits, "bin");
			else
				printf(" %*s  freq (%s)\n", max_digits, "bin", MPIDI_Process.stats.histo.con_type_str);
			for (idx=0; idx < MPIDI_Process.stats.histo.points; idx++) {
				printf("%c%*d  %lld\n", (idx < MPIDI_Process.stats.histo.points-1) ? ' ' : '>', max_digits, MPIDI_Process.stats.histo.limit[idx-(idx == MPIDI_Process.stats.histo.points-1)], MPIDI_Process.stats.histo.count[idx]);
			}
		}
	}
#endif

#ifdef MPID_PSP_HCOLL_STATS
	if (MPIDI_Process.env.enable_hcoll_stats) {

		int op;
		int max_limit;
		int max_digits[mpidi_psp_stats_collops_enum__MAX];
		MPIR_Errflag_t errflag = MPIR_ERR_NONE;

		for (op = 0; op < mpidi_psp_stats_collops_enum__MAX; op ++) {
			max_limit = MPIDI_Process.stats.hcoll.counter[op];
			for (max_digits[op] = 0; max_limit > 0; ++max_digits[op]) {
				max_limit /= 10;
			}
		}
		MPIR_Allreduce_impl(MPI_IN_PLACE, max_digits, mpidi_psp_stats_collops_enum__MAX, MPI_INT, MPI_MAX, MPIR_Process.comm_world, &errflag);
		printf("(r%07d) hcoll stats | Barrier: %*lld | Bcast: %*lld | Reduce: %*lld | Allreduce: %*lld | Allgather: %*lld | Alltoall: %*lld | Alltoallv: %*lld\n",
		       MPIDI_Process.my_pg_rank,
		       max_digits[mpidi_psp_stats_collops_enum__barrier],   MPIDI_Process.stats.hcoll.counter[mpidi_psp_stats_collops_enum__barrier],
		       max_digits[mpidi_psp_stats_collops_enum__bcast],     MPIDI_Process.stats.hcoll.counter[mpidi_psp_stats_collops_enum__bcast],
		       max_digits[mpidi_psp_stats_collops_enum__reduce],    MPIDI_Process.stats.hcoll.counter[mpidi_psp_stats_collops_enum__reduce],
		       max_digits[mpidi_psp_stats_collops_enum__allreduce], MPIDI_Process.stats.hcoll.counter[mpidi_psp_stats_collops_enum__allreduce],
		       max_digits[mpidi_psp_stats_collops_enum__allgather], MPIDI_Process.stats.hcoll.counter[mpidi_psp_stats_collops_enum__allgather],
		       max_digits[mpidi_psp_stats_collops_enum__alltoall],  MPIDI_Process.stats.hcoll.counter[mpidi_psp_stats_collops_enum__alltoall],
		       max_digits[mpidi_psp_stats_collops_enum__alltoallv], MPIDI_Process.stats.hcoll.counter[mpidi_psp_stats_collops_enum__alltoallv]);
	}
#endif
	return 0;
}

int MPID_Finalize(void)
{
	MPIDI_PG_t *pg_ptr;
	int env_finalize_barrier;
	int env_finalize_shutdown;
	int env_finalize_exit;

	MPIR_FUNC_ENTER;

	/*
	  - PSP_FINALIZE_BARRIER (default=0) -- If set to 0, then an experimental
	  sparse synchronization scheme is used. If set to 2, then the PMI's
	  PMI_Barrier() method is used instead. If it is unset or set to 1, the
	  common MPIR_Barrier() is still used.

	  - PSP_FINALIZE_TIMEOUT (default=30) -- Number of seconds that are allowed
	  to elapse in MPI_Finalize() after leaving the first MPIR_Barrier() call
	  until the second barrier call is aborted via a timeout signal.
	  If set to 0, then no timeout and no second barrier are used.

	  - PSP_FINALIZE_SHUTDOWN (default=0) -- If set to >=1, all pscom sockets
	  are already shut down (synchronized) within MPI_Finalize().

	  - PSP_FINALIZE_EXIT (default=0) -- If set to 1, then exit() is called
	  at the very end of MPI_Finalize(). If set to 2, then it is _exit().
	*/
	env_finalize_barrier  = _getenv_i("PSP_FINALIZE_BARRIER", 0);
	env_finalize_shutdown = _getenv_i("PSP_FINALIZE_SHUTDOWN", 0);
	env_finalize_exit     = _getenv_i("PSP_FINALIZE_EXIT", 0);

	if (env_finalize_barrier == 1) {

		/* A sparse synchronization scheme (!experimental!) that just uses the actually established connections: */

		int j;

		pg_ptr = MPIDI_Process.my_pg;

		pscom_test_any();

		for(j=0; j<pg_ptr->size; j++) {

			if( (j != MPIDI_Process.my_pg_rank) &&
			    (( (pg_ptr->cons[j]->type != PSCOM_CON_TYPE_ONDEMAND) && (pg_ptr->cons[j]->state == PSCOM_CON_STATE_RW) ) ||
			     ( (pg_ptr->cons[j]->type == PSCOM_CON_TYPE_ONDEMAND) && (pg_ptr->cons[j]->state != PSCOM_CON_STATE_RW) ) ))
			{
				MPIDI_PSP_SendCtrl(0, MPIR_CONTEXT_INTRA_COLL, MPIDI_Process.my_pg_rank, pg_ptr->cons[j], MPID_PSP_MSGTYPE_FINALIZE_TOKEN);
			}
		}
		for(j=0; j<pg_ptr->size; j++) {

			if( (j != MPIDI_Process.my_pg_rank) &&
			    (( (pg_ptr->cons[j]->type != PSCOM_CON_TYPE_ONDEMAND) && (pg_ptr->cons[j]->state == PSCOM_CON_STATE_RW) ) ||
			     ( (pg_ptr->cons[j]->type == PSCOM_CON_TYPE_ONDEMAND) && (pg_ptr->cons[j]->state != PSCOM_CON_STATE_RW) ) ))
			{
				MPIDI_PSP_RecvCtrl(0, MPIR_CONTEXT_INTRA_COLL, j, pg_ptr->cons[j],  MPID_PSP_MSGTYPE_FINALIZE_TOKEN);
			}
		}

	} else if (env_finalize_barrier == 2) {

		/* Use PMI_Barrier() for synchronization instead of the MPI Barrier (!no MPI progress here!) */

		MPIR_pmi_barrier();

	}

/*	fprintf(stderr, "%d cleanup queue\n", MPIDI_Process.my_pg_rank); */
//	MPID_req_queue_cleanup();

	MPID_PSP_rma_cleanup();

	MPII_Keyval *keyval_ptr;
	MPII_Keyval_get_ptr(MPIDI_Process.shm_attr_key, keyval_ptr);
	MPIR_free_keyval(keyval_ptr);

	if (MPIDI_Process.use_world_model)
	{
		MPI_Info_delete(MPI_INFO_ENV, "cuda_aware");
#ifdef MPID_PSP_MSA_AWARENESS
		if (MPIDI_Process.msa_module_id >= 0) {
			MPI_Info_delete(MPI_INFO_ENV, "msa_module_id");
		}
		if (MPIDI_Process.smp_node_id >= 0 && MPIDI_Process.env.enable_msa_awareness) {
			MPI_Info_delete(MPI_INFO_ENV, "msa_node_id");
		}
#endif
	}

	MPIR_pmi_finalize();

	/* Cleanups */
	pg_ptr = MPIDI_Process.my_pg->next;
	while(pg_ptr) {
		pg_ptr = MPIDI_PG_Destroy(pg_ptr);
	}
	MPIDI_PG_Destroy(MPIDI_Process.my_pg);

	MPL_free(MPIDI_Process.grank2con);
	MPIDI_Process.grank2con = NULL;

	MPL_free(MPIDI_Process.pg_id_name);
	MPIDI_Process.pg_id_name = NULL;

#ifdef MPID_PSP_HISTOGRAM
	MPL_free(MPIDI_Process.stats.histo.limit);
	MPL_free(MPIDI_Process.stats.histo.count);
#endif
	if (env_finalize_shutdown) {
		/* Close _all_ pscom sockets here (NULL is a wildcard for this),
		   which implies a synchronized shutdown of all pscom connections
		   right here (and thus before the pscom's atexit() handler). */
		pscom_close_socket(NULL);
		/* Caution: This feature is currently for internal purposes only!
		   If it should ever be made official, then the ABI version of the
		   pscom should be adapted and checked here for compatibility!
		   ...otherwise, a NULL argument for pscom_close_socket() will lead
		   to a crash with older pscom versions (<= 5.4.8) here.
		*/
	}

	MPIR_FUNC_EXIT;

	/* Exit here? */
	if (env_finalize_exit == 1) {
		exit(MPI_SUCCESS);
	} else if (env_finalize_exit == 2) {
		_exit(MPI_SUCCESS);
	}

	return MPI_SUCCESS;
}
