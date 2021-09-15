/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021      ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <unistd.h>
#include <signal.h>
#include "mpidimpl.h"
#include "pmi.h"

static
int _getenv_i(const char *env_name, int _default)
{
	char *val = getenv(env_name);
	return val ? atoi(val) : _default;
}

static
void sig_finalize_timeout(int signo)
{
	if (_getenv_i("PSP_DEBUG", 0) > 0) {
		fprintf(stderr, "Warning: PSP_FINALIZE_TIMEOUT\n");
	}
	_exit(0);
}

int MPID_Finalize(void)
{
	MPIDI_PG_t *pg_ptr;

	MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPID_FINALIZE);
	MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPID_FINALIZE);

	if(!_getenv_i("PSP_FINALIZE_BARRIER", 1)) {

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

	} else {

		/* The common barrier synchronization across comm_world within MPI Finalize: */

		MPIR_Errflag_t errflag = MPIR_ERR_NONE;
		int timeout;
		// TODO: check THREADPRIV API!

		MPIR_Barrier_impl(MPIR_Process.comm_world, &errflag);

		/* Finalize timeout: Default: 30sec.
		   Overwrite with PSP_FINALIZE_TIMEOUT.
		   Disable with PSP_FINALIZE_TIMEOUT=0 */
		timeout = _getenv_i("PSP_FINALIZE_TIMEOUT", 30);
		if (timeout > 0) {
			signal(SIGALRM, sig_finalize_timeout);
			alarm(timeout);
			MPIR_Barrier_impl(MPIR_Process.comm_world, &errflag);
		}
	}

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
		printf("(r%07d) hcoll stats | Barrier: %*lld | Bcast: %*lld | Reduce: %*lld | Allreduce: %*lld | Allgather: %*lld | Alltoall: %*lld | Alltoallv: %*ld\n",
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

/*	fprintf(stderr, "%d cleanup queue\n", MPIDI_Process.my_pg_rank); */
//	MPID_req_queue_cleanup();

	MPID_PSP_rma_cleanup();

	MPIR_Comm_free_keyval_impl(MPIDI_Process.shm_attr_key);

	MPI_Info_delete(MPI_INFO_ENV, "cuda_aware");
#ifdef MPID_PSP_MSA_AWARENESS
	if(MPIDI_Process.msa_module_id >= 0) {
		MPI_Info_delete(MPI_INFO_ENV, "msa_module_id");
	}
	if(MPIDI_Process.smp_node_id >= 0 && MPIDI_Process.env.enable_msa_awareness) {
		MPI_Info_delete(MPI_INFO_ENV, "msa_node_id");
	}
#endif

/*	fprintf(stderr, "%d PMI_Finalize\n", MPIDI_Process.my_pg_rank); */
	PMI_Finalize();

	/* Release standard communicators */
#ifdef MPID_NEEDS_ICOMM_WORLD
	/* psp don't need icomm. But this might change? */
	MPIR_Comm_release_always(MPIR_Process.icomm_world);
#endif
	MPIR_Comm_release_always(MPIR_Process.comm_self);
	MPIR_Comm_release_always(MPIR_Process.comm_world);

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

	MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPID_FINALIZE);
	return MPI_SUCCESS;
}
