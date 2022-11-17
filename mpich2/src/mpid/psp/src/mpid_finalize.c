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

#include <unistd.h>
#include <signal.h>
#include "mpidimpl.h"

int MPIDI_PSP_finalize_print_stats_cb(void *param ATTRIBUTE((unused)))
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

static
void sig_finalize_timeout(int signo ATTRIBUTE((unused)))
{
	if (MPIDI_Process.env.debug_level > 0) {
		fprintf(stderr, "Warning: PSP_FINALIZE_TIMEOUT\n");
	}
	_exit(0);
}

int MPIDI_PSP_finalize_add_barrier_cb(void *param ATTRIBUTE((unused)))
{
	if (MPIDI_Process.env.finalize.barrier == 1) {

		/* The common barrier synchronization across comm_world within MPI Finalize: */

		MPIR_Errflag_t errflag = MPIR_ERR_NONE;
		int timeout;
		// TODO: check THREADPRIV API!

		MPIR_Barrier_impl(MPIR_Process.comm_world, &errflag);

		/* Finalize timeout: Default: 30sec.
		   Overwrite with PSP_FINALIZE_TIMEOUT.
		   Disable with PSP_FINALIZE_TIMEOUT=0 */
		timeout = MPIDI_Process.env.finalize.timeout;
		if (timeout > 0) {
			signal(SIGALRM, sig_finalize_timeout);
			alarm(timeout);
			MPIR_Barrier_impl(MPIR_Process.comm_world, &errflag);
		}

	} else if (MPIDI_Process.env.finalize.barrier == 2) {

		/* Use PMI_Barrier() for synchronization instead of the MPI Barrier (!no MPI progress here!) */

		MPIR_pmi_barrier();
	}

	return 0;
}

int MPID_Finalize(void)
{
	MPIDI_PG_t *pg_ptr;

	MPIR_FUNC_ENTER;

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
	if(MPIDI_Process.my_pg){
		pg_ptr = MPIDI_Process.my_pg->next;
		while(pg_ptr) {
			pg_ptr = MPIDI_PG_Destroy(pg_ptr);
		}
		MPIDI_PG_Destroy(MPIDI_Process.my_pg);
		MPIDI_Process.my_pg = NULL;
	}
	/*for re-init*/
	MPIDI_Process.next_lpid = 0;

	MPL_free(MPIDI_Process.grank2con);
	MPIDI_Process.grank2con = NULL;

	MPL_free(MPIDI_Process.pg_id_name);
	MPIDI_Process.pg_id_name = NULL;

#ifdef MPID_PSP_HISTOGRAM
	MPL_free(MPIDI_Process.stats.histo.limit);
	MPL_free(MPIDI_Process.stats.histo.count);
#endif
	if (MPIDI_Process.env.finalize.shutdown) {
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
	if (MPIDI_Process.env.finalize.exit == 1) {
		exit(MPI_SUCCESS);
	} else if (MPIDI_Process.env.finalize.exit == 2) {
		_exit(MPI_SUCCESS);
	}

	return MPI_SUCCESS;
}
