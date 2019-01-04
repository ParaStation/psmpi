/*
 * ParaStation
 *
 * Copyright (C) 2006-2009 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include <unistd.h>
#include <signal.h>
#include "mpidimpl.h"
#include "pmi.h"

// This must be the last include before sysmbols are defined:
#include "mpid_visibility.h"


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

	MPIDI_STATE_DECL(MPID_STATE_MPID_FINALIZE);
	MPIDI_FUNC_ENTER(MPID_STATE_MPID_FINALIZE);

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
				MPID_PSP_SendCtrl(0, MPID_CONTEXT_INTRA_COLL, MPIDI_Process.my_pg_rank, pg_ptr->cons[j], MPID_PSP_MSGTYPE_FINALIZE_TOKEN);
			}
		}
		for(j=0; j<pg_ptr->size; j++) {

			if( (j != MPIDI_Process.my_pg_rank) &&
			    (( (pg_ptr->cons[j]->type != PSCOM_CON_TYPE_ONDEMAND) && (pg_ptr->cons[j]->state == PSCOM_CON_STATE_RW) ) ||
			     ( (pg_ptr->cons[j]->type == PSCOM_CON_TYPE_ONDEMAND) && (pg_ptr->cons[j]->state != PSCOM_CON_STATE_RW) ) ))
			{
				MPID_PSP_RecvCtrl(0, MPID_CONTEXT_INTRA_COLL, j, pg_ptr->cons[j],  MPID_PSP_MSGTYPE_FINALIZE_TOKEN);
			}
		}

	} else {

		/* The common barrier synchronization across comm_world within MPI Finalize: */

		MPIR_Errflag_t errflag = MPIR_ERR_NONE;
		int timeout;
		MPIU_THREADPRIV_DECL;
		MPIU_THREADPRIV_GET;

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

#ifdef MPID_PSP_CREATE_HISTOGRAM
	if (MPIDI_Process.env.enable_histogram && MPIDI_Process.histo.points > 0) {

		int idx;
		MPIR_Errflag_t errflag = MPIR_ERR_NONE;

		if (MPIR_Process.comm_world->rank != 0) {
			MPIR_Reduce_impl(MPIDI_Process.histo.count, NULL, MPIDI_Process.histo.points, MPI_LONG_LONG_INT, MPI_SUM, 0, MPIR_Process.comm_world, &errflag);
		} else {
			MPIR_Reduce_impl(MPI_IN_PLACE, MPIDI_Process.histo.count, MPIDI_Process.histo.points, MPI_LONG_LONG_INT, MPI_SUM, 0, MPIR_Process.comm_world, &errflag);

			/* determine digits for formated printing */
			int max_limit = MPIDI_Process.histo.limit[MPIDI_Process.histo.points-1];
			int max_digits;

			for (max_digits = 0; max_limit > 0; ++max_digits) {
				max_limit /= 10;
			}

			/* print the histogram */
			printf("%*s  freq\n", max_digits, "bin");
			for (idx=0; idx < MPIDI_Process.histo.points; idx++) {
				printf("%*d  %lld\n", max_digits, MPIDI_Process.histo.limit[idx], MPIDI_Process.histo.count[idx]);
			}
		}

		MPIU_Free(MPIDI_Process.histo.limit);
		MPIU_Free(MPIDI_Process.histo.count);
	}
#endif


/*	fprintf(stderr, "%d cleanup queue\n", MPIDI_Process.my_pg_rank); */
	MPID_req_queue_cleanup();

	MPID_PSP_rma_cleanup();

	MPIR_Comm_free_keyval_impl(MPIDI_Process.shm_attr_key);

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

	MPIU_Free(MPIDI_Process.grank2con);
	MPIDI_Process.grank2con = NULL;

	MPIU_Free(MPIDI_Process.pg_id_name);
	MPIDI_Process.pg_id_name = NULL;

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
	MPIU_Free(MPIDI_Process.node_id_table);
	MPIDI_Process.node_id_table = NULL;
#endif

	MPIDI_FUNC_EXIT(MPID_STATE_MPID_FINALIZE);
	return MPI_SUCCESS;
}
