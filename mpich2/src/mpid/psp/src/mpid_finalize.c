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
	MPIDI_STATE_DECL(MPID_STATE_MPID_FINALIZE);
	MPIDI_FUNC_ENTER(MPID_STATE_MPID_FINALIZE);
/* ToDo: */
/*	fprintf(stderr, "%d waitall\n", MPIDI_Process.my_pg_rank); */

	{
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
	MPIDI_PG_t* pg_ptr = MPIDI_Process.my_pg->next;
	while(pg_ptr) {
		pg_ptr = MPIDI_PG_Destroy(pg_ptr);
	}
	MPIDI_PG_Destroy(MPIDI_Process.my_pg);

	MPIU_Free(MPIDI_Process.grank2con);
	MPIDI_Process.grank2con = NULL;

	MPIU_Free(MPIDI_Process.pg_id_name);
	MPIDI_Process.pg_id_name = NULL;

#ifdef MPID_PSP_USE_SMP_AWARE_COLLOPS
	MPIU_Free(MPIDI_Process.node_id_table);
	MPIDI_Process.node_id_table = NULL;
#endif

	MPIDI_FUNC_EXIT(MPID_STATE_MPID_FINALIZE);
	return MPI_SUCCESS;
}
