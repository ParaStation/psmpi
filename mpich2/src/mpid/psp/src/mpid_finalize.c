/*
 * ParaStation
 *
 * Copyright (C) 2006-2009 ParTec Cluster Competence Center GmbH, Munich
 *
 * All rights reserved.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include <unistd.h>
#include "mpidimpl.h"
#include "pmi.h"

int MPID_Finalize(void)
{
	MPIDI_STATE_DECL(MPID_STATE_MPID_FINALIZE);
	MPIDI_FUNC_ENTER(MPID_STATE_MPID_FINALIZE);
/* ToDo: */
/*	fprintf(stderr, "%d waitall\n", MPIDI_Process.my_pg_rank); */

	{
		int errflag = 0;
		MPIU_THREADPRIV_DECL;
		MPIU_THREADPRIV_GET;

		MPIR_Barrier_impl(MPIR_Process.comm_world, &errflag);
	}

/*	fprintf(stderr, "%d cleanup queue\n", MPIDI_Process.my_pg_rank); */
	MPID_req_queue_cleanup();

	MPID_PSP_rma_cleanup();

/*	fprintf(stderr, "%d PMI_Finalize\n", MPIDI_Process.my_pg_rank); */
	PMI_Finalize();

	MPIDI_FUNC_EXIT(MPID_STATE_MPID_FINALIZE);
	return MPI_SUCCESS;
}
