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

#include <sched.h>
#include "mpidimpl.h"
#include "mpid_psp_request.h"

#define PSP_NBC_PROGRESS_DELAY 100

int MPIDI_PSP_Wait(MPIR_Request * request)
{
    static unsigned int counter_to_nbc_progress = 0;
    int made_progress;
    pscom_request_t *preq = request->dev.kind.common.pscom_req;

    assert(request->kind != MPIR_REQUEST_KIND__UNDEFINED);
    assert(request->kind != MPIR_REQUEST_KIND__GREQUEST);
    assert(request->kind != MPIR_REQUEST_KIND__COLL);
    assert(request->kind != MPIR_REQUEST_KIND__MPROBE);
    assert(request->kind != MPIR_REQUEST_KIND__LAST);

    MPID_PSP_LOCKFREE_CALL(pscom_wait(preq));

    /* The progress counting here is just for reducing the scheduling impact onto pt2pt
     * latencies if a lot of non-blocking collectives are pending without progress: */
    counter_to_nbc_progress++;
    if (counter_to_nbc_progress == PSP_NBC_PROGRESS_DELAY) {
        MPIDU_Sched_progress(&made_progress);
        if (made_progress)
            counter_to_nbc_progress--;
        else
            counter_to_nbc_progress = 0;
    }

    return MPI_SUCCESS;
}


void MPIDI_PSP_Progress_start(MPID_Progress_state * state)
{
}


#define WAIT_DEBUG(str)

#if !defined(WAIT_DEBUG)
#include <unistd.h>

#define WAIT_DEBUG(str) do {									\
	printf("#%d %s() line %d." str "\n", MPIDI_Process.my_pg_rank, __func__, __LINE__);	\
	sleep(2);										\
} while (0)
#endif

/*  Wait for some communication since 'MPID_Progress_start'  */
int MPIDI_PSP_Progress_wait(MPID_Progress_state * state)
{
    int made_progress = 0;
    int mpi_errno;

    /* Make progress on nonblocking collectives */
    mpi_errno = MPIDU_Sched_progress(&made_progress);
    assert(mpi_errno == MPI_SUCCESS);

    if (!made_progress) {
        /* Make progress on pscom requests */
        MPID_PSP_LOCKFREE_CALL(pscom_wait_any());
    }
    return MPI_SUCCESS;
}


void MPIDI_PSP_Progress_end(MPID_Progress_state * state)
{
}


/*
  MPID_Progress_test - Check for communication

  Return value:
  An mpi error code.

  Notes:
  Unlike 'MPID_Progress_wait', this routine is nonblocking.  Therefore, it
  does not require the use of 'MPID_Progress_start' and 'MPID_Progress_end'.
*/
int MPIDI_PSP_Progress_test(void)
{
    int made_progress = 0;
    int mpi_errno;

    /* Make progress on nonblocking collectives */
    mpi_errno = MPIDU_Sched_progress(&made_progress);
    assert(mpi_errno == MPI_SUCCESS);

    pscom_test_any();
    return MPI_SUCCESS;
}

/*
  MPID_Progress_poke - Allow a progress engine to check for pending
  communication

  Return value:
  An mpi error code.

  Notes:
  This routine provides a way to invoke the progress engine in a polling
  implementation of the ADI.  This routine must be nonblocking.

  A multithreaded implementation is free to define this as an empty macro.

  ch3 use this: #define MPIDI_CH3_Progress_poke() (MPIDI_CH3_Progress_test())
*/
int MPIDI_PSP_Progress_poke(void)
{
    MPID_Progress_test(NULL);
    return MPI_SUCCESS;
}
