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

#include "mpidimpl.h"






/* The following is a temporary hook to ensure that all processes in
   a communicator have a set of process groups.

   All arguments are input (all processes in comm must have gpids)

   First: all processes check to see if they have information on all
   of the process groups mentioned by id in the array of gpids.

   The local result is LANDed with Allreduce.
   If any process is missing process group information, then the
   root process broadcasts the process group information as a string;
   each process then uses this information to update to local process group
   information (in the KVS cache that contains information about
   contacting any process in the process groups).
*/
int MPID_PG_ForwardPGInfo( MPID_Comm *peer_ptr, MPID_Comm *comm_ptr,
			   int nPGids, int gpids[],
			   int root )
{
	/* ToDo: Dont know, what to do here... */
	return MPI_SUCCESS;
}


/* ToDo:
  From src/mpid/ch3/src/mpid_vc.c:
  used in src/mpi/comm/intercomm_merge.c.
*/
int MPID_GPID_Get( MPID_Comm *comm_ptr, int rank, int gpid[] )
{
	int      pgid;
	MPID_VCR vc;

	vc = comm_ptr->vcr[rank];

	/* Get the process group id as an int */
	/*jh MPIDI_PG_IdToNum( vc->pg, &pgid );*/
	pgid = 42;
	gpid[0] = pgid;
	/*jh gpid[1] = vc->pg_rank; */
	gpid[1] = vc->lpid;

	return 0;
}

/* ToDo:
   From src/mpid/ch3/src/mpid_comm_disconnect.c
   used in src/mpi/spawn/comm_disconnect.c
*/
int MPID_Comm_disconnect(MPID_Comm *comm_ptr)
{
    int mpi_errno;
    /* Before releasing the communicator, we need to ensure that all VCs are
       in a stable state.  In particular, if a VC is still in the process of
       connecting, complete the connection before tearing it down */
    /* FIXME: How can we get to a state where we are still connecting a VC but
       the MPIR_Comm_release will find that the ref count decrements to zero
       (it may be that some operation fails to increase/decrease the reference
       count.  A patch could be to increment the reference count while
       connecting, then decrement it.  But the increment in the reference
       count should come
       from the step that caused the connection steps to be initiated.
       Possibility: if the send queue is not empty, the ref count should
       be higher.  */
    /* FIXME: This doesn't work yet */
    /*
    mpi_errno = MPIDI_CH3U_Comm_FinishPending( comm_ptr );
    */

    /* it's more than a comm_release, but ok for now */
    /* FIXME: Describe what more might be required */
    /* MPIU_PG_Printall( stdout ); */
    mpi_errno = MPIR_Comm_release(comm_ptr,1);
    /* If any of the VCs were released by this Comm_release, wait
     for those close operations to complete */
/*jh    MPIDI_CH3U_VC_WaitForClose();*/
    /* MPIU_PG_Printall( stdout ); */

    return mpi_errno;
}


/* This allows each channel to perform final initialization after the
 rest of MPI_Init completes.  */
int MPID_InitCompleted( void )
{
    return MPI_SUCCESS;
}
