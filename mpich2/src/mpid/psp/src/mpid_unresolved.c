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
#include "mpiinfo.h"

/* see intercomm_create.c: */
int MPID_GPID_GetAllInComm(MPID_Comm *comm_ptr, int local_size,
			   int local_gpids[], int *singlePG)
{
	int i;
	int *gpid = local_gpids;
	int lastPGID = -1;

	*singlePG = 1;
	for (i=0; i<comm_ptr->local_size; i++) {
		MPID_GPID_Get(comm_ptr, i, gpid);

		if (lastPGID != gpid[0]) {
			if (i == 0) {
				lastPGID = gpid[0];
			} else {
				*singlePG = 0;
			}
		}
		gpid += 2;
	}
	return 0;
}

/* from intercomm_create.c: */
/* FIXME: A temp for lpids within my comm world */
int MPID_GPID_ToLpidArray(int size, int gpid[], int lpid[])
{
	int i;

	for (i=0; i<size; i++) {
		/* Use lpid part of gpid */
		lpid[i] = gpid[1]; /* *++gpid;  gpid++; */
		gpid += 2;
	}
	return 0;
}

/* from intercomm_create.c: */
/* FIXME: for MPI1, all process ids are relative to MPI_COMM_WORLD.
   For MPI2, we'll need to do something more complex */
int MPID_VCR_CommFromLpids( MPID_Comm *newcomm_ptr,
				       int size, const int lpids[] )
{
    MPID_Comm *commworld_ptr;
    int i;

    commworld_ptr = MPIR_Process.comm_world;
    /* Setup the communicator's vc table: remote group */
    MPID_VCRT_Create( size, &newcomm_ptr->vcrt );
    MPID_VCRT_Get_ptr( newcomm_ptr->vcrt, &newcomm_ptr->vcr );
    for (i=0; i<size; i++) {
	/* For rank i in the new communicator, find the corresponding
	   rank in the comm world (FIXME FOR MPI2) */
	/* printf( "[%d] Remote rank %d has lpid %d\n",
	   MPIR_Process.comm_world->rank, i, lpids[i] ); */
	if (lpids[i] < commworld_ptr->remote_size) {
	    MPID_VCR_Dup( commworld_ptr->vcr[lpids[i]],
			  &newcomm_ptr->vcr[i] );
	}
	else {
	    /* We must find the corresponding vcr for a given lpid */
	    /* FIXME: Error */
	    return 1;
	    /* MPID_VCR_Dup( ???, &newcomm_ptr->vcr[i] ); */
	}
    }
    return 0;
}


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
	/* ToDo: Dont know, what to do here... Hope it is obsolete.*/
	return MPI_SUCCESS;
}


/* This allows each channel to perform final initialization after the
 rest of MPI_Init completes.  */
int MPID_InitCompleted( void )
{
    return MPI_SUCCESS;
}


#define WARN_NOT_IMPLEMENTED						\
do {									\
	static int warned = 0;						\
	if (!warned) {							\
		warned = 1;						\
		fprintf(stderr, "Warning: %s() not implemented\n", __func__); \
	}								\
} while (0)


int MPID_Comm_group_failed(MPID_Comm *comm_ptr, MPID_Group **failed_group_ptr)
{
	WARN_NOT_IMPLEMENTED;
	return MPI_ERR_UNSUPPORTED_OPERATION;
}

int MPID_Comm_reenable_anysource(MPID_Comm *comm,
				 MPID_Group **failed_group_ptr)
{
	WARN_NOT_IMPLEMENTED;
	return MPI_ERR_UNSUPPORTED_OPERATION;
}

int MPID_Comm_remote_group_failed(MPID_Comm *comm, MPID_Group **failed_group_ptr)
{
	WARN_NOT_IMPLEMENTED;
	return MPI_ERR_UNSUPPORTED_OPERATION;
}

int MPID_Comm_failure_get_acked(MPID_Comm *comm, MPID_Group **failed_group_ptr)
{
	WARN_NOT_IMPLEMENTED;
	return MPI_ERR_UNSUPPORTED_OPERATION;
}

int MPID_Comm_get_all_failed_procs(MPID_Comm *comm_ptr, MPID_Group **failed_group, int tag)
{
	WARN_NOT_IMPLEMENTED;
	return MPI_ERR_UNSUPPORTED_OPERATION;
}

int MPID_Comm_revoke(MPID_Comm *comm, int is_remote)
{
	WARN_NOT_IMPLEMENTED;
	return MPI_ERR_UNSUPPORTED_OPERATION;
}

int MPID_Comm_failure_ack(MPID_Comm *comm)
{
	WARN_NOT_IMPLEMENTED;
	return MPI_ERR_UNSUPPORTED_OPERATION;
}

int MPID_Win_set_info(MPID_Win *win, MPID_Info *info)
{
	int mpi_errno = MPI_SUCCESS;
	/* No op, info arguments are ignored by default */
	return mpi_errno;
}

int MPID_Win_get_info(MPID_Win *win, MPID_Info **info_used)
{
	int mpi_errno = MPI_SUCCESS;

	/* Allocate an empty info object */
	mpi_errno = MPIU_Info_alloc(info_used);
	assert(mpi_errno == MPI_SUCCESS);

	return mpi_errno;
}
