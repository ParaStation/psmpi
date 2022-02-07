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

#include "mpidimpl.h"

int MPID_Send(const void * buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
	      MPIR_Comm * comm, int context_offset, MPIR_Request ** request)
{
	int mpi_errno;
/*	printf("#%d ps--- %s() called\n", MPIDI_Process.my_pg_rank, __func__); */

	mpi_errno = MPIDI_PSP_Isend(buf, count, datatype, rank, tag, comm, context_offset, request);

	if (mpi_errno == MPI_SUCCESS) {
		mpi_errno = MPIDI_PSP_Wait(*request);
	}

	return mpi_errno;
}


int MPID_Send_coll(const void *buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
                   MPIR_Comm * comm, int context_offset, MPIR_Request ** request,
                   MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    switch (*errflag) {
    case MPIR_ERR_NONE:
        break;
    case MPIR_ERR_PROC_FAILED:
        MPIR_TAG_SET_PROC_FAILURE_BIT(tag);
        break;
    default:
        MPIR_TAG_SET_ERROR_BIT(tag);
    }

    mpi_errno = MPID_Send(buf, count, datatype, rank, tag, comm, context_offset, request);

    return mpi_errno;
}


int MPID_Ssend(const void * buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
	       MPIR_Comm * comm, int context_offset, MPIR_Request ** request)
{
	int mpi_errno;
/*	printf("#%d ps--- %s() called\n", MPIDI_Process.my_pg_rank, __func__); */

	mpi_errno = MPIDI_PSP_Issend(buf, count, datatype, rank, tag, comm, context_offset, request);

	if (mpi_errno == MPI_SUCCESS) {
		mpi_errno = MPIDI_PSP_Wait(*request);
	}

	return mpi_errno;
}


/* immediate ready send (mapped to immediate send) */
int MPID_Irsend(const void * buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
		MPIR_Comm * comm, int context_offset, MPIR_Request ** request)
{
	return MPID_Isend(buf, count, datatype, rank, tag, comm, context_offset, request);
}


/* ready send (same as send) */
int MPID_Rsend(const void * buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
	       MPIR_Comm * comm, int context_offset, MPIR_Request ** request)
{
	int mpi_errno;

	mpi_errno = MPIDI_PSP_Isend(buf, count, datatype, rank, tag, comm, context_offset, request);

	if (mpi_errno == MPI_SUCCESS) {
		mpi_errno = MPIDI_PSP_Wait(*request);
	}

	return mpi_errno;
}
