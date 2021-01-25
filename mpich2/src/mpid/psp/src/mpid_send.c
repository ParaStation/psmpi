/*
 * ParaStation
 *
 * Copyright (C) 2006-2019 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
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
int MPID_Rsend(const void * buf, int count, MPI_Datatype datatype, int rank, int tag,
	       MPIR_Comm * comm, int context_offset, MPIR_Request ** request)
{
	int mpi_errno;

	mpi_errno = MPIDI_PSP_Isend(buf, count, datatype, rank, tag, comm, context_offset, request);

	if (mpi_errno == MPI_SUCCESS) {
		mpi_errno = MPIDI_PSP_Wait(*request);
	}

	return mpi_errno;
}
