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


MPI_Aint MPID_Aint_add(MPI_Aint base, MPI_Aint disp)
{
	// WTF?
	// result =  MPIU_VOID_PTR_CAST_TO_MPI_AINT ((char*)MPIU_AINT_CAST_TO_VOID_PTR(base) + disp);
	return base + disp;
}


MPI_Aint MPID_Aint_diff(MPI_Aint addr1, MPI_Aint addr2)
{
	// WTF?
	// result =  MPIU_PTR_DISP_CAST_TO_MPI_AINT ((char*)MPIU_AINT_CAST_TO_VOID_PTR(addr1) - (char*)MPIU_AINT_CAST_TO_VOID_PTR(addr2));
	return addr1 - addr2;
}
