/*
 * ParaStation
 *
 * Copyright (C) 2019-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2022 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include "mpitest.h"

static int errors = 0;

void *win_alloc_shared(MPI_Win *win, MPI_Aint size, int disp, MPI_Comm comm)
{
	int qdisp = 0;
	MPI_Aint qsize = 0;
	void *ptr = NULL;
	void *qptr = NULL;

	int comm_rank;
	int comm_size;

	MPI_Comm_size(comm, &comm_size);
	MPI_Comm_rank(comm, &comm_rank);

	MPI_Win_allocate_shared(size, disp, MPI_INFO_NULL, comm, &ptr, win);

	MPI_Win_shared_query(*win, comm_rank, &qsize, &qdisp, &qptr);

	if (qsize != size) {
		printf("(%d) Window sizes do not match: %ld vs. %ld\n", comm_rank, qsize, size);
		errors++;
	}
	if (qdisp != disp) {
		printf("(%d) Window displacement units do not match: %d vs. %d\n", comm_rank, qdisp, disp);
		errors++;
	}
	if (qptr != ptr) {
		printf("(%d) Window pointers do not match: %p vs. %p\n", comm_rank, qptr, ptr);
		errors++;
	}

	return ptr;
}

int main(int argc, char *argv[])
{
	int comm_world_rank;
	int comm_world_size;
	int comm_shmem_rank;
	int comm_shmem_size;
	int comm_dup_rank;
	int comm_dup_size;
	MPI_Comm comm_shmem;
	MPI_Comm comm_dup;
	MPI_Win win_shmem;
	MPI_Win win_dup;

	MTest_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_world_rank);

	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm_shmem);
	MPI_Comm_size(comm_shmem, &comm_shmem_size);
	MPI_Comm_rank(comm_shmem, &comm_shmem_rank);

	win_alloc_shared(&win_shmem, 1024, sizeof(double), comm_shmem);

	MPI_Comm_dup(comm_shmem, &comm_dup);
	MPI_Comm_size(comm_dup, &comm_dup_size);
	MPI_Comm_rank(comm_dup, &comm_dup_rank);

	if (comm_dup_size != comm_shmem_size) {
		printf("Communicator sizes do not match: %d vs. %d\n", comm_dup_size, comm_shmem_size);
		errors++;
	}
	if (comm_dup_rank != comm_shmem_rank) {
		printf("Communicator ranks do not match: %d vs. %d\n", comm_dup_rank, comm_shmem_rank);
		errors++;
	}

	win_alloc_shared(&win_dup, 1024, sizeof(double), comm_shmem);

	MPI_Win_free(&win_dup);
	MPI_Comm_free(&comm_dup);

	MPI_Win_free(&win_shmem);
	MPI_Comm_free(&comm_shmem);

	MTest_Finalize(errors);

	return MTestReturnValue(errors);
}
