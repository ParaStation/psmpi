/*
 * ParaStation
 *
 * Copyright (C) 2019-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2023 ParTec AG, Munich
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
	int color;
	int comm_world_rank;
	int comm_world_size;
	int comm_shmem_rank;
	int comm_shmem_size;
	int comm_split_rank;
	int comm_split_size;
	MPI_Comm comm_shmem;
	MPI_Comm comm_split;
	MPI_Win win_shmem;
	MPI_Win win_split;

	MTest_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_world_rank);

	if (comm_world_rank == 0) {
		MTestPrintfMsg(2, "(%d) The world size is %d\n", comm_world_rank, comm_world_size);
	}

	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm_shmem);
	MPI_Comm_size(comm_shmem, &comm_shmem_size);
	MPI_Comm_rank(comm_shmem, &comm_shmem_rank);

	if (comm_shmem_rank == 0) {
		MTestPrintfMsg(2, "(%d) The shmem size is %d\n", comm_world_rank, comm_shmem_size);
	}

	win_alloc_shared(&win_shmem, 1024, sizeof(double), comm_shmem);

	if (comm_shmem_size > 1) {

		color = comm_shmem_rank % 2;
		MPI_Comm_split(comm_shmem, color, comm_shmem_rank, &comm_split);
		MPI_Comm_size(comm_split, &comm_split_size);
		MPI_Comm_rank(comm_split, &comm_split_rank);

		if (comm_split_rank == 0) {
			MTestPrintfMsg(2, "(%d) The split size is %d\n", comm_world_rank, comm_split_size);
		}

		win_alloc_shared(&win_split, 1024, sizeof(double), comm_split);

		MPI_Win_free(&win_split);
		MPI_Comm_free(&comm_split);
	}

	MPI_Win_free(&win_shmem);
	MPI_Comm_free(&comm_shmem);

	MTest_Finalize(errors);

	return MTestReturnValue(errors);
}
