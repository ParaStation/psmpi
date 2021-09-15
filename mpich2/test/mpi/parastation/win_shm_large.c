/*
 * ParaStation
 *
 * Copyright (C) 2017-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021      ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "mpitest.h"

#define STRIDE 4096
#define SIZE (((long)1<<32)+STRIDE)

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

	if (comm_rank == 0) {
		MPI_Win_allocate_shared(size, disp, MPI_INFO_NULL, comm, &ptr, win);
		if (ptr == NULL) {
			printf("(%d) Window base pointer is NULL (but should not)\n", comm_rank);
			errors++;
		}
		MPI_Win_shared_query(*win, 0, &qsize, &qdisp, &qptr);
		if (qptr != ptr) {
			printf("(%d) Window pointers do not match: %p vs. %p\n", comm_rank, qptr, ptr);
			errors++;
		}
	} else {
		MPI_Win_allocate_shared(0, disp, MPI_INFO_NULL, comm, &ptr, win);
		MPI_Win_shared_query(*win, comm_rank, &qsize, &qdisp, &qptr);
		if (qsize != 0) {
			printf("(%d) Window sizes do not match: %ld vs. 0\n", comm_rank, qsize);
			errors++;
		}
		MPI_Win_shared_query(*win, 0, &qsize, &qdisp, &qptr);
		if (qptr == NULL) {
			printf("(%d) Window pointer to rank 0 is NULL (but should not)\n", comm_rank);
			errors++;
		}
	}

	if (qsize != size) {
		printf("(%d) Window sizes do not match: %ld vs. %ld\n", comm_rank, qsize, size);
		errors++;
	}
	if (qdisp != disp) {
		printf("(%d) Window displacement units do not match: %d vs. %d\n", comm_rank, qdisp, disp);
		errors++;
	}

        return qptr;
}

int main(int argc, char *argv[])
{
	int i;
	size_t j;
	int comm_world_rank;
	int comm_world_size;
	int comm_shmem_rank;
	int comm_shmem_size;
	MPI_Comm comm_shmem;
	MPI_Win win;
	void *ptr;

	MTest_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_world_rank);

	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm_shmem);
	MPI_Comm_size(comm_shmem, &comm_shmem_size);
	MPI_Comm_rank(comm_shmem, &comm_shmem_rank);

	// If SMP-awareness is disabled, it is quite likely that comm_shmem equals COMM_SELF! Just skip this test then...
	if (comm_shmem_size == 1) goto finalize;

	ptr = win_alloc_shared(&win, SIZE, sizeof(char), comm_shmem);

	for (i=0; i < comm_shmem_size; i++) {
		if (comm_shmem_rank == i) {
			for (j=0; j < SIZE; j+=STRIDE) {
				*((char*)(ptr+j)) = (char)i;
			}
		}
		MPI_Barrier(comm_shmem);
		for (j=0; j < SIZE; j+=STRIDE) {
			if (*((char*)(ptr+j)) != (char)i) {
				printf("(%d|%d) Error at %p + %ld = %p: %d vs. %d\n",
				       comm_world_rank, comm_shmem_rank,
				       ptr, j, ptr+j, *((char*)(ptr+j)), (char)i);
				errors++;
			}
		}
		MPI_Barrier(comm_shmem);
	}

	MPI_Win_free(&win);

finalize:
	MPI_Comm_free(&comm_shmem);
	MTest_Finalize(errors);

	return MTestReturnValue(errors);
}
