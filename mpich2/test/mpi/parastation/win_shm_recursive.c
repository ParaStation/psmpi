/*
 * ParaStation
 *
 * Copyright (C) 2019 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 */

#include <mpi.h>
#include <stdio.h>
#include <assert.h>

#define _VERBOSE_ 0

void *win_alloc_shared(MPI_Win *win, int byte, int dims, MPI_Comm comm)
{
        int disp;
        int qdisp;
        int size;
        MPI_Aint qsize;
        void *ptr;
        void *qptr;
	int comm_rank;
	int comm_size;

	MPI_Comm_size(comm, &comm_size);
        MPI_Comm_rank(comm, &comm_rank);

        if(comm_rank == 0) {
                size = dims + byte;
        } else {
                size = 0;
        }

        disp = 1;

        MPI_Win_allocate_shared(size, disp, MPI_INFO_NULL, comm, &ptr, win);

        if(comm_rank == 0) {
                MPI_Win_shared_query(*win, 0, &qsize, &qdisp, &qptr);
                assert(qsize == size);
                assert(qdisp == disp);
                assert(qptr == ptr);
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

        MPI_Init(&argc, &argv);

        MPI_Comm_size(MPI_COMM_WORLD, &comm_world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_world_rank);

	if(_VERBOSE_) {
		if(comm_world_rank == 0) printf("(%d) The world size is %d\n", comm_world_rank, comm_world_size);
	}

        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm_shmem);
	MPI_Comm_size(comm_shmem, &comm_shmem_size);
	MPI_Comm_rank(comm_shmem, &comm_shmem_rank);

	if(_VERBOSE_) {
		if(comm_shmem_rank == 0) printf("(%d) The shmem size is %d\n", comm_world_rank, comm_shmem_size);
	}

	win_alloc_shared(&win_shmem, sizeof(double), 100*100, comm_shmem);

	MPI_Barrier(comm_shmem);

	if(comm_shmem_size > 1) {

		color = comm_shmem_rank % 2;
		MPI_Comm_split(comm_shmem, color, comm_shmem_rank, &comm_split);
		MPI_Comm_size(comm_split, &comm_split_size);
		MPI_Comm_rank(comm_split, &comm_split_rank);

		if(_VERBOSE_) {
			if(comm_shmem_rank == 0) printf("(%d) The split size is %d\n", comm_world_rank, comm_split_size);
		}

		win_alloc_shared(&win_split, sizeof(double), 100*100, comm_split);

		MPI_Barrier(comm_split);

		MPI_Win_free(&win_split);
		MPI_Comm_free(&comm_split);
	}

	MPI_Win_free(&win_shmem);
        MPI_Comm_free(&comm_shmem);

        MPI_Finalize();

	if(comm_world_rank == 0) {
		printf(" No Errors\n");
	}

        return 0;
}
