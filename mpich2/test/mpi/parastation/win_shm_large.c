/*
 * ParaStation
 *
 * Copyright (C) 2017 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 */

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

int comm_world_rank;
int comm_world_size;

MPI_Comm comm_shmem;
int comm_shmem_rank;
int comm_shmem_size;

#define STRIDE 4096
#define SIZE (((long)1<<32)+STRIDE)


void *win_alloc(MPI_Win *win, MPI_Aint size)
{
        int disp;
        int qdisp;
        MPI_Aint qsize;
        void *ptr = NULL;
        void *qptr = NULL;

        disp = 1;

        if(comm_shmem_rank == 0) {
		MPI_Win_allocate_shared(size, disp, MPI_INFO_NULL, comm_shmem, &ptr, win);
		assert(ptr != NULL);
		MPI_Win_shared_query(*win, 0, &qsize, &qdisp, &qptr);
		assert(qptr == ptr);

        } else {
		MPI_Win_allocate_shared(0, disp, MPI_INFO_NULL, comm_shmem, &ptr, win);
		assert(ptr != NULL);
		MPI_Win_shared_query(*win, 0, &qsize, &qdisp, &qptr);
		assert(qptr != NULL);
	}

	assert(qsize == size);
	assert(qdisp == disp);

        return qptr;
}

int main(int argc, char *argv[])
{
        int i;
	size_t j;
	void *ptr;
        MPI_Win win;

        MPI_Init(&argc, &argv);

        MPI_Comm_size(MPI_COMM_WORLD, &comm_world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_world_rank);
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm_shmem);

        MPI_Comm_size(comm_shmem, &comm_shmem_size);
        MPI_Comm_rank(comm_shmem, &comm_shmem_rank);

	assert(sizeof(MPI_Aint) == sizeof(long));
	assert(sizeof(size_t) == sizeof(unsigned long));

	ptr = win_alloc(&win, SIZE);

	for(i=0; i<comm_shmem_size; i++) {
		if(comm_shmem_rank == i); {
			for(j=0; j<SIZE; j+=STRIDE) {
				*((char*)(ptr+j)) = i;
			}
		}
		MPI_Barrier(comm_shmem);
		for(j=0; j<SIZE; j+=STRIDE) {
			assert(*((char*)(ptr+j)) == i);
		}
		MPI_Barrier(comm_shmem);
	}

	MPI_Win_free(&win);        

        MPI_Comm_free(&comm_shmem);

        MPI_Finalize();

	if(comm_world_rank == 0) {
		printf(" No Errors\n");
	}

        return 0;
}
