/*
 * ParaStation
 *
 * Copyright (C) 2016 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 */

#include <mpi.h>
#include <stdio.h>
#include <assert.h>

#define NUM_WINS 10

int comm_world_rank;
int comm_world_size;

MPI_Comm comm_shmem;
int comm_shmem_rank;
int comm_shmem_size;

void *win_alloc(MPI_Win *win, int byte, int dims)
{
        int disp;
        int qdisp;
        int size;
        MPI_Aint qsize;
        void *ptr;
        void *qptr;

        if(comm_shmem_rank == 0) {
                size = dims + byte;
        } else {
                size = 0;
        }
        
        disp = 1;
        
        MPI_Win_allocate_shared(size, disp, MPI_INFO_NULL, comm_shmem, &ptr, win);

        if(comm_shmem_rank == 0) {
                MPI_Win_shared_query(*win, 0, &qsize, &qdisp, &qptr);
                assert(qsize == size);
                assert(qdisp == disp);
                assert(qptr == ptr);
        }

        return ptr;
}

int main(int argc, char *argv[])
{
        int i;
        MPI_Win win_array[NUM_WINS];

        MPI_Init(&argc, &argv);

        MPI_Comm_size(MPI_COMM_WORLD, &comm_world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_world_rank);
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm_shmem);
        
        MPI_Comm_size(comm_shmem, &comm_shmem_size);
        MPI_Comm_rank(comm_shmem, &comm_shmem_rank);

        for(i=0; i<NUM_WINS; i++) {
                win_alloc(&win_array[i], sizeof(double), 100*100);
        }

        for(i=0; i<NUM_WINS; i++) {
                MPI_Win_free(&win_array[i]);
        }

        MPI_Comm_free(&comm_shmem);

        MPI_Finalize();

        return 0;
}
