/*
 * ParaStation
 *
 * Copyright (C) 2017-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2024 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include <stdio.h>

int sbuf[128];
int rbuf[128];

int main(int argc, char *argv[])
{
    int i;
    int errs = 0;
    int world_rank, world_size;
    int split_rank, split_size;
    int remote_size;

    MPI_Comm split_comm;
    MPI_Comm inter_comm;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm_split(MPI_COMM_WORLD, world_rank % 2, 0, &split_comm);

    MPI_Comm_rank(split_comm, &split_rank);
    MPI_Comm_size(split_comm, &split_size);

    MPI_Intercomm_create(split_comm, 0, MPI_COMM_WORLD, (world_rank + 1) % 2, 666, &inter_comm);

    MPI_Comm_remote_size(inter_comm, &remote_size);

    //printf("(%d) split size: %d / remote size: %d\n", world_rank, split_size, remote_size);

    if (world_rank % 2) {

        for (i = 0; i < 128; i++)
            sbuf[i] = i;

        MPI_Send(sbuf, 128, MPI_INT, split_rank, 999, inter_comm);

    } else if (split_rank < remote_size) {

        for (i = 0; i < 128; i++)
            rbuf[i] = 0;

        MPI_Recv(rbuf, 128, MPI_INT, MPI_ANY_SOURCE, 999, inter_comm, MPI_STATUS_IGNORE);

        for (i = 0; i < 128; i++) {
            if (rbuf[i] != i) {
                printf("ERROR: received: %d / expected: %d\n", rbuf[i], i);
                errs++;
            }
        }

    }

    MPI_Reduce((world_rank == 0 ? MPI_IN_PLACE : &errs), &errs, 1, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (world_rank == 0 && !errs) {
        printf(" No Errors\n");
    }

    MPI_Comm_free(&split_comm);
    MPI_Comm_free(&inter_comm);

    MPI_Finalize();
    return 0;
}
