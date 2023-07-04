/*
 * ParaStation
 *
 * Copyright (C) 2021-2023 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpitest.h"

int factor = 2;                 // <- valid factors are within [0..7]
#define SIZE (factor * 1024 * 1024 * (1024 / sizeof(int)) + factor)

int main(int argc, char **argv)
{
    unsigned errs = 0;
    int rank, nprocs;

    size_t i;
    int *buffer;

    MPI_Request request;

    MTest_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc > 1) {
        factor = atoi(argv[1]);
    }

    if ((factor < 0) || (factor > 7)) {
        fprintf(stderr, "ERROR: A valid factor parameter must be within [0..7]! (factor is %d)\n",
                factor);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    buffer = malloc(SIZE * sizeof(int));

    if (!buffer) {
        fprintf(stderr, "ERROR: Could not allocate %ld bytes of memory!\n", SIZE * sizeof(int));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (rank == 0)
        MTestPrintfMsg(1, "*** Checking MPI_Ireduce for messages > ~%d MB\n",
                       (SIZE * sizeof(int)) / (1024 * 1024));

    for (i = 0; i < SIZE; i++)
        buffer[i] = 42;

    if (rank == 0) {
        MPI_Ireduce(MPI_IN_PLACE, buffer, SIZE, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD, &request);
    } else {
        MPI_Ireduce(buffer, buffer, SIZE, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD, &request);
    }

    MPI_Wait(&request, MPI_STATUS_IGNORE);

    if (rank == 0) {
        for (i = 0; i < SIZE; i++) {
            if (buffer[i] != 42 * nprocs) {
                if (errs < 10)
                    fprintf(stderr, "(%d) MPI_Ireduce: Error at position %zu: %d vs. %d\n", rank, i,
                            buffer[i], 42 * nprocs);
                errs++;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    free(buffer);

    MTest_Finalize(errs);

    return MTestReturnValue(errs);
}
