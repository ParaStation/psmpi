/*
 * ParaStation
 *
 * Copyright (C) 2017-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpitest.h"

int factor = 2;
#define SIZE (factor * 1024 * 1024 * (1024 / sizeof(int)) + factor)
#define FACTOR (factor % 2 == 0 ? factor / 2 : factor)

int main(int argc, char **argv)
{
    unsigned errs = 0;
    int rank, nprocs;

    size_t i;
    int *buffer;

    MPI_Request request;
    MPI_Datatype contig_type;
    MPI_Aint lb, extent;

    MTest_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc > 1) {
        factor = atoi(argv[1]);
    }

    if (factor < 0) {
        fprintf(stderr, "ERROR: A valid factor parameter must be positive! (factor is %d)\n",
                factor);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    buffer = malloc(SIZE * sizeof(int));

    if (!buffer) {
        fprintf(stderr, "ERROR: Could not allocate %ld bytes of memory!\n", SIZE * sizeof(int));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    // Create a large datatype (of ~1 GiB +x or, if factor is divisible by 2, of ~2 GiB +x):
    MPI_Type_contiguous(factor % 2 == 0 ? 2 * SIZE / factor : SIZE / factor, MPI_INT, &contig_type);
    MPI_Type_commit(&contig_type);
    MPI_Type_get_extent(contig_type, &lb, &extent);

    // Checking MPI_Ibcast for large messages:

    if (rank == 0)
        MTestPrintfMsg(1,
                       "*** Checking MPI_Ibcast for messages > ~%d MB with a datatype of > ~%ld MB\n",
                       (SIZE * sizeof(int)) / (1024 * 1024), extent / (1024 * 1024));

    if (rank == 0) {
        for (i = 0; i < SIZE; i++)
            buffer[i] = 42;
    } else {
        for (i = 0; i < SIZE; i++)
            buffer[i] = 0;
    }

    MPI_Ibcast(buffer, FACTOR, contig_type, 0, MPI_COMM_WORLD, &request);

    MPI_Wait(&request, MPI_STATUS_IGNORE);

    for (i = 0; i < SIZE; i++) {
        if (buffer[i] != 42) {
            if (errs < 10)
                fprintf(stderr, "(%d) MPI_Ibcast: Error at position %zu: %d vs. %d\n", rank, i,
                        buffer[i], 42);
            errs++;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Type_free(&contig_type);
    free(buffer);

    MTest_Finalize(errs);

    return MTestReturnValue(errs);
}
