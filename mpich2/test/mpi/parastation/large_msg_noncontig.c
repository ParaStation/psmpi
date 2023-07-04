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

int factor = 2;
#define SIZE (factor * 1024 * 1024 * (1024 / sizeof(int)) + factor * 2)

int main(int argc, char **argv)
{
    unsigned errs = 0;
    int rank, nprocs;

    size_t i;
    int *buffer;

    MPI_Win win;
    MPI_Request request;
    MPI_Datatype non_contig_type;

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

    MPI_Win_create(buffer, SIZE * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    MPI_Type_vector(2, SIZE / (2 * factor) /* -> */ -1 /* <- non-contiguous */ ,
                    SIZE / (2 * factor), MPI_INT, &non_contig_type);
    MPI_Type_commit(&non_contig_type);

#if 1   // Checking MPI_Send/Irecv with large non-contiguous messages:

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        MTestPrintfMsg(1, "*** Checking MPI_Send/Irecv with non-contiguous messages > ~%d MB\n",
                       (SIZE * sizeof(int)) / (1024 * 1024));

    if (rank == 0) {
        for (i = 0; i < SIZE; i++)
            buffer[i] = 42;
    } else {
        for (i = 0; i < SIZE; i++)
            buffer[i] = 0;
    }

    MPI_Irecv(buffer, factor, non_contig_type, 0, rank, MPI_COMM_WORLD, &request);
    if (rank == 0) {
        for (i = 0; i < nprocs; i++)
            MPI_Send(buffer, factor, non_contig_type, i, i, MPI_COMM_WORLD);
    }
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    for (i = 0; i < SIZE; i++) {
        if ((rank == 0) ||
            ((i < SIZE - factor) && ((i % (SIZE / (2 * factor) - 1)) - i / (SIZE / factor)))) {
            if ((buffer[i] != 42)) {
                if (errs < 10)
                    fprintf(stderr, "(%d) MPI_Irecv: Error at position %d: %d vs. %d\n", rank, i,
                            buffer[i], 42);
                errs++;
            }
        }
    }
#endif

#if 1   // Checking MPI_Get with large non-contiguous messages:

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        MTestPrintfMsg(1, "*** Checking MPI_Get with non-contiguous messages > ~%d MB\n",
                       (SIZE * sizeof(int)) / (1024 * 1024));

    if (rank == 0) {
        for (i = 0; i < SIZE; i++)
            buffer[i] = 19;
    } else {
        for (i = 0; i < SIZE; i++)
            buffer[i] = 0;
    }

    MPI_Win_fence(0, win);

    if (rank > 0) {
        MPI_Get(buffer, factor, non_contig_type, 0, 0, factor, non_contig_type, win);
    }

    MPI_Win_fence(0, win);

    for (i = 0; i < SIZE; i++) {
        if ((rank == 0) ||
            ((i < SIZE - factor) && ((i % (SIZE / (2 * factor) - 1)) - i / (SIZE / factor)))) {
            if (buffer[i] != 19) {
                if (errs < 10)
                    fprintf(stderr, "(%d) MPI_Get: Error at position %d: %d vs. %d\n", rank, i,
                            buffer[i], 19);
                errs++;
            }
        }
    }
#endif

#if 1   // Checking MPI_Put with large non-contiguous messages:

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        MTestPrintfMsg(1, "*** Checking MPI_Put with non-contiguous messages > ~%d MB\n",
                       (SIZE * sizeof(int)) / (1024 * 1024));

    if (rank == 0) {
        for (i = 0; i < SIZE; i++)
            buffer[i] = 23;
    } else {
        for (i = 0; i < SIZE; i++)
            buffer[i] = 0;
    }

    MPI_Win_fence(0, win);

    if (rank == 0) {
        for (i = 1; i < nprocs; i++) {
            MPI_Put(buffer, factor, non_contig_type, i, 0, factor, non_contig_type, win);
        }
    }

    MPI_Win_fence(0, win);

    for (i = 0; i < SIZE; i++) {
        if ((rank == 0) ||
            ((i < SIZE - factor) && ((i % (SIZE / (2 * factor) - 1)) - i / (SIZE / factor)))) {
            if (buffer[i] != 23) {
                if (errs < 10)
                    fprintf(stderr, "(%d) MPI_Put: Error at position %d: %d vs. %d\n", rank, i,
                            buffer[i], 23);
                errs++;
            }
        }
    }

#endif

#if 1   // Checking MPI_Bcast with large non-contiguous messages:

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        MTestPrintfMsg(1, "*** Checking MPI_Bcast with non-contiguous messages > ~%d MB\n",
                       (SIZE * sizeof(int)) / (1024 * 1024));

    if (rank == 0) {
        for (i = 0; i < SIZE; i++)
            buffer[i] = 137;
    } else {
        for (i = 0; i < SIZE; i++)
            buffer[i] = 0;
    }


    MPI_Bcast(buffer, factor, non_contig_type, 0, MPI_COMM_WORLD);

    for (i = 0; i < SIZE; i++) {
        if ((rank == 0) ||
            ((i < SIZE - factor) && ((i % (SIZE / (2 * factor) - 1)) - i / (SIZE / factor)))) {
            if ((buffer[i] != 137)) {
                if (errs < 10)
                    fprintf(stderr, "(%d) MPI_Bcast: Error at position %zu: %d vs. %d\n", rank, i,
                            buffer[i], 137);
                errs++;
            }
        }
    }
#endif

#if 1   // Checking MPI_Ibcast with large non-contiguous messages:

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        MTestPrintfMsg(1, "*** Checking MPI_Ibcast with non-contiguous messages > ~%d MB\n",
                       (SIZE * sizeof(int)) / (1024 * 1024));

    if (rank == 0) {
        for (i = 0; i < SIZE; i++)
            buffer[i] = 73;
    } else {
        for (i = 0; i < SIZE; i++)
            buffer[i] = 0;
    }


    MPI_Ibcast(buffer, factor, non_contig_type, 0, MPI_COMM_WORLD, &request);

    MPI_Wait(&request, MPI_STATUS_IGNORE);

    for (i = 0; i < SIZE; i++) {
        if ((rank == 0) ||
            ((i < SIZE - factor) && ((i % (SIZE / (2 * factor) - 1)) - i / (SIZE / factor)))) {
            if ((buffer[i] != 73)) {
                if (errs < 10)
                    fprintf(stderr, "(%d) MPI_Ibcast: Error at position %zu: %d vs. %d\n", rank, i,
                            buffer[i], 73);
                errs++;
            }
        }
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_free(&win);
    MPI_Type_free(&non_contig_type);
    free(buffer);

    MTest_Finalize(errs);

    return MTestReturnValue(errs);
}
