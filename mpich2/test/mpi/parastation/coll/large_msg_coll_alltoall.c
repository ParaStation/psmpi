/*
 * ParaStation
 *
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
    int errs = 0;
    int rank, nprocs;

    size_t i;
    int *rbuffer;
    int *sbuffer;

    MPI_Datatype contig_type;
    MPI_Aint lb, extent;

    int *rdispls = NULL;
    int *recvcnts = NULL;
    int *sdispls = NULL;
    int *sendcnts = NULL;

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

    rbuffer = malloc(nprocs * SIZE * sizeof(int));
    sbuffer = malloc(nprocs * SIZE * sizeof(int));

    if (!rbuffer || !sbuffer) {
        fprintf(stderr, "ERROR: Could not allocate %ld bytes of memory!\n",
                nprocs * SIZE * sizeof(int));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    // Create a large datatype (of ~1 GiB +x or, if factor is divisible by 2, of ~2 GiB +x):
    MPI_Type_contiguous(factor % 2 == 0 ? 2 * SIZE / factor : SIZE / factor, MPI_INT, &contig_type);
    MPI_Type_commit(&contig_type);
    MPI_Type_get_extent(contig_type, &lb, &extent);

#if 1   // Checking with MPI_Alltoall:

    if (rank == 0)
        MTestPrintfMsg(1,
                       "*** Checking MPI_Alltoall for messages > ~%d MB with a datatype of > ~%ld MB\n",
                       (SIZE * sizeof(int)) / (1024 * 1024), extent / (1024 * 1024));

    for (i = 0; i < nprocs * SIZE; i++)
        sbuffer[i] = 42 + rank;

    MPI_Alltoall(sbuffer, FACTOR, contig_type, rbuffer, FACTOR, contig_type, MPI_COMM_WORLD);

    for (i = 0; i < nprocs * SIZE; i++) {
        if (rbuffer[i] != 42 + i / SIZE) {
            if (errs < 10)
                fprintf(stderr, "(%d) MPI_Alltoall: Error at position %zu: %d vs. %ld\n", rank, i,
                        rbuffer[i], 42 + i / SIZE);
            errs++;
        }
    }
#endif

#if 1   // Checking with MPI_Alltoallv:

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        MTestPrintfMsg(1,
                       "*** Checking MPI_Alltoallv for messages > ~%d MB with a datatype of > ~%ld MB\n",
                       (SIZE * sizeof(int)) / (1024 * 1024), extent / (1024 * 1024));

    rdispls = malloc(nprocs * sizeof(int));
    recvcnts = malloc(nprocs * sizeof(int));
    sdispls = malloc(nprocs * sizeof(int));
    sendcnts = malloc(nprocs * sizeof(int));
    for (i = 0; i < nprocs; i++) {
        rdispls[i] = i * FACTOR;
        recvcnts[i] = FACTOR;
        sdispls[i] = i * FACTOR;
        sendcnts[i] = FACTOR;
    }

    for (i = 0; i < nprocs * SIZE; i++)
        sbuffer[i] = 19 + rank;

    MPI_Alltoallv(sbuffer, sendcnts, sdispls, contig_type, rbuffer, recvcnts, rdispls, contig_type,
                  MPI_COMM_WORLD);

    for (i = 0; i < nprocs * SIZE; i++) {
        if (rbuffer[i] != 19 + i / SIZE) {
            if (errs < 10)
                fprintf(stderr, "(%d) MPI_Alltoallv: Error at position %zu: %d vs. %ld\n", rank, i,
                        rbuffer[i], 19 + i / SIZE);
            errs++;
        }
    }

    free(sdispls);
    free(sendcnts);
    free(rdispls);
    free(recvcnts);
#endif
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Type_free(&contig_type);
    free(sbuffer);
    free(rbuffer);

    MTest_Finalize(errs);

    return MTestReturnValue(errs);
}
