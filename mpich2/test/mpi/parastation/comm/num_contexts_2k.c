/*
 * ParaStation
 *
 * Copyright (C) 2016-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include <stdio.h>

/* Check for 2k custom/dynamic communicators: (minus COMM_SELF, COMM_WORLD, and ICOMM_WORLD if needed) */
#define NUM_COMMS (2 * 1024 - 3)

/* As 2k is the default for the max number of contexts (see mpich2/src/include/mpir_contextid.h), it's unlikely that this test fails. */

int main(int argc, char *argv[])
{
    unsigned errs = 0;
    int i;
    int world_rank, rank[NUM_COMMS];
    int world_size, size[NUM_COMMS];
    MPI_Comm comm_array[NUM_COMMS];

    MTest_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    for (i = 0; i < NUM_COMMS; i++) {

        int rc;
        int color = (world_rank + 1) % world_size;

        rc = MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &comm_array[i]);

        if (rc != MPI_SUCCESS) {
            if (!errs) {
                fprintf(stderr,
                        "\nERROR: The maximum number of custom/dynamic communicators/contexts is %d but this test checks for %d.\n",
                        i, NUM_COMMS);
            }
            errs++;
        }

        MPI_Comm_rank(comm_array[i], &rank[i]);
        MPI_Comm_size(comm_array[i], &size[i]);
    }

    for (i = 0; i < NUM_COMMS; i++) {
        MPI_Comm_free(&comm_array[i]);
    }

    MTest_Finalize(errs);

    return MTestReturnValue(errs);
}
