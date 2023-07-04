/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2015 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

/*  Extended version of ANL's acc-pairtype test by ParTec. */

/*  Portions of this code were written/modified by ParTec AG
 *  Copyright (C) 2017-2021 ParTec Cluster Competence Center GmbH, Munich
 *  Copyright (C) 2021-2023 ParTec AG, Munich
 */

#include "mpi.h"
#include <stdio.h>

#define DATA_SIZE 25

int main(int argc, char *argv[])
{
    MPI_Win win;
    int errors = 0;
    int rank, nproc, i;
    double *orig_buf;
    double *tar_buf;
    MPI_Datatype vector_dtp;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Alloc_mem(sizeof(double) * DATA_SIZE, MPI_INFO_NULL, &orig_buf);
    MPI_Alloc_mem(sizeof(double) * DATA_SIZE, MPI_INFO_NULL, &tar_buf);

    for (i = 0; i < DATA_SIZE; i++) {
        orig_buf[i] = 1.0;
        tar_buf[i] = 0.5;
    }

    MPI_Type_vector(5 /* count */ , 3 /* blocklength */ , 5 /* stride */ , MPI_DOUBLE, &vector_dtp);
    MPI_Type_commit(&vector_dtp);

    MPI_Win_create(tar_buf, sizeof(double) * DATA_SIZE, sizeof(double), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &win);

    if (rank == 0) {
        MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, win);
        MPI_Accumulate(orig_buf, 1, vector_dtp, 1, 0, 1, vector_dtp, MPI_SUM, win);
        MPI_Win_unlock(1, win);
    }

    MPI_Win_fence(0, win);

    if (rank == 1) {
        for (i = 0; i < DATA_SIZE; i++) {
            if (i % 5 < 3) {
                if (tar_buf[i] != 1.5) {
                    printf("tar_buf[i] = %f (expected 1.5)\n", tar_buf[i]);
                    errors++;
                }
            } else {
                if (tar_buf[i] != 0.5) {
                    printf("tar_buf[i] = %f (expected 0.5)\n", tar_buf[i]);
                    errors++;
                }
            }
        }
    }

    MPI_Type_free(&vector_dtp);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, win);
        MPI_Accumulate(orig_buf, DATA_SIZE, MPI_DOUBLE, 1, 0, DATA_SIZE, MPI_DOUBLE, MPI_SUM, win);
        MPI_Win_unlock(1, win);
    }

    MPI_Win_fence(0, win);

    if (rank == 1) {
        for (i = 0; i < DATA_SIZE; i++) {
            if (i % 5 < 3) {
                if (tar_buf[i] != 2.5) {
                    printf("tar_buf[i] = %f (expected 2.5)\n", tar_buf[i]);
                    errors++;
                }
            } else {
                if (tar_buf[i] != 1.5) {
                    printf("tar_buf[i] = %f (expected 1.5)\n", tar_buf[i]);
                    errors++;
                }
            }
        }
    }

    MPI_Win_free(&win);

    MPI_Free_mem(orig_buf);
    MPI_Free_mem(tar_buf);

    if (rank == 1) {
        if (errors == 0)
            printf(" No Errors\n");
    }

    MPI_Finalize();
    return 0;
}
