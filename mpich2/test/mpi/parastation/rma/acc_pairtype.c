/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2015 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

/*  Extended version of ANL's acc-pairtype test by ParTec. */

/*  Portions of this code were written/modified by ParTec AG
 *  Copyright (C) 2017-2021 ParTec Cluster Competence Center GmbH, Munich
 *  Copyright (C) 2021-2025 ParTec AG, Munich
 */

#include "mpi.h"
#include <stdio.h>

#define DATA_SIZE 25

typedef struct double_int {
    double a;
    int b;
} double_int_t;

int main(int argc, char *argv[])
{
    MPI_Win win;
    int errors = 0;
    int rank, nproc, i;
    double_int_t *orig_buf;
    double_int_t *tar_buf;
    MPI_Datatype vector_dtp;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Alloc_mem(sizeof(double_int_t) * DATA_SIZE, MPI_INFO_NULL, &orig_buf);
    MPI_Alloc_mem(sizeof(double_int_t) * DATA_SIZE, MPI_INFO_NULL, &tar_buf);

    for (i = 0; i < DATA_SIZE; i++) {
        orig_buf[i].a = 1.0;
        orig_buf[i].b = 1;
        tar_buf[i].a = 0.5;
        tar_buf[i].b = 0;
    }

    MPI_Type_vector(5 /* count */ , 3 /* blocklength */ , 5 /* stride */ , MPI_DOUBLE_INT,
                    &vector_dtp);
    MPI_Type_commit(&vector_dtp);

    MPI_Win_create(tar_buf, sizeof(double_int_t) * DATA_SIZE, sizeof(double_int_t), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &win);

    if (rank == 0) {
        MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, win);
        MPI_Accumulate(orig_buf, 1, vector_dtp, 1, 0, 1, vector_dtp, MPI_MAXLOC, win);
        MPI_Win_unlock(1, win);
    }

    MPI_Win_fence(0, win);

    if (rank == 1) {
        for (i = 0; i < DATA_SIZE; i++) {
            if (i % 5 < 3) {
                if (tar_buf[i].a != 1.0 || tar_buf[i].b != 1) {
                    printf("tar_buf[i].a = %f (expected 1.0) | tar_buf[i].b = %d (expected 1)\n",
                           tar_buf[i].a, tar_buf[i].b);
                    errors++;
                }
            } else {
                if (tar_buf[i].a != 0.5 || tar_buf[i].b != 0) {
                    printf("tar_buf[i].a = %f (expected 0.5) | tar_buf[i].b = %d (expected 0)\n",
                           tar_buf[i].a, tar_buf[i].b);
                    errors++;
                }
            }
        }
    }

    MPI_Type_free(&vector_dtp);

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < DATA_SIZE; i++) {
        orig_buf[i].a = 0.0;
        orig_buf[i].b = 2;
    }

    MPI_Win_fence(0, win);

    if (rank == 0) {
        MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, win);
        MPI_Accumulate(orig_buf, DATA_SIZE, MPI_DOUBLE_INT, 1, 0, DATA_SIZE, MPI_DOUBLE_INT,
                       MPI_MINLOC, win);
        MPI_Win_unlock(1, win);
    }

    MPI_Win_fence(0, win);

    if (rank == 1) {
        for (i = 0; i < DATA_SIZE; i++) {
            if (tar_buf[i].a != 0.0 || tar_buf[i].b != 2) {
                printf("tar_buf[i].a = %f (expected 0.0) | tar_buf[i].b = %d (expected 2)\n",
                       tar_buf[i].a, tar_buf[i].b);
                errors++;
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
