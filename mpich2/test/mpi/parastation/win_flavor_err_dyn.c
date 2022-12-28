/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2012 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 *  Portions of this code were written/modified by ParTec AG
 *  Copyright (C) 2016-2021 ParTec Cluster Competence Center GmbH, Munich
 *  Copyright (C) 2021-2023 ParTec AG, Munich
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERR( stmt )                                                               \
    do {                                                                                \
        int err_class, err, rank;                                                       \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                           \
        err = stmt;                                                                     \
        if (err == MPI_SUCCESS) {                                                       \
            printf("%d: Operation succeeded, when it should have failed\n", rank);      \
            errors++;                                                                   \
        } else {                                                                        \
            MPI_Error_class( err, &err_class );                                         \
            if (err_class != MPI_ERR_RMA_FLAVOR)  {                                    \
                char str[MPI_MAX_ERROR_STRING];                                         \
                int  len;                                                               \
                MPI_Error_string(err, str, &len);                                       \
                printf("%d: Expected MPI_ERR_RMA_FLAVOR, got:\n%s\n", rank, str);       \
                errors++;                                                               \
            }                                                                           \
        }                                                                               \
    } while (0);


int main(int argc, char *argv[])
{
    int          rank;
    int          errors = 0, all_errors = 0;
    int          buf;
    MPI_Win      win;
    MPI_Aint     size;
    void*        base;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Win_create(&buf, sizeof(int), sizeof(int),
                    MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    MPI_Win_set_errhandler(win, MPI_ERRORS_RETURN);

    size = 1024;
    base = malloc(1024);

    /* This should fail because the window is not a dynamic one. */
    CHECK_ERR(MPI_Win_attach(win, base, size));

    MPI_Win_free(&win);

    free(base);

    MPI_Reduce(&errors, &all_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0 && all_errors == 0) printf(" No Errors\n");
    MPI_Finalize();

    return 0;
}
