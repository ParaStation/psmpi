/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2012 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 *  Portions of this code were written/modified by ParTec AG
 *  Copyright (C) 2016-2021 ParTec Cluster Competence Center GmbH, Munich
 *  Copyright (C) 2021-2026 ParTec AG, Munich
 *
 */

#include "mpi.h"
#include <stdio.h>

#define CHECK_ERR(stmt)                                                               \
    do {                                                                                \
        int err_class, err, rank;                                                       \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                           \
        err = stmt;                                                                     \
        if (err == MPI_SUCCESS) {                                                       \
            printf("%d: Operation succeeded, when it should have failed\n", rank);      \
            errors++;                                                                   \
        } else {                                                                        \
            MPI_Error_class(err, &err_class);                                         \
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
    int rank;
    int errors = 0, all_errors = 0;
    int buf;
    MPI_Win win;
    MPI_Aint size;
    int disp_unit;
    void *baseptr;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    MPI_Win_set_errhandler(win, MPI_ERRORS_RETURN);

    /* This should fail because the window is of type MPI_WIN_FLAVOR_DYNAMIC,
     * and MPI-4.1 says: "The potential for multiple memory regions in windows
     * created through MPI_WIN_CREATE_DYNAMIC means that these windows cannot
     * be used as input for MPI_WIN_SHARED_QUERY."
     */
    CHECK_ERR(MPI_Win_shared_query(win, 0, &size, &disp_unit, &baseptr));

    MPI_Win_free(&win);

    MPI_Reduce(&errors, &all_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0 && all_errors == 0)
        printf(" No Errors\n");
    MPI_Finalize();

    return 0;
}
