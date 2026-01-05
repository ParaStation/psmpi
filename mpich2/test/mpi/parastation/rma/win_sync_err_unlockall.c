/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2012 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 *  Portions of this code were written/modified by ParTec AG
 *  Copyright (C) 2016-2021 ParTec Cluster Competence Center GmbH, Munich
 *  Copyright (C) 2021-2026 ParTec AG, Munich
 */

#include "mpi.h"
#include <stdio.h>
#include "../errors/rma/win_sync.h"

int main(int argc, char *argv[])
{
    int rank;
    int errors = 0, all_errors = 0;
    int buf;
    MPI_Win win;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Win_create(&buf, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    MPI_Win_set_errhandler(win, MPI_ERRORS_RETURN);

    /* This should fail because the window is not locked. */
    CHECK_ERR(MPI_Win_unlock_all(win));

    MPI_Win_free(&win);

    MPI_Reduce(&errors, &all_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0 && all_errors == 0)
        printf(" No Errors\n");
    MPI_Finalize();

    return 0;
}
