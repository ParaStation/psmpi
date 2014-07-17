/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2012 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */
#include "mpi.h"
#include <stdio.h>
#include "../errors/rma/win_sync.h"

int main(int argc, char *argv[])
{
    int          rank;
    int          errors = 0, all_errors = 0;
    int          buf, my_buf;
    MPI_Win      win;
    MPI_Request  request = MPI_REQUEST_NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Win_create(&buf, sizeof(int), sizeof(int),
                    MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    MPI_Win_set_errhandler(win, MPI_ERRORS_RETURN);

    MPI_Win_fence(0, win);

    /* This should fail because request-based RMA operations are only valid within a passive target epoch! */
    MPI_Rget(&my_buf, 1, MPI_INT, 0, 0, 1, MPI_INT, win, &request);

    if(request != MPI_REQUEST_NULL) MPI_Wait(&request, MPI_STATUS_IGNORE);

    MPI_Win_fence(0, win);

    MPI_Win_free(&win);

    MPI_Reduce(&errors, &all_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0 && all_errors == 0) printf(" No Errors\n");
    MPI_Finalize();

    return 0;
}
