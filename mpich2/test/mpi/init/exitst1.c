/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpi.h"

/*
 * This is a special test to check that mpiexec handles zero/non-zero
 * return status from an application
 */
int main(int argc, char *argv[])
{
    MPI_Init(0, 0);
    MPI_Finalize();
    return 1;
}
