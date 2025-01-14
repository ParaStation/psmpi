/*
 * ParaStation
 *
 * Copyright (C) 2023-2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpi.h"
#include <stdio.h>
#include "mpitest.h"

/*
 * Test the calling of a user-defined error handler for MPI_Session_init()
 * in case of invalid `info` and `session` arguments.
 */

static int calls = 0;
static int errs = 0;

void session_init_errh_func(MPI_Session * session, int *err, ...)
{
    int _err;
    MPI_Error_class(*err, &_err);
    if ((_err != MPI_ERR_ARG) && (_err != MPI_ERR_INFO)) {
        errs++;
        fprintf(stderr, "Unexpected error code\n");
    }
    if (*session != MPI_SESSION_NULL) {
        errs++;
        fprintf(stderr, "Unexpected session pointer\n");
    }
    calls++;
    return;
}

int main(int argc, char *argv[])
{
    MPI_Session session;
    MPI_Errhandler session_init_errh;

    MPI_Session_create_errhandler(session_init_errh_func, &session_init_errh);

    /* Call with invalid `info` and `session` arguments */
    MPI_Session_init((MPI_Info) 0, session_init_errh, &session);
    MPI_Session_init(MPI_INFO_NULL, session_init_errh, NULL);

    /* Finally, do the correct call */
    MPI_Session_init(MPI_INFO_NULL, session_init_errh, &session);

    MPI_Errhandler_free(&session_init_errh);
    MPI_Session_finalize(&session);

    if (calls != 2) {
        errs++;
        fprintf(stderr, "Error handler called %d times (expected exactly 2 calls)\n", calls);
    }

    if (errs == 0) {
        fprintf(stdout, " No Errors\n");
    } else {
        fprintf(stderr, "%d Errors\n", errs);
    }

    return MTestReturnValue(errs);
}
