/*
 * ParaStation
 *
 * Copyright (C) 2025-2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "session_malleability.h"
#include "mpitest.h"

/* Test shrink with MPI Session */

int main(int argc, char *argv[])
{
    int rc, rank, size;
    int terminate = 0;
    MPI_Session session = MPI_SESSION_NULL;
    MPI_Comm comm = MPI_COMM_NULL;

    rc = init_session_and_comm(&session, &comm, &rank, &size);
    CHECK_ERR(rc);

    /* Check test requirements */
    if (size < 4) {
        fprintf(stderr, "At least 4 processes required to continue with shrink operations\n");
        MPI_Abort(comm, 1);
    }

    /* ##### FIRST shrink (2 procs terminate) #####
     * Cutting procs from the end may "accidentally work" because the
     * remaining rank range is not touched. However, removing procs from
     * "the middle" really requires a rank update in the new comm. So
     * this test case is more challenging. */
    if (rank == 1 || rank == 2) {
        terminate = 1;
    }
    rc = prepare_shrink(&session, &comm, terminate, 2, &rank);
    CHECK_ERR(rc);

    /* Terminate procs */
    if (session == MPI_SESSION_NULL) {
        exit(0);
    }

    /* Do something with the shrunk comm */
    rc = do_work(comm);
    CHECK_ERR(rc);

    /* ##### SECOND shrink (1 proc terminates) ##### */
    terminate = 0;
    if (rank == 1) {
        terminate = 1;
    }

    rc = prepare_shrink(&session, &comm, terminate, 1, &rank);
    CHECK_ERR(rc);

    /* Terminate procs */
    if (session == MPI_SESSION_NULL) {
        exit(0);
    }

    /* Do something with the shrunk comm */
    rc = do_work(comm);
    CHECK_ERR(rc);

    /* Final Cleanup */
    rc = cleanup_session_and_comm(&session, &comm);
    CHECK_ERR(rc);

    FINISH_TEST;
}
