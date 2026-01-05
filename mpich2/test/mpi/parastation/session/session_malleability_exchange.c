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

#define SPAWN_PROCS 2
#define PORT_KEY_1 "port_1"
#define COMMAND "./session_malleability_exchange"

/* Test exchange with MPI Session and MPIX_Spawn */

int main(int argc, char *argv[])
{
    int rc, rank, size;
    int terminate = 0;
    MPI_Session session = MPI_SESSION_NULL;
    MPI_Comm comm = MPI_COMM_NULL;

    int initial_size;
    int spawned = 0;

    int root = 0;
    char port[MPI_MAX_PORT_NAME];       /* only for root */
    char *command[] = { COMMAND };      /* only for root */

    rc = init_session_and_comm(&session, &comm, &rank, &size);
    CHECK_ERR(rc);

    /* Is this process a spawned process? */
    rc = MPIX_Spawn_test_parent(&spawned);
    CHECK_MPI_ERR(rc);

    initial_size = size;

    if (!spawned) {
        /* Check conditions for this test */
        if (size < 3) {
            fprintf(stderr, "At least 3 processes required\n");
            MPI_Abort(comm, 1);
        }

        if (rank == 1 || rank == 2) {
            terminate = 1;
        }

        rc = prepare_shrink(&session, &comm, terminate, 2, &rank);
        CHECK_ERR(rc);

        /* Terminate procs */
        if (session == MPI_SESSION_NULL) {
            exit(0);
        }

        if (rank == root) {
            int np = SPAWN_PROCS;
            rc = do_expand(command, MPI_ARGVS_NULL, &np, PORT_KEY_1, port);
            CHECK_ERR(rc);
        }

        /* Expand comm to include new processes */
        rc = expand_comm(root, port, SPAWN_PROCS, &comm);
        CHECK_ERR(rc);

        if (rank == root) {
            MPI_Close_port(port);
        }

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        /* Check if world size stayed the same */
        if (size != initial_size) {
            errs++;
            fprintf(stderr, "After exchange, expected comm size = %d, got %d\n", initial_size,
                    size);
            goto fn_fail;
        }

        /* DO WORK WITH COMM AFTER EXCHANGE OF PROCS */
        rc = do_work(comm);
        CHECK_ERR(rc);

    } else {
        /* Spawned process */
        rc = connect_comm(PORT_KEY_1, &comm);
        CHECK_ERR(rc);

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        /* DO WORK WITH COMM AFTER EXCHANGE OF PROCS */
        rc = do_work(comm);
        CHECK_ERR(rc);
    }

    /* Final Cleanup */
    rc = cleanup_session_and_comm(&session, &comm);
    CHECK_ERR(rc);

    FINISH_TEST;
}
