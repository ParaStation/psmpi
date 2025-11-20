/*
 * ParaStation
 *
 * Copyright (C) 2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "session_malleability.h"
#include "mpitest.h"

#define SPAWN_PROCS 2
#define PORT_KEY_1 "port_1"
#define PORT_KEY_2 "port_2"
#define COMMAND "./session_malleability_expand"

/* Test expansion with MPI Session and MPIX_Spawn */

int main(int argc, char *argv[])
{
    int rc, rank, size;
    MPI_Session session = MPI_SESSION_NULL;
    MPI_Comm comm = MPI_COMM_NULL;

    int spawned = 0;
    int spawn_procs = SPAWN_PROCS;

    int root = 0;
    char port[MPI_MAX_PORT_NAME];       /* only for root */
    char *command[] = { COMMAND };      /* only for root */
    char **spawn_args = NULL;   /* only for root */

    rc = init_session_and_comm(&session, &comm, &rank, &size);
    CHECK_ERR(rc);

    /* Is this process a spawned process? */
    rc = MPIX_Spawn_test_parent(&spawned);
    CHECK_MPI_ERR(rc);

    /* Optional first argument is number of processes to spawn */
    if (argc > 1) {
        spawn_procs = atoi(argv[1]);
    }

    if (!spawned && (spawn_procs > 0)) {
        /* ############# FIRST EXPANSION - PARENT PROCESSES ##################### */
        if (rank == root) {
            create_spawn_argv(spawn_procs, &spawn_args);
            rc = do_expand(command, &spawn_args, &spawn_procs, PORT_KEY_1, port);
            free_spawn_argv(spawn_args);
            CHECK_ERR(rc);
        }

        /* Expand comm to include new processes */
        rc = expand_comm(root, port, spawn_procs, &comm);
        CHECK_ERR(rc);

        if (rank == root) {
            MPI_Close_port(port);
        }

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        /* DO WORK AFTER FIRST EXPANSION */
        rc = do_work(comm);
        CHECK_ERR(rc);

        if (rank == root) {
            /* ######### SECOND EXPANSION ########## */
            create_spawn_argv(0, &spawn_args);
            rc = do_expand(command, &spawn_args, &spawn_procs, PORT_KEY_2, port);
            free_spawn_argv(spawn_args);
            CHECK_ERR(rc);
        }

        /* Expand comm to include new processes */
        rc = expand_comm(root, port, spawn_procs, &comm);
        CHECK_ERR(rc);

        if (rank == root) {
            MPI_Close_port(port);
        }

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        /* DO WORK AFTER SECOND EXPANSION */
        rc = do_work(comm);
        CHECK_ERR(rc);

    } else if (spawned && (spawn_procs > 0)) {
        /* Spawned process from first expansion */
        rc = connect_comm(PORT_KEY_1, &comm);
        CHECK_ERR(rc);

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        /* DO WORK AFTER FIRST EXPANSION */
        rc = do_work(comm);
        CHECK_ERR(rc);

        /* ######### SECOND EXPANSION ########## */
        /* Expand comm to include new processes */
        rc = expand_comm(root, port, spawn_procs, &comm);
        CHECK_ERR(rc);

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        /* DO WORK AFTER SECOND EXPANSION */
        rc = do_work(comm);
        CHECK_ERR(rc);

    } else if (spawned && (spawn_procs == 0)) {
        /* Spawned process from second expansion */
        rc = connect_comm(PORT_KEY_2, &comm);
        CHECK_ERR(rc);

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        /* DO WORK AFTER SECOND EXPANSION */
        rc = do_work(comm);
        CHECK_ERR(rc);
    }

    /* Final Cleanup */
    rc = cleanup_session_and_comm(&session, &comm);
    CHECK_ERR(rc);

    FINISH_TEST;
}
