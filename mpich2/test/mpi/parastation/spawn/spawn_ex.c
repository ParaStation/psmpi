/*
 * ParaStation
 *
 * Copyright (C) 2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

/* Tests for MPIX_Spawn and MPIX_Ispawn, with
 * - MPI World model
 * - MPI Session model
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpitest.h"

#ifdef SESSION_INIT
#ifdef NONBLOCKING
#define COMMAND "./spawn_ex_session_nb"
#else
#define COMMAND "./spawn_ex_session"
#endif
#else
#ifdef NONBLOCKING
#define COMMAND "./spawn_ex_world_nb"
#else
#define COMMAND "./spawn_ex_world"
#endif
#endif

static int np[1] = { 2 };

void parent(int rank, MPI_Comm world_comm, char *port, int *errs)
{
    int rsize, i, err, rc;
    MPI_Comm intercomm = MPI_COMM_NULL;
    char *command[1] = { COMMAND };
    MPI_Info info[1] = { MPI_INFO_NULL };

    if (rank == 0) {
        /* Spawn child processes */
#ifdef NONBLOCKING
        MPI_Request req = MPI_REQUEST_NULL;
        MPI_Status status;
        rc = MPIX_Ispawn(1, command, MPI_ARGVS_NULL, np, info, &req);

        if (rc != MPI_SUCCESS) {
            (*errs)++;
            printf("Error on spawn (code %d)\n", rc);
            /* if spawning failed we end here with error */
            return;
        }

        MPI_Wait(&req, &status);
#else
        rc = MPIX_Spawn(1, command, MPI_ARGVS_NULL, np, info);
#endif

        if (rc != MPI_SUCCESS) {
            (*errs)++;
            printf("Error on spawn (code %d)\n", rc);
            /* if spawning failed we end here with error */
            return;
        }
    }

    /* Accept connections on new port */
    MPI_Comm_accept(port, MPI_INFO_NULL, 0, world_comm, &intercomm);
    MPI_Comm_remote_size(intercomm, &rsize);

    if (rsize != np[0]) {
        (*errs)++;
        printf("Did not create %d processes (got %d)\n", np[0], rsize);
    }

    /* Use intercommunicator for test communication with child procs (see spawn1 test) */
    if (rank == 0) {
        for (i = 0; i < rsize; i++) {
            MPI_Send(&i, 1, MPI_INT, i, 0, intercomm);
        }
        /* We could use intercomm reduce to get the errors from the
         * children, but we'll use a simpler loop to make sure that
         * we get valid data */
        for (i = 0; i < rsize; i++) {
            MPI_Recv(&err, 1, MPI_INT, i, 1, intercomm, MPI_STATUS_IGNORE);
            (*errs) += err;
        }
    }

    MPI_Comm_disconnect(&intercomm);
}

void child(int rank, int size, MPI_Comm world_comm, int *errs)
{
    int rsize, i, err;
    char port[MPI_MAX_PORT_NAME];
    MPI_Comm intercomm = MPI_COMM_NULL;
    MPI_Status status;

    /* Lookup the parent port */
    MPI_Lookup_name("spawn_ex", MPI_INFO_NULL, port);

    /* Connect to parent */
    MPI_Comm_connect(port, MPI_INFO_NULL, 0, world_comm, &intercomm);
    MPI_Comm_remote_size(intercomm, &rsize);

    if (size != np[0]) {
        (*errs)++;
        printf("(Child) Did not create %d processes (got %d)\n", np[0], size);
    }

    /* Use intercommunicator for test communication with parent (see spawn1 test) */
    MPI_Recv(&i, 1, MPI_INT, 0, 0, intercomm, &status);
    if (i != rank) {
        (*errs)++;
        printf("Unexpected rank on child %d (%d)\n", rank, i);
    }
    /* Send the errs back to the parent process */
    MPI_Ssend(errs, 1, MPI_INT, 0, 1, intercomm);

    MPI_Comm_disconnect(&intercomm);
}

int main(int argc, char *argv[])
{
    int errs = 0;
    int rank, size, is_spawned;
    int no_spawns = 1;
    MPI_Comm world_comm;

    /* The test accepts a numerical parameter to set the number
     * of spawns done by the parent, default is 1 */
    if (argc > 1) {
        no_spawns = (int) strtol(argv[1], NULL, 10);
    }
#ifdef SESSION_INIT
    MPI_Session session = MPI_SESSION_NULL;
    MPI_Group group = MPI_GROUP_NULL;
    MPI_Info sinfo = MPI_INFO_NULL;
    const char sf_key[] = "strict_finalize";
    const char sf_value[] = "1";

    MPI_Info_create(&sinfo);
    MPI_Info_set(sinfo, sf_key, sf_value);

    MPI_Session_init(sinfo, MPI_ERRORS_ARE_FATAL, &session);

    MPI_Group_from_session_pset(session, "mpi://WORLD", &group);
    MPI_Comm_create_from_group(group, "tag", MPI_INFO_NULL, MPI_ERRORS_ARE_FATAL, &world_comm);
    MPI_Group_free(&group);
    MPI_Info_free(&sinfo);
#else
    MPI_Init(&argc, &argv);
    world_comm = MPI_COMM_WORLD;
#endif

    MPI_Comm_rank(world_comm, &rank);
    MPI_Comm_size(world_comm, &size);

    MPIX_Spawn_test_parent(&is_spawned);

    if (!is_spawned) {
        /* Parent */
        char port[MPI_MAX_PORT_NAME];

        if (rank == 0) {
            /* Open new port */
            MPI_Open_port(MPI_INFO_NULL, port);

            /* Publish the port so that child processes can look it up */
            MPI_Publish_name("spawn_ex", MPI_INFO_NULL, port);
        }

        for (int i = 0; i < no_spawns; i++) {
            parent(rank, world_comm, port, &errs);
        }

        if (rank == 0) {
            MPI_Close_port(port);
        }
    } else {
        /* Child */
        child(rank, size, world_comm, &errs);
    }

  fn_exit:
#ifdef SESSION_INIT
    MPI_Comm_free(&world_comm);
    MPI_Session_finalize(&session);
#else
    MPI_Finalize();
#endif
    if (!is_spawned && rank == 0 && errs == 0) {
        printf("No Errors\n");
    }
    return MTestReturnValue(errs);
}
