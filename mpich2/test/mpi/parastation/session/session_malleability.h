/*
 * ParaStation
 *
 * Copyright (C) 2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"

int errs = 0;

 /* Macro for MPI error handling */
#define CHECK_MPI_ERR(mpi_errno)            \
    if (mpi_errno != MPI_SUCCESS) {         \
        errs++;                             \
        char errstr[MPI_MAX_ERROR_STRING];  \
        int len = 0;                        \
        MPI_Error_string(rc, errstr, &len); \
        fprintf(stderr, "%s\n", errstr);    \
        goto fn_fail;                       \
    }

/* Macro for jumping to failure label on error */
#define CHECK_ERR(err)                      \
    if (err != MPI_SUCCESS) {               \
        goto fn_fail;                       \
    }

#define FINISH_TEST                              \
  fn_exit:                                       \
    if (rank == 0) {                             \
        if (errs == 0) {                         \
            fprintf(stdout, " No Errors\n");     \
        } else {                                 \
            fprintf(stderr, "%d Errors\n", errs);\
        }                                        \
    }                                            \
    return MTestReturnValue(errs);               \
  fn_fail:                                       \
    if (comm != MPI_COMM_NULL) {                 \
        MPI_Comm_free(&comm);                    \
    }                                            \
    if (session != MPI_SESSION_NULL) {           \
        MPI_Session_finalize(&session);          \
    }                                            \
    goto fn_exit;


/* Reduce operation with a given comm */
static int do_work(MPI_Comm comm)
{
    int rc, rank, size;
    int sum = 0;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Do work - Reduce operation */
    rc = MPI_Reduce(&rank, &sum, 1, MPI_INT, MPI_SUM, 0, comm);
    CHECK_MPI_ERR(rc);

    if (rank == 0) {
        if (sum != (size - 1.0) * (size / 2.0)) {
            fprintf(stderr, "MPI_Reduce: expect %d, got %d\n", (size - 1) * (size / 2), sum);
            errs++;
        }
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

/* Init session and create comm */
static int init_session_and_comm(MPI_Session * session, MPI_Comm * comm, int *rank, int *size)
{
    int rc;
    MPI_Group group = MPI_GROUP_NULL;

    rc = MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_RETURN, session);
    CHECK_MPI_ERR(rc);

    rc = MPI_Session_set_errhandler(*session, MPI_ERRORS_RETURN);
    CHECK_MPI_ERR(rc);

    rc = MPI_Group_from_session_pset(*session, "mpi://WORLD", &group);
    CHECK_MPI_ERR(rc);

    rc = MPI_Comm_create_from_group(group, "org.mpi-forum.mpi-v4_0.example-ex10_8",
                                    MPI_INFO_NULL, MPI_ERRORS_RETURN, comm);
    CHECK_MPI_ERR(rc);

    MPI_Group_free(&group);

    /* Check if comm is initialized */
    if (*comm == MPI_COMM_NULL) {
        errs++;
        fprintf(stderr, "Comm not initialized\n");
        goto fn_fail;
    }

    MPI_Comm_rank(*comm, rank);
    MPI_Comm_size(*comm, size);

    rc = do_work(*comm);
    CHECK_ERR(rc);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

/* Free session and comm */
static int cleanup_session_and_comm(MPI_Session * session, MPI_Comm * comm)
{
    int rc;
    rc = MPI_Comm_free(comm);
    CHECK_MPI_ERR(rc);

    rc = MPI_Session_finalize(session);
    CHECK_MPI_ERR(rc);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

/* Prepare shrink by nterminate procs, create a smaller comm and finalize the session
 * in all terminating procs; provide the new rank */
static int prepare_shrink(MPI_Session * session, MPI_Comm * comm, int terminate,
                          int nterminate, int *new_rank)
{
    int i, idx, rc, old_rank, old_size, new_size;
    int *term_flags = NULL;
    int *term_array = NULL;
    MPI_Info info = MPI_INFO_NULL;
    MPI_Group old_group = MPI_GROUP_NULL;
    MPI_Group shrunk_group = MPI_GROUP_NULL;
    MPI_Comm shrunk_comm = MPI_COMM_NULL;

    MPI_Comm_rank(*comm, &old_rank);
    MPI_Comm_size(*comm, &old_size);

    term_flags = (int *) calloc(sizeof(int), old_size);

    /* Use comm to gather info about which procs are terminating in all procs */
    rc = MPI_Allgather(&terminate, 1, MPI_INT, term_flags, 1, MPI_INT, *comm);
    CHECK_MPI_ERR(rc);

    /* Non-terminating processes need to create a new comm */
    if (!terminate) {
        /* Create an integer array that contains all terminating ranks */
        term_array = calloc(sizeof(int), nterminate);
        idx = 0;
        for (i = 0; i < old_size; i++) {
            if (term_flags[i]) {
                if (idx >= nterminate) {
                    /* error */
                    errs++;
                    fprintf(stderr,
                            "Old rank %d: Found too many terminating ranks: no %d for old rank %d.\n",
                            old_rank, idx, i);
                    rc = -1;
                    goto fn_fail;
                } else {
                    term_array[idx] = i;
                    idx++;
                }
            }
        }
        if (idx < nterminate) {
            /* error */
            errs++;
            fprintf(stderr, "Old rank %d: Did not find %d terminating ranks, found %d instead.\n",
                    old_rank, nterminate, idx);
            rc = -1;
            goto fn_fail;
        }

        rc = MPI_Comm_group(*comm, &old_group);
        CHECK_MPI_ERR(rc);

        /* Create a smaller group that excludes all terminating procs */
        rc = MPI_Group_excl(old_group, nterminate, term_array, &shrunk_group);
        CHECK_MPI_ERR(rc);

        rc = MPI_Group_free(&old_group);
        CHECK_MPI_ERR(rc);

        /* Create a smaller comm from the shrunk group */
        rc = MPI_Comm_create_from_group(shrunk_group, "shrunk_group", MPI_INFO_NULL,
                                        MPI_ERRORS_RETURN, &shrunk_comm);
        CHECK_MPI_ERR(rc);

        rc = MPI_Group_free(&shrunk_group);
        CHECK_MPI_ERR(rc);

        MPI_Comm_size(shrunk_comm, &new_size);
        MPI_Comm_rank(shrunk_comm, new_rank);

        /* Check size of smaller comm */
        if (new_size != old_size - nterminate) {
            /* Error */
            errs++;
            fprintf(stderr, "New rank %d: Expected shrunk comm size = %d, got %d\n", *new_rank,
                    old_size - nterminate, new_size);
            rc = -1;
            goto fn_fail;
        }
    } else {
        *new_rank = -1; /* Terminating procs do not need new rank */
    }

    /* Clean up old comm */
    rc = MPI_Comm_disconnect(comm);
    CHECK_MPI_ERR(rc);

    if (*comm != MPI_COMM_NULL) {
        errs++;
        fprintf(stderr, "Old rank %d: MPI_COMM_DISCONNECT did not set old comm to MPI_COMM_NULL\n",
                old_rank);
        rc = -1;
        goto fn_fail;
    }

    if (!terminate) {
        /* Redirect comm to shrunk comm */
        *comm = shrunk_comm;
    }

    /* Finalize the session in all terminating procs */
    if (terminate) {
        rc = MPI_Session_finalize(session);
        CHECK_MPI_ERR(rc);

        if (*session != MPI_SESSION_NULL) {
            errs++;
            fprintf(stderr,
                    "Old rank %d: MPI_SESSION_FINALIZE did not set session to MPI_SESSION_NULL\n",
                    old_rank);
            rc = -1;
            goto fn_fail;
        }
    }

  fn_exit:
    free(term_flags);
    free(term_array);
    return rc;
  fn_fail:
    goto fn_exit;
}


/* Spawn maxprocs new processes using argv */
static int do_expand(char *array_of_commands[1], char ***array_of_argvs,
                     int *array_of_maxprocs, const char *publish_key, char *port)
{
    int rc = MPI_SUCCESS;
    MPI_Info info[1] = { MPI_INFO_NULL };

    /* Open new port */
    MPI_Open_port(MPI_INFO_NULL, port);

    /* Publish the port so that child processes can look it up */
    MPI_Publish_name(publish_key, MPI_INFO_NULL, port);

    rc = MPIX_Spawn(1, array_of_commands, array_of_argvs, array_of_maxprocs, info);
    CHECK_MPI_ERR(rc);

    /* Check for error */
    if (rc != MPI_SUCCESS) {
        errs++;
        fprintf(stderr, "Internal error in spawn: %d\n", rc);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

/* Create new flat comm including old and new processes (collective over comm)
 * input comm: comm of parent procs
 * output comm: flattened intercomm including parent and spawned procs */
static int expand_comm(int root, char *port, int spawn_procs, MPI_Comm * comm)
{
    int rc, rsize = 0, old_size, new_size, old_rank, new_rank;
    MPI_Comm intercomm = MPI_COMM_NULL;

    MPI_Comm_size(*comm, &old_size);
    MPI_Comm_rank(*comm, &old_rank);

    /* Accept connections on new port, create intercomm */
    rc = MPI_Comm_accept(port, MPI_INFO_NULL, root, *comm, &intercomm);
    CHECK_MPI_ERR(rc);
    rc = MPI_Comm_remote_size(intercomm, &rsize);
    CHECK_MPI_ERR(rc);

    if (rsize != spawn_procs) {
        errs++;
        printf("Parent: Did not create %d processes (got %d)\n", spawn_procs, rsize);
    }

    /* Clean up the old comm */
    rc = MPI_Comm_free(comm);
    CHECK_MPI_ERR(rc);

    /* Flatten the intercomm (low group) */
    rc = MPI_Intercomm_merge(intercomm, 0, comm);
    CHECK_MPI_ERR(rc);

    MPI_Comm_size(*comm, &new_size);
    MPI_Comm_rank(*comm, &new_rank);

    /* Check if size of new comm is correct */
    if (new_size != (old_size + rsize)) {
        errs++;
        printf("Parent: New comm does not have correct size, expected %d, got %d\n",
               old_size + rsize, new_size);
    }

    /* Check if rank in new comm is correct */
    if (new_rank != old_rank) {
        /* low group of intercomm merge should be ordered before remote ranks */
        errs++;
        printf("Parent: Incorrect rank in new comm, expected %d, got %d\n", old_rank, new_rank);
    }

    /* Clean up the intercomm */
    rc = MPI_Comm_disconnect(&intercomm);
    CHECK_MPI_ERR(rc);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

/* Connect spawned procs to parent and create new flat comm with all procs */
static int connect_comm(const char *lookup_key, MPI_Comm * comm)
{
    int rc, rsize, old_size, new_size, old_rank, new_rank;
    char port[MPI_MAX_PORT_NAME];
    MPI_Comm intercomm = MPI_COMM_NULL;

    /* Lookup the parent port */
    rc = MPI_Lookup_name(lookup_key, MPI_INFO_NULL, port);
    CHECK_MPI_ERR(rc);

    MPI_Comm_rank(*comm, &old_rank);
    MPI_Comm_size(*comm, &old_size);

    /* Connect to parent */
    rc = MPI_Comm_connect(port, MPI_INFO_NULL, 0, *comm, &intercomm);
    CHECK_MPI_ERR(rc);
    rc = MPI_Comm_remote_size(intercomm, &rsize);
    CHECK_MPI_ERR(rc);

    /* Clean up the old comm */
    rc = MPI_Comm_free(comm);
    CHECK_MPI_ERR(rc);

    /* Flatten the intercomm (high group) */
    rc = MPI_Intercomm_merge(intercomm, 1, comm);
    CHECK_MPI_ERR(rc);

    MPI_Comm_rank(*comm, &new_rank);
    MPI_Comm_size(*comm, &new_size);

    /* Check the new size */
    if (new_size != old_size + rsize) {
        errs++;
        printf("Child: New comm does not have correct size, expected %d, got %d\n",
               old_size + rsize, new_size);
    }

    /* Check the new rank */
    if (new_rank != (old_rank + rsize)) {
        /* high group in intercomm merge should be ordered after original processes */
        errs++;
        printf("Child: Incorrect rank in new comm, expected %d, got %d\n",
               old_rank + rsize, new_rank);
    }

    /* Clean up the intercomm */
    rc = MPI_Comm_disconnect(&intercomm);
    CHECK_MPI_ERR(rc);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static void create_spawn_argv(int spawn_procs, char ***argv)
{
    char nprocs[MPI_MAX_INFO_KEY];      /* only for root */
    char **args = malloc(2 * sizeof(char *));

    /* Put nprocs value into argv and terminate with NULL */
    snprintf(nprocs, MPI_MAX_INFO_KEY, "%d", spawn_procs);
    args[0] = strdup(nprocs);
    strcpy(args[0], nprocs);
    args[1] = NULL;

    *argv = args;
}

static void free_spawn_argv(char **argv)
{
    /* We know that there is a string stored at position 0 */
    free(argv[0]);
    free(argv);
}
