/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int bufsizes[4] = { 1, 100, 10000, 1000000 };

/* This tests checks whether simple RMA (put)
 * works within a spawned environment.
 */

int main(int argc, char *argv[])
{
    int errs = 0;
    int *errcodes;
    int world_rank, world_size;
    int univ_rank, univ_size;
    MPI_Comm univ_comm = MPI_COMM_NULL;
    MPI_Comm world_comm = MPI_COMM_NULL;
    MPI_Comm parent_comm = MPI_COMM_NULL;
    MPI_Comm spawn_comm = MPI_COMM_NULL;
    MPI_Win win;
    int cs, n, i;
    char *buf;

    MPI_Init(&argc, &argv);

    world_comm = MPI_COMM_WORLD;
    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);

#if 1
    MPI_Comm_get_parent(&parent_comm);

    if (parent_comm == MPI_COMM_NULL) {
        errcodes = malloc(world_size * sizeof(int));
        MPI_Comm_spawn((char *) "./spawn_rma", MPI_ARGV_NULL, world_size, MPI_INFO_NULL, 0,
                       MPI_COMM_WORLD, &spawn_comm, errcodes);
        free(errcodes);
    } else {

        spawn_comm = parent_comm;
    }

    MPI_Intercomm_merge(spawn_comm, parent_comm == MPI_COMM_NULL ? 1 : 0, &univ_comm);

#else
    univ_comm = world_comm;
#endif

    MPI_Comm_rank(univ_comm, &univ_rank);
    MPI_Comm_size(univ_comm, &univ_size);

    MPI_Barrier(univ_comm);

    for (cs = 0; cs < 4; cs++) {

        n = bufsizes[cs];
        if (univ_rank == 0) {
            buf = (char *) malloc(n * univ_size);
            memset(buf, 0, n * univ_size);
        } else {
            buf = (char *) malloc(n);
            memset(buf, (char) univ_rank, n);
        }

        if (!buf) {
            fprintf(stderr, "Unable to allocate %d bytes\n", n);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }


        if (univ_rank == 0) {
            MPI_Win_create(buf, sizeof(char) * n * univ_size, sizeof(char), MPI_INFO_NULL,
                           univ_comm, &win);
        } else {
            MPI_Win_create(NULL, 0, sizeof(char), MPI_INFO_NULL, univ_comm, &win);
        }

        MPI_Win_fence(0, win);

        if (univ_rank != 0) {
            MPI_Put(buf, n, MPI_CHAR, 0, n * univ_rank, n, MPI_CHAR, win);
        }

        MPI_Win_fence(0, win);

        if (univ_rank == 0) {
            for (i = 0; i < univ_size; i++) {
                if (*(buf + i * n) != i) {
                    errs++;
                }
            }
        }

        MPI_Win_free(&win);
        free(buf);
    }

    MPI_Barrier(univ_comm);

  epilogue:
    MPI_Reduce((univ_rank == 0 ? MPI_IN_PLACE : &errs), &errs, 1, MPI_INT, MPI_SUM, 0, univ_comm);
    if (univ_rank == 0) {
        if (errs) {
            printf("found %d errors\n", errs);
        } else {
            printf(" No errors\n");
        }
    }

    if (spawn_comm != MPI_COMM_NULL) {

        MPI_Comm_free(&spawn_comm);
    }

    if (univ_comm != MPI_COMM_NULL && univ_comm != world_comm) {

        MPI_Comm_free(&univ_comm);
    }

    MPI_Finalize();
    return 0;
}
