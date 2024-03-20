/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

/* This test is based on pscw_ordering.c:
 * ->
 *  * This test checks an oddball case for generalized active target
 *  * synchronization where the start occurs before the post.  Since start can
 *  * block until the corresponding post, the group passed to start must be
 *  * disjoint from the group passed to post and processes must avoid a circular
 *  * wait.  Here, odd/even groups are used to accomplish this and the even group
 *  * reverses its start/post calls.
 * <-
 * ...but extends this by a loop of MAX_COUNT pt2pt messages to be sent "backward"
 * from the "first" of the two process groups to the other before this can proceed
 * with the RMA synchronization, which is here conducted in a non-blocking fashion
 * via MPI_Win_test() in the loop.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mpitest.h"

#define MAX_COUNT (nproc)

static const int SQ_LIMIT = 10;
static int SQ_COUNT = 0;
static int SQ_VERBOSE = 0;

#define SQUELCH(X)                              \
  do {                                          \
    if (SQ_COUNT < SQ_LIMIT || SQ_VERBOSE) {    \
      SQ_COUNT++;                               \
      X                                         \
    }                                           \
  } while (0)


int main(int argc, char **argv)
{
    int i, rank, nproc, errors = 0;

    int flag, count = 0;
    int *win_buf;
    MPI_Win win;
    MPI_Status status;

    int odd_nproc, even_nproc;
    int *odd_ranks, *even_ranks;
    MPI_Group odd_group, even_group, world_group;

    MTest_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if ((nproc < 2) || (nproc % 2)) {
        if (rank == 0)
            printf("Error: this test requires an even number of two or more processes\n");
        MPI_Abort(MPI_COMM_WORLD, 100);
    }

    /* Set up odd/even groups and buffers */

    odd_nproc = nproc / 2;
    even_nproc = nproc / 2 + ((nproc % 2 == 0) ? 0 : 1);

    odd_ranks = malloc(sizeof(int) * odd_nproc);
    even_ranks = malloc(sizeof(int) * even_nproc);

    for (i = 0; i < even_nproc; i++)
        even_ranks[i] = i * 2;

    for (i = 0; i < odd_nproc; i++)
        odd_ranks[i] = i * 2 + 1;

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, odd_nproc, odd_ranks, &odd_group);
    MPI_Group_incl(world_group, even_nproc, even_ranks, &even_group);

    /* Create the window */

#ifdef USE_WIN_ALLOCATE
    MPI_Win_allocate(nproc * sizeof(int), sizeof(int), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &win_buf, &win);
#else
    MPI_Alloc_mem(nproc * sizeof(int), MPI_INFO_NULL, &win_buf);
    MPI_Win_create(win_buf, nproc * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
#endif
    MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, win);
    for (i = 0; i < nproc; i++)
        win_buf[i] = -1;
    MPI_Win_unlock(rank, win);

    /* Perform PSCW communication: Odd/even matchup */

    if (rank % 2 == 0) {
        MPI_Win_start(odd_group, 0, win);       /* Even-numbered procs target odd procs */
        MPI_Win_post(odd_group, 0, win);        /* Even procs are targeted by odd procs */

        /* Write to my slot at each target */
        for (i = 0; i < odd_nproc; i++)
            MPI_Put(&rank, 1, MPI_INT, odd_ranks[i], rank, 1, MPI_INT, win);
    } else {
        MPI_Win_post(even_group, 0, win);       /* Odd procs are targeted by even procs */
        MPI_Win_start(even_group, 0, win);      /* Odd-numbered procs target even procs */

        /* Write to my slot at each target */
        for (i = 0; i < even_nproc; i++)
            MPI_Put(&rank, 1, MPI_INT, even_ranks[i], rank, 1, MPI_INT, win);
    }

    if (rank >= nproc / 2) {
        do {
            count++;
            MPI_Recv(&flag, 1, MPI_INT, rank - nproc / 2, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } while (!flag && (count < MAX_COUNT));
    }

    MPI_Win_complete(win);

    for (flag = 0, count = 0; !flag;) {
        MPI_Win_test(win, &flag);
        if ((rank < nproc / 2) && (count < MAX_COUNT)) {
            count++;
            MPI_Send(&flag, 1, MPI_INT, rank + nproc / 2, 42, MPI_COMM_WORLD);
        }
    }

    /* Perform PSCW communication: Odd/odd and even/even matchup */

    if (rank % 2 == 0) {
        MPI_Win_post(even_group, 0, win);       /* Even procs are targeted by even procs */
        MPI_Win_start(even_group, 0, win);      /* Even-numbered procs target even procs */

        /* Write to my slot at each target */
        for (i = 0; i < even_nproc; i++)
            MPI_Put(&rank, 1, MPI_INT, even_ranks[i], rank, 1, MPI_INT, win);
    } else {
        MPI_Win_post(odd_group, 0, win);        /* Odd procs are targeted by odd procs */
        MPI_Win_start(odd_group, 0, win);       /* Odd-numbered procs target odd procs */

        /* Write to my slot at each target */
        for (i = 0; i < odd_nproc; i++)
            MPI_Put(&rank, 1, MPI_INT, odd_ranks[i], rank, 1, MPI_INT, win);
    }

    if (rank < nproc / 2) {
        count = 0;

        do {
            count++;
            MPI_Recv(&flag, 1, MPI_INT, rank + nproc / 2, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } while (!flag && (count < MAX_COUNT));
    }

    MPI_Win_complete(win);

    for (flag = 0, count = 0; !flag;) {
        MPI_Win_test(win, &flag);
        if ((rank >= nproc / 2) && (count < MAX_COUNT)) {
            count++;
            MPI_Send(&flag, 1, MPI_INT, rank - nproc / 2, 42, MPI_COMM_WORLD);
        }
    }

    for (i = 0; i < nproc; i++) {
        if (win_buf[i] != i) {
            errors++;

            SQUELCH(printf("%d: Error -- win_buf[%d] = %d, expected %d\n", rank, i, win_buf[i], i);
);
        }
    }

    MPI_Win_free(&win);
#ifndef USE_WIN_ALLOCATE
    MPI_Free_mem(win_buf);
#endif

    MPI_Group_free(&world_group);
    MPI_Group_free(&odd_group);
    MPI_Group_free(&even_group);

    free(odd_ranks);
    free(even_ranks);

    MTest_Finalize(errors);
    return MTestReturnValue(errors);
}
