/*
 * ParaStation
 *
 * Copyright (C) 2021-2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "mpitest.h"

static
unsigned long get_wtime_sec(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec;
}

#define SIZE (1024)
char BUFFER[SIZE];

int main(int argc, char *argv[])
{
    int flag;
    int i;
    int high;
    int leader;
    int buffer[3];
    int errcodes[2];
    int errs = 0;
    int world_rank;
    int world_size;
    int merge_rank;
    int merge_size;
    int inter_rank;
    int inter_rem_size;
    int inter_loc_size;
    int univ_rank;
    int univ_size;
    MPI_Comm parent_comm = MPI_COMM_NULL;
    MPI_Comm spawn_comm = MPI_COMM_NULL;
    MPI_Comm merge_comm = MPI_COMM_NULL;
    MPI_Comm peer_comm = MPI_COMM_NULL;
    MPI_Comm inter_comm = MPI_COMM_NULL;
    MPI_Comm univ_comm = MPI_COMM_NULL;

    MTest_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        printf("This program needs exactly np = 2 processes! Calling MPI_Abort()...\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    MPI_Comm_get_parent(&parent_comm);

    if (parent_comm == MPI_COMM_NULL) {
        MPI_Comm_spawn((char *) "./spawn_univ_discon", MPI_ARGV_NULL, 2, MPI_INFO_NULL, 0,
                       MPI_COMM_SELF, &spawn_comm, errcodes);

    } else {
        spawn_comm = parent_comm;
    }

    if (parent_comm == MPI_COMM_NULL) {
        high = 1;
    } else {
        high = 0;
    }

    /* Merge each intercomm between the spawned groups into an intracomm: */
    MPI_Intercomm_merge(spawn_comm, high, &merge_comm);

    MPI_Comm_rank(merge_comm, &merge_rank);
    MPI_Comm_size(merge_comm, &merge_size);

    /* Determine the leader (rank 0 & 1 of the origin world): */

    if (parent_comm == MPI_COMM_NULL)
        leader = merge_rank;
    else
        leader = -1;

    MPI_Allgather(&leader, 1, MPI_INT, buffer, 1, MPI_INT, merge_comm);
    for (i = 0; i < merge_size; i++) {
        if (buffer[i] != -1) {
            leader = i;
            break;
        }
    }

    /* Create an intercomm between the two merged intracomms (and use the origin world as bridge/peer communicator): */
    peer_comm = MPI_COMM_WORLD;
    MPI_Intercomm_create(merge_comm, leader, peer_comm, (world_rank + 1) % 2, 123, &inter_comm);

    MPI_Comm_rank(inter_comm, &inter_rank);
    MPI_Comm_size(inter_comm, &inter_loc_size);
    MPI_Comm_remote_size(inter_comm, &inter_rem_size);

    /* Merge the new intercomm into one single univeser: */
    MPI_Intercomm_merge(inter_comm, 0, &univ_comm);

    MPI_Comm_rank(univ_comm, &univ_rank);
    MPI_Comm_size(univ_comm, &univ_size);

    /* The following disconnects() will only decrement the VCR reference counters: */
    /* (and could thus also be replaced by MPI_Comm_free()...) */
    MPI_Comm_disconnect(&inter_comm);
    MPI_Comm_disconnect(&merge_comm);
    MPI_Comm_disconnect(&spawn_comm);

    if (univ_rank == 0) {

        MPI_Send(BUFFER, SIZE, MPI_CHAR, univ_size - 1, 42, univ_comm);

    } else if (univ_rank == univ_size - 1) {

        unsigned long start_time = get_wtime_sec();

        while (1) {
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, univ_comm, &flag, MPI_STATUS_IGNORE);
            if (get_wtime_sec() - start_time > 2)
                break;
        }
        MPI_Recv(BUFFER, SIZE, MPI_CHAR, 0, 42, univ_comm, MPI_STATUS_IGNORE);
    }

    MPI_Comm_disconnect(&univ_comm);

    if (parent_comm == MPI_COMM_NULL) {
        MTest_Finalize(errs);
    } else {
        MPI_Finalize();
    }

    return MTestReturnValue(errs);
}
