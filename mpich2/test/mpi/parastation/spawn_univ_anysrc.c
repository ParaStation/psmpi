/*
 * ParaStation
 *
 * Copyright (C) 2021 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Carsten Clauss <clauss@par-tec.com>
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
	gettimeofday(&tv,NULL);
	return tv.tv_sec;
}

int main( int argc, char *argv[] )
{
	int i, j;
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
	int root = 0;
	MPI_Comm parent_comm = MPI_COMM_NULL;
	MPI_Comm spawn_comm  = MPI_COMM_NULL;
	MPI_Comm merge_comm  = MPI_COMM_NULL;
	MPI_Comm peer_comm   = MPI_COMM_NULL;
	MPI_Comm inter_comm  = MPI_COMM_NULL;
	MPI_Comm univ_comm   = MPI_COMM_NULL;

	MTest_Init( &argc, &argv );

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if(world_size != 2) {
		printf("This program needs exactly np = 2 processes! Calling MPI_Abort()...\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	MPI_Comm_get_parent( &parent_comm );

	if(parent_comm == MPI_COMM_NULL) {
		MPI_Comm_spawn((char*)"./spawn_univ_anysrc", MPI_ARGV_NULL, 2, MPI_INFO_NULL, 0, MPI_COMM_SELF, &spawn_comm, errcodes);

	} else {
		spawn_comm = parent_comm;
	}

	if(parent_comm == MPI_COMM_NULL) {
		high = 1;
	}
	else {
		high = 0;
	}

	/* Merge each intercomm between the spawned groups into an intracomm: */
	MPI_Intercomm_merge(spawn_comm, high, &merge_comm);

	MPI_Comm_rank(merge_comm, &merge_rank);
	MPI_Comm_size(merge_comm, &merge_size);

	/* Determine the leader (rank 0 & 1 of the origin world): */

	if(parent_comm == MPI_COMM_NULL) leader = merge_rank;
	else leader = -1;

	MPI_Allgather(&leader, 1, MPI_INT, buffer, 1, MPI_INT, merge_comm);
	for(i=0; i<merge_size; i++) {
		if(buffer[i] != -1) {
			leader = i;
			break;
		}
	}

	/* Create an intercomm between the two merged intracomms (and use the origin world as bridge/peer communicator): */
	peer_comm = MPI_COMM_WORLD;
	MPI_Intercomm_create(merge_comm, leader, peer_comm, (world_rank+1)%2, 123, &inter_comm);

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

	/* Now, the MPI universe is almost flat: just three worlds forming one universe! */

	MPI_Comm_set_errhandler(univ_comm, MPI_ERRORS_RETURN);

	/* Loop over a set of several slightly different tests: */
	for(j=0; j<8; j++) {

		/* j = 0 : Test, if ANY_SOURCE communication works in this new and flat universe. */
		/* j = 1 : Test, if ANY_SOURCE communication works with 'late senders' in this new and flat universe. */
		/* j = 2 : Test, if ANY_SOURCE communication works with 'late receiver' in this new and flat universe. */
		/* j = 3 : Test, if ANY_SOURCE communication works with additional MPI_Probe() in this new and flat universe. */
		/* j = 4-7 Same tests as 0-3 but with another root... */

		if(j > 3) root = univ_size - 1;

		if(univ_rank == root) {

			int rc;
			int remote_ranks[univ_size];
			MPI_Request send_req;
			MPI_Request recv_reqs[univ_size];
			MPI_Status status_array[univ_size];

			for(i=0; i<univ_size; i++) {
				status_array[i].MPI_ERROR = MPI_SUCCESS;
			}

			if((j%4)==2) {
				int flag;
				unsigned long start_time = get_wtime_sec();
				while (1) {
					MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, univ_comm, &flag, MPI_STATUS_IGNORE);
					if (get_wtime_sec() - start_time > 2) break;
				}
			}

			MPI_Isend(&univ_rank, 1, MPI_INT, root, univ_rank, univ_comm, &send_req);

			for(i=0; i<univ_size; i++) {

				if((j%4)==3) MPI_Probe(MPI_ANY_SOURCE, i, univ_comm, MPI_STATUS_IGNORE);

				MPI_Irecv(&remote_ranks[i], 1, MPI_INT, MPI_ANY_SOURCE, i, univ_comm, &recv_reqs[i]);
			}

			rc = MPI_Waitall(univ_size, recv_reqs, status_array);

			if(rc != MPI_SUCCESS) {
				for(i=1; i<univ_size; i++) {
					printf("Error %d is %d\n", i, status_array[i].MPI_ERROR);
				}
			}

			for(i=1; i<univ_size; i++) {
				if(remote_ranks[i] != i) {
					printf("ERROR: Wrong sender in universe! (got %d /& expected %d)\n", i, remote_ranks[i]);
				}
			}

			MPI_Wait(&send_req, MPI_STATUS_IGNORE);

		} else {
			if((j%4)==1) {
				int flag;
				unsigned long start_time = get_wtime_sec();
				while (1) {
					MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, univ_comm, &flag, MPI_STATUS_IGNORE);
					if (get_wtime_sec() - start_time > 2) break;
				}
			}

			MPI_Send(&univ_rank, 1, MPI_INT, root, univ_rank, univ_comm);
		}

		MPI_Barrier(univ_comm);
	}

	MPI_Comm_disconnect(&univ_comm);

	if (parent_comm == MPI_COMM_NULL) {
		MTest_Finalize(errs);
	} else {
		MPI_Finalize();
	}

	return MTestReturnValue(errs);
}
