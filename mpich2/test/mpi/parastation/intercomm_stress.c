/*
 * ParaStation
 *
 * Copyright (C) 2017-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2022 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_ROUNDS 5

int sbuf[128];
int rbuf[128];

enum { spawn_comm_type, inter_comm_type , split_comm_type };

int main(int argc, char* argv[])
{
	int i, j;
	int errs = 0;
	int world_rank, world_size;
	int split_rank, split_size, split_remote_size;
	int errcodes[4];

	MPI_Comm split_comm;
	MPI_Comm inter_comm;
	MPI_Comm spawn_comm;
	MPI_Comm parent_comm;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	MPI_Comm_get_parent(&parent_comm);

	if(parent_comm == MPI_COMM_NULL) {

		MPI_Comm_spawn( (char*)"./intercomm_stress", MPI_ARGV_NULL, world_size, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &spawn_comm, errcodes );

	} else {

		spawn_comm = parent_comm;
	}

	MPI_Comm_split(MPI_COMM_WORLD, world_rank%2, 0, &split_comm);

	MPI_Comm_rank(split_comm, &split_rank);
	MPI_Comm_size(split_comm, &split_size);

	MPI_Intercomm_create(split_comm, 0, MPI_COMM_WORLD, (world_rank+1)%2, 666, &inter_comm);

	MPI_Comm_remote_size(inter_comm, &split_remote_size);


	if( (world_rank == 0) && (parent_comm == MPI_COMM_NULL) ) {

		int msg_count = 0;
		int comm_type[2*world_size*NUM_ROUNDS];
		MPI_Request requests[2*world_size*NUM_ROUNDS];

		for(j=0; j<NUM_ROUNDS; j++) {

			for(i=1; i<split_size; i++) {

				comm_type[msg_count] = split_comm_type;
				MPI_Irecv(rbuf, 128, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, split_comm, &requests[msg_count++]);
			}

			for(i=1; i<split_remote_size; i++) {

				comm_type[msg_count] = inter_comm_type;
				MPI_Irecv(rbuf, 128, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, inter_comm,&requests[msg_count++]);
			}

			for(i=0; i<world_size; i++) {

				comm_type[msg_count] = spawn_comm_type;
				MPI_Irecv(rbuf, 128, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, spawn_comm, &requests[msg_count++]);
			}
		}

		for(i=0; i<msg_count; i++) {

			int index;
			MPI_Status status;

			MPI_Waitany(msg_count, requests, &index, &status);

			if(comm_type[index] != status.MPI_TAG) {
				fprintf(stderr, "ERROR: Expected TAG = %d but was %d for request %d.\n", comm_type[index], status.MPI_TAG, index);
				errs++;
			}
		}


		MPI_Waitall(msg_count, requests, MPI_STATUS_IGNORE);

		if(!errs) {
			printf(" No Errors\n");
		}

	} else {

		srand(getpid());

		for(j=0; j<NUM_ROUNDS; j++) {

			sleep(rand() % world_size);

			if(parent_comm != MPI_COMM_NULL) {

				MPI_Send(sbuf, 128, MPI_INT, 0, spawn_comm_type, spawn_comm);

			} else if(world_rank % 2) {

				MPI_Send(sbuf, 128, MPI_INT, 0, inter_comm_type, inter_comm);

			} else {				

				MPI_Send(sbuf, 128, MPI_INT, 0, split_comm_type, split_comm);
			}
		}
	}

	MPI_Comm_free(&split_comm);
	MPI_Comm_free(&inter_comm);
	MPI_Comm_disconnect(&spawn_comm);

	MPI_Finalize();
	return 0;
}
