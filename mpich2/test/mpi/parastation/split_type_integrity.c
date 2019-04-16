/*
 * ParaStation
 *
 * Copyright (C) 2017 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _VERBOSE_ 0

int main(int argc, char **argv)
{
	int i, j;
	int errs = 0;
	int world_rank, shm_rank;
	int world_size, shm_size;
	int remote_size;
	int num_procs;
	int num_comms;
	int name_len;
	int checksum;
	int checkres;
	char hostname[MPI_MAX_PROCESSOR_NAME];

	MPI_Comm shm_comm;

	MPI_Init(&argc, &argv);

	MPI_Get_processor_name(hostname, &name_len);

	for(i=0,checksum=0; (i<name_len) && (hostname[i]!=0); i++) {
		checksum += hostname[i] * (i+1);
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// If SMP-awarenes is disabled, it is quite likely that comm_shmem equals COMM_SELF! Just skip this test then...
	if(getenv("PSP_SMP_AWARENESS") && (strcmp(getenv("PSP_SMP_AWARENESS"), "0") == 0)) goto finalize;

	if(_VERBOSE_) {
		if(world_rank == 0) {
			printf("(%d) There are %d ranks with me in COMM_WORLD...\n", world_rank, world_size);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &shm_comm);

	MPI_Comm_rank(shm_comm, &shm_rank);
	MPI_Comm_size(shm_comm, &shm_size);

	if(_VERBOSE_) {
		if(shm_rank == 0) {
			printf("(%d) There are %d ranks with me in comm_shmem...\n", world_rank, shm_size);
		}
	}

	MPI_Allreduce(&checksum, &checkres, 1, MPI_INT, MPI_SUM, shm_comm);

	if(checkres != shm_size * checksum) {
		fprintf(stderr, "(%d) Detected communicator internal anomaly: %d for %s\n", world_rank, checksum, hostname);
		errs++;
	}

	if(world_rank == 0) {

		int* check_array;

		num_comms = 1;
		checkres = checksum;

		for(num_procs = shm_size; num_procs < world_size; ) {

			MPI_Recv(&remote_size, 1, MPI_INT, MPI_ANY_SOURCE, 88, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			num_procs += remote_size;
			num_comms ++;
		}

		check_array = malloc(num_comms * sizeof(int));
		check_array[0] = checksum;

		for(i=1; i<num_comms; i++) {
			MPI_Recv(&check_array[i], 1, MPI_INT, MPI_ANY_SOURCE, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		for(i=0; i<num_comms; i++) {
			for(j=i+1; j<num_comms; j++) {
				if(check_array[i] == check_array[j]) {
					fprintf(stderr, "(%d) Detected inter-communicator anomaly: %d vs. %d\n", world_rank, check_array[i], check_array[j]);
					errs++;
				}
			}
		}

		free(check_array);

	} else {

		if(shm_rank == 0) {
			MPI_Send(&shm_size, 1, MPI_INT, 0, 88, MPI_COMM_WORLD);
			MPI_Send(&checksum, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
		}
	}

	MPI_Reduce((world_rank == 0 ? MPI_IN_PLACE : &errs), &errs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Comm_free(&shm_comm);

finalize:
        MPI_Finalize();

	if(world_rank == 0 && errs == 0) {
		printf(" No Errors\n");
	}

        return 0;
}
