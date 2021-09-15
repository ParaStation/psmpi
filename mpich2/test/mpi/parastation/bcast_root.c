/*
 * ParaStation
 *
 * Copyright (C) 2020-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021      ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

/* This test rotates in a loop through the root rank
 * for Bcast() with a total of np operations.
 */

/* 12288 = MPIR_CVAR_BCAST_SHORT_MSG_SIZE */
#define MAX_MSGLEN 2 * 12288 + 10
#define MAX_PROCS 256

char buf[MAX_PROCS][MAX_MSGLEN];

int main(int argc, char* argv[])
{
	int i, j;
	int msglen;
	int comm_rank;
	int comm_size;
	MPI_Comm comm;

	MPI_Init(&argc, &argv);

	comm = MPI_COMM_WORLD;

	MPI_Comm_size(comm, &comm_size);
	MPI_Comm_rank(comm, &comm_rank);

	if(comm_size > MAX_PROCS) {
		printf("This program can handle up to np = %d (vs. %d started) processes! Calling MPI_Abort()...\n", MAX_PROCS, comm_size);
		MPI_Abort(MPI_COMM_WORLD, -1);
	}


	for(msglen = 1; msglen < MAX_MSGLEN; msglen = msglen*2 + rand() % 10) {
		for (i = 0; i < comm_size; ++i) {
			for(j = 0; j < msglen; j++) {
				if (comm_rank == i) {
					buf[i][j] = comm_rank;
				} else {
					buf[i][j] = -1;
				}
			}			
			MPI_Bcast(buf[i], msglen, MPI_BYTE, i, comm);
		}

		for (i = 0; i < comm_size; ++i) {
			for(j = 0; j < msglen; j++) {
				if(buf[i][j] != i) {
					printf("(%d) ERROR: got %d but expected %d at index %d\n", comm_rank, buf[i][j], i, j);
					MPI_Abort(MPI_COMM_WORLD, -1);
				}
				buf[i][j] = -1;
			}
		}
	}

	if(comm_rank == 0) {
		printf(" No errors\n");
	}

	MPI_Finalize();

	return 0;
}
