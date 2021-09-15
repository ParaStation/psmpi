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

/* This test rotates in a loop through the root rank for Ibcast() with 
 * a total of np operations and then waits in Waitall for their completion.
 */

/* 12288 = MPIR_CVAR_BCAST_SHORT_MSG_SIZE */
#define MAX_MSGLEN 2 * 12288 + 10
#define MAX_PROCS 256

char buf[MAX_PROCS][MAX_MSGLEN];

int main(int argc, char* argv[])
{
	int i, j;
	int msglen;
	int icomm_rank;
	int icomm_size;
	MPI_Comm icomm;
	MPI_Request reqs[MAX_PROCS];
	MPI_Status stats[MAX_PROCS];

	MPI_Init(&argc, &argv);

	icomm = MPI_COMM_WORLD;

	MPI_Comm_size(icomm, &icomm_size);
	MPI_Comm_rank(icomm, &icomm_rank);

	if(icomm_size > MAX_PROCS) {
		printf("This program can handle up to np = %d (vs. %d started) processes! Calling MPI_Abort()...\n", MAX_PROCS, icomm_size);
		MPI_Abort(MPI_COMM_WORLD, -1);
	}


	for(msglen = 1; msglen < MAX_MSGLEN; msglen = msglen*2 + rand() % 10) {
		for (i = 0; i < icomm_size; ++i) {
			for(j = 0; j < msglen; j++) {
				if (icomm_rank == i) {
					buf[i][j] = icomm_rank;
				} else {
					buf[i][j] = -1;
				}
			}			
			MPI_Ibcast(buf[i], msglen, MPI_BYTE, i, icomm, &reqs[i]);
		}

		MPI_Waitall(icomm_size, reqs, stats);

		for (i = 0; i < icomm_size; ++i) {
			for(j = 0; j < msglen; j++) {
				if(buf[i][j] != i) {
					printf("(%d) ERROR: got %d but expected %d at index %d\n", icomm_rank, buf[i][j], i, j);
					MPI_Abort(MPI_COMM_WORLD, -1);
				}
				buf[i][j] = -1;
			}
		}
	}

	if(icomm_rank == 0) {
		printf(" No errors\n");
	}

	MPI_Finalize();

	return 0;
}
