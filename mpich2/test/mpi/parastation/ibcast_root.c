/*
 * ParaStation
 *
 * Copyright (C) 2020-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2022 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "mpitest.h"

/* This test iterates in a loop with increasing message sizes
 * over all possible root ranks for Ibcast() operations and
 * then waits in Waitall for their completion.
 */

/* 12288 = MPIR_CVAR_BCAST_SHORT_MSG_SIZE */
#define MAX_MSGLEN 2 * 12288
#define MAX_PROCS 256

char buf[MAX_PROCS][MAX_MSGLEN];

#define MIN(X,Y) ((X < Y) ? (X) : (Y))

int main(int argc, char* argv[])
{
	unsigned errs = 0;
	int i, j;
	int msglen;
	int icomm_rank;
	int icomm_size;
	MPI_Comm icomm;
	MPI_Request reqs[MAX_PROCS];
	MPI_Status stats[MAX_PROCS];

	MTest_Init(&argc, &argv);

	icomm = MPI_COMM_WORLD;

	MPI_Comm_size(icomm, &icomm_size);
	MPI_Comm_rank(icomm, &icomm_rank);

	if (icomm_size > MAX_PROCS) {
		printf("This program can handle up to np = %d (vs. %d started) processes! Calling MPI_Abort()...\n", MAX_PROCS, icomm_size);
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	/* ensure to test with messages before/after the MPIR_CVAR_BCAST_SHORT_MSG_SIZE threshold (= MAX_MSGLEN / 2), see above */
	for (msglen = 1; msglen <= MAX_MSGLEN; msglen = MIN(MAX_MSGLEN, msglen * 2) + (msglen == MAX_MSGLEN)) {

		for (i = 0; i < icomm_size; ++i) {
			for (j = 0; j < msglen; j++) {
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
			for (j = 0; j < msglen; j++) {
				if (buf[i][j] != i) {
					if (errs < 10) fprintf(stderr, "(%d) ERROR: got %d but expected %d at index %d\n", icomm_rank, buf[i][j], i, j);
					errs++;
				}
				buf[i][j] = -1;
			}
		}
	}

	MTest_Finalize(errs);

	return 0;
}
