/*
 * ParaStation
 *
 * Copyright (C) 2017-2021 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpitest.h"

#define SIZE (2 * 1024 * 1024 * (1024 / sizeof(int)) + 4)

static int errors = 0;

int main(int argc, char** argv)
{
	size_t i;
	int rank, nprocs;
	MPI_Win win;
	MPI_Request request;
	int *buffer;

	MTest_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	buffer = malloc(SIZE * sizeof(int));

	if (rank == 0) {
		for (i = 0; i < SIZE; i++) buffer[i] = 42;
	}

	MPI_Irecv(buffer, SIZE, MPI_INT, 0, rank, MPI_COMM_WORLD, &request);
	if (rank == 0) {
		for (i = 0; i < nprocs; i++)
			MPI_Send(buffer, SIZE, MPI_INT, i, i, MPI_COMM_WORLD);
	}
	MPI_Wait(&request, MPI_STATUS_IGNORE);

	for (i = 0; i < SIZE; i++) {
		if (buffer[i] != 42) {
			printf("(%d) MPI_Irecv: Error at position %zu: %d vs. %d\n", rank, i, buffer[i], 42);
			errors++;
			break;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Win_create(buffer, rank == 0 ? SIZE : 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

	if (rank == 0) {
		for (i = 0; i < SIZE; i++) buffer[i] = 19;
	}

	MPI_Win_fence(0, win);

	if (rank > 0) {
		MPI_Get(buffer, SIZE, MPI_INT, 0, 0, SIZE, MPI_INT, win);
	}

	MPI_Win_fence(0, win);

	for (i = 0; i < SIZE; i++) {
		if (buffer[i] != 19) {
			printf("(%d) MPI_Get: Error at position %zu: %d vs. %d\n", rank, i, buffer[i], 19);
			errors++;
			break;
		}
	}

	if (rank == 0) {
		for (i = 0; i < SIZE; i++) buffer[i] = 23;
	}

	MPI_Win_fence(0, win);

	if (rank == 0) {
		for (i = 1; i < nprocs; i++) {
			MPI_Put(buffer, SIZE, MPI_INT, i, 0, SIZE, MPI_INT, win);
		}
	}

	MPI_Win_fence(0, win);

	for (i = 0; i < SIZE; i++) {
		if (buffer[i] != 23) {
			printf("(%d) MPI_Put: Error at position %zu: %d vs. %d\n", rank, i, buffer[i], 23);
			errors++;
//			break;
		}
	}

	MPI_Win_free(&win);
	free(buffer);

	MPI_Barrier(MPI_COMM_WORLD);

	MTest_Finalize(errors);

	return MTestReturnValue(errors);
}

