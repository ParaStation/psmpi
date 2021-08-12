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
	int *buffer;

	MTest_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	buffer = malloc(SIZE * sizeof(int));

	if (rank == 0) {
		for (i = 0; i < SIZE; i++) buffer[i] = 42;
	}

	MPI_Bcast(buffer, SIZE, MPI_INT, 0, MPI_COMM_WORLD);

	for (i = 0; i < SIZE; i++) {
		if (buffer[i] != 42) {
			printf("(%d) MPI_Bcast: Error at position %zu: %d vs. %d\n", rank, i, buffer[i], 42);
			errors++;
			break;
		}
	}

	MTest_Finalize(errors);

	return MTestReturnValue(errors);
}

