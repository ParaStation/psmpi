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

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define _VERBOSE_ 0

const int DIM = 16 * 1024;

int main (int argc, char **argv)
{
	int errs;
	int rank, size;

	MPI_Win win;
	MPI_Datatype mpi_array;

	size_t i;
	double *a;
	int x = DIM;
	int f = 1;

	MPI_Init (&argc, &argv);

	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	if(size != 2) {
		fprintf(stderr, "error: %s must be started with %d MPI ranks\n", argv[0], size);
		MPI_Finalize();
		return 0;
	}

	MPI_Type_contiguous (f * DIM * DIM + x, MPI_DOUBLE, &mpi_array);
	MPI_Type_commit (&mpi_array);

	a = (double *)malloc((f * DIM * DIM + x) * sizeof (double));

	if (rank == 0) {
		for (i = 0; i < f * DIM * DIM + x; i++) {
			a[i] = (double)i;
		}
	}

	if(rank == 0) {

		if(_VERBOSE_) printf("Sending %ld MB\n", (f * (DIM / 1024) * (DIM / 1024) + (x / 1024)) * sizeof (double));

		MPI_Send (a, 1, mpi_array, 1, 99, MPI_COMM_WORLD);

	} else {

		MPI_Recv (a, 1, mpi_array, 0, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		for (i = 0; i < f * DIM * DIM + x; i++) {
			if (a[i] != (double)i) {
				if(!errs) fprintf (stderr, "MPI_Recv: detected error at position %lld: %f vs. %f\n", (long long int)i, a[i], (double)i);
				errs++;
			}
			a[i] = 0.0;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Win_create(a, rank == 0 ? (f * DIM * DIM + x) * sizeof(double) : 0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
	MPI_Win_fence(0, win);

	if(rank == 1) {

		if(_VERBOSE_) printf("Getting %ld MB\n", (f * (DIM / 1024) * (DIM / 1024) + (x / 1024)) * sizeof (double));

		MPI_Get(a, 1, mpi_array, 0, 0, 1, mpi_array, win);
		MPI_Win_fence(0, win);

		for (i = 0; i < f * DIM * DIM + x; i++) {
			if (a[i] != (double)i) {
				if(!errs) fprintf (stderr, "MPI_Get: detected error at position %lld: %f vs. %f\n", (long long int)i, a[i], (double)i);
				errs++;
			}
			a[i] = 1.0;
		}
	} else {
		MPI_Win_fence(0, win);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if(rank == 1) {

		if(_VERBOSE_) printf("Accumulating %ld MB\n", (f * (DIM / 1024) * (DIM / 1024) + (x / 1024)) * sizeof (double));

		MPI_Accumulate(a, 1, mpi_array, 0, 0, 1, mpi_array, MPI_SUM, win);
		MPI_Win_fence(0, win);

	} else {
		MPI_Win_fence(0, win);

		for (i = 0; i < f * DIM * DIM + x; i++) {
			if (a[i] != ((double)i + 1.0)) {
				if(!errs) fprintf (stderr, "MPI_Accumulate: detected error at position %lld: %f vs. %f\n", (long long int)i, a[i], ((double)i + 1.0));
				errs++;
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Reduce((rank == 0 ? MPI_IN_PLACE : &errs), &errs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(rank == 0) {
		if(!errs) {
			printf(" No Errors\n");
		} else {
			printf(" Found %d mismatches\n", errs);
		}
	}

	free (a);

	MPI_Win_free(&win);

	MPI_Type_free(&mpi_array);

	MPI_Finalize ();

	return 0;
}
