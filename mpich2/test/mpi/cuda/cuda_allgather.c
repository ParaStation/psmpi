/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2003 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 *  Portions of this code were written/modified by ParTec AG
 *  Copyright (C) 2019-2021 ParTec Cluster Competence Center GmbH, Munich
 *  Copyright (C) 2021-2023 ParTec AG, Munich
 */

#include "mpi.h"
#include "mpitest.h"
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define NUM_ELEMENTS 1024

#define MALLOC(x)          malloc(x)
#define FREE(x)            free(x)

#define CUDA_MALLOC(x,y)   cudaMalloc(x,y)
#define CUDA_FREE(x)       cudaFree(x)
#define CUDA_CHECK(call)                                                          \
	do {                                                                          \
		if((call) != cudaSuccess) {                                               \
			cudaError_t err = cudaGetLastError();                                 \
			fprintf(stderr, "CUDA error calling \""#call"\", code is %d\n", err); \
			MPI_Abort(MPI_COMM_WORLD, err);                                       \
		}                                                                         \
	} while(0);

/* ------ Based on coll/allgather3 ------
 * Tests Allgather on array of doubles. Same as allgather2 test
 * but without MPI_IN_PLACE. */

int main(int argc, char **argv)
{
	double *vecout, *invec;
	double *cvecout, *cinvec;
	MPI_Comm comm;
	int count, minsize = 2;
	int i, errs = 0;
	int rank, size;
	size_t insize, sizeout;

	MTest_Init(&argc, &argv);

	while (MTestGetIntracommGeneral(&comm, minsize, 1)) {
		if (comm == MPI_COMM_NULL)
			continue;
		/* Determine the sender and receiver */
		MPI_Comm_rank(comm, &rank);
		MPI_Comm_size(comm, &size);

		for (count = 1; count < 9000; count = count * 2) {
			insize = count * sizeof(double);
			sizeout = size * count * sizeof(double);

			invec = (double *) malloc(insize);
			vecout = (double *) malloc(sizeout);

			CUDA_CHECK(CUDA_MALLOC((void**)&cinvec, insize));
			CUDA_CHECK(CUDA_MALLOC((void**)&cvecout, sizeout));

			/* init buffers and sync with device */
			for (i = 0; i < count; i++) {
				invec[i] = rank * count + i;
			}
			CUDA_CHECK(cudaMemcpy(cinvec, invec, insize, cudaMemcpyHostToDevice));

			/* allgather on device memory */
			MPI_Allgather(cinvec, count, MPI_DOUBLE, cvecout, count, MPI_DOUBLE, comm);

			/* sync with device and check results */
			CUDA_CHECK(cudaMemcpy(vecout, cvecout, sizeout, cudaMemcpyDeviceToHost));
			for (i = 0; i < count * size; i++) {
				if (vecout[i] != i) {
					errs++;
					if (errs < 10) {
						fprintf(stderr, "vecout[%d]=%d\n", i, (int) vecout[i]);
					}
				}
			}
			free(invec);
			free(vecout);
			CUDA_CHECK(CUDA_FREE(cinvec))
			CUDA_CHECK(CUDA_FREE(cvecout))
		}

		MTestFreeComm(&comm);
	}

	/* Do a zero byte gather */
	MPI_Allgather(MPI_IN_PLACE, -1, MPI_DATATYPE_NULL, NULL, 0, MPI_BYTE, MPI_COMM_WORLD);

	MTest_Finalize(errs);
	return MTestReturnValue(errs);
}
