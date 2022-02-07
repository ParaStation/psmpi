/*
 * ParaStation
 *
 * Copyright (C) 2019-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2022 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpi.h"
#include "mpitest.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define BUFSIZE 0x1000
#define MAX_ITER 120

#ifndef SYNC_VIA_PTR_ATTR
#define CUDA_MALLOC(x,y) \
	do { \
		cudaMalloc(x,y); \
	} while(0);
#else
#define CUDA_MALLOC(x,y) \
	do { \
		cudaMalloc(x,y); \
		int sync_op = 1; \
		CUresult res; \
		res = cuPointerSetAttribute(&sync_op,CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)*x); \
		if (res != CUDA_SUCCESS) { \
			fprintf(stderr, "CUDA error calling 'cuPointerSetAttribute', code is %d\n", res); \
		} \
	} while(0);
#endif


#define CUDA_CHECK(call) \
	do { \
		if((call) != cudaSuccess) { \
			cudaError_t err = cudaGetLastError(); \
			fprintf(stderr, "CUDA error calling \""#call"\", code is %d\n", err); \
			MPI_Abort(MPI_COMM_WORLD, err); \
		} \
	} while(0);


int main(int argc, char **argv)
{
	uint8_t *vec, *cvec;
	uint32_t count;
	uint32_t i, errs = 0;
	int rank, size;

	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Status status;

	MTest_Init(&argc, &argv);

	/* Determine the sender and receiver */
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	/* the benchmark has to be executed with an even number of ranks */
	if(size%2) {
		if(rank == 0)
			fprintf(stderr, "ERROR: %s needs an even number of Ranks. Abort!\n",
					argv[0]);
		MPI_Abort(MPI_COMM_WORLD, -1);
	}


	/* allocate buffers and ensure mapping */
	vec = (uint8_t *) malloc(BUFSIZE);
	memset(vec, 0x42, BUFSIZE);
	CUDA_MALLOC((void**)&cvec, BUFSIZE);

	for (i = 0; i<MAX_ITER; ++i) {
		if (rank%2 == 0) {
			memset(vec, i, BUFSIZE);
			CUDA_CHECK(cudaMemcpy(cvec, vec, BUFSIZE, cudaMemcpyHostToDevice));
#ifdef SYNC_VIA_MEMCPY
			CUDA_CHECK(cudaMemcpy(cvec, vec, BUFSIZE, cudaMemcpyHostToDevice));
#elif defined(SYNC_EXPLICITLY)
			/* block until all preceding  requests (i.e., the cudaMemcpy()) have completed */
			CUDA_CHECK(cudaDeviceSynchronize());
#endif
			/* send to the neighbor rank */
			MPI_Send(cvec, BUFSIZE, MPI_CHAR, rank+1, 1, MPI_COMM_WORLD);
		} else {
			MPI_Recv(vec, BUFSIZE, MPI_CHAR, rank-1, 1, MPI_COMM_WORLD, &status);
			for (count = 0; count < BUFSIZE; count++) {
				uint8_t expect = i;
				if (vec[count] != expect) {
					fprintf(stderr, "vec[%d] = 0x%x (expected: 0x%x)\n", count, vec[count], expect);
					errs++;
					exit(-1);
				}
			}

		}
	}

	/* free the buffers */
	free(vec);
	CUDA_CHECK(cudaFree(cvec))

	MTest_Finalize(errs);
	return MTestReturnValue(errs);
}
