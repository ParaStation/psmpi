
/*
 * ParaStation
 *
 * Copyright (C) 2016-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2022 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "mpitest.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define NUM_ELEMENTS 1024

#define MALLOC(x)          malloc(x)
#define FREE(x)            free(x)

#define CUDA_MALLOC(x,y)   cudaMalloc(x,y)
#define CUDA_FREE(x)       cudaFree(x)
#define CUDA_CHECK(call)					\
do {								\
	 if((call) != cudaSuccess) {				\
		 cudaError_t err = cudaGetLastError();		\
		 fprintf(stderr, "CUDA error calling \""#call"\", code is %d\n", err); \
		 MPI_Abort(MPI_COMM_WORLD, err);		\
	 }							\
} while(0);

int rank;
int size;

static
void init_buffers(int* sbuf, int* rbuf, int* csbuf, int* crbuf, int len)
{
	int i;

	for(i=0; i<NUM_ELEMENTS; i++) {
		rbuf[i] = 0;
		sbuf[i] = i + rank;
	}

	CUDA_CHECK(cudaMemcpy(csbuf, sbuf, len, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(crbuf, rbuf, len, cudaMemcpyHostToDevice));
}

int main(int argc, char* argv[])
{
	int i;
	int len;
	int errs;
	int *sbuf;
	int *csbuf;
	int *rbuf;
	int *crbuf;
	int *recvcounts;

	errs = 0;

	MTest_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	len = NUM_ELEMENTS * sizeof(int);

	sbuf = (int*)MALLOC(len);
	rbuf = (int*)MALLOC(len);

	CUDA_CHECK(CUDA_MALLOC((void**)&csbuf, len));
	CUDA_CHECK(CUDA_MALLOC((void**)&crbuf, len));

	MPI_Barrier(MPI_COMM_WORLD);

	// Bcast /////////////////////////////////////////////////////////////////////////

	init_buffers(sbuf, rbuf, csbuf, crbuf, len);
	if(rank == 0) {
		MPI_Bcast(csbuf, NUM_ELEMENTS, MPI_INT, 0, MPI_COMM_WORLD);
	} else {
		MPI_Bcast(crbuf, NUM_ELEMENTS, MPI_INT, 0, MPI_COMM_WORLD);
	}
	CUDA_CHECK(cudaMemcpy(rbuf, crbuf, len, cudaMemcpyDeviceToHost));

	sleep(rank);
	if(rank != 0) {
		for(i=0; i<NUM_ELEMENTS; i++) {
			if (rbuf[i] != i) {
				printf("Bcast Error: rbuf[i] = %d; i=%d\n", rbuf[i], i);
				errs++;
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Scatter ///////////////////////////////////////////////////////////////////////

	init_buffers(sbuf, rbuf, csbuf, crbuf, len);
	MPI_Scatter(csbuf, NUM_ELEMENTS/size, MPI_INT, crbuf, NUM_ELEMENTS/size, MPI_INT, 0, MPI_COMM_WORLD);
	CUDA_CHECK(cudaMemcpy(rbuf, crbuf, len, cudaMemcpyDeviceToHost));

	for(i=0; i<NUM_ELEMENTS/size; i++) {
		int val1 = rbuf[i];
		int val2 = i + rank * NUM_ELEMENTS/size;

		if (val1 != val2) {
			printf("Scatter Error: [rbuf[i] = %d] != [i + rank * NUM_ELEMENTS/size = %d]\n", val1, val2);
			errs++;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Gather ////////////////////////////////////////////////////////////////////////

	init_buffers(sbuf, rbuf, csbuf, crbuf, len);
	MPI_Gather(csbuf, NUM_ELEMENTS/size, MPI_INT, crbuf, NUM_ELEMENTS/size, MPI_INT, 0, MPI_COMM_WORLD);
	CUDA_CHECK(cudaMemcpy(rbuf, crbuf, len, cudaMemcpyDeviceToHost));

	if(rank == 0) {
		for(i=0; i<NUM_ELEMENTS; i++) {
			int val1 = rbuf[i];
			int val2 = (i % (NUM_ELEMENTS/size)) + (i / (NUM_ELEMENTS/size));

			if (val1 != val2) {
				printf("Gather Error: [rbuf[i] = %d] != [(i % (NUM_ELEMENTS/size)) + (i / (NUM_ELEMENTS/size)) = %d]\n", val1, val2);
				errs++;
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Allgather /////////////////////////////////////////////////////////////////////

	init_buffers(sbuf, rbuf, csbuf, crbuf, len);
	MPI_Allgather(csbuf, NUM_ELEMENTS/size, MPI_INT, crbuf, NUM_ELEMENTS/size, MPI_INT, MPI_COMM_WORLD);
	CUDA_CHECK(cudaMemcpy(rbuf, crbuf, len, cudaMemcpyDeviceToHost));

	for(i=0; i<NUM_ELEMENTS; i++) {
		int val1 = rbuf[i];
		int val2 = (i % (NUM_ELEMENTS/size)) + (i / (NUM_ELEMENTS/size));

		if (val1 != val2) {
			printf("Allgather Error: [rbuf[i] = %d] != [(i % (NUM_ELEMENTS/size)) + (i / (NUM_ELEMENTS/size)) = %d]\n", val1, val2);
			errs++;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Alltoall //////////////////////////////////////////////////////////////////////

	init_buffers(sbuf, rbuf, csbuf, crbuf, len);
	MPI_Alltoall(csbuf, NUM_ELEMENTS/size, MPI_INT, crbuf, NUM_ELEMENTS/size, MPI_INT, MPI_COMM_WORLD);
	CUDA_CHECK(cudaMemcpy(rbuf, crbuf, len, cudaMemcpyDeviceToHost));

	for(i=0; i<NUM_ELEMENTS; i++) {
		int val1 = rbuf[i];
		int val2 = (i % (NUM_ELEMENTS/size)) + (i / (NUM_ELEMENTS/size)) + rank * (NUM_ELEMENTS/size);

		if (val1 != val2) {
			printf("Alltoall Error: [rbuf[i] = %d] != [(i % (NUM_ELEMENTS/size)) + (i / (NUM_ELEMENTS/size)) + rank * (NUM_ELEMENTS/size) = %d]\n", val1, val2);
			errs++;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	//////////////////////////////////////////////////////////////////////////////////

	CUDA_CHECK(CUDA_FREE(csbuf));
	CUDA_CHECK(CUDA_FREE(crbuf));

	FREE(sbuf);
	FREE(rbuf);

	MPI_Barrier(MPI_COMM_WORLD);

	MTest_Finalize(errs);

	return MTestReturnValue(errs);
}
