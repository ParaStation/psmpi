/*
 * ParaStation
 *
 * Copyright (C) 2016-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2023 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include "mpitest.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define NUM_ELEMENTS 1024

#define MALLOC(x)          malloc(x)
#define FREE(x)            free(x)

#define RESULT(numranks) ((numranks)*((numranks)-1)/2)

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

	for(i=0; i<NUM_ELEMENTS-1; i++) {
		rbuf[i] = 0;
		sbuf[i] = i + rank;
	}

	rbuf[NUM_ELEMENTS-1] = 0;
	sbuf[NUM_ELEMENTS-1] = rank;

	CUDA_CHECK(cudaMemcpy(csbuf, sbuf, len, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(crbuf, rbuf, len, cudaMemcpyHostToDevice));
}

int main(int argc, char* argv[])
{
	int i;
	int len;
	int *sbuf;
	int *csbuf;
	int *rbuf;
	int *crbuf;
	int *recvcounts;

	int errs = 0;

	MTest_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	len = NUM_ELEMENTS * sizeof(int);

	sbuf = (int*)MALLOC(len);
	rbuf = (int*)MALLOC(len);

	CUDA_CHECK(CUDA_MALLOC((void**)&csbuf, len));
	CUDA_CHECK(CUDA_MALLOC((void**)&crbuf, len));

	MPI_Barrier(MPI_COMM_WORLD);

	// Reduce ////////////////////////////////////////////////////////////////////////

	init_buffers(sbuf, rbuf, csbuf, crbuf, len);
	MPI_Reduce(csbuf, crbuf, NUM_ELEMENTS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	CUDA_CHECK(cudaMemcpy(rbuf, crbuf, len, cudaMemcpyDeviceToHost));

	if(rank == 0) {
		if (rbuf[0] != RESULT(size)) {
			printf("Rank 0 - Reduce Error: [rbuf[0] = %d] != [size*(size-1)/2 = %d]\n", rbuf[0], RESULT(size));
			++errs;
		}
	} else {
		if (rbuf[0] != 0) {
			printf("Rank %d - Reduce Error: [rbuf[0] = %d] != 0\n", rank, rbuf[0]);
			++errs;
		}

		if (rbuf[NUM_ELEMENTS-1] != 0) {
			printf("Rank %d - Reduce Error: [rbuf[NUM_ELEMENTS-1] = %d] != 0\n", rank, rbuf[NUM_ELEMENTS-1]);
			++errs;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Allreduce /////////////////////////////////////////////////////////////////////

	init_buffers(sbuf, rbuf, csbuf, crbuf, len);
	MPI_Allreduce(csbuf, crbuf, NUM_ELEMENTS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	CUDA_CHECK(cudaMemcpy(rbuf, crbuf, len, cudaMemcpyDeviceToHost));

	if (rbuf[0] != RESULT(size)) {
		printf("Rank %d - Allreduce Error: [rbuf[0] = %d] != [size*(size-1)/2) = %d]\n", rank, rbuf[0], RESULT(size));
		++errs;
	}
	if (rbuf[NUM_ELEMENTS-1] != RESULT(size)) {
		printf("Rank %d - Allreduce Error: [rbuf[NUM_ELEMENTS-1] = %d] != [size*(size-1)/2) = %d]\n", rank, rbuf[NUM_ELEMENTS-1], RESULT(size));
		++errs;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Reduce_scatter_block //////////////////////////////////////////////////////////

	init_buffers(sbuf, rbuf, csbuf, crbuf, len);
	MPI_Reduce_scatter_block(csbuf, crbuf, NUM_ELEMENTS/size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	CUDA_CHECK(cudaMemcpy(rbuf, crbuf, len/size, cudaMemcpyDeviceToHost));

	if(rank == 0) {
		if(rbuf[0] != RESULT(size)) {
			printf("Rank 0 - Reduce_scatter_block Error: [rbuf[0] = %d] != [size*(size-1)/2) = %d]\n", rbuf[0], RESULT(size));
			++errs;
		}
	}

	if(rank == size-1) {
		if(rbuf[NUM_ELEMENTS/size-1] != RESULT(size)) {
			printf("Rank %d - Reduce_scatter_block Error: [rbuf[NUM_ELEMENTS/size-1] = %d] != [size*(size-1)/2) = %d]\n", rank, rbuf[NUM_ELEMENTS/size-1], RESULT(size));
			++errs;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Reduce_scatter ////////////////////////////////////////////////////////////////

	recvcounts = (int*)malloc(size*sizeof(int));
	for(i=0; i<size; i++) recvcounts[i] = NUM_ELEMENTS/size;

	init_buffers(sbuf, rbuf, csbuf, crbuf, len);
	MPI_Reduce_scatter(csbuf, crbuf, recvcounts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	CUDA_CHECK(cudaMemcpy(rbuf, crbuf, len/size, cudaMemcpyDeviceToHost));

	if(rank == 0) {
		if(rbuf[0] != RESULT(size)) {
			printf("Rank 0 - Reduce_scatter Error: [rbuf[0] = %d] != [size*(size-1)/2) = %d]\n", rbuf[0], RESULT(size));
			++errs;
		}
	}

	if(rank == size-1) {
		if(rbuf[NUM_ELEMENTS/size-1] != RESULT(size)) {
			printf("Rank %d - Reduce_scatter Error: [rbuf[NUM_ELEMENTS/size-1] = %d] != [size*(size-1)/2) = %d]\n", rank, rbuf[NUM_ELEMENTS/size-1], RESULT(size));
			++errs;
		}
	}

	free(recvcounts);

	MPI_Barrier(MPI_COMM_WORLD);

	// Reduce IN_PLACE ///////////////////////////////////////////////////////////////

	if(rank == 0) {
		init_buffers(rbuf, rbuf, crbuf, crbuf, len);
		MPI_Reduce(MPI_IN_PLACE, crbuf, NUM_ELEMENTS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	} else {
		init_buffers(sbuf, rbuf, csbuf, crbuf, len);
		MPI_Reduce(csbuf, crbuf, NUM_ELEMENTS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	CUDA_CHECK(cudaMemcpy(rbuf, crbuf, len, cudaMemcpyDeviceToHost));

	if(rank == 0) {
		if(rbuf[0] != RESULT(size)) {
			printf("Rank 0 - Reduce IN_PLACE Error: [rbuf[0] = %d] != [size*(size-1)/2) = %d]\n", rbuf[0], RESULT(size));
			++errs;
		}
		if(rbuf[NUM_ELEMENTS-1] != RESULT(size)) {
			printf("Rank 0 - Reduce IN_PLACE Error: [rbuf[0] = %d] != [size*(size-1)/2) = %d]\n", rbuf[NUM_ELEMENTS-1], RESULT(size));
			++errs;
		}
	} else {
		if(rbuf[0] != 0) {
			printf("Rank %d - Reduce IN_PLACE Error: [rbuf[0] = %d] != 0\n", rank, rbuf[0]);
			++errs;
		}
		if(rbuf[NUM_ELEMENTS-1] != 0) {
			printf("Rank %d - Reduce IN_PLACE Error: [rbuf[NUM_ELEMENTS-1] = %d] != 0\n", rank, rbuf[NUM_ELEMENTS-1]);
			++errs;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Allreduce IN_PLACE ////////////////////////////////////////////////////////////

	init_buffers(rbuf, rbuf, crbuf, crbuf, len);
	MPI_Allreduce(MPI_IN_PLACE, crbuf, NUM_ELEMENTS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	CUDA_CHECK(cudaMemcpy(rbuf, crbuf, len, cudaMemcpyDeviceToHost));

	if(rbuf[0] != RESULT(size)) {
		printf("Rank %d - Allreduce IN_PLACE Error: [rbuf[0] = %d] != [size*(size-1)/2 = %d]\n", rank, rbuf[0], RESULT(size));
		++errs;
	}
	if(rbuf[NUM_ELEMENTS-1] != RESULT(size)) {
		printf("Rank %d - Allreduce IN_PLACE Error: [rbuf[NUM_ELEMENTS-1] = %d] != [size*(size-1)/2 = %d]\n", rank, rbuf[NUM_ELEMENTS-1], RESULT(size));
		++errs;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Reduce_scatter_block IN_PLACE /////////////////////////////////////////////////

	init_buffers(rbuf, rbuf, crbuf, crbuf, len);
	MPI_Reduce_scatter_block(MPI_IN_PLACE, crbuf, NUM_ELEMENTS/size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	CUDA_CHECK(cudaMemcpy(rbuf, crbuf, len/size, cudaMemcpyDeviceToHost));

	if(rank == 0) {
		if (rbuf[0] != RESULT(size)) {
			printf("Rank 0 - Reduce_scatter_block IN_PLACE Error: [rbuf[0] = %d] != [size*(size-1)/2 = %d]\n", rbuf[0], RESULT(size));
			++errs;
		}
	}

	if(rank == size-1) {
		int val1 = rbuf[NUM_ELEMENTS/size-1];
		int val2 = RESULT(size);

		if (val1 != val2) {
			printf("Rank %d - Reduce_scatter_block IN_PLACE Error: [rbuf[NUM_ELEMENTS/size-1] = %d] != [size*(size-1)/2 = %d]\n", rank, val1, val2);
			++errs;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Reduce_scatter IN_PLACE ///////////////////////////////////////////////////////

	recvcounts = (int*)malloc(size*sizeof(int));
	for(i=0; i<size; i++) recvcounts[i] = NUM_ELEMENTS/size;
	init_buffers(rbuf, rbuf, crbuf, crbuf, len);

	MPI_Reduce_scatter(MPI_IN_PLACE, crbuf, recvcounts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	CUDA_CHECK(cudaMemcpy(rbuf, crbuf, len/size, cudaMemcpyDeviceToHost));

	if(rank == 0) {
		if (rbuf[0] != RESULT(size)) {
			printf("Rank 0 - Reduce_scatter IN_PLACE Error: [rbuf[0] = %d] != [size*(size-1)/2 = %d]\n", rbuf[0], RESULT(size));
			++errs;
		}
	}

	if(rank == size-1) {
		int val1 = rbuf[NUM_ELEMENTS/size-1];
		int val2 = RESULT(size);

		if (val1 != val2) {
			printf("Rank %d - Reduce_scatter IN_PLACE Error: [rbuf[NUM_ELEMENTS/size-1] = %d] != [size*(size-1)/2 = %d]\n", rank, val1, val2);
			++errs;
		}
	}

	free(recvcounts);

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
