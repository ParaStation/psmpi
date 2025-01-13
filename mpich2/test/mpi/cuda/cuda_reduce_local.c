/*
 * ParaStation
 *
 * Copyright (C) 2016-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2025 ParTec AG, Munich
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

#define CUDA_MALLOC(x,y)   cudaMalloc(x,y)
#define CUDA_FREE(x)       cudaFree(x)
#define CUDA_CHECK(call)					\
do {								\
	 if ((call) != cudaSuccess) {				\
		 cudaError_t err = cudaGetLastError();		\
		 fprintf(stderr, "CUDA error calling \""#call"\", code is %d\n", err); \
		 MPI_Abort(MPI_COMM_WORLD, err);		\
	 }							\
} while (0);

int rank;
int size;

static
void init_buffers(int *sbuf, int *rbuf, int *csbuf, int *crbuf, int len)
{
    int i;

    for (i = 0; i < NUM_ELEMENTS - 1; i++) {
        rbuf[i] = 0;
        sbuf[i] = i + rank;
    }

    rbuf[NUM_ELEMENTS - 1] = 0;
    sbuf[NUM_ELEMENTS - 1] = rank;

    CUDA_CHECK(cudaMemcpy(csbuf, sbuf, len, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(crbuf, rbuf, len, cudaMemcpyHostToDevice));
}

int main(int argc, char *argv[])
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

    sbuf = (int *) MALLOC(len);
    rbuf = (int *) MALLOC(len);

    CUDA_CHECK(CUDA_MALLOC((void **) &csbuf, len));
    CUDA_CHECK(CUDA_MALLOC((void **) &crbuf, len));

    MPI_Barrier(MPI_COMM_WORLD);

    //////////////////////////////////////////////////////////////////////////////////

    init_buffers(sbuf, rbuf, csbuf, crbuf, len);
    MPI_Reduce_local(csbuf, crbuf, NUM_ELEMENTS, MPI_INT, MPI_SUM);
    CUDA_CHECK(cudaMemcpy(rbuf, crbuf, len, cudaMemcpyDeviceToHost));

    if (rank == 0) {
        assert(rbuf[0] == size * (size - 1) / 2);
        assert(rbuf[NUM_ELEMENTS - 1] == size * (size - 1) / 2);
    } else {
        assert(rbuf[0] == 0);
        assert(rbuf[NUM_ELEMENTS - 1] == 0);
    }

    //////////////////////////////////////////////////////////////////////////////////

    MPI_Barrier(MPI_COMM_WORLD);

    CUDA_CHECK(CUDA_FREE(csbuf));
    CUDA_CHECK(CUDA_FREE(crbuf));

    FREE(sbuf);
    FREE(rbuf);

    MPI_Barrier(MPI_COMM_WORLD);

    MTest_Finalize(errs);

    return MTestReturnValue(errs);
}
