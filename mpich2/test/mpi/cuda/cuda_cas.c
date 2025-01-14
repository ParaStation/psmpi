/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2003 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 *  Portions of this code were written/modified by ParTec AG
 *  Copyright (C) 2019-2021 ParTec Cluster Competence Center GmbH, Munich
 *  Copyright (C) 2021-2025 ParTec AG, Munich
 */

#include "mpi.h"
#include "mpitest.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define ITER 100

#define MALLOC(x)          malloc(x)
#define FREE(x)            free(x)

#define CUDA_MALLOC(x,y)   cudaMalloc(x,y)
#define CUDA_FREE(x)       cudaFree(x)
#define CUDA_CHECK(call)                                                          \
	do {                                                                          \
		if ((call) != cudaSuccess) {                                               \
			cudaError_t err = cudaGetLastError();                                 \
			fprintf(stderr, "CUDA error calling \""#call"\", code is %d\n", err); \
			MPI_Abort(MPI_COMM_WORLD, err);                                       \
		}                                                                         \
	} while (0);

static const int SQ_LIMIT = 10;
static int SQ_COUNT = 0;
static int SQ_VERBOSE = 0;

#define SQUELCH(X)                              \
  do {                                          \
    if (SQ_COUNT < SQ_LIMIT || SQ_VERBOSE) {    \
      SQ_COUNT++;                               \
      X                                         \
    }                                           \
  } while (0)


/* ------ Based on rma/compare_and_swap.c  ------ */
int main(int argc, char **argv)
{
    int i, rank, nproc;
    int errors = 0;
    int *val_ptr;
    int *cval_ptr;
    MPI_Win win;

    MTest_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    val_ptr = malloc(sizeof(int));
    CUDA_CHECK(CUDA_MALLOC((void **) &cval_ptr, sizeof(int)));

    *val_ptr = 0;
    CUDA_CHECK(cudaMemcpy(cval_ptr, val_ptr, sizeof(int), cudaMemcpyHostToDevice));

    MPI_Win_create(cval_ptr, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    /* Test self communication */

    for (i = 0; i < ITER; i++) {
        int next = i + 1, result = -1;
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
        MPI_Compare_and_swap(&next, &i, &result, MPI_INT, rank, 0, win);
        MPI_Win_unlock(rank, win);
        if (result != i) {
            SQUELCH(printf("%d->%d -- Error: next=%d compare=%d result=%d val=%d\n", rank,
                           rank, next, i, result, *val_ptr););
            errors++;
        }
    }

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
    *val_ptr = 0;
    CUDA_CHECK(cudaMemcpy(cval_ptr, val_ptr, sizeof(int), cudaMemcpyHostToDevice));
    MPI_Win_unlock(rank, win);

    MPI_Barrier(MPI_COMM_WORLD);

    /* Test neighbor communication */

    for (i = 0; i < ITER; i++) {
        int next = i + 1;
        int *result, *cresult;

        result = (int *) malloc(sizeof(int));
        CUDA_CHECK(CUDA_MALLOC((void **) &cresult, sizeof(int)));

        *result = -1;
        CUDA_CHECK(cudaMemcpy(cresult, result, sizeof(int), cudaMemcpyHostToDevice));

        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, (rank + 1) % nproc, 0, win);
        MPI_Compare_and_swap(&next, &i, cresult, MPI_INT, (rank + 1) % nproc, 0, win);
        MPI_Win_unlock((rank + 1) % nproc, win);

        CUDA_CHECK(cudaMemcpy(result, cresult, sizeof(int), cudaMemcpyDeviceToHost));
        if (*result != i) {
            SQUELCH(printf("%d->%d -- Error: next=%d compare=%d result=%d val=%d\n", rank,
                           (rank + 1) % nproc, next, i, *result, *val_ptr););
            errors++;
        }

        CUDA_CHECK(CUDA_FREE(cresult));
    }

    fflush(NULL);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
    *val_ptr = 0;
    CUDA_CHECK(cudaMemcpy(cval_ptr, val_ptr, sizeof(int), cudaMemcpyHostToDevice));
    MPI_Win_unlock(rank, win);
    MPI_Barrier(MPI_COMM_WORLD);


    /* Test contention */

    if (rank != 0) {
        for (i = 0; i < ITER; i++) {
            int next = i + 1, result = -1;
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
            MPI_Compare_and_swap(&next, &i, &result, MPI_INT, 0, 0, win);
            MPI_Win_unlock(0, win);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(val_ptr, cval_ptr, sizeof(int), cudaMemcpyDeviceToHost));

    if (rank == 0 && nproc > 1) {
        if (*val_ptr != ITER) {
            SQUELCH(printf("%d - Error: expected=%d val=%d\n", rank, ITER, *val_ptr););
            errors++;
        }
    }

    MPI_Win_free(&win);

    free(val_ptr);
    CUDA_CHECK(CUDA_FREE(cval_ptr));
    MTest_Finalize(errors);

    return MTestReturnValue(errors);
}
