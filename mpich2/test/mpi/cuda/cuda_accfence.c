/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2003 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 *  Portions of this code were written/modified by ParTec AG
 *  Copyright (C) 2019-2021 ParTec Cluster Competence Center GmbH, Munich
 *  Copyright (C) 2021-2024 ParTec AG, Munich
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
		if ((call) != cudaSuccess) {                                               \
			cudaError_t err = cudaGetLastError();                                 \
			fprintf(stderr, "CUDA error calling \""#call"\", code is %d\n", err); \
			MPI_Abort(MPI_COMM_WORLD, err);                                       \
		}                                                                         \
	} while (0);


#ifndef MAX_INT
#define MAX_INT 0x7fffffff
#endif

/* ------ Based on rma/accfence2 ------
static char MTEST_Descrip[] = "Test MPI_Accumulate with fence";
*/

int main(int argc, char *argv[])
{
    int errs = 0;
    int rank, size, source;
    int minsize = 2, count, i;
    MPI_Comm comm;
    MPI_Win win;
    int *winbuf, *sbuf;
    int *cwinbuf, *csbuf;
    size_t winbufsize, sbufsize;

    MTest_Init(&argc, &argv);

    /* The following illustrates the use of the routines to
     * run through a selection of communicators and datatypes.
     * Use subsets of these for tests that do not involve combinations
     * of communicators, datatypes, and counts of datatypes */
    while (MTestGetIntracommGeneral(&comm, minsize, 1)) {
        if (comm == MPI_COMM_NULL)
            continue;
        /* Determine the sender and receiver */
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        source = 0;

        for (count = 1; count < 65000; count = count * 2) {
            /* We compare with an integer value that can be as large as
             * size * (count * count + (1/2)*(size-1))
             * For large machines (size large), this can exceed the
             * maximum integer for some large values of count.  We check
             * that in advance and break this loop if the above value
             * would exceed MAX_INT.  Specifically,
             *
             * size*count*count + (1/2)*size*(size-1) > MAX_INT
             * count*count > (MAX_INT/size - (1/2)*(size-1))
             */
            if (count * count > (MAX_INT / size - (size - 1) / 2))
                break;

            winbufsize = count * sizeof(int);
            sbufsize = count * sizeof(int);

            winbuf = (int *) malloc(winbufsize);
            sbuf = (int *) malloc(sbufsize);

            CUDA_CHECK(CUDA_MALLOC((int **) &cwinbuf, winbufsize));
            CUDA_CHECK(CUDA_MALLOC((int **) &csbuf, sbufsize));


            /* init buffers and sync with device */
            for (i = 0; i < count; i++)
                winbuf[i] = 0;
            for (i = 0; i < count; i++)
                sbuf[i] = rank + i * count;
            CUDA_CHECK(cudaMemcpy(cwinbuf, winbuf, winbufsize, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(csbuf, sbuf, sbufsize, cudaMemcpyHostToDevice));

            /* accumulate on device memory */
            MPI_Win_create(cwinbuf, winbufsize, sizeof(int), MPI_INFO_NULL, comm, &win);
            MPI_Win_fence(0, win);
            MPI_Accumulate(csbuf, count, MPI_INT, source, 0, count, MPI_INT, MPI_SUM, win);
            MPI_Win_fence(0, win);

            /* sync with device and check results */
            CUDA_CHECK(cudaMemcpy(winbuf, cwinbuf, winbufsize, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(sbuf, csbuf, sbufsize, cudaMemcpyDeviceToHost));
            if (rank == source) {
                /* Check the results */
                for (i = 0; i < count; i++) {
                    int result = i * count * size + (size * (size - 1)) / 2;
                    if (winbuf[i] != result) {
                        if (errs < 10) {
                            fprintf(stderr,
                                    "Winbuf[%d] = %d, expected %d (count = %d, size = %d)\n", i,
                                    winbuf[i], result, count, size);
                        }
                        errs++;
                    }
                }
            }
            free(winbuf);
            free(sbuf);
            CUDA_CHECK(CUDA_FREE(cwinbuf))
                CUDA_CHECK(CUDA_FREE(csbuf))
                MPI_Win_free(&win);
        }
        MTestFreeComm(&comm);
    }

    MTest_Finalize(errs);
    return MTestReturnValue(errs);
}
