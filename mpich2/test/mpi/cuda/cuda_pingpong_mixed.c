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

#define WARM_UP  16
#define PING_TAG  0
#define PONG_TAG  1

#define DEFAULT_MSGS_START 1
#define DEFAULT_MSGS_STOP  (65536 * 16)
#define DEFAULT_MSGS_REPS  100

#define INTEGRITY_CHECK

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

double PingPong(int ping, int pong, int len, int reps, int mode, int *errs)
{
    int iter;
    void *sbuf, *_sbuf, *csbuf;
    void *rbuf, *_rbuf, *crbuf;

    double stop_time;
    double start_time;

    sbuf = MALLOC(len);
    _sbuf = sbuf;
    rbuf = MALLOC(len);
    _rbuf = rbuf;

    memset(sbuf, 0, len);
    memset(rbuf, 0, len);

#ifdef INTEGRITY_CHECK
    {
        int i;
        for (i = 0; i < len; i++) {
            ((unsigned char *) sbuf)[i] = i % 256;
        }
    }
#endif

    if (mode) {

        CUDA_CHECK(CUDA_MALLOC(&csbuf, len));
        CUDA_CHECK(CUDA_MALLOC(&crbuf, len));

        CUDA_CHECK(cudaMemcpy(csbuf, sbuf, len, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(crbuf, rbuf, len, cudaMemcpyHostToDevice));

        sbuf = csbuf;

        if (mode == 1) {
            rbuf = crbuf;
        }
    }

    if (rank == ping) {

        for (iter = 0; iter < reps + WARM_UP; iter++) {

            if (iter == WARM_UP) {

                start_time = MPI_Wtime();
            }

            MPI_Send(sbuf, len, MPI_BYTE, pong, PING_TAG, MPI_COMM_WORLD);
            MPI_Recv(rbuf, len, MPI_BYTE, pong, PONG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        stop_time = MPI_Wtime();
    }

    if (rank == pong) {

        for (iter = 0; iter < reps + WARM_UP; iter++) {

            if (iter == WARM_UP) {

                start_time = MPI_Wtime();
            }

            MPI_Recv(rbuf, len, MPI_BYTE, ping, PING_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(sbuf, len, MPI_BYTE, ping, PONG_TAG, MPI_COMM_WORLD);
        }

        stop_time = MPI_Wtime();
    }
#ifdef INTEGRITY_CHECK
    {
        int i;

        if (mode == 1) {
            CUDA_CHECK(cudaMemcpy(_rbuf, crbuf, len, cudaMemcpyDeviceToHost));
        }

        for (i = 0; i < len; i++) {
            if (((unsigned char *) _rbuf)[i] != i % 256) {
                printf("INTEGRITY CHECK ERROR: %d vs. %d at %d\n", ((unsigned char *) _rbuf)[i],
                       i % 256, i);
                ++(*errs);
            }
        }
    }
#endif

    FREE(_sbuf);
    FREE(_rbuf);

    if (mode) {
        CUDA_CHECK(CUDA_FREE(csbuf));
        CUDA_CHECK(CUDA_FREE(crbuf));
    }

    return (stop_time - start_time);
}

int main(int argc, char *argv[])
{
    int current_size;
    int msgs_start = DEFAULT_MSGS_START;
    int msgs_stop = DEFAULT_MSGS_STOP;
    int msgs_reps = DEFAULT_MSGS_REPS;

    int errs = 0;

    MTest_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);

    if (size < 2) {
        fprintf(stderr,
                "##### ERROR: Too few MPI ranks since %s needs at least two processes! Abort!\n",
                argv[0]);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (rank == 0) {
        if (size > 2) {
            printf
                ("# *** WARNING: Too many MPI ranks since %s only needs 2 procs! --> The %d additional procs are now going to wait in MPI_Barrier...\n",
                 argv[0], size - 2);
            fflush(stdout);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (current_size = msgs_start; current_size <= msgs_stop; current_size *= 2) {

        /* Host to Host */
        PingPong(0, 1, current_size, msgs_reps, 0, &errs);

        /* Device to Device */
        PingPong(0, 1, current_size, msgs_reps, 1, &errs);

        /* Mixed */
        PingPong(0, 1, current_size, msgs_reps, 2, &errs);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MTest_Finalize(errs);

    return MTestReturnValue(errs);
}
