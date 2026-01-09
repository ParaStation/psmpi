/*
 * ParaStation
 *
 * Copyright (C) 2016-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
extern "C" {
#include "mpitest.h"
}
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#endif

#define ThreadsPerBlock 1

// Total domain size:
#define domain_size_x 256
#define domain_size_y 256

#ifdef __NVCC__
__global__ void laplace(float *old_values, float *new_values, int n, int m)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	if ((i > 0) && (i < n-1) &&
	    (j > 0) && (j < m-1)) {
		new_values[i*m + j] = 0.25 *
			(old_values[(i-1)*m + j] +
			 old_values[(i+1)*m + j] +
			 old_values[i*m + j-1]   +
			 old_values[i*m + j+1]   );
	}
}

__global__ void swap_arrays(float *old_values, float *new_values, int n, int m)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	if ((i > 0) && (i < n-1) &&
	    (j > 0) && (j < m-1)) {
		new_values[i*m + j] = old_values[i*m + j];
	}
}
#endif

#define CUDA_CHECK(call, fatal)                                         \
	do {                                                            \
		if ((cuda_error == cudaSuccess) && (((call) != cudaSuccess) || ((cuda_error = cudaGetLastError()) != cudaSuccess))) { \
			MTestPrintfMsg(0, #call" returned: %s\n", cudaGetErrorString(cuda_error)); \
			if (fatal) MTestError("Fatal CUDA error! Calling MPI_Abort()...\n");			\
		}                                                       \
	} while(0);


double start_time, stop_time;

int main(int argc, char **argv)
{
	int errs = 0;
	int t, Tmax = 10000;

	int i, j;
	int my_rank;
	int num_ranks;

	int n = domain_size_x;
	int m = domain_size_y;

	double residual = 0.0;

	float** new_values;
	float** old_values;
	float* new_values_blob;
	float* old_values_blob;

	float* send_buffer_up;
	float* send_buffer_dn;
	float* recv_buffer_up;
	float* recv_buffer_dn;

#ifdef __NVCC__
	cudaError_t cuda_error = cudaSuccess;
	float* __new_values_blob;
	float* __old_values_blob;
	void* ptr;
#endif

	MPI_Request* upper_send_req;
	MPI_Request* lower_send_req;

	MPI_Request* upper_recv_req;
	MPI_Request* lower_recv_req;

	MPI_Status stat_array[4];
	MPI_Request req_array[4] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL };

	upper_send_req = req_array;
	lower_send_req = req_array + 1;
	upper_recv_req = req_array + 2;
	lower_recv_req = req_array + 3;


	MTest_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	n = domain_size_x / num_ranks;
	if (my_rank == num_ranks-1) n += domain_size_x % num_ranks;

	m += 2;
	n += 2;

	MTestPrintfMsg(3, "(%d) %d x %d / (%d x %d)\n", my_rank, domain_size_x, domain_size_y, n, m);

	new_values = (float**)malloc(n * sizeof(float*));
	new_values_blob = (float *)malloc(n * m * sizeof(float));

	old_values = (float**)malloc(n * sizeof(float*));
	old_values_blob = (float *)malloc(n * m * sizeof(float));

	new_values[0] = new_values_blob;
	old_values[0] = old_values_blob;
	for (i = 1; i < n; i++) {

		new_values[i] = new_values[i-1] + m;
		old_values[i] = old_values[i-1] + m;
	}

#ifdef __NVCC__
	CUDA_CHECK(cudaMalloc((void**)&ptr, n * m * sizeof(float)), 1);
	__new_values_blob = (float*)ptr;
	CUDA_CHECK(cudaMalloc((void**)&ptr, n * m * sizeof(float)), 1);
	__old_values_blob = (float*)ptr;
#endif

	MTestPrintfMsg(1, "(%d) Memory allocated!\n", my_rank);


	for (i = 0; i < n; i++) {

		for (j = 0; j < m; j++) {

			new_values[i][j] = 0.0;
		}
	}

	if (my_rank == 0) {

		for (j = 0; j < m; j++) {

			new_values[0][j] = 100.0;
		}
	}

	if (my_rank == num_ranks-1) {

		for (j = 0; j < m; j++) {

			new_values[n-1][j] = 100.0;
		}
	}

	for (i = 0; i < n; i++) {

		new_values[i][0]   = 100.0;
		new_values[i][m-1] = 100.0;
	}

	for (i = 0; i < n; i++) {

		for (j = 0; j < m; j++) {

			old_values[i][j] = new_values[i][j];
		}
	}

#ifdef __NVCC__
	CUDA_CHECK(cudaMemcpy(__new_values_blob, new_values_blob, n * m * sizeof(float), cudaMemcpyHostToDevice), 0);
	CUDA_CHECK(cudaMemcpy(__old_values_blob, old_values_blob, n * m * sizeof(float), cudaMemcpyHostToDevice), 0);
	MPI_Barrier(MPI_COMM_WORLD);

	send_buffer_up = &__new_values_blob[m];
	send_buffer_dn = &__new_values_blob[(n-2)*m];
	recv_buffer_up = &__old_values_blob[0];
	recv_buffer_dn = &__old_values_blob[(n-1)*m];

#else
	send_buffer_up = new_values[1];
	send_buffer_dn = new_values[n-2];
	recv_buffer_up = old_values[0];
	recv_buffer_dn = old_values[n-1];
#endif

	MTestPrintfMsg(1, "(%d) Arrays initialized!\n", my_rank);

	start_time = MPI_Wtime();

	for (t = 0; t <= Tmax; t++){

#ifdef __NVCC__
		laplace<<< m, n >>>(__old_values_blob, __new_values_blob, n, m);
		CUDA_CHECK(cudaDeviceSynchronize(), 0);
#else
		for (i = 1; i < n-1; i++) {

			for (j = 1; j < m-1; j++) {

				new_values[i][j] = 0.25 * ( old_values[i-1][j] + old_values[i+1][j] +
							    old_values[i][j-1] + old_values[i][j+1] );
			}
		}
#endif

		if (my_rank != 0) {

			MPI_Isend(send_buffer_up, m, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD, upper_send_req);
			MPI_Irecv(recv_buffer_up, m, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD, upper_recv_req);
		}

		if (my_rank != num_ranks-1) {

			MPI_Irecv(recv_buffer_dn, m, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD, lower_recv_req);
			MPI_Isend(send_buffer_dn, m, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD, lower_send_req);
		}

		MPI_Waitall(4, req_array, stat_array);

		if (t < Tmax) {

#ifdef __NVCC__
			swap_arrays<<< m, n >>>(__new_values_blob, __old_values_blob, n, m);
			CUDA_CHECK(cudaDeviceSynchronize(), 0);
#else
			for (i = 1; i < n-1; i++) {

				for (j = 1; j < m-1; j++) {

					old_values[i][j] = new_values[i][j];
				}
			}
#endif
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

#ifdef __NVCC__
	CUDA_CHECK(cudaMemcpy(new_values_blob, __new_values_blob, n * m * sizeof(float), cudaMemcpyDeviceToHost), 1);
	CUDA_CHECK(cudaMemcpy(old_values_blob, __old_values_blob, n * m * sizeof(float), cudaMemcpyDeviceToHost), 1);
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	for (i = 1; i < n-1; i++) {

		for (j = 1; j < m-1; j++) {

			residual += fabs(old_values[i][j] - new_values[i][j]);
		}
	}

	MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	stop_time = MPI_Wtime();

	MTestPrintfMsg(1, "(%d) Algorithm completed!\n", my_rank);
	MTestPrintfMsg(2, "(%d) Residual = %f / Needed time: %f\n", my_rank, residual, stop_time - start_time);

	free(new_values);
	free(new_values_blob);
	free(old_values);
	free(old_values_blob);

#ifdef __NVCC__
	CUDA_CHECK(cudaFree(__new_values_blob), 0);
	CUDA_CHECK(cudaFree(__old_values_blob), 0);
#endif

	MTest_Finalize(errs);
	return MTestReturnValue(errs);
}
