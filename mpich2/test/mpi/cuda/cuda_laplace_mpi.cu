/*
 * ParaStation
 *
 * Copyright (C) 2016 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:      Carsten Clauss <clauss@par-tec.com>
 */

#include <mpi.h>

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

#if 1
#define printf(...)
#endif

#define domain_size_x 256
#define domain_size_y 256

#ifdef __NVCC__
__global__ void laplace(float **old_values, float **new_values)
{
	int i = threadIdx.x + 1;
	int j = blockIdx.x + 1;

	new_values[i][j] = 0.25 * (old_values[i-1][j] + old_values[i+1][j] + old_values[i][j-1] + old_values[i][j+1]);
}

__global__ void swap_arrays(float **old_values, float **new_values)
{
	int i = threadIdx.x + 1;
	int j = blockIdx.x + 1;

	old_values[i][j] = new_values[i][j];
}

__global__ void set_pointers(float **lines, float *blob, int n, int m)
{
	int i;

	lines[0] = blob;

	for (i = 1; i < n+2; i++) {

		lines[i] = lines[i-1] + (m+2);
	}
}
#endif

int main(int argc, char **argv)
{
	int t, Tmax = 10000;

	int i, j;
	int my_rank;
	int num_ranks;

	int n, N = domain_size_x;
	int m, M = domain_size_y;

	double residual = 0.0;

	void* ptr;

	float** new_values;
	float** old_values;
	float* new_values_blob;
	float* old_values_blob;

	float** __new_values;
	float** __old_values;
	float* __new_values_blob;
	float* __old_values_blob;

	double start_time, stop_time;

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


	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);


	m = M;
	n = N/num_ranks;
	if(my_rank == num_ranks-1) n += N%num_ranks;

	printf("(%d) %d x %d / (%d x %d)\n", my_rank, N, M, n, m);


	new_values = (float**)malloc((n+2)*sizeof(float*));
	new_values_blob = (float *)malloc((n+2)*(m+2)*sizeof(float));

	old_values = (float**)malloc((n+2)*sizeof(float*));
	old_values_blob = (float *)malloc((n+2)*(m+2)*sizeof(float));

	new_values[0] = new_values_blob;
	old_values[0] = old_values_blob;
	for (i = 1; i < n+2; i++) {

		new_values[i] = new_values[i-1] + (m+2);
		old_values[i] = old_values[i-1] + (m+2);
	}

#ifdef __NVCC__
	cudaMalloc((void**)&ptr, (n+2)*sizeof(float*));
	__new_values = (float**)ptr;
	cudaMalloc((void**)&ptr, (n+2)*(m+2)*sizeof(float));
	__new_values_blob = (float*)ptr;

	cudaMalloc((void**)&ptr, (n+2)*sizeof(float*));
	__old_values = (float**)ptr;
	cudaMalloc((void**)&ptr, (n+2)*(m+2)*sizeof(float));
	__old_values_blob = (float*)ptr;

	set_pointers<<< 1, 1 >>>(__old_values, __old_values_blob, n, m);
	set_pointers<<< 1, 1 >>>(__new_values, __new_values_blob, n, m);
#endif

	printf("(%d) Memory allocated!\n", my_rank); fflush(stdout);


	for(i=0; i<n+2; i++) {

		for(j=0; j<m+2; j++) {

			new_values[i][j] = 0.0;
		}
	}

	if(my_rank == 0) {

		for(j=0; j<m+2; j++) {

			new_values[0][j] = 100.0;
		}
	}

	if(my_rank == num_ranks-1) {

		for(j=0; j<m+2; j++) {

			new_values[n+1][j] = 100.0;
		}
	}

	for(i=0; i<n+2; i++) {

		new_values[i][0]   = 100.0;
		new_values[i][N+1] = 100.0;
	}

	for(i=0; i<n+2; i++) {

		for(j=0; j<m+2; j++) {

			old_values[i][j] = new_values[i][j];
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

#ifdef __NVCC__
	cudaMemcpy(__new_values_blob, new_values_blob, (n+2)*(m+2)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(__old_values_blob, old_values_blob, (n+2)*(m+2)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(new_values, __new_values, (n+2)*sizeof(float*), cudaMemcpyDeviceToHost);
	cudaMemcpy(old_values, __old_values, (n+2)*sizeof(float*), cudaMemcpyDeviceToHost);
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	printf("(%d) Arrays initialized!\n", my_rank); fflush(stdout);

	start_time = MPI_Wtime();

	for(t=0; t<=Tmax; t++){

#ifdef __NVCC__
		laplace<<< m, n >>>(__old_values, __new_values);
#else
		for(i=1; i<n+1; i++) {

			for(j=1; j<m+1; j++) {

				new_values[i][j] = 0.25 * ( old_values[i-1][j] + old_values[i+1][j] +
							    old_values[i][j-1] + old_values[i][j+1] );
			}
		}
#endif
		if(my_rank != 0) {

			MPI_Isend(new_values[1], m+2, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD, upper_send_req);
			MPI_Irecv(old_values[0], m+2, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD, upper_recv_req);
		}

		if(my_rank != num_ranks-1) {

			MPI_Irecv(old_values[n+1], m+2, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD, lower_recv_req);
			MPI_Isend(new_values[n],   m+2, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD, lower_send_req);
		}

		MPI_Waitall(4, req_array, stat_array);

		if(t<Tmax) {
#ifdef __NVCC__
			swap_arrays<<< m, n >>>(__old_values, __new_values);
#else
			for(i=1; i<n+1; i++) {

				for(j=1; j<m+1; j++) {

					old_values[i][j]=new_values[i][j];
				}
			}
#endif
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

#ifdef __NVCC__
	cudaMemcpy(new_values_blob, __new_values_blob, (n+2)*(m+2)*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(old_values_blob, __old_values_blob, (n+2)*(m+2)*sizeof(float), cudaMemcpyDeviceToHost);

	new_values[0] = new_values_blob;
	old_values[0] = old_values_blob;
	for (i = 1; i < n+2; i++) {

		new_values[i] = new_values[i-1] + (m+2);
		old_values[i] = old_values[i-1] + (m+2);
	}

	MPI_Barrier(MPI_COMM_WORLD);
#endif

	for(i=1; i<n+1; i++) {

		for(j=1; j<m+1; j++) {

			residual += fabs(old_values[i][j] - new_values[i][j]);
		}
	}

	MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	stop_time = MPI_Wtime();

	printf("(%d) Algorithm completed! Residual = %f / Needed time: %f\n", my_rank, residual, stop_time - start_time); fflush(stdout);

	free(new_values);
	free(new_values_blob);
	free(old_values);
	free(old_values_blob);

#ifdef __NVCC__
	cudaFree(__new_values);
	cudaFree(__new_values_blob);
	cudaFree(__old_values);
	cudaFree(__old_values_blob);
#endif

	MPI_Finalize();

	if(my_rank == 0) {
		fprintf(stdout, "No Errors\n"); fflush(stdout);
	}

	return 0;
}
