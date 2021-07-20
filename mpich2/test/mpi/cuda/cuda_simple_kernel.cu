
/* Derived from NVIDIA's simpleMPI CUDA example:
*
*  1. Generate some random numbers on one node.
*  2. Dispatch them to all nodes.
*  3. Compute their square root on each node's GPU.
*  4. Compute the average of the results using MPI.

*/

#include <mpi.h>
extern "C" {
#include "mpitest.h"
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define BLOCK_SIZE 256
#define GRID_SIZE 10000

__global__ void compute_square_roots(float *in_data, float *out_data)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	out_data[tid] = sqrt(in_data[tid]);
}


#define CUDA_CHECK(call, fatal)                                         \
	do {                                                            \
		if ((cuda_error == cudaSuccess) && (((call) != cudaSuccess) || ((cuda_error = cudaGetLastError()) != cudaSuccess))) { \
			MTestPrintfMsg(0, #call" returned: %s\n", cudaGetErrorString(cuda_error)); \
			if (fatal) MTestError("Fatal CUDA error! Calling MPI_Abort()...\n");			\
		}                                                       \
	} while(0);


int main(int argc, char *argv[])
{
	int errs = 0;
	cudaError_t cuda_error = cudaSuccess;

	int i;
	int rank, np;
	int block_size;
	int grid_size;
	int local_size;
	int total_size;
	float *total_data;
	float *local_data;
	float *dev_in_data;
	float *dev_out_data;
	float local_sum;
	float total_sum;

	MTest_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	block_size = BLOCK_SIZE;
	grid_size = GRID_SIZE;
	local_size = block_size * grid_size;
	total_size = local_size * np;

	if (rank == 0) {
		total_data = (float*) malloc(total_size * sizeof(float));
		for (i = 0; i < total_size; i++) {
			total_data[i] = (float)rand() / RAND_MAX;
		}
	}

	local_data = (float*) malloc(local_size * sizeof(float));

	MPI_Scatter(total_data, local_size, MPI_FLOAT, local_data, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		free(total_data);
	}

	CUDA_CHECK(cudaMalloc((void **)&dev_in_data, block_size * grid_size * sizeof(float)), 1);
	CUDA_CHECK(cudaMalloc((void **)&dev_out_data, block_size * grid_size * sizeof(float)), 1);

	CUDA_CHECK(cudaMemcpy(dev_in_data, local_data, block_size * grid_size * sizeof(float), cudaMemcpyHostToDevice), 0);

	compute_square_roots<<<grid_size, block_size>>>(dev_in_data, dev_out_data);
	CUDA_CHECK(cudaDeviceSynchronize(), 0);

	cudaMemcpy(local_data, dev_out_data, block_size * grid_size * sizeof(float), cudaMemcpyDeviceToHost);

	CUDA_CHECK(cudaFree(dev_in_data), 0);
	CUDA_CHECK(cudaFree(dev_out_data), 0);

	for (i = 0 , local_sum = 0.0; i < local_size; i++) {
		local_sum += local_data[i];
	}

	MPI_Reduce(&local_sum, &total_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	free(local_data);


	MTest_Finalize(errs);
	return MTestReturnValue(errs);

}
