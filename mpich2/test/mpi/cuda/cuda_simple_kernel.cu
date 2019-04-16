
/* Derived from NVIDIA's simpleMPI CUDA example:
*
*  1. Generate some random numbers on one node.
*  2. Dispatch them to all nodes.
*  3. Compute their square root on each node's GPU.
*  4. Compute the average of the results using MPI.

*/

#include <mpi.h>
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


int main(int argc, char *argv[])
{
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

	MPI_Init(&argc, &argv);

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

	cudaMalloc((void **)&dev_in_data, block_size * grid_size * sizeof(float));
	cudaMalloc((void **)&dev_out_data, block_size * grid_size * sizeof(float));

	cudaMemcpy(dev_in_data, local_data, block_size * grid_size * sizeof(float), cudaMemcpyHostToDevice);

	compute_square_roots<<<grid_size, block_size>>>(dev_in_data, dev_out_data);

	cudaMemcpy(local_data, dev_out_data, block_size * grid_size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_in_data);
	cudaFree(dev_out_data);

	for (i = 0 , local_sum = 0.0; i < local_size; i++) {
		local_sum += local_data[i];
	}

	MPI_Reduce(&local_sum, &total_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	free(local_data);

	MPI_Finalize();

	if (rank == 0) {
		printf("No Errors\n");
		fflush(stdout);
	}

	return 0;
}
