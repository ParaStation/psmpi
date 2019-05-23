/*
 * ParaStation
 *
 * Copyright (C) 2019 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Simon Pickartz <pickartz@par-tec.com>
 */


#include "mpi.h"
#include "mpitest.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>


#define NUM_ELEMENTS (2048)
#define MAX_ERR (0.0001)

#define CUDA_MALLOC(x,y)   cudaMalloc(x,y)
#define CUDA_FREE(x)       cudaFree(x)
#define CUDA_CHECK(call)                                                          \
	do {                                                                          \
		if((call) != cudaSuccess) {                                               \
			cudaError_t err = cudaGetLastError();                                 \
			fprintf(stderr, "CUDA error calling \""#call"\", code is %d\n", err); \
			MPI_Abort(MPI_COMM_WORLD, err);                                       \
		}                                                                         \
	} while(0);

void column_add_op(void *in, void *inout, int *len, MPI_Datatype *dtype)
{
	double *invec = in;
	double *inoutvec = inout;
	int nints, naddresses, ntypes;
	int combiner;

	MPI_Type_get_envelope(*dtype, &nints, &naddresses, &ntypes, &combiner);

	int vecargs[nints];
	MPI_Aint vecaddrs[naddresses];
	MPI_Datatype vectypes[ntypes];

	MPI_Type_get_contents(*dtype, nints, naddresses, ntypes,
			vecargs, vecaddrs, vectypes);

	int count    = vecargs[0];
	int blocklen = vecargs[1];
	int stride   = vecargs[2];

	for ( int i=0; i<count; i++ ) {
		for ( int j=0; j<blocklen; j++) {
			inoutvec[i*stride+j] += invec[i*stride+j];
		}
	}
}

void init_comm_bufs(double **inmat, double **outmat, double *inmat_blob,
		double *outmat_blob, double *cinmat_blob, double *coutmat_blob)
{
	int i, j;
	uint64_t mat_bytes = NUM_ELEMENTS*NUM_ELEMENTS*sizeof(double);

	for (i = 0; i<NUM_ELEMENTS; ++i) {
		for (j = 0; j<NUM_ELEMENTS; ++j) {
			inmat[i][j] = 1;
			outmat[i][j] = 0;
		}
	}

	/* copy to the device memory */
	CUDA_CHECK(cudaMemcpy(cinmat_blob, inmat_blob, mat_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(coutmat_blob, outmat_blob, mat_bytes, cudaMemcpyHostToDevice));
}

int main(int argc, char *argv[])
{
	int rank, size, i, j, errs = 0;
	uint64_t mat_bytes = NUM_ELEMENTS*NUM_ELEMENTS*sizeof(double);
	double **inmat, **outmat;
	double *inmat_blob, *outmat_blob, *cinmat_blob, *coutmat_blob;

	/* initialize MPI */
	MTest_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* allocate memory on the host and the device */
	inmat = (double **)malloc(NUM_ELEMENTS*sizeof(double*));
	outmat = (double **)malloc(NUM_ELEMENTS*sizeof(double*));
	inmat_blob = (double *)malloc(mat_bytes);
	outmat_blob = (double *)malloc(mat_bytes);
	CUDA_MALLOC((void**)&cinmat_blob, mat_bytes);
	CUDA_MALLOC((void**)&coutmat_blob, mat_bytes);

	/* initialize the matrices and sync with device */
	inmat[0] = inmat_blob;
	outmat[0] = outmat_blob;
	for (i=1; i<NUM_ELEMENTS; ++i) {
		inmat[i] = inmat[i-i] + NUM_ELEMENTS;
		outmat[i] = outmat[i-i] + NUM_ELEMENTS;
	}

	init_comm_bufs(inmat, outmat, inmat_blob, outmat_blob, cinmat_blob, coutmat_blob);

	/* create a column datatype */
	MPI_Datatype column_t;
	MPI_Type_vector(NUM_ELEMENTS, 1, NUM_ELEMENTS, MPI_DOUBLE, &column_t);
	MPI_Type_commit(&column_t);

	/* create column add operation */
	MPI_Op column_add;
	MPI_Op_create(column_add_op, 1, &column_add);

	// Allreduce ///////////////////////////////////////////////////////////////

	/* perform the reduction */
	MPI_Allreduce(cinmat_blob, coutmat_blob, 1, column_t, column_add, MPI_COMM_WORLD);

	/* copy from the device memory */
	CUDA_CHECK(cudaMemcpy(inmat_blob, cinmat_blob, mat_bytes, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(outmat_blob, coutmat_blob, mat_bytes, cudaMemcpyDeviceToHost));

	/* check the results */
	for (i=0; i < NUM_ELEMENTS ; ++i) {
		/* check first column */
		if (fabs(outmat[i][0] - size) > MAX_ERR) {
			fprintf(stderr, "outmat[%d][0] = %f (!= 0)\n", i, outmat[i][0]);
			errs++;
		}

		/* check second column */
		for (j=1; j<NUM_ELEMENTS; ++j) {
			if (fabs(outmat[i][j]) > MAX_ERR) {
				fprintf(stderr, "outmat[%d][%d] = %f (!= %d)\n", i, j, outmat[i][j], size);
				errs++;
			}
		}
	}

	// Allreduce (MPI_IN_PLACE) ////////////////////////////////////////////////

	init_comm_bufs(inmat, outmat, inmat_blob, outmat_blob, cinmat_blob, coutmat_blob);

	/* perform the reduction */
	MPI_Allreduce(MPI_IN_PLACE, cinmat_blob, 1, column_t, column_add, MPI_COMM_WORLD);

	/* copy from the device memory */
	CUDA_CHECK(cudaMemcpy(inmat_blob, cinmat_blob, mat_bytes, cudaMemcpyDeviceToHost));

	/* check the results */
	for (i=0; i < NUM_ELEMENTS ; ++i) {
		/* check first column */
		if (fabs(inmat[i][0] - size) > MAX_ERR) {
			fprintf(stderr, "outmat[%d][0] = %f (!= %d)\n", i, inmat[i][0], size);
			errs++;
		}

		/* check second column */
		for (j=1; j<NUM_ELEMENTS; ++j) {
			if (fabs(inmat[i][j] - 1) > MAX_ERR) {
				fprintf(stderr, "outmat[%d][%d] = %f (!= 1)\n", i, j, inmat[i][j]);
				errs++;
			}
		}
	}

	/* cleanup */
	MPI_Op_free(&column_add);
	MPI_Type_free(&column_t);
	CUDA_CHECK(CUDA_FREE(cinmat_blob));
	CUDA_CHECK(CUDA_FREE(coutmat_blob));

	MTest_Finalize(errs);
	return MTestReturnValue(errs);
}
