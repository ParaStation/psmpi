/*
 * ParaStation
 *
 * Copyright (C) 2020 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 */

#include <mpi.h>
#include <mpi-ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[])
{
	int world_rank;
	int world_size;
	int module_id;
	int flag;
	char value[MPI_MAX_INFO_VAL];
	int rc = 0;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	rc = MPI_Info_get(MPI_INFO_ENV, "cuda_aware", MPI_MAX_INFO_VAL, value, &flag);

	if (rc) {
		printf("\nMPI_Info_get on \"cuda_aware\" did NOT return MPI_SUCCESS!");
	}

#if defined (MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT

	if (!flag) {
		printf("\nThis MPI environment is NOT CUDA-aware -- but it should!\n");
		rc=1;
	} else {
		if (!strncmp(value, "true", MPI_MAX_INFO_VAL) == 0) {
			printf("\nMPI_INFO_ENV did not return \"cuda_aware\" = \"true\" -- but %s instead!\n", value);
			rc++;
		} else {
			if (!MPIX_Query_cuda_support()) {
				printf("\nMPIX_Query_cuda_support() said CUDA awareness is NOT enabled -- but it should! %s\n", value);
				rc++;
			}
		}
	}
#else
	if (flag && (strncmp(value, "true", MPI_MAX_INFO_VAL) == 0)) {
		printf("\nThis MPI environment is CUDA-aware -- but it should NOT!\n");
		rc=1;
	}
#endif

	if (rc == 0) printf(" No errors\n");

	MPI_Finalize();
}
