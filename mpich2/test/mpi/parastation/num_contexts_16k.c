/*
 * ParaStation
 *
 * Copyright (C) 2016 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 */

#include <mpi.h>
#include <mpi-ext.h>
#include <stdio.h>

/* Check for 16k custom/dynamic communicators: (minus COMM_SELF, COMM_WORLD, and ICOMM_WORLD if needed) */
#define NUM_COMMS (16 * 1024 - 3)

/* If this test fails, then try to set MPIR_CONTEXT_SUBCOMM_WIDTH and MPIR_CONTEXT_DYNAMIC_PROC_WIDTH to (0) in mpich2/src/include/mpiimpl.h */

int main(int argc, char* argv[])
{
	int i;
	int world_rank, rank[NUM_COMMS];
	int world_size, size[NUM_COMMS];
	MPI_Comm comm_array[NUM_COMMS];

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

#if !defined(MPIX_TOPOLOGY_AWARENESS) || (MPIX_TOPOLOGY_AWARENESS == 0)

	for(i=0; i<NUM_COMMS; i++) {

		int rc;
		int color = (world_rank + 1) % world_size;

		rc = MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &comm_array[i]);

		if(rc != MPI_SUCCESS) {
			printf("\nThe maximum number of custom/dynamic communicators/contexts is %d but this test checks for %d.\n", i, NUM_COMMS);
			printf("Try to set MPIR_CONTEXT_SUBCOMM_WIDTH and MPIR_CONTEXT_DYNAMIC_PROC_WIDTH to (0) in mpich2/src/include/mpiimpl.h to get more contexts.\n");
			printf("This test is known to fail with CH3 device as well as if MPID_PSP_TOPOLOGY_AWARENESS is enabled.\n\n");
			MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
			MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &comm_array[i]);
		}

		MPI_Comm_rank(comm_array[i], &rank[i]);
		MPI_Comm_size(comm_array[i], &size[i]);
	}

	for(i=0; i<NUM_COMMS; i++) {
		MPI_Comm_free(&comm_array[i]);
	}
#endif

	printf(" No errors\n");

	MPI_Finalize();
}
