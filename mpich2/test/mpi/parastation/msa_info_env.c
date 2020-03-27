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


int main(int argc, char* argv[])
{
	int world_rank;
	int world_size;
	int module_id;
	int msa_enabled;
	char* env_str;
	int flag;
	char value[MPI_MAX_INFO_VAL];
	int rc = 0;

	env_str = getenv("PSP_MSA_AWARENESS");
	if(env_str && (atoi(env_str) == 1)) {
		msa_enabled = 1;
	} else {
		msa_enabled = 0;
	}

	env_str = getenv("PSP_MSA_MODULE_ID");
	if(env_str) {
		module_id = atoi(env_str);
	} else {
		module_id = -1;
	}


	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	rc += MPI_Info_get(MPI_INFO_ENV, "msa_module_id", MPI_MAX_INFO_VAL, value, &flag);

#if defined(MPIX_TOPOLOGY_AWARENESS) && MPIX_TOPOLOGY_AWARENESS
	if (flag) { /* This MPI environment is modularity-aware! */

		if (msa_enabled) {

			if ( (module_id > -1) && (atoi(value) != module_id) ) {

				printf("\nModule ID was explicitly set to %d but MPI_INFO_ENV returned %d / %s\n", module_id, atoi(value), value);
				rc++;
			}
		} else {
			if (atoi(value) > 0) {

				printf("\nModularity awareness was disabled but MPI_INFO_ENV returned an ID > 0 (%d / %s)\n", atoi(value), value);
				rc++;
			}
		}
	}
#else
	if (flag) { /* This MPI environment is modularity-aware -- but it should NOT! */
		printf("\nThis MPI environment is modularity-aware -- but it should NOT!\n");
		rc++;
	}
#endif

	if (rc == 0) printf(" No errors\n");

	MPI_Finalize();
}
