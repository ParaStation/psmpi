/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2022 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpi.h"
#include "mpitest.h"
#include <stdio.h>
#include <stdlib.h>

static int bufsizes[4] = { 1, 100, 10000, 1000000 };

/* This tests checks for the synchronizing behavior
 * of Issend() within a spawned environment.
 */

int main( int argc, char *argv[] )
{
	int errs = 0;
	int src, dest;
	int flag;
	int *errcodes;
	int univ_rank, univ_size;
	int world_rank, world_size;
	MPI_Comm      univ_comm   = MPI_COMM_NULL;
	MPI_Comm      world_comm  = MPI_COMM_NULL;
	MPI_Comm      spawn_comm  = MPI_COMM_NULL;
	MPI_Comm      parent_comm = MPI_COMM_NULL;
	MPI_Status    status;
	MPI_Request   req;
	int cs, n, t;
	char *buf;

	MTest_Init( &argc, &argv );

	world_comm = MPI_COMM_WORLD;
	MPI_Comm_rank(world_comm, &world_rank);
	MPI_Comm_size(world_comm, &world_size);

#if 1
	MPI_Comm_get_parent(&parent_comm);

	if(parent_comm == MPI_COMM_NULL) {
		errcodes = malloc(world_size*sizeof(int));
		MPI_Comm_spawn((char*)"./spawn_ssend", MPI_ARGV_NULL, world_size, MPI_INFO_NULL, 0, world_comm, &spawn_comm, errcodes);
		free(errcodes);
	} else {
		spawn_comm = parent_comm;
	}

	MPI_Intercomm_merge(spawn_comm, parent_comm == MPI_COMM_NULL ? 1 : 0, &univ_comm);

#else
	univ_comm = world_comm;
#endif

	MPI_Comm_rank(univ_comm, &univ_rank);
	MPI_Comm_size(univ_comm, &univ_size);

	src  = 0;
	dest = univ_size - 1;

	for (cs=0; cs<4; cs++) {
		n = bufsizes[cs];
		buf = (char*)malloc(n);

		if (!buf) {
			fprintf(stderr, "Unable to allocate %d bytes\n", n);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		MPI_Barrier(univ_comm);

		if (univ_rank == src) {

			MPI_Issend(buf, n, MPI_CHAR, dest, 999, univ_comm, &req);
			MPI_Barrier(univ_comm);

			for(t=0; t<3; t++) {

				//		    sleep(1);
				MPI_Test(&req, &flag, &status);

				if(flag) {
					errs++;
					break;
				}
			}

			MPI_Send(&n, 1, MPI_INT, dest, 888, univ_comm);
			MPI_Wait(&req, &status);
		}
		else if (univ_rank == dest)
		{
			MPI_Barrier(univ_comm);
			MPI_Recv(&n, 1, MPI_INT, src, 888, univ_comm, &status);
			MPI_Recv(buf, n, MPI_CHAR, src, 999, univ_comm, &status);
		}
		else {
			MPI_Barrier(univ_comm);
		}

		MPI_Barrier(univ_comm);
		free(buf);
	}

epilogue:
	MPI_Reduce((univ_rank == 0 ? MPI_IN_PLACE : &errs), &errs, 1, MPI_INT, MPI_SUM, 0, univ_comm);
	if (univ_rank == 0) {
		if (errs) {
			printf("found %d errors\n", errs);
		}
	}

	if(spawn_comm != MPI_COMM_NULL) {

		MPI_Comm_free(&spawn_comm);
	}

	if(univ_comm != MPI_COMM_NULL && univ_comm != world_comm) {

		MPI_Comm_free(&univ_comm);
	}

	if (parent_comm == MPI_COMM_NULL) {
		MTest_Finalize(errs);
	} else {
		MPI_Finalize();
	}

	return MTestReturnValue(errs);

	return 0;
}
