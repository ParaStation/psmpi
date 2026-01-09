/*
 * ParaStation
 *
 * Copyright (C) 2019-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi-ext.h>
#include <sys/types.h>
#include <unistd.h>

#define NUM_PROCS 8

#ifndef _VERBOSE_
#define _VERBOSE_ 0
#endif

int my_module_id = -1;

int main(int argc, char *argv[])
{
    int errors = 0;
    int all_errors;
    int world_rank;
    int world_size;
    char value[MPI_MAX_INFO_VAL];
    int flag;
    int module_id;
    MPI_Comm split_comm;
    MPI_Comm split_type_comm;
    int split_comm_size;
    int split_type_comm_size;

    srand(getpid());
    my_module_id = rand() % (NUM_PROCS / 2);

    sprintf(value, "%d", my_module_id);
    setenv("PSP_MSA_MODULE_ID", value, 1);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != NUM_PROCS) {
        printf("WARNING: This test expects %d processes to be started!\n", NUM_PROCS);
    }

    /* Test MPI_INFO_ENV entry: */

    MPI_Info_get(MPI_INFO_ENV, "msa_module_id", MPI_MAX_INFO_VAL, value, &flag);

    if (flag) {
        module_id = atoi(value);
        if (_VERBOSE_) {
            printf("(%d) Found \"msa_module_id = %d\n", world_rank, module_id);
        }

        if (my_module_id != module_id) {
            printf("ERROR: my_module_id=%d vs. module_id=%d\n", my_module_id, module_id);
            errors++;
        }

    } else {
#if defined(MPIX_MSA_AWARENESS) && MPIX_MSA_AWARENESS
        module_id = 0;
        if (_VERBOSE_) {
            printf("(%d) Found no entry for \"msa_module_id\"\n", world_rank);
        }
#endif
    }

    MPI_Comm_split(MPI_COMM_WORLD, module_id, 0, &split_comm);

#if defined(MPIX_MSA_AWARENESS) && MPIX_MSA_AWARENESS
    /* Test MPIX_COMM_TYPE_MODULE: */
    if (_VERBOSE_) {
        printf("(%d) Calling MPI_Comm_split_type with type = MPIX_COMM_TYPE_MODULE...\n",
               world_rank);
    }
    MPI_Comm_split_type(MPI_COMM_WORLD, MPIX_COMM_TYPE_MODULE, 0, MPI_INFO_NULL, &split_type_comm);

    if (split_type_comm != MPI_COMM_NULL) {
        MPI_Comm_size(split_comm, &split_comm_size);
        MPI_Comm_size(split_type_comm, &split_type_comm_size);

        if (_VERBOSE_) {
            printf("(%d) Communicator sizes: %d / %d\n", world_rank, split_comm_size,
                   split_type_comm_size);
        }

        if (split_comm_size != split_type_comm_size) {
            printf("ERROR: split_comm_size = %d vs. split_type_comm_size = %d\n", split_comm_size,
                   split_type_comm_size);
            errors++;
        }

        if (1) {
            char *envval = getenv("PSP_MSA_AWARENESS");

            if ((envval == NULL) || (atoi(envval) == 0)) {
                if (split_type_comm_size != NUM_PROCS) {
                    printf
                        ("ERROR: Without MSA awareness, all procs should belong to the same (single) module! (%d != %d)\n",
                         split_type_comm_size, NUM_PROCS);
                    errors++;
                }
            }
        }

        MPI_Comm_free(&split_type_comm);

    } else {
        if (module_id != MPI_UNDEFINED) {
            printf("ERROR: Something went wrong in communicato creation!\n");
        }
    }
#else
    if (_VERBOSE_) {
        printf("(%d) No topology awareness detected... (MPIX_MSA_AWARENESS not set)\n", world_rank);
    }
    if (flag && (module_id != 0)) {
        printf("ERROR: Topology awareness not enabled but module_id = %d detected\n", module_id);
        errors++;
    }
#endif

    MPI_Comm_free(&split_comm);

    MPI_Reduce(&errors, &all_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (world_rank == 0 && all_errors == 0)
        printf(" No Errors\n");

    MPI_Finalize();
}
