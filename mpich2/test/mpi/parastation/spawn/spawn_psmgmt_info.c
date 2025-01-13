/*
 * ParaStation
 *
 * Copyright (C) 2024-2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpi.h"
#include "mpitest.h"
#include <stdio.h>
#include <stdlib.h>

#define ENV_VAR_NAME "MYAPP_TESTENV"
#define ENV_VAR_VALUE "hello"

/* This test provides examples for using spawn info keys specific for
 * ParaStation Management. The keys only have an effect if ParaStation
 * Management and PMIx (pspmix) are used with psmpi.
 */
int main(int argc, char *argv[])
{
    int errs = 0, err;
    int rank, size, rsize, i;
    int np = 2;
    int errcodes[2];
    MPI_Comm parentcomm, intercomm;
    MPI_Info spawninfo;
    int can_spawn;
    char myenv[1024];

    MTest_Init(&argc, &argv);

    errs += MTestSpawnPossible(&can_spawn);

    if (can_spawn) {
        MPI_Comm_get_parent(&parentcomm);

        if (parentcomm == MPI_COMM_NULL) {
            /* Set both keys mpiexecopts and srunopts here.
             * Only one of them will take effect depending on the launcher
             * in use (either mpiexec or srun). */

            char mpiexecopts[1024];
            char srunopts[1024];

            snprintf(mpiexecopts, 1024, "--env=%s %s", ENV_VAR_NAME, ENV_VAR_VALUE);
            snprintf(srunopts, 1024, "--export=ALL,%s=%s", ENV_VAR_NAME, ENV_VAR_VALUE);

            MPI_Info_create(&spawninfo);
            MPI_Info_set(spawninfo, (char *) "mpiexecopts", mpiexecopts);
            MPI_Info_set(spawninfo, (char *) "srunopts", srunopts);

            MPI_Comm_spawn((char *) "spawn_psmgmt_info", MPI_ARGV_NULL, np,
                           spawninfo, 0, MPI_COMM_WORLD, &intercomm, errcodes);
            MPI_Info_free(&spawninfo);
        } else {
            intercomm = parentcomm;
        }

        /* We now have a valid intercomm */

        MPI_Comm_remote_size(intercomm, &rsize);
        MPI_Comm_size(intercomm, &size);
        MPI_Comm_rank(intercomm, &rank);

        if (parentcomm == MPI_COMM_NULL) {
            /* Parent */
            if (rsize != np) {
                errs++;
                printf("Did not create %d processes (got %d)\n", np, rsize);
            }

            if (rank == 0) {
                for (i = 0; i < rsize; i++) {
                    MPI_Send(&i, 1, MPI_INT, i, 0, intercomm);
                }
                /* We could use intercomm reduce to get the errors from the
                 * children, but we'll use a simpler loop to make sure that
                 * we get valid data */
                for (i = 0; i < rsize; i++) {
                    MPI_Recv(&err, 1, MPI_INT, i, 1, intercomm, MPI_STATUS_IGNORE);
                    errs += err;
                }
                for (i = 0; i < rsize; i++) {
                    MPI_Recv(myenv, sizeof(myenv), MPI_CHAR, i, 2, intercomm, MPI_STATUS_IGNORE);
                    if ((myenv == NULL) || (strcmp(myenv, ENV_VAR_VALUE) != 0)) {
                        printf
                            ("Expected environment variable value %s of child rank %d but got %s\n",
                             ENV_VAR_VALUE, i, myenv);
                        errs++;
                    }
                }
            }
        } else {
            /* Child */
            char cname[MPI_MAX_OBJECT_NAME];
            int rlen;

            if (size != np) {
                errs++;
                printf("(Child) Did not create %d processes (got %d)\n", np, size);
            }

            /* Check if mpiexecopts/ srunopts info key had expected effect */
            char *val = getenv(ENV_VAR_NAME);
            if (val == NULL) {
                errs++;
                printf("Env value not found\n");
            } else if (strcmp(val, ENV_VAR_VALUE) != 0) {
                errs++;
                printf("Did not get correct env value. Expected %s, got %s.\n", ENV_VAR_VALUE, val);
            }

            MPI_Recv(&i, 1, MPI_INT, 0, 0, intercomm, MPI_STATUS_IGNORE);
            if (i != rank) {
                errs++;
                printf("Unexpected rank on child %d (%d)\n", rank, i);
            }
            /* Send our notion of environment value to the parent */
            if (val == NULL) {
                /* env value not found */
                MPI_Send("not found", strlen("not found") + 1, MPI_CHAR, 0, 2, intercomm);
            } else {
                MPI_Send(val, strlen(val) + 1, MPI_CHAR, 0, 2, intercomm);
            }

            /* Send the errs back to the parent process */
            MPI_Ssend(&errs, 1, MPI_INT, 0, 1, intercomm);
        }

        /* It isn't necessary to free the intercomm, but it should not hurt */
        MPI_Comm_free(&intercomm);

        /* Note that the MTest_Finalize get errs only over COMM_WORLD */
        /* Note also that both the parent and child will generate "No Errors"
         * if both call MTest_Finalize */
        if (parentcomm == MPI_COMM_NULL) {
            MTest_Finalize(errs);
        } else {
            MPI_Finalize();
        }
    } else {
        MTest_Finalize(errs);
    }

    return MTestReturnValue(errs);
}
