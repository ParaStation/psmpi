/*
 * ParaStation
 *
 * Copyright (C) 2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MSG_SIZE 8
#define ITER     10000

/*
This test is aimed at evaluating the performance of any source queue.
In this test rank 0 will post one anysrc req first and N extra receives.
Other ranks will send messages to rank 0 and last send will match the anysrc req posted by rank 0.

N is the extra receive after anysource receive and default is set to 1000
User can set N with the following command (x>2)
mpirun -n x ./many_req_after_anysrc_req N
*/

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            fprintf(stderr, "Run with more than 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int num_extra_recvs = 1000;

    if (argc > 1)
        num_extra_recvs = atoi(argv[1]);

    char sendbuf[MSG_SIZE];
    char recvbuf[MSG_SIZE];

    MPI_Request *reqs = malloc((num_extra_recvs + 1) * (size - 1) * sizeof(MPI_Request));

    double total = 0.0;

    for (int iter = 0; iter < ITER; iter++) {

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            int req_counter = 0;

            for (int i = 0; i < size - 1; i++) {
                /* ANY_SOURCE recv */
                MPI_Irecv(recvbuf,
                          MSG_SIZE,
                          MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &reqs[req_counter++]);
            }
            for (int j = 0; j < size - 1; j++) {
                /* many exact-source recvs */
                for (int i = 0; i < num_extra_recvs; i++) {
                    MPI_Irecv(recvbuf,
                              MSG_SIZE,
                              MPI_BYTE,
                              j + 1, (j + 1) * i + 100, MPI_COMM_WORLD, &reqs[req_counter++]);
                }
            }

            double t0 = MPI_Wtime();

            MPI_Waitall((num_extra_recvs + 1) * (size - 1), &reqs[0], MPI_STATUSES_IGNORE);

            double t1 = MPI_Wtime();

            total += (t1 - t0);
        }

        else {
            for (int i = 0; i < num_extra_recvs; i++)
                MPI_Send(sendbuf, MSG_SIZE, MPI_BYTE, 0, rank * i + 100, MPI_COMM_WORLD);
            /* send message to match the anysource req posted at rank 0 */
            MPI_Send(sendbuf, MSG_SIZE, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        /* rank 0 prints the latency */
        printf("extra_recvs=%d\n", num_extra_recvs);
        printf("avg latency = %.3f us\n", total * 1e6 / ITER);
    }

    free(reqs);

    MPI_Finalize();
    return 0;
}
