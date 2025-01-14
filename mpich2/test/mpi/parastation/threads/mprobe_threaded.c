/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  Portions of this code were written/modified by ParTec AG
 *
 *  Copyright (C) 2016-2021 ParTec Cluster Competence Center GmbH, Munich
 *  Copyright (C) 2021-2025 ParTec AG, Munich
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "mpi.h"
#include "mpitest.h"

#define NUM_ITER 10000
#define NUM_THREADS 4

/* assert-like macro that bumps the err count and emits a message */
#define check(x_)							\
	do {								\
		if (!(x_)) {						\
			++errs;						\
			if (errs < 10) {				\
				fprintf(stderr, "check failed: (%s), line %d\n", #x_, __LINE__); \
			}						\
		}							\
	} while (0)


void *thread_func(void *args)
{
    int errs = 0;
    int *_errs;
    int i, count, tag;
    MPI_Message msg;
    MPI_Status status;
    int recvbuf;
    int flag;

    for (i = 0; i < NUM_ITER; i++) {

        if (rand() % 2) {
            MPI_Mprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &msg, &status);
        } else {
            do {
                MPI_Improbe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &msg, &status);
            } while (!flag);
        }

        check(status.MPI_SOURCE == 0);
        check(msg != MPI_MESSAGE_NULL);

        tag = status.MPI_TAG;
        check(tag >= 19 || tag <= 42);

        count = -1;
        MPI_Get_count(&status, MPI_INT, &count);
        check(count == 1);

        MPI_Mrecv(&recvbuf, 1, MPI_INT, &msg, &status);
        check(tag == status.MPI_TAG);

        if (!(tag % 2)) {
            check(recvbuf == 0xdeadbeef);
        } else {
            check(recvbuf == 0xfeedface);
        }

        count = -1;
        MPI_Get_count(&status, MPI_INT, &count);
        check(count == 1);
    }

    _errs = malloc(sizeof(int));
    *_errs = errs;
    return (void *) _errs;
}

int main(int argc, char **argv)
{
    int errs = 0;
    int rank, size;
    int provided;
    int i, tag;
    int sendbuf[2];
    void *retval;
    pthread_t handles[NUM_THREADS];

    MTest_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        printf("this test requires at least 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* all processes besides ranks 0 & 1 aren't used by this test */
    if (rank >= 2) {
        goto epilogue;
    }

    if (provided == MPI_THREAD_MULTIPLE) {
        if (rank == 0) {
            sendbuf[0] = 0xdeadbeef;
            sendbuf[1] = 0xfeedface;

            for (i = 0; i < NUM_ITER * NUM_THREADS; i++) {
                tag = 19 + rand() % (42 - 19);
                MPI_Send(&sendbuf[tag % 2], 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
            }
        } else {
            for (i = 0; i < NUM_THREADS; ++i) {
                pthread_create(&handles[i], NULL, thread_func, NULL);
            }
            for (i = 0; i < NUM_THREADS; ++i) {
                pthread_join(handles[i], &retval);
                if (retval != PTHREAD_CANCELED) {
                    errs += *(int *) retval;
                    free(retval);
                }
            }
        }
    }

  epilogue:
    MTest_Finalize(errs);

    return MTestReturnValue(errs);
}
