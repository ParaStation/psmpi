/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2012 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 *  Portions of this code were written/modified by ParTec AG
 *  Copyright (C) 2016-2021 ParTec Cluster Competence Center GmbH, Munich
 *  Copyright (C) 2021      ParTec AG, Munich
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "mpitest.h"

/* This is a temporary #ifdef to control whether we test this functionality.  A
 * configure-test or similar would be better.  Eventually the MPI-3 standard
 * will be released and this can be gated on a MPI_VERSION check */
#if !defined(USE_STRICT_MPI) && defined(MPICH)
#define TEST_MPROBE_ROUTINES 1
#define TEST_MPROBE_WITH_PTHREADS 1
#endif

/* assert-like macro that bumps the err count and emits a message */
#define check(x_)                                                                 \
    do {                                                                          \
        if (!(x_)) {                                                              \
            ++errs;                                                               \
            if (errs < 10) {                                                      \
                fprintf(stderr, "check failed: (%s), line %d\n", #x_, __LINE__); \
            }                                                                     \
        }                                                                         \
    } while (0)


#if defined(TEST_MPROBE_WITH_PTHREADS) && defined(TEST_MPROBE_ROUTINES)
#include <pthread.h>
#define NUM_ITER 100
#define NUM_THREADS 4
void* thread_func(void* args)
{
    int errs = 0;
    int i, count;
    MPI_Message msg;
    MPI_Status status;
    int recvbuf;

    for(i=0; i<NUM_ITER; i++) {

        MPI_Mprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &msg, &status);

        check(status.MPI_SOURCE == 0);
        check(status.MPI_TAG == 5 || status.MPI_TAG == 6);
        check(msg != MPI_MESSAGE_NULL);

        count = -1;
        MPI_Get_count(&status, MPI_INT, &count);
        check(count == 1);

        MPI_Mrecv(&recvbuf, 1, MPI_INT, &msg, &status);
        check(status.MPI_TAG == 5 || status.MPI_TAG == 6);
        if(status.MPI_TAG == 5)
            check(recvbuf == 0xdeadbeef);
        else
            check(recvbuf == 0xfeedface);

        count = -1;
        MPI_Get_count(&status, MPI_INT, &count);
        check(count == 1);
    }
}
#endif

int main(int argc, char **argv)
{
    int errs = 0;
    int found, completed;
    int rank, size;
    int sendbuf[8], recvbuf[8];
    int count, count1, count2;
#ifdef TEST_MPROBE_ROUTINES
    MPI_Message msg, msg1, msg2;
#endif
    MPI_Request rreq;
    MPI_Request sreq1, sreq2;
    MPI_Status s, s1, s2;
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

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

#ifdef TEST_MPROBE_ROUTINES
    /* test 0: simple send & mprobe+mrecv */
    if (rank == 0) {
        sendbuf[0] = 0xdeadbeef;
        sendbuf[1] = 0xfeedface;
        MPI_Send(sendbuf, 2, MPI_INT, 1, 5, MPI_COMM_WORLD);
        MPI_Send(sendbuf, 1, MPI_INT, 1, 6, MPI_COMM_WORLD);
        MPI_Send(sendbuf, 1, MPI_INT, 1, 5, MPI_COMM_WORLD);
    }
    else {
        memset(&s1, 0xab, sizeof(MPI_Status));
        memset(&s2, 0xab, sizeof(MPI_Status));
        /* the error field should remain unmodified */
        s1.MPI_ERROR = MPI_ERR_DIMS;
        s2.MPI_ERROR = MPI_ERR_TOPOLOGY;

        msg1 = MPI_MESSAGE_NULL;
        MPI_Mprobe(0, 5, MPI_COMM_WORLD, &msg1, &s1);
        check(s1.MPI_SOURCE == 0);
        check(s1.MPI_TAG == 5);
        check(s1.MPI_ERROR == MPI_ERR_DIMS);
        check(msg1 != MPI_MESSAGE_NULL);

        count1 = -1;
        MPI_Get_count(&s1, MPI_INT, &count1);
        check(count1 == 2);

        msg2 = MPI_MESSAGE_NULL;
        MPI_Mprobe(0, 6, MPI_COMM_WORLD, &msg2, &s2);
        check(s2.MPI_SOURCE == 0);
        check(s2.MPI_TAG == 6);
        check(s2.MPI_ERROR == MPI_ERR_TOPOLOGY);
        check(msg2 != MPI_MESSAGE_NULL);

        count2 = -1;
        MPI_Get_count(&s2, MPI_INT, &count2);
        check(count2 == 1);

        msg = MPI_MESSAGE_NULL;
        MPI_Mprobe(0, 5, MPI_COMM_WORLD, &msg, &s);
        check(s.MPI_SOURCE == 0);
        check(s.MPI_TAG == 5);
        check(msg != MPI_MESSAGE_NULL);

        count = -1;
        MPI_Get_count(&s, MPI_INT, &count);
        check(count == 1);

        recvbuf[0] = 0x01234567;
        recvbuf[1] = 0x89abcdef;

        MPI_Mrecv(recvbuf, count2, MPI_INT, &msg2, &s2);
        check(recvbuf[0] == 0xdeadbeef);
        check(s2.MPI_SOURCE == 0);
        check(s2.MPI_TAG == 6);
        check(s2.MPI_ERROR == MPI_ERR_TOPOLOGY);
        check(msg2 == MPI_MESSAGE_NULL);

        recvbuf[0] = 0x01234567;
        recvbuf[1] = 0x89abcdef;

        MPI_Mrecv(recvbuf, count1, MPI_INT, &msg1, &s1);
        check(recvbuf[0] == 0xdeadbeef);
        check(recvbuf[1] == 0xfeedface);
        check(s1.MPI_SOURCE == 0);
        check(s1.MPI_TAG == 5);
        check(s1.MPI_ERROR == MPI_ERR_DIMS);
        check(msg1 == MPI_MESSAGE_NULL);

        MPI_Mrecv(recvbuf, count, MPI_INT, &msg, &s);
        check(recvbuf[0] == 0xdeadbeef);
        check(s.MPI_SOURCE == 0);
        check(s.MPI_TAG == 5);
        check(msg == MPI_MESSAGE_NULL);

        count = -1;
        MPI_Get_count(&s, MPI_INT, &count);
        check(count == 1);

    }

    /* test 1: simple send & mprobe+imrecv */
    if (rank == 0) {
        sendbuf[0] = 0xdeadbeef;
        sendbuf[1] = 0xfeedface;
        MPI_Send(sendbuf, 2, MPI_INT, 1, 5, MPI_COMM_WORLD);
    }
    else {
        memset(&s1, 0xab, sizeof(MPI_Status));
        memset(&s2, 0xab, sizeof(MPI_Status));
        /* the error field should remain unmodified */
        s1.MPI_ERROR = MPI_ERR_DIMS;
        s2.MPI_ERROR = MPI_ERR_TOPOLOGY;

        msg = MPI_MESSAGE_NULL;
        MPI_Mprobe(0, 5, MPI_COMM_WORLD, &msg, &s1);
        check(s1.MPI_SOURCE == 0);
        check(s1.MPI_TAG == 5);
        check(s1.MPI_ERROR == MPI_ERR_DIMS);
        check(msg != MPI_MESSAGE_NULL);

        count = -1;
        MPI_Get_count(&s1, MPI_INT, &count);
        check(count == 2);

        rreq = MPI_REQUEST_NULL;
        recvbuf[0] = 0x01234567;
        recvbuf[1] = 0x89abcdef;
        MPI_Imrecv(recvbuf, count, MPI_INT, &msg, &rreq);
        check(rreq != MPI_REQUEST_NULL);
        MPI_Wait(&rreq, &s2);
        check(recvbuf[0] == 0xdeadbeef);
        check(recvbuf[1] == 0xfeedface);
        check(s2.MPI_SOURCE == 0);
        check(s2.MPI_TAG == 5);
        check(s2.MPI_ERROR == MPI_ERR_TOPOLOGY);
        check(msg == MPI_MESSAGE_NULL);
    }

    /* test 2: simple send & improbe+mrecv */
    if (rank == 0) {
        sendbuf[0] = 0xdeadbeef;
        sendbuf[1] = 0xfeedface;
        MPI_Send(sendbuf, 2, MPI_INT, 1, 5, MPI_COMM_WORLD);
    }
    else {
        memset(&s1, 0xab, sizeof(MPI_Status));
        memset(&s2, 0xab, sizeof(MPI_Status));
        /* the error field should remain unmodified */
        s1.MPI_ERROR = MPI_ERR_DIMS;
        s2.MPI_ERROR = MPI_ERR_TOPOLOGY;

        msg = MPI_MESSAGE_NULL;
        do {
            check(msg == MPI_MESSAGE_NULL);
            MPI_Improbe(0, 5, MPI_COMM_WORLD, &found, &msg, &s1);
        } while (!found);
        check(msg != MPI_MESSAGE_NULL);
        check(s1.MPI_SOURCE == 0);
        check(s1.MPI_TAG == 5);
        check(s1.MPI_ERROR == MPI_ERR_DIMS);

        count = -1;
        MPI_Get_count(&s1, MPI_INT, &count);
        check(count == 2);

        recvbuf[0] = 0x01234567;
        recvbuf[1] = 0x89abcdef;
        MPI_Mrecv(recvbuf, count, MPI_INT, &msg, &s2);
        check(recvbuf[0] == 0xdeadbeef);
        check(recvbuf[1] == 0xfeedface);
        check(s2.MPI_SOURCE == 0);
        check(s2.MPI_TAG == 5);
        check(s2.MPI_ERROR == MPI_ERR_TOPOLOGY);
        check(msg == MPI_MESSAGE_NULL);
    }

    /* test 3: simple send & improbe+imrecv */
    if (rank == 0) {
        sendbuf[0] = 0xdeadbeef;
        sendbuf[1] = 0xfeedface;
        MPI_Send(sendbuf, 2, MPI_INT, 1, 5, MPI_COMM_WORLD);
    }
    else {
        memset(&s1, 0xab, sizeof(MPI_Status));
        memset(&s2, 0xab, sizeof(MPI_Status));
        /* the error field should remain unmodified */
        s1.MPI_ERROR = MPI_ERR_DIMS;
        s2.MPI_ERROR = MPI_ERR_TOPOLOGY;

        msg = MPI_MESSAGE_NULL;
        do {
            check(msg == MPI_MESSAGE_NULL);
            MPI_Improbe(0, 5, MPI_COMM_WORLD, &found, &msg, &s1);
        } while (!found);
        check(msg != MPI_MESSAGE_NULL);
        check(s1.MPI_SOURCE == 0);
        check(s1.MPI_TAG == 5);
        check(s1.MPI_ERROR == MPI_ERR_DIMS);

        count = -1;
        MPI_Get_count(&s1, MPI_INT, &count);
        check(count == 2);

        rreq = MPI_REQUEST_NULL;
        MPI_Imrecv(recvbuf, count, MPI_INT, &msg, &rreq);
        check(rreq != MPI_REQUEST_NULL);
        MPI_Wait(&rreq, &s2);
        check(recvbuf[0] == 0xdeadbeef);
        check(recvbuf[1] == 0xfeedface);
        check(s2.MPI_SOURCE == 0);
        check(s2.MPI_TAG == 5);
        check(s2.MPI_ERROR == MPI_ERR_TOPOLOGY);
        check(msg == MPI_MESSAGE_NULL);
    }

    /* test 4: mprobe+mrecv with MPI_PROC_NULL */
    {
        memset(&s1, 0xab, sizeof(MPI_Status));
        memset(&s2, 0xab, sizeof(MPI_Status));
        /* the error field should remain unmodified */
        s1.MPI_ERROR = MPI_ERR_DIMS;
        s2.MPI_ERROR = MPI_ERR_TOPOLOGY;

        msg = MPI_MESSAGE_NULL;
        MPI_Mprobe(MPI_PROC_NULL, 5, MPI_COMM_WORLD, &msg, &s1);
        check(s1.MPI_SOURCE == MPI_PROC_NULL);
        check(s1.MPI_TAG == MPI_ANY_TAG);
        check(s1.MPI_ERROR == MPI_ERR_DIMS);
        check(msg == MPI_MESSAGE_NO_PROC);

        count = -1;
        MPI_Get_count(&s1, MPI_INT, &count);
        check(count == 0);

        recvbuf[0] = 0x01234567;
        recvbuf[1] = 0x89abcdef;
        MPI_Mrecv(recvbuf, count, MPI_INT, &msg, &s2);
        /* recvbuf should remain unmodified */
        check(recvbuf[0] == 0x01234567);
        check(recvbuf[1] == 0x89abcdef);
        /* should get back "proc null status" */
        check(s2.MPI_SOURCE == MPI_PROC_NULL);
        check(s2.MPI_TAG == MPI_ANY_TAG);
        check(s2.MPI_ERROR == MPI_ERR_TOPOLOGY);
        check(msg == MPI_MESSAGE_NULL);
        count = -1;
        MPI_Get_count(&s2, MPI_INT, &count);
        check(count == 0);
    }

    /* test 5: mprobe+imrecv with MPI_PROC_NULL */
    {
        memset(&s1, 0xab, sizeof(MPI_Status));
        memset(&s2, 0xab, sizeof(MPI_Status));
        /* the error field should remain unmodified */
        s1.MPI_ERROR = MPI_ERR_DIMS;
        s2.MPI_ERROR = MPI_ERR_TOPOLOGY;

        msg = MPI_MESSAGE_NULL;
        MPI_Mprobe(MPI_PROC_NULL, 5, MPI_COMM_WORLD, &msg, &s1);
        check(s1.MPI_SOURCE == MPI_PROC_NULL);
        check(s1.MPI_TAG == MPI_ANY_TAG);
        check(s1.MPI_ERROR == MPI_ERR_DIMS);
        check(msg == MPI_MESSAGE_NO_PROC);
        count = -1;
        MPI_Get_count(&s1, MPI_INT, &count);
        check(count == 0);

        rreq = MPI_REQUEST_NULL;
        recvbuf[0] = 0x01234567;
        recvbuf[1] = 0x89abcdef;
        MPI_Imrecv(recvbuf, count, MPI_INT, &msg, &rreq);
        check(rreq != MPI_REQUEST_NULL);
        completed = 0;
        MPI_Test(&rreq, &completed, &s2); /* single test should always succeed */
        check(completed);
        /* recvbuf should remain unmodified */
        check(recvbuf[0] == 0x01234567);
        check(recvbuf[1] == 0x89abcdef);
        /* should get back "proc null status" */
        check(s2.MPI_SOURCE == MPI_PROC_NULL);
        check(s2.MPI_TAG == MPI_ANY_TAG);
        check(s2.MPI_ERROR == MPI_ERR_TOPOLOGY);
        check(msg == MPI_MESSAGE_NULL);
        count = -1;
        MPI_Get_count(&s2, MPI_INT, &count);
        check(count == 0);
    }

    /* test 6: improbe+mrecv with MPI_PROC_NULL */
    {
        memset(&s1, 0xab, sizeof(MPI_Status));
        memset(&s2, 0xab, sizeof(MPI_Status));
        /* the error field should remain unmodified */
        s1.MPI_ERROR = MPI_ERR_DIMS;
        s2.MPI_ERROR = MPI_ERR_TOPOLOGY;

        msg = MPI_MESSAGE_NULL;
        found = 0;
        MPI_Improbe(MPI_PROC_NULL, 5, MPI_COMM_WORLD, &found, &msg, &s1);
        check(found);
        check(msg == MPI_MESSAGE_NO_PROC);
        check(s1.MPI_SOURCE == MPI_PROC_NULL);
        check(s1.MPI_TAG == MPI_ANY_TAG);
        check(s1.MPI_ERROR == MPI_ERR_DIMS);
        count = -1;
        MPI_Get_count(&s1, MPI_INT, &count);
        check(count == 0);

        recvbuf[0] = 0x01234567;
        recvbuf[1] = 0x89abcdef;
        MPI_Mrecv(recvbuf, count, MPI_INT, &msg, &s2);
        /* recvbuf should remain unmodified */
        check(recvbuf[0] == 0x01234567);
        check(recvbuf[1] == 0x89abcdef);
        /* should get back "proc null status" */
        check(s2.MPI_SOURCE == MPI_PROC_NULL);
        check(s2.MPI_TAG == MPI_ANY_TAG);
        check(s2.MPI_ERROR == MPI_ERR_TOPOLOGY);
        check(msg == MPI_MESSAGE_NULL);
        count = -1;
        MPI_Get_count(&s2, MPI_INT, &count);
        check(count == 0);
    }

    /* test 7: improbe+imrecv with MPI_PROC_NULL */
    {
        memset(&s1, 0xab, sizeof(MPI_Status));
        memset(&s2, 0xab, sizeof(MPI_Status));
        /* the error field should remain unmodified */
        s1.MPI_ERROR = MPI_ERR_DIMS;
        s2.MPI_ERROR = MPI_ERR_TOPOLOGY;

        msg = MPI_MESSAGE_NULL;
        MPI_Improbe(MPI_PROC_NULL, 5, MPI_COMM_WORLD, &found, &msg, &s1);
        check(found);
        check(msg == MPI_MESSAGE_NO_PROC);
        check(s1.MPI_SOURCE == MPI_PROC_NULL);
        check(s1.MPI_TAG == MPI_ANY_TAG);
        check(s1.MPI_ERROR == MPI_ERR_DIMS);
        count = -1;
        MPI_Get_count(&s1, MPI_INT, &count);
        check(count == 0);

        rreq = MPI_REQUEST_NULL;
        MPI_Imrecv(recvbuf, count, MPI_INT, &msg, &rreq);
        check(rreq != MPI_REQUEST_NULL);
        completed = 0;
        MPI_Test(&rreq, &completed, &s2); /* single test should always succeed */
        check(completed);
        /* recvbuf should remain unmodified */
        check(recvbuf[0] == 0x01234567);
        check(recvbuf[1] == 0x89abcdef);
        /* should get back "proc null status" */
        check(s2.MPI_SOURCE == MPI_PROC_NULL);
        check(s2.MPI_TAG == MPI_ANY_TAG);
        check(s2.MPI_ERROR == MPI_ERR_TOPOLOGY);
        check(msg == MPI_MESSAGE_NULL);
        count = -1;
        MPI_Get_count(&s2, MPI_INT, &count);
        check(count == 0);
    }

    /* test 8: MPI_COMM_SELF isend & mprobe+mrecv */
    {
        sendbuf[0] = 0xdeadbeef;
        sendbuf[1] = 0xfeedface;
        MPI_Isend(&sendbuf[0], 1, MPI_INT, 0, 5, MPI_COMM_SELF, &sreq1);
        MPI_Isend(&sendbuf[1], 1, MPI_INT, 0, 6, MPI_COMM_SELF, &sreq2);

        msg1 = MPI_MESSAGE_NULL;
        MPI_Mprobe(0, 5, MPI_COMM_SELF, &msg1, &s1);
        check(s1.MPI_SOURCE == 0);
        check(s1.MPI_TAG == 5);
        check(msg1 != MPI_MESSAGE_NULL);

        count1 = -1;
        MPI_Get_count(&s1, MPI_INT, &count1);
        check(count1 == 1);

        msg2 = MPI_MESSAGE_NULL;
        MPI_Mprobe(0, 6, MPI_COMM_SELF, &msg2, &s2);
        check(s2.MPI_SOURCE == 0);
        check(s2.MPI_TAG == 6);
        check(msg2 != MPI_MESSAGE_NULL);

        count2 = -1;
        MPI_Get_count(&s2, MPI_INT, &count2);
        check(count2 == 1);

        MPI_Mrecv(&recvbuf[0], 1, MPI_INT, &msg2, &s2);
        check(recvbuf[0] == 0xfeedface);

        MPI_Mrecv(&recvbuf[1], 1, MPI_INT, &msg1, &s1);
        check(recvbuf[1] == 0xdeadbeef);

        MPI_Wait(&sreq1, &s1);
        MPI_Wait(&sreq2, &s2);
    }

    /* TODO a full range of message sizes should be tested too */

    /* simple test to ensure that c2f/f2c routines are present (initially missed
     * in MPICH impl) */
    {
        MPI_Fint f_handle = 0xdeadbeef;
        f_handle = MPI_Message_c2f(MPI_MESSAGE_NULL);
        msg = MPI_Message_f2c(f_handle);
        check(f_handle != 0xdeadbeef);
        check(msg == MPI_MESSAGE_NULL);

        /* PMPI_ versions should also exists */
        f_handle = 0xdeadbeef;
        f_handle = PMPI_Message_c2f(MPI_MESSAGE_NULL);
        msg = PMPI_Message_f2c(f_handle);
        check(f_handle != 0xdeadbeef);
        check(msg == MPI_MESSAGE_NULL);
    }

#ifdef TEST_MPROBE_WITH_PTHREADS
    if(provided == MPI_THREAD_MULTIPLE)
    {
        int i;
        if (rank == 0) {
            sendbuf[0] = 0xdeadbeef;
            sendbuf[1] = 0xfeedface;
            for(i=0; i<NUM_ITER*NUM_THREADS; i++) {
                MPI_Send(&sendbuf[0], 1, MPI_INT, 1, 5, MPI_COMM_WORLD);
                MPI_Send(&sendbuf[1], 1, MPI_INT, 1, 6, MPI_COMM_WORLD);
            }
        } else {
            pthread_t handles[NUM_THREADS];
            for (i = 0; i<NUM_THREADS; ++i)
		pthread_create(&handles[i], NULL, thread_func, NULL);
            for (i = 0; i<NUM_THREADS; ++i)
		pthread_join(handles[i], NULL);
        }
    }
#endif

#endif /* TEST_MPROBE_ROUTINES */

epilogue:
    MPI_Reduce((rank == 0 ? MPI_IN_PLACE : &errs), &errs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        if (errs) {
            printf("found %d errors\n", errs);
        }
        else {
            printf(" No errors\n");
        }
    }

    MPI_Finalize();

    return 0;
}

