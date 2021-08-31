/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "queue/zm_mpbqueue.h"
#define TEST_NELEMTS  64
#define NITER (1024*32)

/*-------------------------------------------------------------------------
 * Function: run
 *
 * Purpose: Test the correctness of queue operations by counting the number
 *  of dequeued elements to the expected number
 *
 * Return: Success: 0
 *         Failure: 1
 *-------------------------------------------------------------------------
 */
static inline void run() {
    unsigned test_counter = 0;
    struct zm_mpbqueue queue;
    double t1, t2;
    int maxthreads = omp_get_max_threads();

    printf("nbuckets,nthreads,throughput\n");

    int nthreads, nbuckets;
    for (nbuckets = 1; nbuckets <= maxthreads; nbuckets *= 2) {
        for (nthreads = 2; nthreads <= maxthreads; nthreads ++) {
            zm_mpbqueue_init(&queue, nbuckets);
            int nelem_enq, nelem_deq;
            nelem_enq = TEST_NELEMTS/(nthreads-1);
            nelem_deq = (nthreads-1)*nelem_enq;

            t1 = omp_get_wtime();

            #pragma omp parallel num_threads(nthreads)
            {
                int tid, producer_b, trg_bucket;
                size_t input = 1;
                tid = omp_get_thread_num();
                producer_b = (tid != 0);
                trg_bucket = tid % nbuckets;
                int elem;

                for(int i = 0; i<NITER; i++) {
                    if(producer_b) { /* producer */
                        for(elem=0; elem < nelem_enq; elem++) {
                            zm_mpbqueue_enqueue(&queue, (void*) input, trg_bucket);
                        }
                    } else {           /* consumer */
                        while(test_counter < nelem_deq) {
                            void* elem = NULL;
                            zm_mpbqueue_dequeue(&queue, (void**)&elem);
                            if ((elem != NULL) && ((size_t)elem == 1)) {
                                #pragma omp atomic
                                    test_counter++;
                            }
                        }
                    }
                }
            }

            t2 = omp_get_wtime();
            printf("%d,%d,%.3f\n", nbuckets, nthreads, (double)nelem_deq*NITER/(t2-t1));
        }
    }

} /* end run() */

int main(int argc, char **argv) {
  run();
} /* end main() */

