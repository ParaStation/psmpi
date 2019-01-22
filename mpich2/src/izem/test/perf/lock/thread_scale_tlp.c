/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <assert.h>
#include "zmtest_abslock.h"

#define TEST_NITER 100000

char cache_lines[640] = {0};
int indices [] = {3,6,1,7,0,2,9,4,8,5};

static void test_thruput()
{
    unsigned nthreads = omp_get_max_threads();

    zm_abslock_t lock;
    zm_abslock_init(&lock);
    int cur_nthreads = 0;
    printf("#Thread \t HP:Thruput[acqs/s] \t LP Thruput[acqs/s]\n");
    for(cur_nthreads=1; cur_nthreads <= nthreads; cur_nthreads++) {
        double start_times[cur_nthreads];
        double stop_times[cur_nthreads];
    #pragma omp parallel num_threads(cur_nthreads)
    {
        int tid = omp_get_thread_num();
        int iter;
        start_times[tid] = omp_get_wtime();
        for(iter=0; iter<TEST_NITER; iter++){
            int err;
            if(tid % 2 == 0)
                zm_abslock_acquire(&lock);
            else
                zm_abslock_acquire_l(&lock);

            /* Computation */

            for(int i = 0; i < 10; i++)
                 cache_lines[indices[i]] += cache_lines[indices[9-i]];

            zm_abslock_release(&lock);
        }
        stop_times[tid] = omp_get_wtime();
    } /* End of omp parallel*/
        double htimes = 0.0, ltimes = 0.0;
        int i;
        for(i=0; i < cur_nthreads; i++) {
            if (i % 2 == 0) htimes += (stop_times[i] - start_times[i]);
            else            ltimes += (stop_times[i] - start_times[i]);
        }
        int hthreads = (cur_nthreads % 2 == 0) ? cur_nthreads / 2 : (cur_nthreads + 1) / 2;
        int lthreads = (cur_nthreads % 2 == 0) ? hthreads : hthreads - 1;
        assert(hthreads + lthreads == cur_nthreads);
        if(lthreads > 0)
            printf("%d \t %lf \t %lf\n", cur_nthreads, ((double)TEST_NITER*hthreads)/htimes, ((double)TEST_NITER*lthreads)/ltimes);
        else
            printf("%d \t %f \t %f\n", cur_nthreads, ((double)TEST_NITER*hthreads)/htimes, -1.0);
    }

} /* end test_locked_counter() */

int main(int argc, char **argv)
{
  test_thruput();
  return 0;
} /* end main() */

