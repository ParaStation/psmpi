/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <omp.h>
#include <sched.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <getopt.h>
#include "zmtest_abslock.h"

#define TEST_NITER (1<<22)
#define WARMUP_ITER 128

#define CACHELINE_SZ 64

char *cache_lines;
unsigned array_len = 10;

zm_abslock_t lock;

#if defined (ZM_BIND_MANUAL)
void bind_compact(){
  int tid = omp_get_thread_num();
  /* Compute the target core */
  int tgt_core = tid;

  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(tgt_core, &set);

  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) < 0) {
      perror("pthread_setaffinity_np");
  }
}
#else
#define bind_compact()
#endif

static void test_thruput()
{
    memset(cache_lines, 0, CACHELINE_SZ * array_len);
    unsigned nthreads = omp_get_max_threads();

    zm_abslock_init(&lock);
    int cur_nthreads;
    /* Throughput = lock acquisitions per second */
    printf("nthreads,thruput,lat\n");
    for(cur_nthreads=1; cur_nthreads <= nthreads; cur_nthreads+= ((cur_nthreads==1) ? 1 : 2)) {
        double start_time, stop_time;
        #pragma omp parallel num_threads(cur_nthreads)
        {

            bind_compact();

            int tid = omp_get_thread_num();
            assert(tid >= 0);
            unsigned private_seed = (unsigned) tid;

            /* Warmup */
            for(int iter=0; iter < WARMUP_ITER; iter++) {
                zm_abslock_acquire(&lock);
                /* Computation */
                for(int i = 0; i < array_len; i++) {
                     unsigned read_index = rand_r(&private_seed) % array_len;
                     unsigned write_index = rand_r(&private_seed) % array_len;
                     cache_lines[write_index] += cache_lines[read_index] + 1;
                }
                zm_abslock_release(&lock);
            }
            #pragma omp barrier
            #pragma omp single
            {
                start_time = omp_get_wtime();
            }
            #pragma omp for schedule(static)
            for(int iter = 0; iter < TEST_NITER; iter++) {
                zm_abslock_acquire(&lock);
                /* Computation */
                for(int i = 0; i < array_len; i++) {
                     unsigned read_index = rand_r(&private_seed) % array_len;
                     unsigned write_index = rand_r(&private_seed) % array_len;
                     cache_lines[write_index] += cache_lines[read_index] + 1;
                }
                zm_abslock_release(&lock);
            }
        }
        stop_time = omp_get_wtime();
        double elapsed_time = stop_time - start_time;
        double thruput = (double)TEST_NITER/elapsed_time;
        double latency = elapsed_time*1e9/TEST_NITER; // latency in nanoseconds
        printf("%d,%.2lf,%.2lf\n", cur_nthreads, thruput, latency);
    }

}

void usage() {
   printf("usage: ./<binary name> [-s <size of shared array>]\n");
}

int main(int argc, char **argv)
{

    int c;

    while((c = getopt(argc, argv, "s:")) != -1) {
        switch (c) {
            case 's':
                array_len = atoi(optarg);
                break;
            default:
                usage();
        }
    }

    posix_memalign((void**)&cache_lines, 64, CACHELINE_SZ * array_len);

    test_thruput();
    return 0;
}

