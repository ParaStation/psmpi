/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <lock/zm_lock.h>
#include <cond/zm_scount.h>

/* This benchmark schedules threads in a round-robin (rr)
   fashion. Each thread has an id and increments the counter when its id equals
the counter value. If not, the thread waits on its condition variable. Thread i
waits on cond_var[i] and thead [(i - 1) % NTHREADS] wakes it up. */

#define TEST_NTHREADS 4
#define TEST_NITER 10

struct zm_scount scounts[TEST_NTHREADS];
int counter = 0;
zm_lock_t glock;
pthread_barrier_t barrier;

static void* run(void *arg) {
    int tid = (intptr_t) arg;
    int iter;
    for(iter=0; iter<TEST_NITER; iter++) {
        pthread_barrier_wait(&barrier);
        if(tid == 0)
           counter = 0;
        zm_scount_init(&scounts[tid], tid);
        pthread_barrier_wait(&barrier);

        zm_lock_acquire(&glock);
        zm_scount_wait(&scounts[tid], &glock);
        if(counter != tid) {
            printf("Error: expected counter=tid=%d but counter=%d and tid=%d\n", counter, counter, tid);
            abort();
        }
        counter++;
        for(int i = 1; i<TEST_NTHREADS; i++) {
            int trg = (tid + i) % TEST_NTHREADS;
            int out_count; /* ignored */
            zm_scount_signal(&scounts[trg], &out_count);
        }

        zm_lock_release(&glock);   /* Release the lock */
    }
    return 0;
}

static void test_prio_chain() {
    void *res;
    pthread_t threads[TEST_NTHREADS];
    zm_lock_init(&glock);
    pthread_barrier_init(&barrier, NULL /*attr*/, TEST_NTHREADS);

    int th;
    for (th=0; th<TEST_NTHREADS; th++) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(th, &cpuset);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
        pthread_create(&threads[th], &attr, run, (void*)(intptr_t)th);
    }
    for (th=0; th<TEST_NTHREADS; th++)
        pthread_join(threads[th], &res);

    zm_lock_destroy(&glock);
    for (int i = 0; i < TEST_NTHREADS; i++)
        zm_scount_destroy(&scounts[i]);

    printf("Pass\n");

} /* end test_lock_thruput() */

int main(int argc, char **argv)
{
  test_prio_chain();
} /* end main() */

