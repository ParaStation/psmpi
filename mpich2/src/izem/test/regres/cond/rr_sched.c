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
#include <cond/zm_cond.h>

/* This benchmark schedules threads in a round-robin (rr)
   fashion. Each thread has an id and increments the counter when its id equals
the counter value. If not, the thread waits on its condition variable. Thread i
waits on cond_var[i] and thead [(i - 1) % NTHREADS] wakes it up. */

#define TEST_NTHREADS 4
#define TEST_NITER 1000

zm_cond_t cond_vars[TEST_NTHREADS];
int counter = 0;
zm_lock_t glock;

static void* run(void *arg) {
    int tid = (intptr_t) arg;
    int iter;
    for(iter=0; iter<TEST_NITER; iter++) {
        zm_lock_acquire(&glock);
        do {
            if (counter == tid) {
                counter = (counter + 1) % TEST_NTHREADS;
                zm_cond_signal(&cond_vars[(tid + 1) % TEST_NTHREADS]);
                break;
            } else {
                zm_cond_wait(&cond_vars[tid], &glock);
            }
        } while (1);
        zm_lock_release(&glock);   /* Release the lock */
    }
    return 0;
}

static void test_rr_sched() {
    void *res;
    pthread_t threads[TEST_NTHREADS];

    for (int i = 0; i < TEST_NTHREADS; i++)
        zm_cond_init(&cond_vars[i]);
    zm_lock_init(&glock);

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
        zm_cond_destroy(&cond_vars[i]);

    printf("Pass\n");

} /* end test_lock_thruput() */

int main(int argc, char **argv)
{
  test_rr_sched();
} /* end main() */

