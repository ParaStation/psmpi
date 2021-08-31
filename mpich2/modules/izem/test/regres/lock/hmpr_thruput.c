/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <lock/zm_hmpr.h>

#define TEST_NTHREADS 2
#define TEST_NITER 1000

struct zm_hmpr lock;

char cache_lines[640] = {0};
int indices [] = {3,6,1,7,0,2,9,4,8,5};

struct zm_hmpr_pnode pnodes[TEST_NTHREADS];
int counter = 0;

static void* run(void *arg) {
     int tid = (intptr_t) arg;
     int iter;
     for(iter=0; iter<TEST_NITER; iter++) {
         zm_hmpr_acquire(&lock, &pnodes[tid]);
         counter++;
         int trg = (tid + 1) % TEST_NTHREADS;
         zm_hmpr_raise_prio(&pnodes[trg]);
         zm_hmpr_release(&lock, &pnodes[tid]);
     }
     return 0;
}

/*-------------------------------------------------------------------------
 * Function: test_lock_throughput
 *
 * Purpose: Test the lock thruput for an empty critical section
 *
 * Return: Success: 0
 *         Failure: 1
 *-------------------------------------------------------------------------
 */
static void test_lock_thruput() {
    void *res;
    pthread_t threads[TEST_NTHREADS];

    for (int i = 0; i < TEST_NTHREADS; i++) {
        pnodes[i].p = i;
        pnodes[i].qnode = NULL;
    }

    zm_hmpr_init(&lock);

    counter = 0;

    int th;
    for (th=0; th<TEST_NTHREADS; th++) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(th, &cpuset);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
        pthread_create(&threads[th], &attr, run, (void*)(intptr_t)th);
        pthread_attr_destroy(&attr);
    }
    for (th=0; th<TEST_NTHREADS; th++)
        pthread_join(threads[th], &res);

    zm_hmpr_destroy(&lock);
    if (counter == TEST_NTHREADS * TEST_NITER)
        printf("Pass\n");
    else
        printf("Fail\n");

} /* end test_lock_thruput() */

int main(int argc, char **argv)
{
  test_lock_thruput();
} /* end main() */

