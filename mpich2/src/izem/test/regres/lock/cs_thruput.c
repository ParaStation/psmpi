/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <zmtest_abslock.h>

#define TEST_NTHREADS 1
#define TEST_NITER 1000

char cache_lines[640] = {0};
int indices [] = {3,6,1,7,0,2,9,4,8,5};

static void* run(void *arg) {
     int iter;
     zm_abslock_t *lock = (zm_abslock_t*) arg;
     for(iter=0; iter<TEST_NITER; iter++) {
         int err =  zm_abslock_acquire(lock);
         if(err==0) {  /* Lock successfully acquired */
            for(int i = 0; i < 10; i++)
                 cache_lines[indices[i]] += cache_lines[indices[9-i]];
             zm_abslock_release(lock);   /* Release the lock */
         } else {
            fprintf(stderr, "Error: couldn't acquire the lock\n");
            exit(1);
         }
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

    zm_abslock_t lock;
    zm_abslock_init(&lock);

    int th;
    for (th=0; th<TEST_NTHREADS; th++) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(th, &cpuset);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
        pthread_create(&threads[th], &attr, run, (void*) &lock);
        pthread_attr_destroy(&attr);
    }
    for (th=0; th<TEST_NTHREADS; th++)
        pthread_join(threads[th], &res);

    zm_abslock_destroy(&lock);

    printf("Pass\n");

} /* end test_lock_thruput() */

int main(int argc, char **argv)
{
  test_lock_thruput();
} /* end main() */

