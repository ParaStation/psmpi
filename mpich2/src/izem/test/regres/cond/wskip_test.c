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
#include <cond/zm_wskip.h>

#define TEST_NTHREADS 15
#define TEST_NITER 10

struct zm_mcs_qnode *nodes[TEST_NTHREADS];
zm_atomic_uint_t counter;
pthread_barrier_t barrier;
zm_mcs_t filter;

static void* run(void *arg) {
    int tid = (intptr_t) arg;
    int iter;
    for(iter=0; iter<TEST_NITER; iter++) {
        pthread_barrier_wait(&barrier);
        if(tid == 0)
           zm_atomic_store(&counter, 0, zm_memord_release);
        pthread_barrier_wait(&barrier);
        zm_wskip_wait(filter, &nodes[tid]);
        zm_atomic_fetch_add(&counter, 1, zm_memord_acq_rel);
        int wait_for = 0;
        if(tid == 0) {
            int tmp = zm_atomic_load(&counter, zm_memord_acquire);
            for(int i = 0; i<TEST_NTHREADS/2; i++) {
                int trg = (tid + i + 1) % TEST_NTHREADS;
                /* This status check is not 100% reliable, hangs might occur occasionaly.
                   FIXME: a reliable check would require skip() to return whether a thread
                   got woken up before incrementing wait_for */
                if (nodes[trg] != NULL && zm_atomic_load(&nodes[trg]->status, zm_memord_acquire) == 0) {
                    wait_for++;
                    zm_wskip_skip(nodes[trg]);
                }
            }
            while(zm_atomic_load(&counter, zm_memord_acquire) - tmp < wait_for)
                ; /* wait for threads that skpped the line */
        }
        zm_wskip_wake(filter, nodes[tid]);
    }
    return 0;
}

static void test_prio_chain() {
    void *res;
    int ret;
    pthread_t threads[TEST_NTHREADS];
    zm_wskip_init(&filter);
    pthread_barrier_init(&barrier, NULL /*attr*/, TEST_NTHREADS);

    int th;
    for (th=0; th<TEST_NTHREADS; th++) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(th, &cpuset);
        ret = pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
        if(ret != 0){
            switch(ret) {
                case EINVAL: assert(0 && "setaffinity: EINVAL");
                case ENOMEM: assert(0 && "setaffinity: ENOMEM");
            }
        }

        ret = pthread_create(&threads[th], &attr, run, (void*)(intptr_t)th);
        if(ret != 0){
            switch(ret) {
                case EAGAIN: assert(0 && "thread_create: EAGAIN");
                case EINVAL: assert(0 && "thread_create: EINVAL");
                case EPERM:  assert(0 && "thread_create EPERM");
            }
        }
    }
    for (th=0; th<TEST_NTHREADS; th++)
        pthread_join(threads[th], &res);

    zm_wskip_destroy(&filter);

    printf("Pass\n");

} /* end test_lock_thruput() */

int main(int argc, char **argv)
{
  test_prio_chain();
} /* end main() */

