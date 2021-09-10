/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdlib.h>
#include <hwloc.h>
#include "lock/zm_mcs.h"

struct zm_mcs {
    zm_atomic_ptr_t lock;
    struct zm_mcs_qnode *local_nodes;
    hwloc_topology_t topo;
};

static zm_thread_local int tid = -1;

/* Check the actual affinity mask assigned to the thread */
static inline void check_affinity(hwloc_topology_t topo) {
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    int set_length;
    hwloc_get_cpubind(topo, cpuset, HWLOC_CPUBIND_THREAD);
    set_length = hwloc_get_nbobjs_inside_cpuset_by_type(topo, cpuset, HWLOC_OBJ_PU);
    hwloc_bitmap_free(cpuset);

    if(set_length != 1) {
        printf("IZEM:HMCS:ERROR: thread bound to more than one HW thread!\n");
        exit(EXIT_FAILURE);
    }
}

static inline int get_hwthread_id(hwloc_topology_t topo){
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_obj_t obj;
    hwloc_get_cpubind(topo, cpuset, HWLOC_CPUBIND_THREAD);
    obj = hwloc_get_obj_inside_cpuset_by_type(topo, cpuset, HWLOC_OBJ_PU, 0);
    hwloc_bitmap_free(cpuset);
    return obj->logical_index;
}

static void* new_lock() {
    int max_threads;
    struct zm_mcs_qnode *qnodes;


    struct zm_mcs *L;
    posix_memalign((void **) &L, ZM_CACHELINE_SIZE, sizeof(struct zm_mcs));

    hwloc_topology_init(&L->topo);
    hwloc_topology_load(L->topo);

    max_threads = hwloc_get_nbobjs_by_type(L->topo, HWLOC_OBJ_PU);

    posix_memalign((void **) &qnodes, ZM_CACHELINE_SIZE, sizeof(struct zm_mcs_qnode) * max_threads);

    zm_atomic_store(&L->lock, (zm_ptr_t)ZM_NULL, zm_memord_release);
    L->local_nodes = qnodes;

    return L;
}

/* Main routines */
static inline int acquire_c(struct zm_mcs *L, zm_mcs_qnode_t* I) {
    zm_atomic_store(&I->next, ZM_NULL, zm_memord_release);
    zm_mcs_qnode_t* pred = (zm_mcs_qnode_t*)zm_atomic_exchange_ptr(&L->lock, (zm_ptr_t)I, zm_memord_acq_rel);
    if((zm_ptr_t)pred != ZM_NULL) {
        zm_atomic_store(&I->status, ZM_LOCKED, zm_memord_release);
        zm_atomic_store(&pred->next, (zm_ptr_t)I, zm_memord_release);
        while(zm_atomic_load(&I->status, zm_memord_acquire) != ZM_UNLOCKED)
            ; /* SPIN */
    }
    return 0;
}

static inline int tryacq_c(struct zm_mcs *L, zm_mcs_qnode_t* I, int *success) {
    int acquired  = 0;
    zm_atomic_store(&I->next, ZM_NULL, zm_memord_release);
    zm_ptr_t expected = ZM_NULL;
    if(zm_atomic_compare_exchange_strong(&L->lock,
                                         &expected,
                                         (zm_ptr_t)I,
                                         zm_memord_acq_rel,
                                         zm_memord_acquire))
        acquired = 1;
    *success = acquired;
    return 0;
}

/* Release the lock */
static inline int release_c(struct zm_mcs *L, zm_mcs_qnode_t *I) {
    if (zm_atomic_load(&I->next, zm_memord_acquire) == ZM_NULL) {
        zm_mcs_qnode_t *tmp = I;
        if(zm_atomic_compare_exchange_strong(&L->lock,
                                             (zm_ptr_t*)&tmp,
                                             ZM_NULL,
                                             zm_memord_acq_rel,
                                             zm_memord_acquire))
            return 0;
        while(zm_atomic_load(&I->next, zm_memord_acquire) == ZM_NULL)
            ; /* SPIN */
    }
    zm_atomic_store(&((zm_mcs_qnode_t*)zm_atomic_load(&I->next, zm_memord_acquire))->status, ZM_UNLOCKED, zm_memord_release);
    return 0;
}

static inline int nowaiters_c(struct zm_mcs *L, zm_mcs_qnode_t *I) {
    return (zm_atomic_load(&I->next, zm_memord_acquire) == ZM_NULL);
}

/* Context-less API */
static inline int mcs_acquire(struct zm_mcs *L) {
    if (zm_unlikely(tid == -1)) {
        check_affinity(L->topo);
        tid = get_hwthread_id(L->topo);
    }
    acquire_c(L, &L->local_nodes[tid]);
    return 0;
}

static inline int mcs_tryacq(struct zm_mcs *L, int *success) {
    if (zm_unlikely(tid == -1)) {
        check_affinity(L->topo);
        tid = get_hwthread_id(L->topo);
    }
    return tryacq_c(L, &L->local_nodes[tid], success);
}

static inline int mcs_release(struct zm_mcs *L) {
    assert(tid >= 0);
    return release_c(L, &L->local_nodes[tid]);
}

static inline int mcs_nowaiters(struct zm_mcs *L) {
    assert(tid >= 0);
    return nowaiters_c(L, &L->local_nodes[tid]);
}

/* Context-full API */
static inline int mcs_acquire_c(struct zm_mcs *L, zm_mcs_qnode_t* I) {
    return acquire_c(L, I);
}

int mcs_tryacq_c(struct zm_mcs *L, zm_mcs_qnode_t* I, int *success) {
    return tryacq_c(L, I, success);
}


static inline int mcs_release_c(struct zm_mcs *L, zm_mcs_qnode_t *I) {
    return release_c(L, I);
}

static inline int mcs_nowaiters_c(struct zm_mcs *L, zm_mcs_qnode_t *I) {
    return nowaiters_c(L, I);
}

static inline int free_lock(struct zm_mcs *L)
{
    free(L->local_nodes);
    hwloc_topology_destroy(L->topo);
    return 0;
}


int zm_mcs_init(zm_mcs_t *handle) {
    void *p = new_lock();
    *handle  = (zm_mcs_t) p;
    return 0;
}

int zm_mcs_destroy(zm_mcs_t *L) {
    free_lock((struct zm_mcs*)(*L));
    return 0;
}


/* Context-less API */
int zm_mcs_acquire(zm_mcs_t L) {
    /*
      It is prohibited to convert intptr_t (=zm_mcs_t) to a non-void pointer.
      Converting intptr_t to void*, then void* to any pointer type is permitted.
      cf. https://stackoverflow.com/questions/34291377/converting-a-non-void-pointer-to-uintptr-t-and-vice-versa
    */
    return mcs_acquire((struct zm_mcs*)(void *)L) ;
}

int zm_mcs_tryacq(zm_mcs_t L, int *success) {
    return mcs_tryacq((struct zm_mcs*)(void *)L, success) ;
}

int zm_mcs_release(zm_mcs_t L) {
    return mcs_release((struct zm_mcs*)(void *)L) ;
}

int zm_mcs_nowaiters(zm_mcs_t L) {
    return mcs_nowaiters((struct zm_mcs*)(void *)L) ;
}

/* Context-full API */
int zm_mcs_acquire_c(zm_mcs_t L, zm_mcs_qnode_t* I) {
    return mcs_acquire_c((struct zm_mcs*)(void *)L, I) ;
}

int zm_mcs_tryacq_c(zm_mcs_t L, zm_mcs_qnode_t* I, int *success) {
    return mcs_tryacq_c((struct zm_mcs*)(void *)L, I, success) ;
}

int zm_mcs_release_c(zm_mcs_t L, zm_mcs_qnode_t *I) {
    return mcs_release_c((struct zm_mcs*)(void *)L, I) ;
}

int zm_mcs_nowaiters_c(zm_mcs_t L, zm_mcs_qnode_t *I) {
    return mcs_nowaiters_c((struct zm_mcs*)(void *)L, I) ;
}
