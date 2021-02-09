/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdlib.h>
#include <hwloc.h>
#include "lock/zm_mcs.h"
#include "cond/zm_wskip.h"

#define ZM_WAIT 0
#define ZM_WAKE 1
#define ZM_SKIP 2
#define ZM_RECYCLE 3
#define ZM_CHECK 4

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
        printf("IZEM:WSKIP:ERROR: thread bound to more than one HW thread!\n");
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

static void* new_wskip() {
    int max_threads;
    struct zm_mcs_qnode *qnodes;


    struct zm_mcs *L;
    posix_memalign((void **) &L, ZM_CACHELINE_SIZE, sizeof(struct zm_mcs));

    hwloc_topology_init(&L->topo);
    hwloc_topology_load(L->topo);

    max_threads = hwloc_get_nbobjs_by_type(L->topo, HWLOC_OBJ_PU);

    posix_memalign((void **) &qnodes, ZM_CACHELINE_SIZE, sizeof(struct zm_mcs_qnode) * max_threads);
    for (int i = 0; i < max_threads; i ++)
        zm_atomic_store(&qnodes[i].status, ZM_RECYCLE, zm_memord_release);

    zm_atomic_store(&L->lock, (zm_ptr_t)ZM_NULL, zm_memord_release);
    L->local_nodes = qnodes;

    return L;
}

/* This routine is just to insert myself into the queue and block
 * whoever comes after me. */
static inline int enq(struct zm_mcs *L, zm_mcs_qnode_t* I, int *wait) {
    *wait = 1;
    zm_mcs_qnode_t* pred;
    int status = zm_atomic_exchange_int(&I->status, ZM_WAIT, zm_memord_acq_rel);
    /* wake() passed this node and is in the processs of setting it to RECYCLE */
    if(status == ZM_CHECK) {
        while (status != ZM_RECYCLE)
            status = zm_atomic_load(&I->status, zm_memord_acquire); /* wait */
        zm_atomic_store(&I->status, ZM_WAIT, zm_memord_release);
    }

    if (status == ZM_RECYCLE) {
        zm_atomic_store(&I->next, ZM_NULL, zm_memord_release);
        pred = (zm_mcs_qnode_t*)zm_atomic_exchange_ptr(&L->lock, (zm_ptr_t)I, zm_memord_acq_rel);
        if((zm_ptr_t)pred == ZM_NULL) {
            zm_atomic_store(&I->status, ZM_WAKE, zm_memord_release);
            *wait = 0;
            return 0;
        }
        zm_atomic_store(&pred->next, (zm_ptr_t)I, zm_memord_release);
    }

    return 0;
}

/* Main routines */
static inline int wait(struct zm_mcs *L, zm_mcs_qnode_t* I) {
    int wait = 0;
    /* First, insert the qnode into the queue */
    enq(L,I, &wait);
    /* wait in line if necessary */
    if (wait)
        while(zm_atomic_load(&I->status, zm_memord_acquire) != ZM_WAKE &&
              zm_atomic_load(&I->status, zm_memord_acquire) != ZM_RECYCLE)
            ; /* SPIN */

    return 0;
}

/* Release the lock */
static inline int wake(struct zm_mcs *L, zm_mcs_qnode_t *I) {

    int status = ZM_WAKE;
    if(!zm_atomic_compare_exchange_strong(&I->status,
                                         &status,
                                         ZM_RECYCLE,
                                         zm_memord_acq_rel,
                                         zm_memord_acquire))
        return 0;

    zm_mcs_qnode_t *cur_node = I;
    zm_mcs_qnode_t *next = (zm_mcs_qnode_t*)zm_atomic_load(&cur_node->next, zm_memord_acquire);
    zm_mcs_qnode_t *pred = NULL;
    /* traverse queue until end or encountering a node that wasn't skipped */
    while ((zm_ptr_t)next != ZM_NULL) {
        int status = ZM_WAIT;
        if(zm_atomic_compare_exchange_strong(&next->status,
                                         &status,
                                         ZM_WAKE,
                                         zm_memord_acq_rel,
                                         zm_memord_acquire))
            break;
        zm_atomic_store(&next->status, ZM_CHECK, zm_memord_release);
        /* modify next for reverse traversal later */
        zm_atomic_store(&cur_node->next, pred, zm_memord_release);
        pred = cur_node;
        cur_node = next;
        next = (zm_mcs_qnode_t*)zm_atomic_load(&next->next, zm_memord_acquire);
    }
    /* reverse traversal for recycling */
    zm_atomic_store(&cur_node->status, ZM_RECYCLE, zm_memord_release);
    zm_mcs_qnode_t *rev = pred;
    while((zm_ptr_t)rev != ZM_NULL) {
        zm_atomic_store(&rev->status, ZM_RECYCLE, zm_memord_release);
        rev = (zm_mcs_qnode_t*)zm_atomic_load(&rev->next, zm_memord_acquire);
    }

    if ((zm_ptr_t)next == ZM_NULL) {
        zm_mcs_qnode_t *tmp = cur_node;
        if(zm_atomic_compare_exchange_strong(&L->lock,
                                             (zm_ptr_t*)&tmp,
                                             ZM_NULL,
                                             zm_memord_acq_rel,
                                             zm_memord_acquire))
            return 0;
        while(zm_atomic_load(&cur_node->next, zm_memord_acquire) == ZM_NULL)
            ; /* SPIN */
        zm_atomic_store(&((zm_mcs_qnode_t*)zm_atomic_load(&cur_node->next, zm_memord_acquire))->status, ZM_WAKE, zm_memord_release);
    }
    zm_atomic_store(&I->next, NULL, zm_memord_release);

    return 0;
}

static inline int skip(zm_mcs_qnode_t *I) {
    int status = ZM_WAIT;
    zm_atomic_compare_exchange_strong(&I->status,
                                      &status,
                                      ZM_SKIP,
                                      zm_memord_acq_rel,
                                      zm_memord_acquire);
    return 0;
}

static inline int nowaiters(struct zm_mcs *L, zm_mcs_qnode_t *I) {
    return (zm_atomic_load(&I->next, zm_memord_acquire) == ZM_NULL);
}

int wskip_wait(struct zm_mcs *L, zm_mcs_qnode_t** I) {
    if (zm_unlikely(tid == -1)) {
        check_affinity(L->topo);
        tid = get_hwthread_id(L->topo);
    }
    *I= &L->local_nodes[tid];
    return wait(L, *I);
}

int wskip_enq(struct zm_mcs *L, zm_mcs_qnode_t *I) {
    int wait; /* unused */
    return enq(L, I, &wait);
}

int wskip_wake(struct zm_mcs *L, zm_mcs_qnode_t *I) {
    return wake(L, I);
}

int wskip_skip(zm_mcs_qnode_t *I) {
    return skip(I);
}

int wskip_nowaiters(struct zm_mcs *L, zm_mcs_qnode_t *I) {
    return nowaiters(L, I);
}

static inline int free_wskip(struct zm_mcs *L)
{
    free(L->local_nodes);
    hwloc_topology_destroy(L->topo);
    return 0;
}


int zm_wskip_init(zm_mcs_t *handle) {
    void *p = new_wskip();
    *handle  = (zm_mcs_t) p;
    return 0;
}

int zm_wskip_destroy(zm_mcs_t *L) {
    free_wskip((struct zm_mcs*)(*L));
    return 0;
}

int zm_wskip_wait(zm_mcs_t L, zm_mcs_qnode_t** I) {
    return wskip_wait((struct zm_mcs*)(void *)L, I);
}

int zm_wskip_enq(zm_mcs_t L, zm_mcs_qnode_t* I) {
    return wskip_enq((struct zm_mcs*)(void *)L, I);
}

int zm_wskip_wake(zm_mcs_t L, zm_mcs_qnode_t *I) {
    return wskip_wake((struct zm_mcs*)(void *)L, I);
}

int zm_wskip_skip(zm_mcs_qnode_t *I) {
    return wskip_skip(I);
}

int zm_wskip_nowaiters(zm_mcs_t L, zm_mcs_qnode_t *I) {
    return wskip_nowaiters((struct zm_mcs*)(void *)L, I);
}

