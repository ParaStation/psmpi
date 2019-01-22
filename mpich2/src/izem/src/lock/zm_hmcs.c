/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * The original version of this code was contributed by Milind Chabbi
 * based on the work when he was at Rice University. It relies on the
 * HMCS lock description in [1] and the fast path described in [2].
 *
 * [1] Chabbi, Milind, Michael Fagan, and John Mellor-Crummey. "High
 * performance locks for multi-level NUMA systems." In Proceedings of
 * the ACM SIGPLAN Symposium on Principles and Practice of Parallel
 * Programming (PPoPP'15), ACM, 2015.
 *
 * [2] Chabbi, Milind, and John Mellor-Crummey. "Contention-conscious,
 * locality-preserving locks." In Proceedings of the 21st ACM SIGPLAN
 * Symposium on Principles and Practice of Parallel Programming (PPoPP'16,
 * p. 22. ACM, 2016.
 */

#include "lock/zm_lock_types.h"
#include <hwloc.h>

#ifndef DEFAULT_THRESHOLD
#define DEFAULT_THRESHOLD 256
#endif

#ifndef HMCS_DEFAULT_MAX_LEVELS
#define HMCS_DEFAULT_MAX_LEVELS 3
#endif

#define WAIT (0xffffffff)
#define COHORT_START (0x1)
#define ACQUIRE_PARENT (0xcffffffc)

#ifndef TRUE
#define TRUE 1
#else
#error "TRUE already defined"
#endif

#ifndef FALSE
#define FALSE 0
#else
#error "TRUE already defined"
#endif

/* Atomic operation shorthands. The memory ordering defaults to:
 * 1- Acquire ordering for loads
 * 2- Release ordering for stores
 * 3- Acquire+Release ordering for read-modify-write operations
 * */

#define LOAD(addr)                  zm_atomic_load(addr, zm_memord_acquire)
#define STORE(addr, val)            zm_atomic_store(addr, val, zm_memord_release)
#define SWAP(addr, desire)          zm_atomic_exchange_ptr(addr, desire, zm_memord_acq_rel)
#define CAS(addr, expect, desire)   zm_atomic_compare_exchange_strong(addr,\
                                                                      expect,\
                                                                      desire,\
                                                                      zm_memord_acq_rel,\
                                                                      zm_memord_acquire)

struct hnode{
    unsigned threshold __attribute__((aligned(ZM_CACHELINE_SIZE)));
    struct hnode * parent __attribute__((aligned(ZM_CACHELINE_SIZE)));
    zm_atomic_ptr_t lock __attribute__((aligned(ZM_CACHELINE_SIZE)));
    zm_mcs_qnode_t node __attribute__((aligned(ZM_CACHELINE_SIZE)));

}__attribute__((aligned(ZM_CACHELINE_SIZE)));

struct leaf{
    struct hnode * cur_node;
    struct hnode * root_node;
    zm_mcs_qnode_t I;
    int curDepth;
    int took_fast_path;
};

struct lock{
    // Assumes tids range from [0.. max_threads)
    // Assumes that tid 0 is close to tid and so on.
    struct leaf ** leaf_nodes __attribute__((aligned(ZM_CACHELINE_SIZE)));
    hwloc_topology_t topo;
    int levels;
};

static zm_thread_local int tid = -1;

/* TODO: automate hardware topology detection
 * instead of the below hard-coded method */

/* Check the actual affinity mask assigned to the thread */
static void check_affinity(hwloc_topology_t topo) {
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

static inline void reuse_qnode(zm_mcs_qnode_t *I){
    STORE(&I->status, WAIT);
    STORE(&I->next, ZM_NULL);
}

static void* new_hnode() {
    int err;
    void *storage;
    err = posix_memalign(&storage, ZM_CACHELINE_SIZE, sizeof(struct hnode));
    if (err != 0) {
        printf("posix_memalign failed in HMCS : new_hnode \n");
        exit(EXIT_FAILURE);
    }
    return storage;
}

/* TODO: Macro or Template this for fast comprison */
static inline unsigned get_threshold(struct hnode *L) {
    return L->threshold;
}

static inline void normal_mcs_release_with_value(struct hnode * L, zm_mcs_qnode_t *I, unsigned val){

    zm_mcs_qnode_t *succ = (zm_mcs_qnode_t *)LOAD(&I->next);
    if(succ) {
        STORE(&succ->status, val);
        return;
    }
    zm_mcs_qnode_t *tmp = I;
    if (CAS(&(L->lock), (zm_ptr_t*)&tmp,ZM_NULL))
        return;
    while(succ == NULL)
        succ = (zm_mcs_qnode_t *)LOAD(&I->next); /* SPIN */
    STORE(&succ->status, val);
    return;
}

static inline void acquire_root(struct hnode * L, zm_mcs_qnode_t *I) {
    // Prepare the node for use.
    reuse_qnode(I);
    zm_mcs_qnode_t *pred = (zm_mcs_qnode_t*) SWAP(&(L->lock), (zm_ptr_t)I);

    if(!pred) {
        // I am the first one at this level
        return;
    }

    STORE(&pred->next, I);
    while(LOAD(&I->status) == WAIT)
        ; /* SPIN */
    return;
}

static inline void tryacq_root(struct hnode * L, zm_mcs_qnode_t *I, int *success) {
    zm_ptr_t expected = ZM_NULL;
    // Prepare the node for use.
    reuse_qnode(I);
    *success = CAS(&(L->lock), &expected, (zm_ptr_t)I);

    return;
}

static inline void release_root(struct hnode * L, zm_mcs_qnode_t *I) {
    // Top level release is usual MCS
    // At the top level MCS we always writr COHORT_START since
    // 1. It will release the lock
    // 2. Will never grow large
    // 3. Avoids a read from I->status
    normal_mcs_release_with_value(L, I, COHORT_START);
}

static inline int nowaiters_root(struct hnode * L, zm_mcs_qnode_t *I) {
    return (LOAD(&I->next) == ZM_NULL);
}

static inline void acquire_helper(int level, struct hnode * L, zm_mcs_qnode_t *I) {
    // Trivial case = root level
    if (level == 1)
        acquire_root(L, I);
    else {
        // Prepare the node for use.
        reuse_qnode(I);
        zm_mcs_qnode_t* pred = (zm_mcs_qnode_t*)SWAP(&(L->lock), (zm_ptr_t)I);
        if(!pred) {
            // I am the first one at this level
            // begining of cohort
            STORE(&I->status, COHORT_START);
            // acquire at next level if not at the top level
            acquire_helper(level - 1, L->parent, &(L->node));
            return;
        } else {
            STORE(&pred->next, I);
            for(;;){
                unsigned myStatus = LOAD(&I->status);
                if(myStatus < ACQUIRE_PARENT) {
                    return;
                }
                if(myStatus == ACQUIRE_PARENT) {
                    // beginning of cohort
                    STORE(&I->status, COHORT_START);
                    // This means this level is acquired and we can start the next level
                    acquire_helper(level - 1, L->parent, &(L->node));
                    return;
                }
                // spin back; (I->status == WAIT)
            }
        }
    }
}

static inline void release_helper(int level, struct hnode * L, zm_mcs_qnode_t *I) {
    // Trivial case = root level
    if (level == 1) {
        release_root(L, I);
    } else {
        unsigned cur_count = LOAD(&(I->status)) ;
        zm_mcs_qnode_t * succ;

        // Lower level releases
        if(cur_count == get_threshold(L)) {
            // NO KNOWN SUCCESSORS / DESCENDENTS
            // reached threshold and have next level
            // release to next level
            release_helper(level - 1, L->parent, &(L->node));
            //COMMIT_ALL_WRITES();
            // Tap successor at this level and ask to spin acquire next level lock
            normal_mcs_release_with_value(L, I, ACQUIRE_PARENT);
            return;
        }

        succ = (zm_mcs_qnode_t*)LOAD(&(I->next));
        // Not reached threshold
        if(succ) {
            STORE(&succ->status, cur_count + 1);
            return; // released
        }
        // No known successor, so release
        release_helper(level - 1, L->parent, &(L->node));
        // Tap successor at this level and ask to spin acquire next level lock
        normal_mcs_release_with_value(L, I, ACQUIRE_PARENT);
    }
}

static inline int nowaiters_helper(int level, struct hnode * L, zm_mcs_qnode_t *I) {
    if (level == 1 ) {
        return nowaiters_root(L,I);
    } else {
        if(LOAD(&I->next) != ZM_NULL)
            return FALSE;
        else
            return nowaiters_helper(level - 1, L->parent, &(L->node));
    }
}

static void* new_leaf(struct hnode *h, int depth) {
    int err;
    struct leaf *leaf;
    err = posix_memalign((void **) &leaf, ZM_CACHELINE_SIZE, sizeof(struct leaf));
    if (err != 0) {
        printf("posix_memalign failed in HMCS : new_leaf \n");
        exit(EXIT_FAILURE);
    }
    leaf->cur_node = h;
    leaf->curDepth = depth;
    leaf->took_fast_path = FALSE;
    struct hnode *tmp, *root_node;
    for(tmp = leaf->cur_node; tmp->parent != NULL; tmp = tmp->parent);
    root_node = tmp;
    leaf->root_node = root_node;
    return leaf;
}

static inline void acquire_from_leaf(int level, struct leaf *L){
    if((zm_ptr_t)L->cur_node->lock == ZM_NULL
    && (zm_ptr_t)L->root_node->lock == ZM_NULL) {
        // go FP
        L->took_fast_path = TRUE;
        acquire_root(L->root_node, &L->I);
        return;
    }
    acquire_helper(level, L->cur_node, &L->I);
    return;
}

static inline void tryacq_from_leaf(int level, struct leaf *L, int *success){
    *success = 0;
    if((zm_ptr_t)L->cur_node->lock == ZM_NULL
    && (zm_ptr_t)L->root_node->lock == ZM_NULL) {
        tryacq_root(L->root_node, &L->I, success);
        if (*success)
            L->took_fast_path = TRUE;
    }
    return;
}

static inline void release_from_leaf(int level, struct leaf *L){
    //myrelease(cur_node, I);
    if(L->took_fast_path) {
        release_root(L->root_node, &L->I);
        L->took_fast_path = FALSE;
        return;
    }
    release_helper(level, L->cur_node, &L->I);
    return;
}

static inline int nowaiters_from_leaf(int level, struct leaf *L){
    // Shouldnt this be nowaiters(root_node, I)?
    if(L->took_fast_path) {
        return nowaiters_root(L->cur_node, &L->I);
    }

    return nowaiters_helper(level, L->cur_node, &L->I);
}

static int get_hwthread_id(hwloc_topology_t topo){
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_obj_t obj;
    hwloc_get_cpubind(topo, cpuset, HWLOC_CPUBIND_THREAD);
    obj = hwloc_get_obj_inside_cpuset_by_type(topo, cpuset, HWLOC_OBJ_PU, 0);
    hwloc_bitmap_free(cpuset);
    return obj->logical_index;
}

static void set_hierarchy(struct lock *L, int *max_threads, int** particip_per_level) {
    int max_depth, levels = 0, max_levels = HMCS_DEFAULT_MAX_LEVELS, explicit_levels = 0;
    char tmp[20];
    char *s = getenv("ZM_HMCS_MAX_LEVELS");
    if (s != NULL)
        max_levels = atoi(s);
    int depths[max_levels];
    int idx = 0;
    /* advice to users: run `hwloc-ls -s --no-io --no-icaches` and choose
     * depths of interest in ascending order. The first depth must be `0' */

    s = getenv("ZM_HMCS_EXPLICIT_LEVELS");
    if (s != NULL) {
        strcpy(tmp, s);
        explicit_levels = 1;
        char* token;
        token = strtok(tmp,",");
        while(token != NULL) {
            depths[idx] = atoi(token);
            if (idx == 0)
                assert(depths[idx] == 0 && "the first depth must be machine level (i.e., depth 0), run `hwloc-ls -s --no-io --no-icaches` and choose appropriate depth values");
            idx++;
            token = strtok(NULL,",");
        }
        assert(idx == max_levels);
    }

    hwloc_topology_init(&L->topo);
    hwloc_topology_load(L->topo);

    *max_threads = hwloc_get_nbobjs_by_type(L->topo, HWLOC_OBJ_PU);

    max_depth = hwloc_topology_get_depth(L->topo);
    assert(max_depth >= 2); /* At least Machine and Core levels exist */

    *particip_per_level = (int*) malloc(max_levels * sizeof(int));
    int prev_nobjs = -1;
    if(!explicit_levels) {
        for (int d = max_depth - 2; d > 1; d--) {
            int cur_nobjs = hwloc_get_nbobjs_by_depth(L->topo, d);
            /* Check if this level has a hierarchical impact */
            if(cur_nobjs != prev_nobjs) {
                prev_nobjs = cur_nobjs;
                (*particip_per_level)[levels] = (*max_threads)/cur_nobjs;
                levels++;
                if(levels >= max_levels - 1)
                    break;
            }
        }
        (*particip_per_level)[levels] = *max_threads;
        levels++;
    } else {
        for(int i = max_levels - 1; i >= 0; i--){
            int d = depths[i];
            int cur_nobjs = hwloc_get_nbobjs_by_depth(L->topo, d);
            /* Check if this level has a hierarchical impact */
            if(cur_nobjs != prev_nobjs) {
                prev_nobjs = cur_nobjs;
                (*particip_per_level)[levels] = (*max_threads)/cur_nobjs;
                levels++;
            } else {
                assert(0 && "plz choose levels that have a hierarchical impact");
            }
        }
    }

    L->levels = levels;
}

static void free_hierarchy(int* particip_per_level){
    free(particip_per_level);
}

static void* new_lock(){

    struct lock *L;
    posix_memalign((void **) &L, ZM_CACHELINE_SIZE, sizeof(struct lock));

    int max_threads;
    int *participants_at_level;
    set_hierarchy(L, &max_threads, &participants_at_level);

    // Total locks needed = participantsPerLevel[1] + participantsPerLevel[2] + .. participantsPerLevel[levels-1] + 1
    // Save affinity
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_get_cpubind(L->topo, cpuset, HWLOC_CPUBIND_THREAD);

    int total_locks_needed = 0;
    int levels = L->levels;

    for (int i=0; i < levels; i++) {
        total_locks_needed += max_threads / participants_at_level[i] ;
    }
    struct hnode ** lock_locations;
    posix_memalign((void **) &lock_locations, ZM_CACHELINE_SIZE, sizeof(struct hnode*) * total_locks_needed);
    struct leaf ** leaf_nodes;
    posix_memalign((void **) &leaf_nodes, ZM_CACHELINE_SIZE, sizeof(struct leaf*) * max_threads);

    int threshold = DEFAULT_THRESHOLD;
    char *s = getenv("ZM_HMCS_THRESHOLD");
    if (s != NULL)
        threshold = atoi(s);

    hwloc_obj_t obj;
    for(int tid = 0 ; tid < max_threads; tid ++){
        obj = hwloc_get_obj_by_type (L->topo, HWLOC_OBJ_PU, tid);
        hwloc_set_cpubind(L->topo, obj->cpuset, HWLOC_CPUBIND_THREAD);
        // Pin me to hw-thread-id = tid
        int last_lock_location_end = 0;
        for(int cur_level = 0 ; cur_level < levels; cur_level++){
            if (tid%participants_at_level[cur_level] == 0) {
                // master, initialize the lock
                int lock_location = last_lock_location_end + tid/participants_at_level[cur_level];
                last_lock_location_end += max_threads/participants_at_level[cur_level];
                struct hnode * cur_hnode = new_hnode();
                cur_hnode->threshold = threshold;
                cur_hnode->parent = NULL;
                cur_hnode->lock = ZM_NULL;
                lock_locations[lock_location] = cur_hnode;
            }
        }
    }

    // setup parents
    for(int tid = 0 ; tid < max_threads; tid ++){
        obj = hwloc_get_obj_by_type (L->topo, HWLOC_OBJ_PU, tid);
        hwloc_set_cpubind(L->topo, obj->cpuset, HWLOC_CPUBIND_THREAD);
        int last_lock_location_end = 0;
        for(int cur_level = 0 ; cur_level < levels - 1; cur_level++){
            if (tid%participants_at_level[cur_level] == 0) {
                int lock_location = last_lock_location_end + tid/participants_at_level[cur_level];
                last_lock_location_end += max_threads/participants_at_level[cur_level];
                int parentLockLocation = last_lock_location_end + tid/participants_at_level[cur_level+1];
                lock_locations[lock_location]->parent = lock_locations[parentLockLocation];
            }
        }
        leaf_nodes[tid] = (struct leaf*)new_leaf(lock_locations[tid/participants_at_level[0]], levels);
    }
    free(lock_locations);
    free_hierarchy(participants_at_level);
    // Restore affinity
    hwloc_set_cpubind(L->topo, cpuset, HWLOC_CPUBIND_THREAD);
    L->leaf_nodes = leaf_nodes;

    hwloc_bitmap_free(cpuset);

    return L;
}

static void search_nodes_rec(struct hnode *node, struct hnode **nodes_to_free, int *num_ptrs, int max_threads) {
    int i;
    if(node != NULL) {
        for(i = 0; i < *num_ptrs; i++) {
            if(node == nodes_to_free[i])
                break; /* already marked to be free'd */
        }
        if(i == *num_ptrs) { /* newly encountered pointer */
            nodes_to_free[*num_ptrs] = node;
            (*num_ptrs)++;
            assert(*num_ptrs < 2*max_threads);
        }
        search_nodes_rec(node->parent, nodes_to_free, num_ptrs, max_threads);
    }
}

static void free_lock(struct lock* L) {
    int max_threads = hwloc_get_nbobjs_by_type(L->topo, HWLOC_OBJ_PU);
    int num_ptrs = 0;
    struct hnode **nodes_to_free = (struct hnode**) malloc(2*max_threads*sizeof(struct hnode*));
    for (int tid = 0; tid < max_threads; tid++) {
        search_nodes_rec(L->leaf_nodes[tid]->cur_node, nodes_to_free, &num_ptrs, max_threads);
        free(L->leaf_nodes[tid]);
    }
    free(L->leaf_nodes);
    for(int i = 0; i < num_ptrs; i++)
        free(nodes_to_free[i]);
    free(nodes_to_free);
    hwloc_topology_destroy(L->topo);
    free(L);
}

static inline void hmcs_acquire(struct lock *L){
    if (zm_unlikely(tid == -1)) {
        check_affinity(L->topo);
        tid = get_hwthread_id(L->topo);
    }
    acquire_from_leaf(L->levels, L->leaf_nodes[tid]);
}

static inline void hmcs_tryacq(struct lock *L, int *success){
    if (zm_unlikely(tid == -1)) {
        check_affinity(L->topo);
        tid = get_hwthread_id(L->topo);
    }
    tryacq_from_leaf(L->levels, L->leaf_nodes[tid], success);
}

static inline void hmcs_release(struct lock *L){
    release_from_leaf(L->levels, L->leaf_nodes[tid]);
}

static inline int hmcs_nowaiters(struct lock *L){
    return nowaiters_from_leaf(L->levels, L->leaf_nodes[tid]);
}

int zm_hmcs_init(zm_hmcs_t * handle) {
    void *p = new_lock();
    *handle  = (zm_hmcs_t) p;
    return 0;
}

int zm_hmcs_destroy(zm_hmcs_t *L) {
    free_lock((struct lock*)(*L));
    return 0;
}

int zm_hmcs_acquire(zm_hmcs_t L){
    hmcs_acquire((struct lock*)L);
    return 0;
}

int zm_hmcs_tryacq(zm_hmcs_t L, int *success){
    hmcs_tryacq((struct lock*)L, success);
    return 0;
}
int zm_hmcs_release(zm_hmcs_t L){
    hmcs_release((struct lock*)L);
    return 0;
}
int zm_hmcs_nowaiters(zm_hmcs_t L){
    return hmcs_nowaiters((struct lock*)L);
}

