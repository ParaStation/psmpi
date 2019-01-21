/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */
#include <assert.h>
#include "queue/zm_mpbqueue.h"
#include "queue/zm_swpqueue.h"

#define EMPTY_BUCKET 0
#define NONEMPTY_BUCKET 1
#define INCONSISTENT_BUCKET 2

#define MIN_BACKOFF 1
#define MAX_BACKOFF 1024

#define LOAD(addr)                  zm_atomic_load(addr, zm_memord_acquire)
#define STORE(addr, val)            zm_atomic_store(addr, val, zm_memord_release)
#define SWAP(addr, desire)          zm_atomic_exchange_ptr(addr, desire, zm_memord_acq_rel)
#define CAS(addr, expect, desire)   zm_atomic_compare_exchange_strong(addr,\
                                                                      expect,\
                                                                      desire,\
                                                                      zm_memord_acq_rel,\
                                                                      zm_memord_acquire)

static inline int get_set_state(struct zm_mpbqueue *q, int offset) {
    int llong_width  = (int) sizeof(long long);
    int nbucket_sets = q->nbuckets/llong_width;
    int bucket_setsz = llong_width/sizeof(char);

    long long *bucket_state_sets = (long long *)q->bucket_states;
    int *backoff_counters = q->backoff_counters;
    int *backoff_bounds = q->backoff_bounds;;
    int j, state = EMPTY_BUCKET;
    if(LOAD(&bucket_state_sets[offset]) > EMPTY_BUCKET) {
        state = NONEMPTY_BUCKET;
    } else {
        backoff_counters[offset]++;
        if(backoff_counters[offset] >= backoff_bounds[offset]) {
            for(j = 0; j < bucket_setsz; j++) {
                int bucket_idx = offset * bucket_setsz + j;
                if(!zm_swpqueue_isempty_weak(&q->buckets[bucket_idx])) {
                    q->bucket_states[bucket_idx] = NONEMPTY_BUCKET;
                    state = NONEMPTY_BUCKET;
                    break;
                }
            }
            if(j >= bucket_setsz) {
                backoff_counters[offset] = 0;
                if(backoff_bounds[offset] < MAX_BACKOFF)
                    backoff_bounds[offset] *= 2;
            }
        }
    }
    return state;
}


int zm_mpbqueue_init(struct zm_mpbqueue *q, int nbuckets) {
    /* multiples of 8 allow single operation, 64bit-width emptiness check  */
    /* TODO: replace the below assert with error handling */
    int llong_width = (int) sizeof(long long);
    /* Adjust nbuckets to allow < llong_width number of buckets */
    /* FIXME: the user can supply a bucket id >= the original nbucket, which is erroneous */
    if(nbuckets < llong_width)
        nbuckets = llong_width;
    assert(nbuckets % llong_width == 0);

    /* allocate and initialize the buckets as SWP-based MPSC queues */
    zm_swpqueue_t *buckets = (zm_swpqueue_t*) malloc(sizeof(zm_swpqueue_t) * nbuckets);
    for(int i = 0; i < nbuckets; i++)
        zm_swpqueue_init(&buckets[i]);
    /* initialze the state of all buckests to empty (0) */
    long long *bucket_state_sets = (long long *) malloc(sizeof(zm_atomic_char_t) * nbuckets);
    int *backoff_counters        = (int *) malloc(sizeof(int) * (nbuckets/llong_width));
    int *backoff_bounds          = (int *) malloc(sizeof(int) * (nbuckets/llong_width));
    for(int i = 0; i < nbuckets/llong_width; i++) {
        bucket_state_sets[i] = EMPTY_BUCKET;
        backoff_counters[i] = 0;
        backoff_bounds[i] = MIN_BACKOFF;
    }

    q->buckets = buckets;
    q->nbuckets = nbuckets;
    q->backoff_counters = backoff_counters;
    q->backoff_bounds = backoff_bounds;
    q->bucket_states = (char*) bucket_state_sets;

    return 0;
}

int zm_mpbqueue_enqueue(struct zm_mpbqueue* q, void *data, int bucket_idx) {

    /* Push to the queue at bucket_idx*/
    zm_swpqueue_enqueue(&q->buckets[bucket_idx], data);

    return 0;
}

int zm_mpbqueue_dequeue(struct zm_mpbqueue* q, void **data) {

    *data = NULL;

    /* Check for a nonempty bucket in sets of bucket_setsz */
    int llong_width  = (int) sizeof(long long);
    int nbucket_sets = q->nbuckets/llong_width;
    int bucket_setsz = llong_width/sizeof(char);

    int i;
    for(i = 0; i < nbucket_sets; i++) {
        int offset = (q->last_bucket_set + i) % nbucket_sets;
        if(get_set_state(q, offset) > EMPTY_BUCKET) {
            int j;
            for(j = 0; j < bucket_setsz; j++) {
                int bucket_idx = offset * bucket_setsz + j;
                if(!zm_swpqueue_isempty_weak(&q->buckets[bucket_idx])) {
                    zm_swpqueue_dequeue(&q->buckets[bucket_idx], data);
                    break;
                } else {
                    q->bucket_states[bucket_idx] = EMPTY_BUCKET;
                }
            }
            if(j < bucket_setsz)
                break;
        }
    }
    q->last_bucket_set = (q->last_bucket_set + i) % nbucket_sets;
    return 1;
}

int zm_mpbqueue_dequeue_bulk(struct zm_mpbqueue* q, void **data, int in_count, int *out_count) {
    /* Check for a nonempty bucket in sets of bucket_setsz */
    int llong_width  = (int) sizeof(zm_atomic_llong_t);
    int nbucket_sets = q->nbuckets/llong_width;
    int bucket_setsz = llong_width/sizeof(zm_atomic_char_t);

    zm_atomic_llong_t *bucket_state_sets = (zm_atomic_llong_t *)q->bucket_states;
    int i, out_idx = 0;
    for(i = 0; i < nbucket_sets; i++) {
        int offset = (q->last_bucket_set + i) % nbucket_sets;
        if(LOAD(&bucket_state_sets[offset]) > EMPTY_BUCKET) {
            int j;
            for(j = 0; j < bucket_setsz; j++) {
                int bucket_idx = offset * bucket_setsz + j;
                if (LOAD(&q->bucket_states[bucket_idx]) == NONEMPTY_BUCKET) {
                    zm_swpqueue_dequeue(&q->buckets[bucket_idx], &data[out_idx]);
                    out_idx++;
                    if(zm_swpqueue_isempty_weak(&q->buckets[bucket_idx])) {
                        if(zm_swpqueue_isempty_strong(&q->buckets[bucket_idx]))
                            STORE(&q->bucket_states[bucket_idx], EMPTY_BUCKET);
                    }
                    if(out_idx >= (in_count -1))
                        break;
                }
            }
            if(j < bucket_setsz)
                break;
        }
    }
    q->last_bucket_set = (q->last_bucket_set + i) % nbucket_sets;
    *out_count = out_idx + 1;
    return 1;
}

/* dequeue in bulk from a defined range of buckets [start,stop[
 * */

int zm_mpbqueue_dequeue_range(struct zm_mpbqueue* q, void **data, int start, int stop, int in_count, int *out_count) {
    assert ((start >= 0) && (stop <= q->nbuckets) && "[start,stop[ out of range");
    int j, count = 0;
    for(j = start; j < stop; j++) {
        if(zm_swpqueue_isempty_weak(&q->buckets[j]))
            continue;
        zm_swpqueue_dequeue(&q->buckets[j], &data[count]);
        count++;
        if(count >= in_count)
            break;
    }
    *out_count = count;
    return 1;
}
