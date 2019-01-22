/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdlib.h>
#include "cond/zm_ccond.h"

/* CAS-based condition variable */

int zm_ccond_init(struct zm_ccond *C)
{
    zm_atomic_store(&C->flag, ZM_COND_CLEAR, zm_memord_release);
    return 0;
}

int zm_ccond_destroy(struct zm_ccond *C)
{
    return 0;
}

int zm_ccond_wait(struct zm_ccond *C, zm_lock_t *L) {
    zm_atomic_store(&C->flag, ZM_COND_WAIT, zm_memord_release);
    zm_lock_release(L);
    while(zm_atomic_load(&C->flag, zm_memord_acquire) == ZM_COND_WAIT)
        ; /* SPIN */
    zm_lock_acquire(L);
   return 0;
}

int zm_ccond_wait_c(struct zm_ccond *C, zm_lock_t *L, zm_lock_ctxt_t *ctxt) {
    zm_atomic_store(&C->flag, ZM_COND_WAIT, zm_memord_release);
    zm_lock_release_c(L, ctxt);
    while(zm_atomic_load(&C->flag, zm_memord_acquire) == ZM_COND_WAIT)
        ; /* SPIN */
    zm_lock_acquire_c(L, ctxt);
   return 0;
}

int zm_ccond_signal(struct zm_ccond *C) {
    zm_atomic_store(&C->flag, ZM_COND_CLEAR, zm_memord_release);
    return 0;
}

int zm_ccond_bcast(struct zm_ccond *C) {
    return zm_ccond_signal(C);
}
