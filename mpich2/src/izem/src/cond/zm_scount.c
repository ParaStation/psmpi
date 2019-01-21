/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdlib.h>
#include "cond/zm_scount.h"
#include "cond/zm_ccond.h"

int zm_scount_init(struct zm_scount *C, int count)
{
    zm_atomic_store(&C->count, count, zm_memord_release);
    zm_ccond_init(&C->cvar);

    return 0;
}

int zm_scount_destroy(struct zm_scount *C)
{
    return zm_ccond_destroy(&C->cvar);
}

int zm_scount_wait(struct zm_scount *C, zm_lock_t *L) {
    int ret = 0;
    if(C->count > 0)
        ret = zm_ccond_wait(&C->cvar, L);
    return ret;
}

int zm_scount_signal(struct zm_scount *C, int *out_count) {
    int ret = 0;
    if(C->count > 0) {
        C->count--;
        if(C->count == 0)
            ret = zm_ccond_signal(&C->cvar);
    }
    *out_count = C->count;
    return ret;
}

/* Forced wake up regardless of the counter */
int zm_scond_signalf(struct zm_scount *C) {
    int ret = 0;
    /* wakeup signal regardless of the counter value */
    ret = zm_ccond_signal(&C->cvar);

    return ret;
}
