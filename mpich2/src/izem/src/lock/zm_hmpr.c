/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdlib.h>
#include "lock/zm_hmcs.h"
#include "lock/zm_hmpr.h"
#include "cond/zm_wskip.h"

int zm_hmpr_init(struct zm_hmpr *L) {
    int ret;
    ret = zm_hmcs_init(&L->lock);
    ret = zm_wskip_init(&L->waitq);
    return ret;
}

int zm_hmpr_destroy(struct zm_hmpr *L) {
    int ret;
    ret = zm_hmcs_destroy(&L->lock);
    ret = zm_wskip_destroy(&L->waitq);
    return ret;
}

int zm_hmpr_acquire(struct zm_hmpr *L, struct zm_hmpr_pnode *N) {
    int ret;
    if (N->p == 0) {
        ret = zm_hmcs_acquire(L->lock);
        assert(ret == 0);
        if (!L->go_straight) {
            ret = zm_wskip_enq(L->waitq, &L->znode);
            assert(ret == 0);
            L->go_straight = 1;
        }
    } else {
        ret = zm_wskip_wait(L->waitq, &N->qnode);
        ret = zm_hmcs_acquire(L->lock);
        L->low_p_acq = 1;
    }
    return ret;
}

int zm_hmpr_release(struct zm_hmpr *L, struct zm_hmpr_pnode *N) {
    int ret;
    if (!L->low_p_acq) {
        if (zm_hmcs_nowaiters(L->lock)) {
            L->go_straight = 0;
            ret = zm_wskip_wake(L->waitq, &L->znode);
        }
        zm_hmcs_release(L->lock);
    } else {
        L->low_p_acq = 0;
        zm_hmcs_release(L->lock);
        ret = zm_wskip_wake(L->waitq, N->qnode);
    }
    return ret;
}

int zm_hmpr_raise_prio(struct zm_hmpr_pnode *N) {
    if (N->p > 0) {
        N->p--;
        if ((N->p == 0) && (N->qnode != NULL))
            return zm_wskip_skip(N->qnode);
    }
    return 0;
}
