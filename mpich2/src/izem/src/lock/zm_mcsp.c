/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdlib.h>
#include "lock/zm_mcsp.h"

int zm_mcsp_init(zm_mcsp_t *L) {
    zm_mcs_init(&L->high_p);
    L->go_straight = 0;
    L->low_p_acq = 0;
    zm_ticket_init(&L->filter);
    zm_mcs_init(&L->low_p);
    return 0;
}

int zm_mcsp_acquire(zm_mcsp_t *L) {
    zm_mcs_acquire(L->high_p);
    if (!L->go_straight) {
        zm_ticket_acquire(&L->filter);
        L->go_straight = 1;
    }
    return 0;
}

int zm_mcsp_tryacq(zm_mcsp_t *L, int *success) {
    zm_mcs_tryacq(L->high_p, success);
    if (success) {
        if (!L->go_straight) {
            zm_ticket_tryacq(&L->filter, success);
            if (success)
                L->go_straight = 1;
            else
                zm_mcs_release(L->high_p);
        }
    }
    return 0;
}

int zm_mcsp_acquire_low(zm_mcsp_t *L) {
    zm_mcs_acquire(L->low_p);
    zm_ticket_acquire(&L->filter);
    L->low_p_acq = 1;
    return 0;
}

int zm_mcsp_tryacq_low(zm_mcsp_t *L, int *success) {
    zm_mcs_tryacq(L->low_p, success);
    if (success) {
        zm_ticket_tryacq(&L->filter, success);
        if (success)
                L->low_p_acq = 1;
        else
                zm_mcs_release(L->low_p);
    }
    return 0;
}

int zm_mcsp_release(zm_mcsp_t *L) {
    if (!L->low_p_acq) {
        if (zm_mcs_nowaiters(L->high_p)) {
            L->go_straight = 0;
            zm_ticket_release(&L->filter);
        }
        zm_mcs_release(L->high_p);
    } else {
        L->low_p_acq = 0;
        zm_ticket_release(&L->filter);
        zm_mcs_release(L->low_p);
    }
    return 0;
}

int zm_mcsp_acquire_c(zm_mcsp_t *L, zm_mcs_qnode_t* I) {
    zm_mcs_acquire_c(L->high_p, I);
    if (!L->go_straight) {
        zm_ticket_acquire(&L->filter);
        L->go_straight = 1;
    }
    return 0;
}

int zm_mcsp_acquire_low_c(zm_mcsp_t *L, zm_mcs_qnode_t* I) {
    zm_mcs_acquire_c(L->low_p, I);
    zm_ticket_acquire(&L->filter);
    L->low_p_acq = 1;
    return 0;
}

int zm_mcsp_release_c(zm_mcsp_t *L, zm_mcs_qnode_t *I) {
    if (!L->low_p_acq) {
        if (zm_mcs_nowaiters_c(L->high_p, I)) {
            L->go_straight = 0;
            zm_ticket_release(&L->filter);
        }
        zm_mcs_release_c(L->high_p, I);
    } else {
        L->low_p_acq = 0;
        zm_ticket_release(&L->filter);
        zm_mcs_release_c(L->low_p, I);
    }
    return 0;
}

int zm_mcsp_destroy(zm_mcsp_t *L) {
    zm_mcs_destroy(&L->high_p);
    zm_ticket_destroy(&L->filter);
    zm_mcs_destroy(&L->low_p);
    return 0;
}
