/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef _ZM_TLP_H
#define _ZM_TLP_H
#include "lock/zm_lock_types.h"

#include "lock/zm_ticket.h"
#if (ZM_TLP_HIGH_P == ZM_MCS) || (ZM_TLP_LOW_P  == ZM_MCS)
#include "lock/zm_mcs.h"
#endif
#if (ZM_TLP_HIGH_P == ZM_HMCS) || (ZM_TLP_LOW_P  == ZM_HMCS)
#include "lock/zm_hmcs.h"
#endif

int zm_tlp_init(zm_tlp_t *);
int zm_tlp_destroy(zm_tlp_t *);

int zm_tlp_acquire(zm_tlp_t* lock);
int zm_tlp_acquire_low(zm_tlp_t* lock);
int zm_tlp_tryacq(zm_tlp_t* lock, int*);
int zm_tlp_tryacq_low(zm_tlp_t* lock, int*);
int zm_tlp_release(zm_tlp_t* lock);

int zm_tlp_acquire_c(zm_tlp_t* lock, zm_mcs_qnode_t*);
int zm_tlp_acquire_low_c(zm_tlp_t* lock, zm_mcs_qnode_t*);
int zm_tlp_tryacq_c(zm_tlp_t* lock, zm_mcs_qnode_t*, int*);
int zm_tlp_tryacq_low_c(zm_tlp_t* lock, zm_mcs_qnode_t*, int*);
int zm_tlp_release_c(zm_tlp_t* lock, zm_mcs_qnode_t*);

#endif /* _ZM_TLP_H */
