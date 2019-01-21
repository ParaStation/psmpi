/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef _ZM_WSKIP_H
#define _ZM_WSKIP_H

#include "lock/zm_lock_types.h"

int zm_wskip_init(zm_mcs_t*);
int zm_wskip_destroy(zm_mcs_t*);
int zm_wskip_wait(zm_mcs_t, zm_mcs_qnode_t**);
int zm_wskip_enq(zm_mcs_t, zm_mcs_qnode_t*);
int zm_wskip_wake(zm_mcs_t, zm_mcs_qnode_t*);
int zm_wskip_skip(zm_mcs_qnode_t*);
int zm_wskip_nowaiters(zm_mcs_t, zm_mcs_qnode_t*);

#endif /* _ZM_WSKIP_H */
