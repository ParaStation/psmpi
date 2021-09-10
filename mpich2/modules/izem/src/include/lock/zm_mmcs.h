/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef _ZM_MMCS_H
#define _ZM_MMCS_H
#include "lock/zm_lock_types.h"

int zm_mmcs_init(zm_mmcs_t *);
int zm_mmcs_destroy(zm_mmcs_t *);
int zm_mmcs_acquire(zm_mmcs_t *, zm_mcs_qnode_t*);
int zm_mmcs_release(zm_mmcs_t *, zm_mcs_qnode_t**);
int zm_mmcs_nowaiters(zm_mmcs_t *);

#endif /* _ZM_MMCS_H */
