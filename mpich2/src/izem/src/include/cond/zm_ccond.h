/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef _ZM_CCOND_H
#define _ZM_CCOND_H

#include "cond/zm_cond_types.h"

int zm_ccond_init(struct zm_ccond *C);
int zm_ccond_destroy(struct zm_ccond *C);
int zm_ccond_wait(struct zm_ccond *C, zm_lock_t *L);
int zm_ccond_wait_c(struct zm_ccond *C, zm_lock_t *L, zm_lock_ctxt_t *ctxt);
int zm_ccond_signal(struct zm_ccond *C);
int zm_ccond_bcast(struct zm_ccond *C);

#endif /* _ZM_CCOND_H */
