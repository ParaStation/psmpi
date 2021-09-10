/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef _ZM_HMPR_H
#define _ZM_HMPR_H
#include "lock/zm_lock_types.h"

int zm_hmpr_init(struct zm_hmpr *);
int zm_hmpr_destroy(struct zm_hmpr *);
int zm_hmpr_acquire(struct zm_hmpr *, struct zm_hmpr_pnode *);
int zm_hmpr_release(struct zm_hmpr *, struct zm_hmpr_pnode *);
int zm_hmpr_raise_prio(struct zm_hmpr_pnode *);

#endif /* _ZM_HMPR_H */
