/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef _ZM_SCOND_H
#define _ZM_SCOND_H

#include "cond/zm_cond_types.h"

int zm_scount_init(struct zm_scount *C, int);
int zm_scount_destroy(struct zm_scount *C);
int zm_scount_wait(struct zm_scount *C, zm_lock_t *L);
int zm_scount_signal(struct zm_scount *C, int*);
int zm_scount_signalf(struct zm_scount *C);

#endif /* _ZM_SCOND_H */
