/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef _ZM_COND_TYPES_H
#define _ZM_COND_TYPES_H

#include "common/zm_common.h"
#include "lock/zm_lock.h"

#define ZM_COND_CLEAR 0
#define ZM_COND_WAIT  1


struct zm_ccond {
    zm_atomic_uint_t flag;
};

struct zm_scount {
    zm_atomic_uint_t count;
    struct zm_ccond cvar;
};


#endif /* _IZEM_COND_TYPES_H */
