/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */
#include "zm_mcs.h"
#include "zm_ticket.h"

int zm_mcsp_init(zm_mcsp_t *);
int zm_mcsp_destroy(zm_mcsp_t *);

int zm_mcsp_acquire(zm_mcsp_t *);
int zm_mcsp_acquire_low(zm_mcsp_t*);
int zm_mcsp_tryacq(zm_mcsp_t *, int*);
int zm_mcsp_tryacq_low(zm_mcsp_t*, int*);
int zm_mcsp_release(zm_mcsp_t *);

int zm_mcsp_acquire_c(zm_mcsp_t *, zm_mcs_qnode_t*);
int zm_mcsp_acquire_low_c(zm_mcsp_t*, zm_mcs_qnode_t*);
int zm_mcsp_release_c(zm_mcsp_t *, zm_mcs_qnode_t *);
