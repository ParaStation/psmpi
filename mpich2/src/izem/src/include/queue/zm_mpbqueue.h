/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef _ZM_MPBQUEUE_H
#define _ZM_MPBQUEUE_H
#include <stdlib.h>
#include <stdio.h>
#include "queue/zm_queue_types.h"

/* mpbqueue: MPB: Multiple Producer Bucket queue. Concurrent queue where both enqueue and dequeue operations
 * are protected with the same global lock (thus, the gl prefix) */

int zm_mpbqueue_init(struct zm_mpbqueue *, int);
int zm_mpbqueue_enqueue(struct zm_mpbqueue* q, void *data, int);
int zm_mpbqueue_dequeue(struct zm_mpbqueue* q, void **data);
int zm_mpbqueue_dequeue_bulk(struct zm_mpbqueue* q, void*[], int, int*);
int zm_mpbqueue_dequeue_range(struct zm_mpbqueue* q, void*[], int, int, int, int*);

#endif /* _ZM_MPBQUEUE_H */
