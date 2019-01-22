/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "queue/zm_faqueue.h"
#include "mem/zm_hzdptr.h"
#include <assert.h>

/*
     Helper functions
 */

static inline zm_faseg_t *zm_faseg_alloc(zm_ulong_t seg_id) {
    int i;
    zm_faseg_t *seg = malloc (sizeof(zm_faseg_t));
    seg->id = seg_id;
    zm_atomic_store(&seg->next, ZM_NULL, zm_memord_release);
    for(i = 0; i < ZM_MAX_FASEG_SIZE; i++)
        seg->cells[i].data = ZM_FAQUEUE_ALPHA;
    return seg;
}

static inline void zm_faseg_free(zm_faseg_t *seg) {
    zm_hzdptr_t *hzdptrs = zm_hzdptr_get();
    hzdptrs[0] <- (zm_ptr_t)seg;
    zm_hzdptr_retire((zm_ptr_t)seg);
}

static inline zm_faseg_t *zm_faseg_advance(zm_faseg_t *seg, zm_ulong_t seg_id){
    assert(seg->id == seg_id-1);
    zm_hzdptr_t *hzdptrs = zm_hzdptr_get();
    zm_faseg_t *next =  (zm_faseg_t*) zm_atomic_load(&seg->next,
                                                           zm_memord_acquire);
    hzdptrs[0] = (zm_ptr_t)next;
    if((zm_ptr_t)next == ZM_NULL) {
        zm_faseg_t *tmp_seg = zm_faseg_alloc (seg_id);
        if (!zm_atomic_compare_exchange_strong(&seg->next,
                                               (zm_ptr_t*)&next,
                                               (zm_ptr_t)tmp_seg,
                                               zm_memord_acq_rel,
                                               zm_memord_acquire))
            zm_faseg_free(tmp_seg);

        next = (zm_faseg_t*) zm_atomic_load(&seg->next,
                                                zm_memord_acquire);
    }
    assert(next->id == seg_id);
    return next;
}

/* Search, and allocate a segment if necessary, the cell that cell_id belongs to*/
static inline zm_facell_t *zm_facell_find(zm_faseg_t *seg, zm_ulong_t cell_id) {
    zm_faseg_t *cur_seg = seg;
    zm_ulong_t trg_seg = cell_id/ZM_MAX_FASEG_SIZE;

    for (zm_ulong_t i = cur_seg->id; i < trg_seg; i++)
        cur_seg = zm_faseg_advance (cur_seg, i + 1);
    assert(cur_seg->id == trg_seg);

    return &cur_seg->cells[cell_id % ZM_MAX_FASEG_SIZE];
}

/*
   Body of the routines
 */

int zm_faqueue_init(zm_faqueue_t *q) {
    q->head = 0;
    zm_atomic_store(&q->tail, 0, zm_memord_release);
    q->seg_head = (zm_ptr_t)(void *)zm_faseg_alloc(0);
    zm_atomic_store(&q->seg_tail, q->seg_head, zm_memord_release);
    return 0;
}

int zm_faqueue_enqueue(zm_faqueue_t* q, void *data) {
    zm_ulong_t cell_id = zm_atomic_fetch_add(&q->tail, 1, zm_memord_acq_rel);
    zm_facell_t *cell = zm_facell_find ((zm_faseg_t*)q->seg_head, cell_id);
    cell->data = data;
    return 0;
}

int zm_faqueue_dequeue(zm_faqueue_t* q, void **data) {
    *data = NULL;
    zm_ulong_t cell_disp = q->head % ZM_MAX_FASEG_SIZE;
    zm_ulong_t seg_id = q->head/ ZM_MAX_FASEG_SIZE;
    zm_faseg_t *seg_head = (zm_faseg_t*)q->seg_head;
    zm_faseg_t *next =  (zm_faseg_t*) zm_atomic_load(&seg_head->next,
                                                     zm_memord_acquire);
    /* q empty or next segment not yet allocated by enqueuers */
    if ((seg_id > seg_head->id) && (next == NULL))
        return 0;
    else if (seg_id > seg_head->id) {   /* next segment allocated */
        assert(seg_id == seg_head->id + 1);
        q->seg_head = (zm_ptr_t)next;             /* advance seg_head */
        zm_faseg_free(seg_head);                 /* free no longer used segment */
        seg_head = next;
        assert(seg_id == seg_head->id);
    }

    if ((seg_head->cells[cell_disp].data == ZM_FAQUEUE_ALPHA)) /* general case of empty q */
        return 0;
    else {
        *data = seg_head->cells[cell_disp].data;
        q->head++;
    }
    return 1;
}
