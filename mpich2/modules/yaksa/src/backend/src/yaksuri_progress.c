/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <pthread.h>
#include "yaksa.h"
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri.h"
#include "yutlist.h"

static yaksuri_request_s *pending_reqs = NULL;
static pthread_mutex_t progress_mutex = PTHREAD_MUTEX_INITIALIZER;

static int icopy(yaksuri_gpudriver_id_e id, const void *inbuf, void *outbuf, uintptr_t bytes,
                 yaksi_info_s * info, int device)
{
    int rc = YAKSA_SUCCESS;

    yaksi_type_s *byte_type;
    rc = yaksi_type_get(YAKSA_TYPE__BYTE, &byte_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksuri_global.gpudriver[id].hooks->ipack(inbuf, outbuf, bytes, byte_type, info, device);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int ipack(yaksuri_gpudriver_id_e id, const void *inbuf, void *outbuf, uintptr_t count,
                 yaksi_type_s * type, yaksi_info_s * info, int device)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].hooks->ipack(inbuf, outbuf, count, type, info, device);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int iunpack(yaksuri_gpudriver_id_e id, const void *inbuf, void *outbuf, uintptr_t count,
                   yaksi_type_s * type, yaksi_info_s * info, int device)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].hooks->iunpack(inbuf, outbuf, count, type, info, device);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int check_p2p_comm(yaksuri_gpudriver_id_e id, int indev, int outdev, bool * is_enabled)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].hooks->check_p2p_comm(indev, outdev, is_enabled);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int event_record(yaksuri_gpudriver_id_e id, int device, void **event)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].hooks->event_record(device, event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int event_query(yaksuri_gpudriver_id_e id, void *event, int *completed)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].hooks->event_query(event, completed);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int add_dependency(yaksuri_gpudriver_id_e id, int device1, int device2)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].hooks->add_dependency(device1, device2);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int alloc_chunk(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                       yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;

    assert(subreq);
    assert(subreq->kind == YAKSURI_SUBREQ_KIND__MULTI_CHUNK);

    /* allocate the chunk */
    *chunk = (yaksuri_subreq_chunk_s *) malloc(sizeof(yaksuri_subreq_chunk_s));

    (*chunk)->count_offset = subreq->u.multiple.issued_count;
    uintptr_t count_per_chunk = YAKSURI_TMPBUF_EL_SIZE / subreq->u.multiple.type->size;
    if ((*chunk)->count_offset + count_per_chunk <= subreq->u.multiple.count) {
        (*chunk)->count = count_per_chunk;
    } else {
        (*chunk)->count = subreq->u.multiple.count - (*chunk)->count_offset;
    }

    (*chunk)->event = NULL;

    DL_APPEND(subreq->u.multiple.chunks, (*chunk));

    return rc;
}

static int simple_release(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                          yaksuri_subreq_chunk_s * chunk)
{
    int rc = YAKSA_SUCCESS;

    /* cleanup */
    for (int i = 0; i < chunk->num_tmpbufs; i++) {
        rc = yaksu_buffer_pool_elem_free(chunk->tmpbufs[i].pool, chunk->tmpbufs[i].buf);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    DL_DELETE(subreq->u.multiple.chunks, chunk);
    free(chunk);

    if (subreq->u.multiple.chunks == NULL) {
        DL_DELETE(reqpriv->subreqs, subreq);
        yaksi_type_free(subreq->u.multiple.type);
        free(subreq);
    }
    if (reqpriv->subreqs == NULL) {
        HASH_DEL(pending_reqs, reqpriv);
        yaksu_atomic_decr(&reqpriv->request->cc);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_d2d_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                            yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    assert(reqpriv->request->backend.inattr.device != reqpriv->request->backend.outattr.device);

    *chunk = NULL;

    bool is_enabled;
    rc = check_p2p_comm(id, reqpriv->request->backend.inattr.device,
                        reqpriv->request->backend.outattr.device, &is_enabled);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (is_enabled) {
        /* p2p is enabled: we need a temporary buffer on the source device */
        void *d_buf;
        rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.
                                          gpudriver[id].device[reqpriv->request->backend.inattr.
                                                               device], &d_buf);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (d_buf == NULL)
            goto fn_exit;

        /* we have the temporary buffer, so we can safely issue this
         * operation */
        rc = alloc_chunk(reqpriv, subreq, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);

        (*chunk)->num_tmpbufs = 1;
        (*chunk)->tmpbufs[0].buf = d_buf;
        (*chunk)->tmpbufs[0].pool =
            yaksuri_global.gpudriver[id].device[reqpriv->request->backend.inattr.device];

        /* first pack data from the origin buffer into the temporary buffer */
        const char *sbuf =
            (const char *) subreq->u.multiple.inbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->extent;

        rc = ipack(id, sbuf, d_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
                   reqpriv->request->backend.inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        /* second copy the data into the target device */
        char *dbuf =
            (char *) subreq->u.multiple.outbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->size;

        rc = icopy(id, d_buf, dbuf, (*chunk)->count * subreq->u.multiple.type->size, reqpriv->info,
                   reqpriv->request->backend.inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_record(id, reqpriv->request->backend.inattr.device, &(*chunk)->event);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        /* p2p is not enabled: we need two temporary buffers, one on
         * the source device and one on the host */
        void *d_buf, *rh_buf;

        rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.
                                          gpudriver[id].device[reqpriv->request->backend.inattr.
                                                               device], &d_buf);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (d_buf == NULL)
            goto fn_exit;

        rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].host, &rh_buf);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (rh_buf == NULL) {
            if (d_buf) {
                rc = yaksu_buffer_pool_elem_free(yaksuri_global.
                                                 gpudriver[id].device[reqpriv->request->backend.
                                                                      inattr.device], d_buf);
                YAKSU_ERR_CHECK(rc, fn_fail);
            }
            goto fn_exit;
        }

        /* we have the temporary buffers, so we can safely issue this
         * operation */
        rc = alloc_chunk(reqpriv, subreq, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);

        (*chunk)->num_tmpbufs = 2;
        (*chunk)->tmpbufs[0].buf = d_buf;
        (*chunk)->tmpbufs[0].pool =
            yaksuri_global.gpudriver[id].device[reqpriv->request->backend.inattr.device];
        (*chunk)->tmpbufs[1].buf = rh_buf;
        (*chunk)->tmpbufs[1].pool = yaksuri_global.gpudriver[id].host;

        /* first pack data from the origin buffer into the temporary buffer */
        const char *sbuf =
            (const char *) subreq->u.multiple.inbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->extent;

        rc = ipack(id, sbuf, d_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
                   reqpriv->request->backend.inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        /* second copy the data into the temporary host buffer */
        rc = icopy(id, d_buf, rh_buf, (*chunk)->count * subreq->u.multiple.type->size,
                   reqpriv->info, reqpriv->request->backend.inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        /* third DMA from the host temporary buffer to the target device */
        rc = add_dependency(id, reqpriv->request->backend.inattr.device,
                            reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        char *dbuf =
            (char *) subreq->u.multiple.outbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->size;

        rc = icopy(id, rh_buf, dbuf, (*chunk)->count * subreq->u.multiple.type->size,
                   reqpriv->info, reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_record(id, reqpriv->request->backend.outattr.device, &(*chunk)->event);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_d2rh_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                             yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    *chunk = NULL;

    /* we need a temporary buffer on the source device */
    void *d_buf;
    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.
                                      gpudriver[id].device[reqpriv->request->backend.inattr.device],
                                      &d_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (d_buf == NULL)
        goto fn_exit;

    /* we have the temporary buffer, so we can safely issue this
     * operation */
    rc = alloc_chunk(reqpriv, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*chunk)->num_tmpbufs = 1;
    (*chunk)->tmpbufs[0].buf = d_buf;
    (*chunk)->tmpbufs[0].pool =
        yaksuri_global.gpudriver[id].device[reqpriv->request->backend.inattr.device];

    /* first pack data from the origin buffer into the temporary buffer */
    const char *sbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    rc = ipack(id, sbuf, d_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
               reqpriv->request->backend.inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* second copy the data into the destination buffer */
    char *dbuf;
    dbuf =
        (char *) subreq->u.multiple.outbuf + (*chunk)->count_offset * subreq->u.multiple.type->size;

    rc = icopy(id, d_buf, dbuf, (*chunk)->count * subreq->u.multiple.type->size, reqpriv->info,
               reqpriv->request->backend.inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = event_record(id, reqpriv->request->backend.inattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_d2urh_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                              yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    *chunk = NULL;

    /* we need two temporary buffers, one on the source device and one
     * on the host */
    void *d_buf, *rh_buf;

    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.
                                      gpudriver[id].device[reqpriv->request->backend.inattr.device],
                                      &d_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (d_buf == NULL)
        goto fn_exit;

    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].host, &rh_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (rh_buf == NULL) {
        if (d_buf) {
            rc = yaksu_buffer_pool_elem_free(yaksuri_global.
                                             gpudriver[id].device[reqpriv->request->backend.inattr.
                                                                  device], d_buf);
            YAKSU_ERR_CHECK(rc, fn_fail);
        }
        goto fn_exit;
    }

    /* we have the temporary buffers, so we can safely issue this
     * operation */
    rc = alloc_chunk(reqpriv, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*chunk)->num_tmpbufs = 2;
    (*chunk)->tmpbufs[0].buf = d_buf;
    (*chunk)->tmpbufs[0].pool =
        yaksuri_global.gpudriver[id].device[reqpriv->request->backend.inattr.device];
    (*chunk)->tmpbufs[1].buf = rh_buf;
    (*chunk)->tmpbufs[1].pool = yaksuri_global.gpudriver[id].host;

    /* first pack data from the origin buffer into the temporary buffer */
    const char *sbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    rc = ipack(id, sbuf, d_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
               reqpriv->request->backend.inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* second copy the data into the temporary host buffer */
    rc = icopy(id, d_buf, rh_buf, (*chunk)->count * subreq->u.multiple.type->size,
               reqpriv->info, reqpriv->request->backend.inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = event_record(id, reqpriv->request->backend.inattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_d2urh_release(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                              yaksuri_subreq_chunk_s * chunk)
{
    int rc = YAKSA_SUCCESS;

    yaksi_type_s *byte_type;
    rc = yaksi_type_get(YAKSA_TYPE__BYTE, &byte_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    char *dbuf;
    dbuf = (char *) subreq->u.multiple.outbuf + chunk->count_offset * subreq->u.multiple.type->size;
    rc = yaksuri_seq_ipack(chunk->tmpbufs[1].buf, dbuf,
                           chunk->count * subreq->u.multiple.type->size, byte_type, reqpriv->info);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = simple_release(reqpriv, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_h2d_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                            yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    *chunk = NULL;

    /* we need a host temporary buffer */
    void *rh_buf;

    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].host, &rh_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (rh_buf == NULL)
        goto fn_exit;

    /* we have the temporary buffers, so we can safely issue this
     * operation */
    rc = alloc_chunk(reqpriv, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*chunk)->num_tmpbufs = 1;
    (*chunk)->tmpbufs[0].buf = rh_buf;
    (*chunk)->tmpbufs[0].pool = yaksuri_global.gpudriver[id].host;

    /* first pack data from the origin buffer into the temporary buffer */
    const char *sbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    rc = yaksuri_seq_ipack(sbuf, rh_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* second copy the data into the target device */
    char *dbuf;
    dbuf =
        (char *) subreq->u.multiple.outbuf + (*chunk)->count_offset * subreq->u.multiple.type->size;

    rc = icopy(id, rh_buf, dbuf, (*chunk)->count * subreq->u.multiple.type->size, reqpriv->info,
               reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = event_record(id, reqpriv->request->backend.outattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_d2d_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                              yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    assert(reqpriv->request->backend.inattr.device != reqpriv->request->backend.outattr.device);

    *chunk = NULL;

    bool is_enabled;
    rc = check_p2p_comm(id, reqpriv->request->backend.inattr.device,
                        reqpriv->request->backend.outattr.device, &is_enabled);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (is_enabled) {
        /* p2p is enabled: we need a temporary buffer on the destination device */
        void *d_buf;
        rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.
                                          gpudriver[id].device[reqpriv->request->backend.outattr.
                                                               device], &d_buf);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (d_buf == NULL)
            goto fn_exit;

        /* we have the temporary buffer, so we can safely issue this
         * operation */
        rc = alloc_chunk(reqpriv, subreq, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);

        (*chunk)->num_tmpbufs = 1;
        (*chunk)->tmpbufs[0].buf = d_buf;
        (*chunk)->tmpbufs[0].pool =
            yaksuri_global.gpudriver[id].device[reqpriv->request->backend.outattr.device];

        /* first copy the data from the origin buffer into the
         * temporary buffer */
        const char *sbuf;
        sbuf = (const char *) subreq->u.multiple.inbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->size;

        rc = icopy(id, sbuf, d_buf, (*chunk)->count * subreq->u.multiple.type->size, reqpriv->info,
                   reqpriv->request->backend.inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        /* second unpack the data into the destination buffer */
        rc = add_dependency(id, reqpriv->request->backend.inattr.device,
                            reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        char *dbuf;
        dbuf = (char *) subreq->u.multiple.outbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->extent;

        rc = iunpack(id, d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
                     reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_record(id, reqpriv->request->backend.outattr.device, &(*chunk)->event);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        /* p2p is not enabled: we need two temporary buffers, one on
         * the destination device and one on the host */
        void *d_buf, *rh_buf;

        rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.
                                          gpudriver[id].device[reqpriv->request->backend.outattr.
                                                               device], &d_buf);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (d_buf == NULL)
            goto fn_exit;

        rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].host, &rh_buf);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (rh_buf == NULL) {
            if (d_buf) {
                rc = yaksu_buffer_pool_elem_free(yaksuri_global.
                                                 gpudriver[id].device[reqpriv->request->backend.
                                                                      outattr.device], d_buf);
                YAKSU_ERR_CHECK(rc, fn_fail);
            }
            goto fn_exit;
        }

        /* we have the temporary buffers, so we can safely issue this
         * operation */
        rc = alloc_chunk(reqpriv, subreq, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);

        (*chunk)->num_tmpbufs = 2;
        (*chunk)->tmpbufs[0].buf = d_buf;
        (*chunk)->tmpbufs[0].pool =
            yaksuri_global.gpudriver[id].device[reqpriv->request->backend.outattr.device];
        (*chunk)->tmpbufs[1].buf = rh_buf;
        (*chunk)->tmpbufs[1].pool = yaksuri_global.gpudriver[id].host;

        /* first copy data from the origin buffer into the temporary host buffer */
        const char *sbuf;
        sbuf = (const char *) subreq->u.multiple.inbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->size;

        rc = icopy(id, sbuf, rh_buf, (*chunk)->count * subreq->u.multiple.type->size, reqpriv->info,
                   reqpriv->request->backend.inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        /* second copy the data from the temporary host buffer into the
         * temporary destination device buffer */
        rc = add_dependency(id, reqpriv->request->backend.inattr.device,
                            reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = icopy(id, rh_buf, d_buf, (*chunk)->count * subreq->u.multiple.type->size,
                   reqpriv->info, reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        /* third unpack from the temporary device buffer to the destination buffer */
        char *dbuf;
        dbuf = (char *) subreq->u.multiple.outbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->extent;

        rc = iunpack(id, d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type,
                     reqpriv->info, reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_record(id, reqpriv->request->backend.outattr.device, &(*chunk)->event);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_rh2d_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                               yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    *chunk = NULL;

    /* we need a temporary buffer on the destination device */
    void *d_buf;
    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.
                                      gpudriver[id].device[reqpriv->request->backend.outattr.
                                                           device], &d_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (d_buf == NULL)
        goto fn_exit;

    /* we have the temporary buffer, so we can safely issue this
     * operation */
    rc = alloc_chunk(reqpriv, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*chunk)->num_tmpbufs = 1;
    (*chunk)->tmpbufs[0].buf = d_buf;
    (*chunk)->tmpbufs[0].pool =
        yaksuri_global.gpudriver[id].device[reqpriv->request->backend.outattr.device];

    /* first copy the data from the origin buffer into the temporary
     * device buffer */
    const char *sbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->size;

    rc = icopy(id, sbuf, d_buf, (*chunk)->count * subreq->u.multiple.type->size, reqpriv->info,
               reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* second unpack the data into the destination buffer */
    char *dbuf;
    dbuf = (char *) subreq->u.multiple.outbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    rc = iunpack(id, d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
                 reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = event_record(id, reqpriv->request->backend.outattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_urh2d_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                                yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    *chunk = NULL;

    /* we need two temporary buffers, one on the destination device
     * and one on the host */
    void *d_buf, *rh_buf;

    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.
                                      gpudriver[id].device[reqpriv->request->backend.outattr.
                                                           device], &d_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (d_buf == NULL)
        goto fn_exit;

    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].host, &rh_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (rh_buf == NULL) {
        if (d_buf) {
            rc = yaksu_buffer_pool_elem_free(yaksuri_global.
                                             gpudriver[id].device[reqpriv->request->backend.outattr.
                                                                  device], d_buf);
            YAKSU_ERR_CHECK(rc, fn_fail);
        }
        goto fn_exit;
    }

    /* we have the temporary buffer, so we can safely issue this
     * operation */
    rc = alloc_chunk(reqpriv, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*chunk)->num_tmpbufs = 2;
    (*chunk)->tmpbufs[0].buf = d_buf;
    (*chunk)->tmpbufs[0].pool =
        yaksuri_global.gpudriver[id].device[reqpriv->request->backend.outattr.device];
    (*chunk)->tmpbufs[1].buf = rh_buf;
    (*chunk)->tmpbufs[1].pool = yaksuri_global.gpudriver[id].host;

    /* first copy the data into a temporary host buffer */
    const char *sbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->size;

    yaksi_type_s *byte_type;
    rc = yaksi_type_get(YAKSA_TYPE__BYTE, &byte_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksuri_seq_ipack(sbuf, rh_buf, (*chunk)->count * subreq->u.multiple.type->size,
                           byte_type, reqpriv->info);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* second copy the data from the origin buffer into the temporary
     * buffer */
    rc = icopy(id, rh_buf, d_buf, (*chunk)->count * subreq->u.multiple.type->size, reqpriv->info,
               reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* third unpack the data into the destination buffer */
    char *dbuf;
    dbuf = (char *) subreq->u.multiple.outbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    rc = iunpack(id, d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
                 reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = event_record(id, reqpriv->request->backend.outattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_d2h_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                              yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    *chunk = NULL;

    /* we need a temporary buffer on the host */
    void *rh_buf;
    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].host, &rh_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (rh_buf == NULL)
        goto fn_exit;

    /* we have the temporary buffer, so we can safely issue this
     * operation */
    rc = alloc_chunk(reqpriv, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*chunk)->num_tmpbufs = 1;
    (*chunk)->tmpbufs[0].buf = rh_buf;
    (*chunk)->tmpbufs[0].pool = yaksuri_global.gpudriver[id].host;

    /* first copy the data from the origin buffer into the temporary
     * host buffer */
    const char *sbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->size;

    rc = icopy(id, sbuf, rh_buf, (*chunk)->count * subreq->u.multiple.type->size, reqpriv->info,
               reqpriv->request->backend.inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = event_record(id, reqpriv->request->backend.inattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_d2h_release(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                              yaksuri_subreq_chunk_s * chunk)
{
    int rc = YAKSA_SUCCESS;

    char *dbuf;
    dbuf =
        (char *) subreq->u.multiple.outbuf + chunk->count_offset * subreq->u.multiple.type->extent;
    rc = yaksuri_seq_iunpack(chunk->tmpbufs[0].buf, dbuf, chunk->count, subreq->u.multiple.type,
                             reqpriv->info);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = simple_release(reqpriv, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_progress_enqueue(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                             yaksi_info_s * info, yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_request_s *reqpriv = (yaksuri_request_s *) request->backend.priv;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    assert(yaksuri_global.gpudriver[id].hooks);
    assert(request->backend.inattr.type == YAKSUR_PTR_TYPE__GPU ||
           request->backend.outattr.type == YAKSUR_PTR_TYPE__GPU);

    reqpriv->info = info;

    /* if the GPU reqpriv cannot support this type, return */
    bool is_supported;
    rc = yaksuri_global.gpudriver[id].hooks->pup_is_supported(type, &is_supported);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (!is_supported) {
        rc = YAKSA_ERR__NOT_SUPPORTED;
        goto fn_exit;
    }

    yaksuri_subreq_s *subreq;
    subreq = (yaksuri_subreq_s *) malloc(sizeof(yaksuri_subreq_s));

    int (*pupfn) (yaksuri_gpudriver_id_e id, const void *inbuf, void *outbuf, uintptr_t count,
                  yaksi_type_s * type, yaksi_info_s * info, int device);
    if (reqpriv->optype == YAKSURI_OPTYPE__PACK) {
        pupfn = ipack;
    } else {
        pupfn = iunpack;
    }

    uintptr_t threshold;
    if (reqpriv->optype == YAKSURI_OPTYPE__PACK) {
        threshold = yaksuri_global.gpudriver[id].hooks->get_iov_pack_threshold(info);
    } else {
        threshold = yaksuri_global.gpudriver[id].hooks->get_iov_unpack_threshold(info);
    }

    if (request->backend.inattr.type == YAKSUR_PTR_TYPE__GPU) {
        if (request->backend.outattr.type == YAKSUR_PTR_TYPE__GPU &&
            request->backend.inattr.device == request->backend.outattr.device) {

            subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
            rc = pupfn(id, inbuf, outbuf, count, type, info, request->backend.inattr.device);
            YAKSU_ERR_CHECK(rc, fn_fail);

            rc = event_record(id, request->backend.inattr.device, &subreq->u.single.event);
            YAKSU_ERR_CHECK(rc, fn_fail);

            goto enqueue_subreq;
        }

        if (request->backend.outattr.type == YAKSUR_PTR_TYPE__MANAGED) {

            subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
            rc = pupfn(id, inbuf, outbuf, count, type, info, request->backend.inattr.device);
            YAKSU_ERR_CHECK(rc, fn_fail);

            rc = event_record(id, request->backend.inattr.device, &subreq->u.single.event);
            YAKSU_ERR_CHECK(rc, fn_fail);

            goto enqueue_subreq;
        }

        if (request->backend.outattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST &&
            (type->is_contig || type->size / type->num_contig >= threshold)) {

            subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
            rc = pupfn(id, inbuf, outbuf, count, type, info, request->backend.inattr.device);
            YAKSU_ERR_CHECK(rc, fn_fail);

            rc = event_record(id, request->backend.inattr.device, &subreq->u.single.event);
            YAKSU_ERR_CHECK(rc, fn_fail);

            goto enqueue_subreq;
        }
    } else if (request->backend.inattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) {

        if (request->backend.outattr.type == YAKSUR_PTR_TYPE__GPU &&
            (type->is_contig || type->size / type->num_contig >= threshold)) {

            subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
            rc = pupfn(id, inbuf, outbuf, count, type, info, request->backend.outattr.device);
            YAKSU_ERR_CHECK(rc, fn_fail);

            rc = event_record(id, request->backend.outattr.device, &subreq->u.single.event);
            YAKSU_ERR_CHECK(rc, fn_fail);

            goto enqueue_subreq;
        }
    } else if (request->backend.inattr.type == YAKSUR_PTR_TYPE__MANAGED) {

        if (request->backend.outattr.type == YAKSUR_PTR_TYPE__GPU) {

            subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
            rc = pupfn(id, inbuf, outbuf, count, type, info, request->backend.outattr.device);
            YAKSU_ERR_CHECK(rc, fn_fail);

            rc = event_record(id, request->backend.outattr.device, &subreq->u.single.event);
            YAKSU_ERR_CHECK(rc, fn_fail);

            goto enqueue_subreq;
        }
    }

    /* we can only take on types where at least one count of the type
     * fits into our temporary buffers. */
    if (type->size > YAKSURI_TMPBUF_EL_SIZE) {
        rc = YAKSA_ERR__NOT_SUPPORTED;
        free(subreq);
        goto fn_exit;
    }

    subreq->kind = YAKSURI_SUBREQ_KIND__MULTI_CHUNK;

    subreq->u.multiple.inbuf = inbuf;
    subreq->u.multiple.outbuf = outbuf;
    subreq->u.multiple.count = count;
    subreq->u.multiple.type = type;
    subreq->u.multiple.issued_count = 0;
    subreq->u.multiple.chunks = NULL;

    yaksu_atomic_incr(&type->refcount);

    if (reqpriv->optype == YAKSURI_OPTYPE__PACK) {
        if (request->backend.inattr.type == YAKSUR_PTR_TYPE__GPU &&
            request->backend.outattr.type == YAKSUR_PTR_TYPE__GPU) {
            subreq->u.multiple.acquire = pack_d2d_acquire;
            subreq->u.multiple.release = simple_release;
        } else if (request->backend.inattr.type == YAKSUR_PTR_TYPE__GPU) {
            if (request->backend.outattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) {
                subreq->u.multiple.acquire = pack_d2rh_acquire;
                subreq->u.multiple.release = simple_release;
            } else {
                subreq->u.multiple.acquire = pack_d2urh_acquire;
                subreq->u.multiple.release = pack_d2urh_release;
            }
        } else if (request->backend.outattr.type == YAKSUR_PTR_TYPE__GPU) {
            subreq->u.multiple.acquire = pack_h2d_acquire;
            subreq->u.multiple.release = simple_release;
        }
    } else {
        if (request->backend.inattr.type == YAKSUR_PTR_TYPE__GPU &&
            request->backend.outattr.type == YAKSUR_PTR_TYPE__GPU) {
            subreq->u.multiple.acquire = unpack_d2d_acquire;
            subreq->u.multiple.release = simple_release;
        } else if (request->backend.inattr.type == YAKSUR_PTR_TYPE__GPU) {
            subreq->u.multiple.acquire = unpack_d2h_acquire;
            subreq->u.multiple.release = unpack_d2h_release;
        } else if (request->backend.outattr.type == YAKSUR_PTR_TYPE__GPU) {
            if (request->backend.inattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) {
                subreq->u.multiple.acquire = unpack_rh2d_acquire;
                subreq->u.multiple.release = simple_release;
            } else {
                subreq->u.multiple.acquire = unpack_urh2d_acquire;
                subreq->u.multiple.release = simple_release;
            }
        }
    }

  enqueue_subreq:
    pthread_mutex_lock(&progress_mutex);
    DL_APPEND(reqpriv->subreqs, subreq);

    /* if the request is not in our pending list, add it */
    yaksuri_request_s *req;
    HASH_FIND_PTR(pending_reqs, &request, req);
    if (req == NULL) {
        HASH_ADD_PTR(pending_reqs, request, reqpriv);
        yaksu_atomic_incr(&request->cc);
    }
    pthread_mutex_unlock(&progress_mutex);

    rc = yaksuri_progress_poke();
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_progress_poke(void)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id;

    /* A progress poke is in two steps.  In the first step, we check
     * for event completions, finish any post-processing and retire
     * any temporary resources.  In the second steps, we issue out any
     * pending operations. */

    pthread_mutex_lock(&progress_mutex);

    yaksi_type_s *byte_type;
    rc = yaksi_type_get(YAKSA_TYPE__BYTE, &byte_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /**********************************************************************/
    /* Step 1: Check for completions */
    /**********************************************************************/
    yaksuri_request_s *reqpriv, *tmp;
    HASH_ITER(hh, pending_reqs, reqpriv, tmp) {
        id = reqpriv->gpudriver_id;
        assert(reqpriv->subreqs);

        yaksuri_subreq_s *subreq, *tmp2;
        DL_FOREACH_SAFE(reqpriv->subreqs, subreq, tmp2) {
            if (subreq->kind == YAKSURI_SUBREQ_KIND__SINGLE_CHUNK) {
                int completed;
                rc = event_query(id, subreq->u.single.event, &completed);
                YAKSU_ERR_CHECK(rc, fn_fail);

                if (!completed)
                    continue;

                DL_DELETE(reqpriv->subreqs, subreq);
                free(subreq);
                if (reqpriv->subreqs == NULL) {
                    HASH_DEL(pending_reqs, reqpriv);
                    yaksu_atomic_decr(&reqpriv->request->cc);
                }
            } else {
                yaksuri_subreq_chunk_s *chunk, *tmp3;
                DL_FOREACH_SAFE(subreq->u.multiple.chunks, chunk, tmp3) {
                    int completed;
                    rc = event_query(id, chunk->event, &completed);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    if (!completed)
                        continue;

                    rc = subreq->u.multiple.release(reqpriv, subreq, chunk);
                    YAKSU_ERR_CHECK(rc, fn_fail);
                }
            }
        }
    }

    /**********************************************************************/
    /* Step 2: Issue new operations */
    /**********************************************************************/
    HASH_ITER(hh, pending_reqs, reqpriv, tmp) {
        id = reqpriv->gpudriver_id;
        assert(reqpriv->subreqs);

        yaksuri_subreq_s *subreq, *tmp2;
        DL_FOREACH_SAFE(reqpriv->subreqs, subreq, tmp2) {
            if (subreq->kind == YAKSURI_SUBREQ_KIND__SINGLE_CHUNK)
                continue;

            while (subreq->u.multiple.issued_count < subreq->u.multiple.count) {
                yaksuri_subreq_chunk_s *chunk;

                rc = subreq->u.multiple.acquire(reqpriv, subreq, &chunk);
                YAKSU_ERR_CHECK(rc, fn_fail);

                if (chunk == NULL)
                    goto fn_exit;

                subreq->u.multiple.issued_count += chunk->count;
            }
        }
    }

  fn_exit:
    pthread_mutex_unlock(&progress_mutex);
    return rc;
  fn_fail:
    goto fn_exit;
}
