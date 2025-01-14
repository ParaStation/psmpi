/*
 * ParaStation
 *
 * Copyright (C) 2006-2019 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#ifndef _MPID_PSP_REQUEST_H_
#define _MPID_PSP_REQUEST_H_

#include <assert.h>

static inline void MPIDI_PSP_Request_set_completed(MPIR_Request * req)
{
    *(req->cc_ptr) = 0;
}

static inline void MPID_PSP_Subrequest_add(MPIR_Request * req)
{
    /* ToDo: should be explicit atomic */
    (*(req->cc_ptr))++;
}

static inline int MPID_PSP_Subrequest_completed(MPIR_Request * req)
{
    /* ToDo: should be explicit atomic */
    (*(req->cc_ptr))--;

    int completed = (*(req->cc_ptr)) == 0;

    struct MPID_DEV_Request_common *dev_req = &req->dev.kind.common;
    if (completed && dev_req->completion_notification) {
        /* decrement the completion notification counter */
        MPIR_cc_dec(dev_req->completion_notification);

        /*release reference of subrequest */
        MPIR_Request_free(req);
    }

    return completed;
}

static inline void MPIDI_PSP_Request_init(MPIR_Request * req, MPIR_Comm * comm)
{
    MPIR_Object_set_ref(req, 1);
    req->cc = 1;
    req->cc_ptr = &req->cc;
    req->comm = comm;
}

static inline void MPIDI_PSP_Request_status_init(MPIR_Request * req)
{
    req->status.MPI_SOURCE = MPI_UNDEFINED;
    req->status.MPI_TAG = MPI_UNDEFINED;
    req->status.MPI_ERROR = MPI_SUCCESS;

    /* combined MPIR_STATUS_SET_COUNT and MPIR_STATUS_SET_CANCEL_BIT: */
    req->status.count_lo = 0;
    req->status.count_hi_and_cancelled = 0;
}

static inline void MPIDI_PSP_Request_pscom_req_create(MPIR_Request * req)
{
    struct MPID_DEV_Request_common *creq = NULL;

    /* allocate the pscom request */
    creq = &req->dev.kind.common;
    creq->pscom_req = PSCOM_REQUEST_CREATE();
    creq->pscom_req->user->type.sr.mpid_req = req;
}

static inline void MPIDI_PSP_Request_pscom_req_init(MPIR_Request * req)
{
    pscom_request_t *preq = NULL;

    /* initialize the pscom_request_t */
    preq = req->dev.kind.common.pscom_req;
    preq->connection = NULL;
    preq->socket = NULL;
    preq->ops.recv_accept = NULL;
    preq->ops.io_done = NULL;
    preq->xheader_len = 0;
    preq->data_len = 0;
    preq->data = 0;
}
#endif /* _MPID_PSP_REQUEST_H_ */
