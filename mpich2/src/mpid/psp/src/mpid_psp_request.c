/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2024 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"
#include "mpid_psp_datatype.h"
#include "mpid_psp_packed_msg.h"
#include "mpid_psp_request.h"
#include "mpid_debug.h"


/* ------------------------------------------------------------------------- */
/* Soeme request-specific create/destroy hooks                               */
/* ------------------------------------------------------------------------- */
static inline void MPIDI_PSP_Request_persistent_create_hook(MPIR_Request * req)
{
    struct MPID_DEV_Request_persistent *preq = &req->dev.kind.persistent;
    preq->datatype = 0;
    preq->comm = NULL;
}

static inline void MPIDI_PSP_Request_partitioned_create_hook(MPIR_Request * req)
{
    struct MPID_DEV_Request_partitioned *preq = &req->dev.kind.partitioned;
    preq->datatype = 0;
}

static inline void MPIDI_PSP_Request_send_create_hook(MPIR_Request * req)
{
    struct MPID_DEV_Request_send *sreq = &req->dev.kind.send;
    sreq->msg.tmp_buf = NULL;
}

static inline void MPIDI_PSP_Request_send_destroy_hook(MPIR_Request * req)
{
    assert(req->dev.kind.send.msg.tmp_buf == NULL);
}

static inline void MPIDI_PSP_Request_persistent_destroy_hook(MPIR_Request * req)
{
    struct MPID_DEV_Request_persistent *preq = &req->dev.kind.persistent;
    if (preq->datatype) {
        MPID_PSP_Datatype_release(preq->datatype);
    }
    /* TODO: this should be done by the MPIR layer */
//      if (preq->comm) {
//              MPIR_Comm_release(preq->comm);
//      }
}

static inline void MPIDI_PSP_Request_partitioned_destroy_hook(MPIR_Request * req)
{
    struct MPID_DEV_Request_partitioned *preq = &req->dev.kind.partitioned;
    if (preq->datatype) {
        MPID_PSP_Datatype_release(preq->datatype);
    }
}


/* TODO: this should be done by the MPIR layer */
/*
static inline
void MPID_DEV_Request_coll_destroy(MPIR_Request *req)
{
	if (req->comm) {
		MPIR_Comm_release(req->comm);
	}
}
*/


/* ------------------------------------------------------------------------- */
/* Device interface part                                                     */
/* ------------------------------------------------------------------------- */
void MPID_Request_create_hook(MPIR_Request * req)
{
    if (!req)
        return;

    MPIDI_PSP_Request_pscom_req_create(req);

    // MPIDI_PSP_Request_init(req, NULL); // <- assignments all already done in MPIR_Request_create()
    assert(MPIR_Object_get_ref(req) == 1);
    assert(req->cc == 1);
    assert(req->cc_ptr == &req->cc);
    assert(req->comm == NULL);

    MPIDI_PSP_Request_status_init(req);

    MPIDI_PSP_Request_pscom_req_init(req);

/*	req->dev.datatype_ptr = NULL;
	req->dev.cancel_pending = FALSE;
	req->dev.target_win_handle = MPI_WIN_NULL;
	req->dev.source_win_handle = MPI_WIN_NULL;
	req->dev.single_op_opt = 0;
	req->dev.lock_queue_entry = NULL;
	req->dev.dtype_info = NULL;
	req->dev.dataloop = NULL;
	req->dev.rdma_iov_count = 0;
	req->dev.rdma_iov_offset = 0;
*/

    struct MPID_DEV_Request_common *dev_req = &req->dev.kind.common;
    dev_req->completion_notification = NULL;

    /* request-specific initialization */
    switch (req->kind) {
        case MPIR_REQUEST_KIND__SEND:
        case MPIR_REQUEST_KIND__RMA:
            MPIDI_PSP_Request_send_create_hook(req);
            break;
        case MPIR_REQUEST_KIND__PREQUEST_RECV:
        case MPIR_REQUEST_KIND__PREQUEST_SEND:
        case MPIR_REQUEST_KIND__PREQUEST_COLL:
            MPIDI_PSP_Request_persistent_create_hook(req);
            break;
        case MPIR_REQUEST_KIND__PART_RECV:
        case MPIR_REQUEST_KIND__PART_SEND:
            MPIDI_PSP_Request_partitioned_create_hook(req);
            break;
        case MPIR_REQUEST_KIND__RECV:
        case MPIR_REQUEST_KIND__GREQUEST:
        case MPIR_REQUEST_KIND__COLL:
            break;
        case MPIR_REQUEST_KIND__MPROBE:
            {
                struct MPID_DEV_Request_mprobe *mreq = &req->dev.kind.mprobe;
                mreq->mprobe_tag = NULL;
                break;
            }
        case MPIR_REQUEST_KIND__PART:
        case MPIR_REQUEST_KIND__UNDEFINED:
        case MPIR_REQUEST_KIND__LAST:
            assert(0);
            break;
    }

}

void MPID_Request_destroy_hook(MPIR_Request * req)
{
    switch (req->kind) {
        case MPIR_REQUEST_KIND__SEND:
        case MPIR_REQUEST_KIND__RMA:
            MPIDI_PSP_Request_send_destroy_hook(req);
            break;
        case MPIR_REQUEST_KIND__PREQUEST_RECV:
        case MPIR_REQUEST_KIND__PREQUEST_SEND:
        case MPIR_REQUEST_KIND__PREQUEST_COLL:
            MPIDI_PSP_Request_persistent_destroy_hook(req);
            break;
        case MPIR_REQUEST_KIND__PART_RECV:
        case MPIR_REQUEST_KIND__PART_SEND:
            MPIDI_PSP_Request_partitioned_destroy_hook(req);
            break;
        case MPIR_REQUEST_KIND__RECV:
        case MPIR_REQUEST_KIND__COLL:
        case MPIR_REQUEST_KIND__MPROBE:
        case MPIR_REQUEST_KIND__GREQUEST:
            break;
        case MPIR_REQUEST_KIND__PART:
        case MPIR_REQUEST_KIND__UNDEFINED:
        case MPIR_REQUEST_KIND__LAST:
            assert(0);
            break;
    }

    struct MPID_DEV_Request_common *creq = &req->dev.kind.common;
    assert(creq->pscom_req && (creq->pscom_req->state & PSCOM_REQ_STATE_DONE));

    Dprintf("destroy request req=%p type=%d", req, req->kind);

    pscom_request_free(creq->pscom_req);
    creq->pscom_req = NULL;
}

void MPID_Request_free_hook(MPIR_Request * req)
{
    return;
}

int MPID_Request_complete(MPIR_Request * req)
{
    if (MPID_PSP_Subrequest_completed(req)) {
        MPIR_Request_free(req);
    }

    return MPI_SUCCESS;
}
