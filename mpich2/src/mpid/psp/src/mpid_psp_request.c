/*
 * ParaStation
 *
 * Copyright (C) 2006-2019 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include "mpidimpl.h"
#include "mpid_psp_datatype.h"
#include "mpid_psp_packed_msg.h"
#include "mpid_psp_request.h"
#include "mpid_debug.h"


/* ------------------------------------------------------------------------- */
/* Soeme request-specific create/destroy hooks                               */
/* ------------------------------------------------------------------------- */
static inline
void MPIDI_PSP_Request_persistent_create_hook(MPIR_Request *req)
{
	struct MPID_DEV_Request_persistent *preq = &req->dev.kind.persistent;
	preq->datatype = 0;
	preq->comm = NULL;
}

static inline
void MPIDI_PSP_Request_send_create_hook(MPIR_Request *req)
{
	struct MPID_DEV_Request_send *sreq = &req->dev.kind.send;
	sreq->msg.tmp_buf = NULL;
}

static inline
void MPIDI_PSP_Request_send_destroy_hook(MPIR_Request *req)
{
	assert(req->dev.kind.send.msg.tmp_buf == NULL);
}

static inline
void MPIDI_PSP_Request_persistent_destroy_hook(MPIR_Request *req)
{
	struct MPID_DEV_Request_persistent *preq = &req->dev.kind.persistent;
	if (preq->datatype) {
		MPID_PSP_Datatype_release(preq->datatype);
	}
	/* TODO: this should be done by the MPIR layer */
//	if (preq->comm) {
//		MPIR_Comm_release(preq->comm);
//	}
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
void MPID_Request_create_hook(MPIR_Request *req)
{
	struct MPID_DEV_Request_common *creq = NULL;
	pscom_request_t *preq = NULL;

	if (!req) return;

	/* allocate the pscom request */
	creq = &req->dev.kind.common;
	creq->pscom_req = PSCOM_REQUEST_CREATE();
	creq->pscom_req->user->type.sr.mpid_req = req;


	/* MPID_Request_construct(req); no need?*/
	MPIR_Object_set_ref(req, 1);

	/* initialize the MPIR_Request */
	req->cc = 1;
	req->cc_ptr = &req->cc;
	req->status.MPI_SOURCE = MPI_UNDEFINED;
	req->status.MPI_TAG = MPI_UNDEFINED;
	req->status.MPI_ERROR = MPI_SUCCESS;
	/* combined MPIR_STATUS_SET_COUNT and MPIR_STATUS_SET_CANCEL_BIT: */
	req->status.count_lo = 0;
	req->status.count_hi_and_cancelled = 0;
	req->comm = NULL;
	req->u.nbc.errflag = MPIR_ERR_NONE;

	/* initialize the pscom_request_t */
	preq = creq->pscom_req;
	preq->connection = NULL;
	preq->socket = NULL;
	preq->ops.recv_accept = NULL;
	preq->ops.io_done = NULL;
	preq->xheader_len = 0;
	preq->data_len = 0;
	preq->data = 0;


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

	/* request-specific initialization */
	switch (req->kind) {
		case MPIR_REQUEST_KIND__SEND:
			MPIDI_PSP_Request_send_create_hook(req);
			break;
		case MPIR_REQUEST_KIND__PREQUEST_RECV:
		case MPIR_REQUEST_KIND__PREQUEST_SEND:
			MPIDI_PSP_Request_persistent_create_hook(req);
			break;
		case MPIR_REQUEST_KIND__RECV:
		case MPIR_REQUEST_KIND__GREQUEST:
		case MPIR_REQUEST_KIND__RMA:
		case MPIR_REQUEST_KIND__COLL:
		case MPIR_REQUEST_KIND__MPROBE:
			break;
		case MPIR_REQUEST_KIND__UNDEFINED:
		case MPIR_REQUEST_KIND__LAST:
			assert(0);
			break;
	}

}

void MPID_Request_destroy_hook(MPIR_Request *req)
{
	switch (req->kind) {
	case MPIR_REQUEST_KIND__SEND:
		MPIDI_PSP_Request_send_destroy_hook(req);
		break;
	case MPIR_REQUEST_KIND__PREQUEST_RECV:
	case MPIR_REQUEST_KIND__PREQUEST_SEND:
		MPIDI_PSP_Request_persistent_destroy_hook(req);
		break;
	case MPIR_REQUEST_KIND__RECV:
	case MPIR_REQUEST_KIND__RMA:
	case MPIR_REQUEST_KIND__COLL:
	case MPIR_REQUEST_KIND__MPROBE:
	case MPIR_REQUEST_KIND__GREQUEST:
		break;
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

void MPID_Request_free_hook(MPIR_Request *req)
{
    return;
}

int MPID_Request_complete(MPIR_Request *req)
{
	if(MPID_PSP_Subrequest_completed(req)) {
		MPIR_Request_free(req);
	}

	return MPI_SUCCESS;
}
