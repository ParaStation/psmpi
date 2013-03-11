/*
 * ParaStation
 *
 * Copyright (C) 2007-2010 ParTec Cluster Competence Center GmbH, Munich
 *
 * All rights reserved.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include "mpidimpl.h"


int MPID_Cancel_recv(MPID_Request * rreq)
{
	pscom_request_t *req = rreq->dev.kind.recv.common.pscom_req;

	if (req && pscom_cancel_recv(req)) {
		rreq->status.cancelled = 1;
	}

	return MPI_SUCCESS;
}


int MPID_Cancel_send(MPID_Request * sreq)
{
	pscom_request_t *req = sreq->dev.kind.send.common.pscom_req;

	if (req) {
		if (pscom_cancel_send(req)) {
			sreq->status.cancelled = 1;
		} else {
			MPID_PSCOM_XHeader_t *xhead = &req->xheader.user.common;

			if (xhead->type == MPID_PSP_MSGTYPE_DATA_REQUEST_ACK) {
				/* request is a synchronous send. */
				MPID_PSP_SendCtrl(xhead->tag, xhead->context_id, sreq->comm->rank,
						  req->connection, MPID_PSP_MSGTYPE_CANCEL_DATA_REQUEST_ACK);
			}
		}
	}

	return MPI_SUCCESS;
}
