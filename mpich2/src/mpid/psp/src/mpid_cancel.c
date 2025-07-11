/*
 * ParaStation
 *
 * Copyright (C) 2007-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"

int MPID_Cancel_recv(MPIR_Request * rreq)
{
    pscom_request_t *req = rreq->dev.kind.recv.common.pscom_req;

    if (req && pscom_cancel_recv(req)) {
        MPIR_STATUS_SET_CANCEL_BIT(rreq->status, TRUE);
    }

    return MPI_SUCCESS;
}


int MPID_Cancel_send(MPIR_Request * sreq)
{
    pscom_request_t *req = sreq->dev.kind.send.common.pscom_req;

    if (req) {
        if (pscom_cancel_send(req)) {
            MPIR_STATUS_SET_CANCEL_BIT(sreq->status, TRUE);
        } else {
            MPIDI_PSP_PSCOM_Xheader_t *xhead = &req->xheader.user.common;

            if (xhead->type == MPID_PSP_MSGTYPE_DATA_REQUEST_ACK) {
                /* request is a synchronous send. */
                MPIDI_PSP_SendCtrl(xhead->tag, xhead->context_id, sreq->comm->rank,
                                   req->connection, MPID_PSP_MSGTYPE_CANCEL_DATA_REQUEST_ACK);
            }
#if 0
/*
 |  Cancelling of non-synchronous messages is disabled because
 |  with PSP_UNEXPECTED_RECEIVES=0 (default) the absence of the
 |  expected cancel-ack may lead to a deadlock...
 */
            else {
                /* request is NOT a synchronous send. */

                /* remember that this message is to be cancelled: */
                xhead->type = MPID_PSP_MSGTYPE_DATA_CANCELLED;

                /* wait for the ack cancel: */
                MPID_PSP_RecvAck(sreq);

                /* send the anti-send message: */
                MPIDI_PSP_SendCtrl(xhead->tag, xhead->context_id, sreq->comm->rank,
                                   req->connection, MPID_PSP_MSGTYPE_CANCEL_DATA_REQUEST_ACK);
            }
#endif
        }
    }

    return MPI_SUCCESS;
}
