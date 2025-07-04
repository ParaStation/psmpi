/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"
#include "mpid_psp_request.h"
#include "mpid_psp_packed_msg.h"
#include "mpid_psp_datatype.h"

void MPIDI_PSP_SendCtrl(int tag, int context_id, int src_rank, pscom_connection_t * con,
                        enum MPID_PSP_MSGTYPE msgtype)
{
    MPIDI_PSP_PSCOM_Xheader_t xhead;

    /*
     * printf("%s(): send ctrl (tag:%d, cid:%d, srank:%d) %s to %s\n",
     * __func__, tag, context_id, src_rank, mpid_msgtype_str(msgtype),
     * pscom_con_info_str(&con->remote_con_info));
     */

    xhead.tag = tag;
    xhead.context_id = context_id;
    xhead.type = msgtype;
    xhead._reserved_ = 0;
    xhead.src_rank = src_rank;

    pscom_send(con, &xhead, sizeof(xhead), NULL, 0);
}


static
int accept_ctrl(pscom_request_t * req,
                pscom_connection_t * connection, pscom_header_net_t * header_net)
{
    MPIDI_PSP_PSCOM_Xheader_t *xhead = &req->xheader.user.common;
    MPIDI_PSP_PSCOM_Xheader_t *xhead_net = &header_net->xheader->user.common;
/*
	printf("accept_ctrl %d-%d? tag %d-%d, srcrank %d-%d, context_id %d-%d\n",
	       xhead->type, xhead_net->type,
	       xhead->tag, xhead_net->tag,
	       xhead->src_rank, xhead_net->src_rank,
	       xhead->context_id, xhead_net->context_id);
*/
    return (header_net->xheader_len == sizeof(*xhead)) &&
        (xhead->type == xhead_net->type) &&
        (xhead->src_rank == xhead_net->src_rank) &&
        (xhead->tag == xhead_net->tag) && (xhead->context_id == xhead_net->context_id);
}

static
void prepare_ctrl_recvreq(pscom_request_t * req, int tag, int recvcontext_id, int src_rank,
                          pscom_connection_t * con, enum MPID_PSP_MSGTYPE msgtype)
{
    MPIDI_PSP_PSCOM_Xheader_t *xhead = &req->xheader.user.common;

    /* prepare the xheader */
    xhead->tag = tag;
    xhead->context_id = recvcontext_id;
    xhead->type = msgtype;
    xhead->_reserved_ = 0;
    xhead->src_rank = src_rank;

    /* prepare the pscom request */
    req->ops.recv_accept = accept_ctrl;
    req->data = NULL;
    req->data_len = 0;
    req->xheader_len = sizeof(*xhead);
    req->connection = con;

    if (src_rank == MPI_ANY_SOURCE) {
        req->socket = MPIR_Process.comm_world->pscom_socket;
    }
}

void MPIDI_PSP_RecvCtrl(int tag, int recvcontext_id, int src_rank, pscom_connection_t * con,
                        enum MPID_PSP_MSGTYPE msgtype)
{
    pscom_request_t *req = PSCOM_REQUEST_CREATE();

    prepare_ctrl_recvreq(req, tag, recvcontext_id, src_rank, con, msgtype);

    pscom_post_recv(req);
    MPID_PSP_LOCKFREE_CALL(pscom_wait(req));
    pscom_request_free(req);
}

void MPIDI_PSP_RecvPartitionedCtrl(int tag, int context_id, int src_rank,
                                   pscom_connection_t * con, enum MPID_PSP_MSGTYPE msgtype)
{
    pscom_request_t *req = PSCOM_REQUEST_CREATE();

    MPIDI_PSP_PSCOM_Xheader_part_t *xheader = &req->xheader.user.part;

    // set xheader
    xheader->common.tag = tag;
    xheader->common.context_id = context_id;
    xheader->common.type = msgtype;
    xheader->common._reserved_ = 0;
    xheader->common.src_rank = src_rank;

    req->ops.recv_accept = accept_ctrl;
    req->ops.io_done = pscom_request_free;
    req->data = NULL;
    req->data_len = 0;
    req->xheader_len = sizeof(*xheader);
    req->connection = con;

    if (src_rank == MPI_ANY_SOURCE) {
        req->socket = MPIR_Process.comm_world->pscom_socket;
    }

    pscom_post_recv(req);
}

void MPIDI_PSP_IprobeCtrl(int tag, int recvcontext_id, int src_rank, pscom_connection_t * con,
                          enum MPID_PSP_MSGTYPE msgtype, int *flag)
{
    pscom_request_t *req = PSCOM_REQUEST_CREATE();

    prepare_ctrl_recvreq(req, tag, recvcontext_id, src_rank, con, msgtype);

    *flag = pscom_iprobe(req);
    pscom_request_free(req);
}

void MPIDI_PSP_SendRmaCtrl(MPIR_Win * win_ptr, MPIR_Comm * comm, pscom_connection_t * con,
                           int dest_rank, enum MPID_PSP_MSGTYPE msgtype)
{
    MPIDI_PSP_PSCOM_Xheader_rma_lock_t xhead;

    MPID_Win_rank_info *ri = win_ptr->rank_info + dest_rank;

    xhead.common.tag = 0;
    xhead.common.context_id = comm->context_id;
    xhead.common.type = msgtype;
    xhead.common._reserved_ = 0;
    xhead.common.src_rank = comm->rank;
    xhead.win_ptr = ri->win_ptr;

    pscom_send(con, &xhead, sizeof(xhead), NULL, 0);
}


void MPIDI_PSP_SendPartitionedCtrl(int tag, int context_id, int src_rank,
                                   pscom_connection_t * con, MPI_Aint sdata_size, int requests,
                                   MPIR_Request * sreq, MPIR_Request * rreq,
                                   enum MPID_PSP_MSGTYPE msgtype)
{
    MPIDI_PSP_PSCOM_Xheader_part_t xheader;

    // set xheader
    xheader.common.tag = tag;
    xheader.common.context_id = context_id;
    xheader.common.type = msgtype;
    xheader.common._reserved_ = 0;
    xheader.common.src_rank = src_rank;

    // partitioned communication specific infos
    xheader.sdata_size = sdata_size;
    xheader.requests = requests;
    xheader.sreq_ptr = sreq;
    xheader.rreq_ptr = rreq;

    pscom_send(con, &xheader, sizeof(xheader), NULL, 0);
}
