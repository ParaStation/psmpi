/*
 * ParaStation
 *
 * Copyright (C) 2006-2010 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include "mpidimpl.h"
#include "mpid_psp_packed_msg.h"
#include "mpid_psp_request.h"
#include "mpid_psp_datatype.h"


static inline
void sendrequest_common_done(pscom_request_t *preq)
{
	/* This is an pscom.io_done call. Global lock state undefined! */
	MPID_Request *req = preq->user->type.sr.mpid_req;
	if (pscom_req_successful(preq)) {
		req->status.MPI_ERROR = MPI_SUCCESS;
	} else if (preq->state & PSCOM_REQ_STATE_CANCELED) {
		req->status.MPI_ERROR = MPI_SUCCESS;
		MPIR_STATUS_SET_CANCEL_BIT(req->status, TRUE);
	} else {
		static char state_str[100];
		snprintf(state_str, 100, "request state:%s", pscom_req_state_str(preq->state));
		req->status.MPI_ERROR = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL,
							     "mpid_isend_done", __LINE__,
							     MPI_ERR_OTHER, "**write",
							     "**write %s", state_str);
	}

	if (preq->xheader.user.send.common.type == MPID_PSP_MSGTYPE_DATA_REQUEST_ACK) {
		/* from synchronous send. */
		if (pscom_req_successful(preq)) {
			/* Also wait for the ack (or ack cancel) */
			MPID_PSP_RecvAck(req);
		}
	}

/*	assert(*(req->cc_ptr) == 1);  */
	MPID_PSP_Subrequest_completed(req);
	MPID_PSP_Request_dequeue(req, MPID_REQUEST_SEND);
}


static
void sendrequest_done(pscom_request_t *preq)
{
	/* This is an pscom.io_done call. Global lock state undefined! */
	MPID_Request *req = preq->user->type.sr.mpid_req;

	MPID_PSP_packed_msg_cleanup(&req->dev.kind.send.msg);

	sendrequest_common_done(preq);
}


static
void prepare_xheader(MPID_Request *req, int tag, MPID_Comm * comm, int context_offset, enum MPID_PSP_MSGTYPE type)
{
	pscom_request_t	*preq = req->dev.kind.common.pscom_req;
	MPID_PSCOM_XHeader_Send_t *xheader = &preq->xheader.user.send;

	xheader->common.tag = tag;
	xheader->common.context_id = comm->context_id + context_offset;
	xheader->common.type = type;
	xheader->common._reserved_ = 0;
	xheader->common.src_rank = comm->rank;
	preq->xheader_len = sizeof(*xheader);

	req->status.MPI_TAG = tag;
	req->status.MPI_SOURCE = MPI_PROC_NULL;
}


static
void prepare_destination(MPID_Request *req, MPID_Comm * comm, int rank)
{
	pscom_request_t *preq = req->dev.kind.common.pscom_req;

	preq->connection = MPID_PSCOM_rank2connection(comm, rank);
}


static
void prepare_cleanup(MPID_Request *req)
{
	pscom_request_t *preq = req->dev.kind.common.pscom_req;

	preq->ops.io_done = sendrequest_done;
}


static
int prepare_data(MPID_Request *req, const void *buf, int count, MPI_Datatype datatype)
{
	struct MPID_DEV_Request_send *sreq = &req->dev.kind.send;
	pscom_request_t *preq = sreq->common.pscom_req;

	int ret;

	/* Data */
	ret = MPID_PSP_packed_msg_prepare(buf, count, datatype, &sreq->msg);
	if (unlikely(ret != MPI_SUCCESS)) goto err_create_packed_msg;

	preq->data_len = sreq->msg.msg_sz;
	preq->data = sreq->msg.msg;

	MPIR_STATUS_SET_COUNT(req->status, preq->data_len);
	MPIR_STATUS_SET_CANCEL_BIT(req->status, FALSE)

	return MPI_SUCCESS;
	/* --- */
err_create_packed_msg:
	return ret;
}


static
void copy_data(MPID_Request *req, const void *buf, int count, MPI_Datatype datatype)
{
	struct MPID_DEV_Request_send *sreq = &req->dev.kind.send;
	MPID_PSP_packed_msg_pack(buf, count, datatype, &sreq->msg);
}


static inline
int MPID_PSP_Sendtype(const void * buf, int count, MPI_Datatype datatype, int rank,
		      int tag, MPID_Comm * comm, int context_offset,
		      MPID_Request ** request, enum MPID_PSP_MSGTYPE type)
{
	int mpi_errno = MPI_SUCCESS;
	MPID_Request *req;

/*
  printf("#%d ps--- %s() called\n", MPIDI_Process.my_pg_rank, __func__);

  printf("#%d buf %p, count %d, datatype 0x%0x, rank %d, tag %d, comm %p, off %d\n",
  MPIDI_Process.my_pg_rank, buf, count, datatype, rank, tag, comm, context_offset);
  printf("#%d ctx.id %d ctx.rank %d, ctx.name %s\n",
  MPIDI_Process.my_pg_rank, comm->context_id, comm->rank, comm->name);
*/

	req = MPID_DEV_Request_send_create(comm);
	if (unlikely(!req)) goto err_request_send_create;

	if (rank >= 0) {
		mpi_errno = prepare_data(req, buf, count, datatype);
		if (unlikely(mpi_errno != MPI_SUCCESS)) goto err_prepare_data_failed;

		copy_data(req, buf, count, datatype);

		prepare_xheader(req, tag, comm, context_offset, type);
		prepare_destination(req, comm, rank);
		prepare_cleanup(req);

		MPID_PSP_Request_enqueue(req);

		pscom_post_send(req->dev.kind.send.common.pscom_req);

	} else switch (rank) {
	case MPI_PROC_NULL:
		MPIR_Status_set_procnull(&req->status);
		MPID_PSP_Request_set_completed(req);
		break;
	case MPI_ANY_SOURCE:
	case MPI_ROOT:
	default:
		/* printf("#%d ps--- %s(): MPI_ERR_RANK: rank = %d, comm->size=%d, comm->name=%s\n",
		   MPIDI_Process.my_pg_rank, __func__, rank, comm->local_size, comm->name ? comm->name : "?"); */
		mpi_errno = MPI_ERR_RANK;
	}

	assert(mpi_errno == MPI_SUCCESS);
	*request = req;

	return MPI_SUCCESS;
	/* --- */
err_request_send_create:
	mpi_errno = MPI_ERR_NO_MEM;
err_prepare_data_failed:
	MPID_DEV_Request_release_ref(req, MPID_REQUEST_SEND);
	return mpi_errno;
}


void MPID_PSP_SendCtrl(int tag, int context_id, int src_rank, pscom_connection_t *con, enum MPID_PSP_MSGTYPE msgtype)
{
	MPID_PSCOM_XHeader_t xhead;

	/*
	printf("%s(): send ctrl (tag:%d, cid:%d, srank:%d) %s to %s\n",
	       __func__, tag, context_id, src_rank, mpid_msgtype_str(msgtype),
	       pscom_con_info_str(&con->remote_con_info));
	*/

	xhead.tag = tag;
	xhead.context_id = context_id;
	xhead.type = msgtype;
	xhead._reserved_ = 0;
	xhead.src_rank = src_rank;

	pscom_send(con, &xhead, sizeof(xhead), NULL, 0);
}


static
int accept_ctrl(pscom_request_t *req,
		pscom_connection_t *connection,
		pscom_header_net_t *header_net)
{
	MPID_PSCOM_XHeader_t *xhead = &req->xheader.user.common;
	MPID_PSCOM_XHeader_t *xhead_net = &header_net->xheader->user.common;
/*
	printf("accept_ctrl %d-%d? tag %d-%d, srcrank %d-%d, context_id %d-%d\n",
	       xhead->type, xhead_net->type,
	       xhead->tag, xhead_net->tag,
	       xhead->src_rank, xhead_net->src_rank,
	       xhead->context_id, xhead_net->context_id);
*/
	return  (header_net->xheader_len == sizeof(*xhead)) &&
		(xhead->type == xhead_net->type) &&
		(xhead->src_rank == xhead_net->src_rank) &&
		(xhead->tag == xhead_net->tag) &&
		(xhead->context_id == xhead_net->context_id);
}


void MPID_PSP_RecvCtrl(int tag, int recvcontext_id, int src_rank, pscom_connection_t *con, enum MPID_PSP_MSGTYPE msgtype)
{
	pscom_request_t *req = PSCOM_REQUEST_CREATE();
	MPID_PSCOM_XHeader_t *xhead = &req->xheader.user.common;

	xhead->tag = tag;
	xhead->context_id = recvcontext_id;
	xhead->type = msgtype;
	xhead->_reserved_ = 0;
	xhead->src_rank = src_rank;

	if(src_rank != MPI_ANY_SOURCE) {
		req->connection = con;
	} else {
		req->connection = NULL;
		req->socket = MPIR_Process.comm_world->pscom_socket;
	}

	req->ops.recv_accept = accept_ctrl;
	req->data = NULL;
	req->data_len = 0;
	req->xheader_len = sizeof(*xhead);

	pscom_post_recv(req);
	MPID_PSP_LOCKFREE_CALL(pscom_wait(req));
	pscom_request_free(req);
}


int MPID_Isend(const void * buf, MPI_Aint count, MPI_Datatype datatype, int rank,
	       int tag, MPID_Comm * comm, int context_offset, MPID_Request ** request)
{
	int mpi_errno;
	mpi_errno = MPID_PSP_Sendtype(buf, count, datatype, rank, tag,
				      comm, context_offset, request, MPID_PSP_MSGTYPE_DATA);
	return mpi_errno;
}


int MPID_Issend(const void * buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
		MPID_Comm * comm, int context_offset, MPID_Request ** request)
{
	int mpi_errno;
	mpi_errno = MPID_PSP_Sendtype(buf, count, datatype, rank, tag,
				      comm, context_offset, request, MPID_PSP_MSGTYPE_DATA_REQUEST_ACK);

	return mpi_errno;
}
