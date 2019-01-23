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

#if !defined(PSCOM_ALLIN) || defined(PSCOM_ALLIN_INCLUDE_TOKEN)

#include <assert.h>
#include "mpidimpl.h"
#include "mpid_psp_request.h"
#include "mpid_psp_packed_msg.h"
#include "mpid_psp_datatype.h"

// This must be the last include before sysmbols are defined:
#include "mpid_visibility.h"


static
int cb_accept_data(pscom_request_t *request,
		   pscom_connection_t *connection,
		   pscom_header_net_t *header_net)
{
	MPIR_Request *req = request->user->type.sr.mpid_req;
	struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv;
	MPID_PSCOM_XHeader_t *xhead = &header_net->xheader->user.common;

	return  (xhead->type <= MPID_PSP_MSGTYPE_DATA_REQUEST_ACK) &&
		((xhead->tag == rreq->tag) || (rreq->tag == MPI_ANY_TAG)) &&
		(xhead->context_id == rreq->context_id);
}


static
int cb_accept_ack(pscom_request_t *request,
		  pscom_connection_t *connection,
		  pscom_header_net_t *header_net)
{
	MPID_PSCOM_XHeader_t *xhead = &request->xheader.user.common;

	MPID_PSCOM_XHeader_t *xhead_net = &header_net->xheader->user.common;

	return  ((xhead_net->type == MPID_PSP_MSGTYPE_DATA_ACK) ||
		 (xhead_net->type == MPID_PSP_MSGTYPE_CANCEL_DATA_ACK)) &&
		(xhead_net->tag == xhead->tag) &&
		(xhead_net->context_id == xhead->context_id);
}


static
void cb_io_done_ack(pscom_request_t *request)
{
	/* This is an pscom.io_done call. Global lock state undefined! */
	MPID_PSCOM_XHeader_t *xhead = &request->xheader.user.common;

	/* Todo: Test for pscom_req_successful(request) ? */
	MPIR_Request *send_req = request->user->type.sr.mpid_req;

	if (xhead->type == MPID_PSP_MSGTYPE_CANCEL_DATA_ACK) {
		MPIR_STATUS_SET_CANCEL_BIT(send_req->status, TRUE);
	}

	MPID_PSP_Subrequest_completed(send_req);
	MPIR_Request_free(send_req);
	request->user->type.sr.mpid_req = NULL;
	pscom_request_free(request);
}


static inline
void receive_done(pscom_request_t *request)
{
	/* This is an pscom.io_done call. Global lock state undefined! */
	MPIR_Request *req = request->user->type.sr.mpid_req;
	MPID_PSCOM_XHeader_t *xhead = &request->xheader.user.common;

	MPIR_STATUS_SET_COUNT(req->status, request->header.data_len); /* status.count == datalen, or == datalen/sizeof(mpitype) ?? */
	req->status.MPI_SOURCE = xhead->src_rank;
	req->status.MPI_TAG = xhead->tag;
	/* req->status.MPI_ERROR has already been preset to MPI_SUCCESS in prepare_recvreq() */
	/* ...and may by now be overwritten with MPI_ERR_TYPE in receive_done_noncontig() */
	if (pscom_req_successful(request)) {
		assert(request->xheader_len == request->header.xheader_len);

		if (unlikely(xhead->type == MPID_PSP_MSGTYPE_DATA_REQUEST_ACK)) {
			/* synchronous send : send ack */
			MPID_PSP_SendCtrl(xhead->tag, xhead->context_id, req->comm->rank,
					  request->connection, MPID_PSP_MSGTYPE_DATA_ACK);
		}
	} else if (request->state & PSCOM_REQ_STATE_TRUNCATED) {
		assert (request->header.data_len > request->data_len);
		req->status.MPI_ERROR = MPI_ERR_TRUNCATE;
	} else if (request->state & PSCOM_REQ_STATE_CANCELED) {
		/* ToDo: MPI_ERROR = MPI_SUCCESS on cancelled ? */
		/* req->status.MPI_ERROR = MPI_SUCCESS; */
		MPIR_STATUS_SET_CANCEL_BIT(req->status, TRUE);
	} else {
		static char state_str[100];
		snprintf(state_str, 100, "request state:%s", pscom_req_state_str(request->state));
		req->status.MPI_ERROR = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL,
							     "mpid_irecv_done", __LINE__,
							     MPI_ERR_OTHER, "**read",
							     "**read %s", state_str);
	}

	MPID_PSP_Subrequest_completed(req);
	MPIR_Request_free(req);
}


static
void receive_done_noncontig(pscom_request_t *request)
{
	/* This is an pscom.io_done call. Global lock state undefined! */
	MPIR_Request *req = request->user->type.sr.mpid_req;
	struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv;

	if (pscom_req_successful(request) || (request->state & PSCOM_REQ_STATE_TRUNCATED)) {
		req->status.MPI_ERROR = MPID_PSP_packed_msg_unpack(rreq->addr, rreq->count, rreq->datatype,
								   &rreq->msg, request->header.data_len);
	}

	/* Noncontig receive request */
	/* cleanup temp buffer and datatype */
	MPID_PSP_packed_msg_cleanup_datatype(&rreq->msg, rreq->datatype);

	receive_done(request);
}


static
int cb_accept_cancel_data(pscom_request_t *request,
			  pscom_connection_t *connection,
			  pscom_header_net_t *header_net)
{
	MPID_PSCOM_XHeader_t *xhead = &request->xheader.user.common;
	MPID_PSCOM_XHeader_t *xhead_net = &header_net->xheader->user.common;

	return ((xhead_net->type == MPID_PSP_MSGTYPE_DATA_REQUEST_ACK) ||
		(xhead_net->type == MPID_PSP_MSGTYPE_DATA_CANCELLED)) &&
		(xhead_net->tag == xhead->tag) &&
		(xhead_net->context_id == xhead->context_id);
}


static
void MPID_do_recv_cancel_data_request_ack(pscom_request_t *cancel_req)
{
	/* reuse cancel_req to eatup the generated request */
	MPID_PSCOM_XHeader_t *xhead = &cancel_req->xheader.user.common;

	cancel_req->ops.recv_accept = cb_accept_cancel_data;
	cancel_req->ops.io_done = NULL; /* delay pscom_request_free() / see below! */

	pscom_post_recv(cancel_req);

	if (!(cancel_req->state & PSCOM_REQ_STATE_IO_STARTED)) {
		/* post_recv should find a generated request. If not, we
		   cannot cancel, because ack is already send. So we
		   cancel the cancel.
		*/
		pscom_cancel_recv(cancel_req);
		pscom_request_free(cancel_req);
#if 0
/*
 |  Cancelling of non-synchronous messages is disabled because
 |  with PSP_UNEXPECTED_RECEIVES=0 (default) the absence of the
 |  expected cancel-ack may lead to a deadlock...
 */
		if (xhead->type != MPID_PSP_MSGTYPE_DATA_REQUEST_ACK) { /* XXX */
			/* this was NOT a synchronous send. */
			/* send common ack to signal the failed cancel request: */
			MPID_PSP_SendCtrl(xhead->tag, xhead->context_id, MPI_PROC_NULL,
					  cancel_req->connection, MPID_PSP_MSGTYPE_DATA_ACK);
		}
#endif
	} else {
		/* send cancel ack */
		MPID_PSP_SendCtrl(xhead->tag, xhead->context_id, MPI_PROC_NULL,
				  cancel_req->connection, MPID_PSP_MSGTYPE_CANCEL_DATA_ACK);
		if (pscom_req_is_done(cancel_req)) {
			pscom_request_free(cancel_req);
		} else {
			/* Free the cancel_req when done */
			cancel_req->ops.io_done = pscom_request_free;
		}
	}
}


static
pscom_request_t *MPID_do_recv_forward_to(void (*io_done)(pscom_request_t *req), pscom_header_net_t *header_net)
{
	pscom_request_t *req = PSCOM_REQUEST_CREATE();

	assert(header_net->xheader_len <= sizeof(req->xheader));

	req->xheader_len = header_net->xheader_len;
	req->ops.io_done = io_done;

	return req;
}


static
pscom_request_t *receive_dispatch(pscom_connection_t *connection,
				  pscom_header_net_t *header_net)
{
	MPID_PSCOM_XHeader_t *xhead = &header_net->xheader->user.common;

	if (xhead->type == MPID_PSP_MSGTYPE_DATA) {
		/* fastpath */
		return NULL;
	}

	switch (xhead->type) {
	case MPID_PSP_MSGTYPE_RMA_PUT:
		return MPID_do_recv_rma_put(connection, &header_net->xheader->user.put);

	case MPID_PSP_MSGTYPE_RMA_ACCUMULATE:
		return MPID_do_recv_rma_accumulate(connection, header_net);

	case MPID_PSP_MSGTYPE_DATA_REQUEST_ACK:
		break;

	case MPID_PSP_MSGTYPE_CANCEL_DATA_REQUEST_ACK:
		return MPID_do_recv_forward_to(MPID_do_recv_cancel_data_request_ack, header_net);

	case MPID_PSP_MSGTYPE_RMA_GET_REQ:
		return MPID_do_recv_rma_get_req(connection, &header_net->xheader->user.get_req);

	case MPID_PSP_MSGTYPE_RMA_LOCK_EXCLUSIVE_REQUEST:
		return MPID_do_recv_forward_to(MPID_do_recv_rma_lock_exclusive_req, header_net);

	case MPID_PSP_MSGTYPE_RMA_LOCK_SHARED_REQUEST:
		return MPID_do_recv_forward_to(MPID_do_recv_rma_lock_shared_req, header_net);

	case MPID_PSP_MSGTYPE_RMA_UNLOCK_REQUEST:
		return MPID_do_recv_forward_to(MPID_do_recv_rma_unlock_req, header_net);

	case MPID_PSP_MSGTYPE_RMA_FLUSH_REQUEST:
		return MPID_do_recv_forward_to(MPID_do_recv_rma_flush_req, header_net);

	case MPID_PSP_MSGTYPE_RMA_INTERNAL_LOCK_REQUEST:
		return MPID_do_recv_forward_to(MPID_do_recv_rma_lock_internal_req, header_net);

	case MPID_PSP_MSGTYPE_RMA_INTERNAL_UNLOCK_REQUEST:
		return MPID_do_recv_forward_to(MPID_do_recv_rma_unlock_internal_req, header_net);
	}

	return NULL;
}


void MPID_enable_receive_dispach(pscom_socket_t *socket)
{
	if (!socket->ops.default_recv) {
		socket->ops.default_recv = receive_dispatch;
	} else {
		assert(socket->ops.default_recv == receive_dispatch);
	}
}


static
void prepare_recvreq(MPIR_Request *req, int tag, MPIR_Comm * comm, int context_offset)
{
	struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv;
	pscom_request_t *preq = rreq->common.pscom_req;

	rreq->tag = tag;
	rreq->context_id = comm->recvcontext_id + context_offset;
	preq->ops.recv_accept = cb_accept_data;
	preq->xheader_len = sizeof(MPID_PSCOM_XHeader_Send_t);
	req->status.MPI_ERROR = MPI_SUCCESS;
}


static
void prepare_probereq(MPIR_Request *req, int tag, MPIR_Comm * comm, int context_offset)
{
	struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv;
	pscom_request_t *preq = rreq->common.pscom_req;

	prepare_recvreq(req, tag, comm, context_offset);
	preq->ops.recv_accept = cb_accept_data;
}


static
void prepare_data(MPIR_Request *req, void * buf, int count, MPI_Datatype datatype)
{
	struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv;
	pscom_request_t *preq = rreq->common.pscom_req;
	int ret;

	ret = MPID_PSP_packed_msg_prepare(buf, count, datatype, &rreq->msg);
	if (unlikely(ret != MPI_SUCCESS)) goto err_alloc_tmpbuf;

	preq->data = rreq->msg.msg;
	preq->data_len = rreq->msg.msg_sz;

	return;
	/* --- */
err_alloc_tmpbuf: /* ToDo: */
	fprintf(stderr, "MPL_malloc() failed\n");
	exit(1);
}


static
void prepare_cleanup(MPIR_Request *req, void * buf, int count, MPI_Datatype datatype)
{
	struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv;
	pscom_request_t *preq = rreq->common.pscom_req;

	preq->ops.io_done = receive_done;

	if (MPID_PSP_packed_msg_need_unpack(&rreq->msg)) {
		rreq->addr = buf;
		rreq->count = count;
		rreq->datatype = datatype;
		MPID_PSP_Datatype_add_ref(datatype);

		preq->ops.io_done = receive_done_noncontig;
	}
}


static
void prepare_source(MPIR_Request *req, pscom_connection_t *con, pscom_socket_t *sock)
{
	struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv;
	pscom_request_t *preq = rreq->common.pscom_req;

	preq->connection = con;
	preq->socket = sock;
}


int MPID_Irecv(void * buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
	       MPIR_Comm * comm, int context_offset, MPIR_Request ** request)
{
	MPIR_Request *req;
	pscom_connection_t *con;
	pscom_socket_t *sock;
/*
	printf("#%d ps--- %s() called\n", MPIDI_Process.my_pg_rank, __func__);
	printf("#%d buf %p, count %d, datatype 0x%0x, rank %d, tag %d, comm %p, off %d\n",
	       MPIDI_Process.my_pg_rank, buf, count, datatype, rank, tag, comm, context_offset);
	printf("#%d ctx.id %d ctx.rank %d, ctx.name %s\n",
	       MPIDI_Process.my_pg_rank, comm->context_id, comm->rank, comm->name);
*/
	req = MPIR_Request_create(MPIR_REQUEST_KIND__RECV);
	if (unlikely(!req)) goto err_request_recv_create;
	req->comm = comm;
	MPIR_Comm_add_ref(comm);

	prepare_recvreq(req, tag, comm, context_offset);

	con = MPID_PSCOM_rank2connection(comm, rank);
	sock = comm->pscom_socket;

	if (con || (rank == MPI_ANY_SOURCE)) {

		prepare_data(req, buf, count, datatype);
		prepare_source(req, con, sock);
		prepare_cleanup(req, buf, count, datatype);

		MPIR_Request_add_ref(req);

		pscom_post_recv(req->dev.kind.recv.common.pscom_req);

	} else switch (rank) {
	case MPI_PROC_NULL:
		MPIR_Status_set_procnull(&req->status);
		MPIDI_PSP_Request_set_completed(req);
		break;
	case MPI_ROOT:
	default:
		/* printf("%s(): MPI_ERR_RANK: rank = %d, comm->size=%d, comm->name=%s\n",
		   __func__, rank, comm->local_size, comm->name ? comm->name : "?"); */
		goto err_rank;
	}

	*request = req;

	return MPI_SUCCESS;
	/* --- */
 err_request_recv_create:
	return  MPI_ERR_NO_MEM;
	/* --- */
 err_rank:
	MPIR_Request_free(req);
	return  MPI_ERR_RANK;
}


void MPID_PSP_RecvAck(MPIR_Request *send_req)
{
	pscom_request_t *preq;
	pscom_request_t *preq_send;
	MPID_PSCOM_XHeader_t *xhead;

	preq = PSCOM_REQUEST_CREATE();
	assert(preq != NULL);

	preq_send = send_req->dev.kind.send.common.pscom_req;

	preq->xheader_len = sizeof(*xhead);
	preq->ops.recv_accept = cb_accept_ack;
	preq->ops.io_done = cb_io_done_ack;
	preq->connection = preq_send->connection;
	assert(preq->connection != NULL);

	/* Copy xheader from send request */
	xhead = &preq->xheader.user.common;
	*xhead = preq_send->xheader.user.common;

	preq->user->type.sr.mpid_req = send_req;

	MPID_PSP_Subrequest_add(send_req);   /* Subrequest_completed(sendreq) and */
	MPIR_Request_add_ref(send_req);  /* Request_release_ref(sendreq) in cb_receive_ack() */

	pscom_post_recv(preq);
}


static
void set_probe_status(pscom_request_t *req, MPI_Status *status)
{
	if (!status || status == MPI_STATUS_IGNORE) return;

	MPIR_STATUS_SET_COUNT(*status, req->header.data_len);
	MPIR_STATUS_SET_CANCEL_BIT(*status, (req->state & PSCOM_REQ_STATE_CANCELED) ? TRUE : FALSE);
	status->MPI_SOURCE = req->xheader.user.common.src_rank;
	status->MPI_TAG    = req->xheader.user.common.tag;
	/* status->MPI_ERROR  = MPI_SUCCESS; */
}


int MPID_Probe(int rank, int tag, MPIR_Comm * comm, int context_offset, MPI_Status * status)
{
	pscom_connection_t *con;
	pscom_socket_t *sock;
/*
	printf("#%d ps--- %s() called\n", MPIDI_Process.my_pg_rank, __func__);
	printf("#%d buf %p, count %d, datatype 0x%0x, rank %d, tag %d, comm %p, off %d\n",
	       MPIDI_Process.my_pg_rank, buf, count, datatype, rank, tag, comm, context_offset);
	printf("#%d ctx.id %d ctx.rank %d, ctx.name %s\n",
	       MPIDI_Process.my_pg_rank, comm->context_id, comm->rank, comm->name);
*/

	con = MPID_PSCOM_rank2connection(comm, rank);
	sock = comm->pscom_socket;

	if (con || (rank == MPI_ANY_SOURCE)) {
		MPIR_Request *req;
		req = MPIR_Request_create(MPIR_REQUEST_KIND__RECV);
		if (unlikely(!req)) goto err_request_recv_create;
		req->comm = comm;
		MPIR_Comm_add_ref(comm);

		prepare_probereq(req, tag, comm, context_offset);

		prepare_source(req, con, sock);

		MPID_PSP_LOCKFREE_CALL(pscom_probe(req->dev.kind.recv.common.pscom_req));

		set_probe_status(req->dev.kind.recv.common.pscom_req, status);

		MPID_PSP_Subrequest_completed(req);
		MPIR_Request_free(req);
	} else switch (rank) {
	case MPI_PROC_NULL:
		MPIR_Status_set_procnull(status);
		break;
	case MPI_ROOT:
	default:
		/* printf("#%d ps--- %s(): MPI_ERR_RANK: rank = %d, comm->size=%d, comm->name=%s\n",
		   MPIDI_Process.my_pg_rank, __func__, rank, comm->local_size, comm->name ? comm->name : "?"); */
		goto err_rank;
	}

	return MPI_SUCCESS;
	/* --- */
 err_request_recv_create:
	return  MPI_ERR_NO_MEM;
	/* --- */
 err_rank:
	return  MPI_ERR_RANK;
}


int MPID_Iprobe(int rank, int tag, MPIR_Comm * comm, int context_offset, int * flag, MPI_Status * status)
{
	pscom_connection_t *con;
	pscom_socket_t *sock;
/*
	printf("#%d ps--- %s() called\n", MPIDI_Process.my_pg_rank, __func__);
	printf("#%d buf %p, count %d, datatype 0x%0x, rank %d, tag %d, comm %p, off %d\n",
	       MPIDI_Process.my_pg_rank, buf, count, datatype, rank, tag, comm, context_offset);
	printf("#%d ctx.id %d ctx.rank %d, ctx.name %s\n",
	       MPIDI_Process.my_pg_rank, comm->context_id, comm->rank, comm->name);
*/

	con = MPID_PSCOM_rank2connection(comm, rank);
	sock = comm->pscom_socket;

	if (con || (rank == MPI_ANY_SOURCE)) {
		MPIR_Request *req;
		req = MPIR_Request_create(MPIR_REQUEST_KIND__RECV);
		if (unlikely(!req)) goto err_request_recv_create;
		req->comm = comm;
		MPIR_Comm_add_ref(comm);

		prepare_probereq(req, tag, comm, context_offset);

		prepare_source(req, con, sock);

		*flag = pscom_iprobe(req->dev.kind.recv.common.pscom_req);
		if (*flag) {
			set_probe_status(req->dev.kind.recv.common.pscom_req, status);
		}

		MPID_PSP_Subrequest_completed(req);
		MPIR_Request_free(req);
	} else switch (rank) {
	case MPI_PROC_NULL:
		MPIR_Status_set_procnull(status);
		*flag = 1;
		break;
	case MPI_ROOT:
	default:
		/* printf("#%d ps--- %s(): MPI_ERR_RANK: rank = %d, comm->size=%d, comm->name=%s\n",
		   MPIDI_Process.my_pg_rank, __func__, rank, comm->local_size, comm->name ? comm->name : "?"); */
		goto err_rank;
	}

	return MPI_SUCCESS;
	/* --- */
 err_request_recv_create:
	return  MPI_ERR_NO_MEM;
	/* --- */
 err_rank:
	return  MPI_ERR_RANK;
}

#include "mpid_mprobe.c"

#endif
