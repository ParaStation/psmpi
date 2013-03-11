/*
 * ParaStation
 *
 * Copyright (C) 2006-2010 ParTec Cluster Competence Center GmbH, Munich
 *
 * All rights reserved.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include "mpidimpl.h"
#include "mpid_psp_datatype.h"
#include "mpid_psp_packed_msg.h"
#include "mpid_psp_request.h"


typedef struct {
	MPID_PSCOM_XHeader_Rma_get_req_t xheader;
} mpid_rma_get_req_t;


static
int accept_rma_get_answer(pscom_request_t *request,
			  pscom_connection_t *connection,
			  pscom_header_net_t *header_net)
{
	MPID_PSCOM_XHeader_t *xhead_net = &header_net->xheader->user.common;

	return (xhead_net->type == MPID_PSP_MSGTYPE_RMA_GET_ANSWER);
}


static
void io_done_rma_get_answer(pscom_request_t *request)
{
	/* This is an pscom.io_done call. Global lock state undefined! */
	pscom_request_get_answer_recv_t *ga = &request->user->type.get_answer_recv;

	if (pscom_req_successful(request)) {
		MPID_PSP_packed_msg_unpack(ga->origin_addr, ga->origin_count, ga->origin_datatype,
					   &ga->msg, request->header.data_len);
	}

	MPID_PSP_packed_msg_cleanup_datatype(&ga->msg, ga->origin_datatype);
	/* ToDo: This is not threadsave */
	ga->win_ptr->rma_local_pending_cnt--;

	pscom_request_free(request);
}


int MPID_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
	     int target_rank, MPI_Aint target_disp, int target_count,
	     MPI_Datatype target_datatype, MPID_Win *win_ptr)
{
	MPID_PSP_Datatype_info dt_info;
	MPID_Win_rank_info *ri = win_ptr->rank_info + target_rank;
#if 0
	fprintf(stderr, "int MPID_Get(origin_addr: %p, origin_count: %d, origin_datatype: %08x,"
		" target_rank: %d, target_disp: %d, target_count: %d, target_datatype: %08x,"
		" *win_ptr: %p)\n",
		origin_addr, origin_count, origin_datatype,
		target_rank, target_disp, target_count,
		target_datatype, win_ptr);
#endif
	/* Datatype */
	MPID_PSP_Datatype_get_info(target_datatype, &dt_info);

	char *target_buf = (char *) ri->base_addr + ri->disp_unit * target_disp;

	if (0 && MPID_PSP_Datatype_is_contig(&dt_info)) { /* ToDo: reenable pscom buildin rma_write */
/*
		// Contig message. Use pscom buildin rma
		pscom_request_t *req = pscom_request_create(0);

		req->data_len = msg.msg_sz;
		req->data = msg.msg;
		req->connection = ri->con;
		req->ops.io_done = pscom_request_free;
		req->xheader.rma_write.dest = target_buf;

		pscom_post_rma_write(req);

		// win_ptr->rma_puts_accs[target_rank]++; // ToDo: Howto receive this?
*/
	} else {
		unsigned int	encode_dt_size	= MPID_PSP_Datatype_get_size(&dt_info);
		unsigned int	xheader_len	= sizeof(MPID_PSCOM_XHeader_Rma_get_req_t) + encode_dt_size;

		pscom_request_t *req = pscom_request_create(xheader_len, 0);
		MPID_PSCOM_XHeader_Rma_get_req_t *xheader = &req->xheader.user.get_req;

		MPID_PSP_Datatype_encode(&dt_info, &xheader->encoded_type);

		{ /* Post a receive */
			pscom_request_t *rreq = PSCOM_REQUEST_CREATE();
			pscom_request_get_answer_recv_t *ga = &rreq->user->type.get_answer_recv;

			MPID_PSP_packed_msg_prepare(origin_addr, origin_count, origin_datatype,
						    &ga->msg);
			ga->origin_addr = origin_addr;
			ga->origin_count = origin_count;
			ga->origin_datatype = origin_datatype;
			ga->win_ptr = win_ptr;
			MPID_PSP_Datatype_add_ref(origin_datatype);

			rreq->data_len = ga->msg.msg_sz;
			rreq->data = ga->msg.msg;
			rreq->xheader_len = sizeof(MPID_PSCOM_XHeader_Rma_get_answer_t);

			rreq->ops.recv_accept = accept_rma_get_answer;
			rreq->ops.io_done = io_done_rma_get_answer;
			rreq->connection = ri->con;

			pscom_post_recv(rreq);
		}

		xheader->common.tag = 0;
		xheader->common.context_id = 0;
		xheader->common.type = MPID_PSP_MSGTYPE_RMA_GET_REQ;
		xheader->common._reserved_ = 0;
		xheader->common.src_rank = win_ptr->rank;

		/* xheader->target_disp = target_disp; */
		xheader->target_count = target_count;
		xheader->target_buf = target_buf;
		/* xheader->epoch = ri->epoch_origin; */
		xheader->win_ptr = ri->win_ptr; /* remote win_ptr */

		req->xheader_len = xheader_len;
		req->ops.io_done = pscom_request_free;
		req->connection = ri->con;
		req->data_len = 0;

		pscom_post_send(req);

		win_ptr->rma_local_pending_cnt++;
	}

	return MPI_SUCCESS;
}


static
void io_done_get_answer_send(pscom_request_t *req)
{
	/* This is an pscom.io_done call. Global lock state undefined! */
	pscom_request_get_answer_send_t *gas = &req->user->type.get_answer_send;

	MPID_PSP_packed_msg_cleanup(&gas->msg);

	pscom_request_free(req);
}


static
void io_done_get_answer_recv(pscom_request_t *req)
{
	/* This is an pscom.io_done call. Global lock state undefined! */
	/* save orginal xheader: */
	MPID_PSCOM_XHeader_Rma_get_req_t *xhead_get = &req->xheader.user.get_req;

	MPI_Datatype datatype = req->user->type.get_answer_send.datatype;
	/* reuse req for the answer: */
	pscom_request_get_answer_send_t *gas = &req->user->type.get_answer_send;
	MPID_PSCOM_XHeader_Rma_get_answer_t *xhead_answ = &req->xheader.user.get_answer;
	int ret;

	ret = MPID_PSP_packed_msg_prepare(xhead_get->target_buf, xhead_get->target_count,
					  datatype, &gas->msg);
	assert(ret == MPI_SUCCESS);
	MPID_PSP_packed_msg_pack(xhead_get->target_buf, xhead_get->target_count,
				 datatype, &gas->msg);

	MPID_PSP_Datatype_release(datatype);

	xhead_answ->common.tag = xhead_get->common.tag;
	xhead_answ->common.context_id = xhead_get->common.context_id;
	xhead_answ->common.type = MPID_PSP_MSGTYPE_RMA_GET_ANSWER;
	xhead_answ->common._reserved_ = 0;
	xhead_answ->common.src_rank = -1;

	req->xheader_len = sizeof(*xhead_answ);
	req->data = gas->msg.msg;
	req->data_len = gas->msg.msg_sz;

	req->ops.io_done = io_done_get_answer_send;
	/* req->connection = connection; <- set in MPID_do_recv_rma_get_req() */

	pscom_post_send(req);
}



pscom_request_t *MPID_do_recv_rma_get_req(pscom_connection_t *connection, MPID_PSCOM_XHeader_Rma_get_req_t *xhead_get)
{
	pscom_request_t *req = PSCOM_REQUEST_CREATE();

	req->xheader_len = sizeof(*xhead_get);
	req->data = NULL;
	req->data_len = 0;

	req->ops.io_done = io_done_get_answer_recv;

	/* save datatype */
	req->user->type.get_answer_send.datatype = MPID_PSP_Datatype_decode(xhead_get->encoded_type);

	return req;
}
