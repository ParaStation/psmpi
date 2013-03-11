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


/*
 *  RMA Put
 */

static
void rma_put_done(pscom_request_t *req)
{
	/* This is an pscom.io_done call. Global lock state undefined! */
	MPID_PSP_packed_msg_cleanup(&req->user->type.put_send.msg);
	/* ToDo: this is not threadsave */
	req->user->type.put_send.win_ptr->rma_local_pending_cnt--;
	pscom_request_free(req);
}


int MPID_Put(void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
	     int target_rank, MPI_Aint target_disp, int target_count,
	     MPI_Datatype target_datatype, MPID_Win *win_ptr)
{
	int ret;
	MPID_PSP_Datatype_info dt_info;
	MPID_PSP_packed_msg_t msg;
	MPID_Win_rank_info *ri = win_ptr->rank_info + target_rank;
	char *target_buf;
#if 0
	fprintf(stderr, "int MPID_Put(origin_addr: %p, origin_count: %d, origin_datatype: %08x,"
		" target_rank: %d, target_disp: %d, target_count: %d, target_datatype: %08x,"
		" *win_ptr: %p)\n",
		origin_addr, origin_count, origin_datatype,
		target_rank, target_disp, target_count,
		target_datatype, win_ptr);
#endif
	/* Datatype */
	MPID_PSP_Datatype_get_info(target_datatype, &dt_info);

	/* Data */
	ret = MPID_PSP_packed_msg_prepare(origin_addr, origin_count, origin_datatype, &msg);
	if (unlikely(ret != MPI_SUCCESS)) goto err_create_packed_msg;

	MPID_PSP_packed_msg_pack(origin_addr, origin_count, origin_datatype, &msg);

	target_buf = (char *) ri->base_addr + ri->disp_unit * target_disp;


	if (0 && MPID_PSP_Datatype_is_contig(&dt_info)) { /* ToDo: reenable pscom buildin rma_write */
		/* Contig message. Use pscom buildin rma */
		pscom_request_t *req = pscom_request_create(0, 0);

		req->data_len = msg.msg_sz;
		req->data = msg.msg;
		req->connection = ri->con;

		/* ToDo: need a new io_done. inside io_done, call MPID_PSP_packed_msg_cleanup(msg)!!! */
		req->ops.io_done = pscom_request_free;
		req->xheader.rma_write.dest = target_buf;

		pscom_post_rma_write(req);

		/* win_ptr->rma_puts_accs[target_rank]++; / ToDo: Howto receive this? */
	} else {
		unsigned int	encode_dt_size	= MPID_PSP_Datatype_get_size(&dt_info);
		unsigned int	xheader_len	= sizeof(MPID_PSCOM_XHeader_Rma_put_t) + encode_dt_size;
		pscom_request_t *req = pscom_request_create(xheader_len, sizeof(pscom_request_put_send_t));
		MPID_PSCOM_XHeader_Rma_put_t *xheader = &req->xheader.user.put;

		req->user->type.put_send.msg = msg;
		req->user->type.put_send.win_ptr = win_ptr;

		MPID_PSP_Datatype_encode(&dt_info, &xheader->encoded_type);

		xheader->common.tag = 0;
		xheader->common.context_id = 0;
		xheader->common.type = MPID_PSP_MSGTYPE_RMA_PUT;
		xheader->common._reserved_ = 0;
		xheader->common.src_rank = win_ptr->rank;

		/* xheader->target_disp = target_disp; */
		xheader->target_count = target_count;
		xheader->target_buf = target_buf;
		/* xheader->epoch = ri->epoch_origin; */
		xheader->win_ptr = ri->win_ptr; /* remote win_ptr */

		req->xheader_len = xheader_len;

		req->data = msg.msg;
		req->data_len = msg.msg_sz;

		req->ops.io_done = rma_put_done;

		req->connection = ri->con;

		win_ptr->rma_local_pending_cnt++;
		win_ptr->rma_puts_accs[target_rank]++;

		pscom_post_send(req);
	}

	return MPI_SUCCESS;
	/* --- */
err_create_packed_msg:
	return ret;
}


static
void rma_put_receive_done(pscom_request_t *req)
{
	/* This is an pscom.io_done call. Global lock state undefined! */
	pscom_request_put_recv_t *rpr = &req->user->type.put_recv;
	MPID_PSCOM_XHeader_Rma_put_t *xhead_rma = &req->xheader.user.put;
/*
	printf("Packed des:(%d) %s\n", req->data_len,
	       pscom_dumpstr(req->data, pscom_min(req->data_len, 64)));
	fflush(stdout);
*/

	MPID_PSP_packed_msg_unpack(xhead_rma->target_buf, xhead_rma->target_count, rpr->datatype,
				   &rpr->msg, req->data_len);
	/* if noncontig, cleanup temp buffer and datatype */
	MPID_PSP_packed_msg_cleanup_datatype(&rpr->msg, rpr->datatype);

	/* ToDo: This is not treadsave. */
	xhead_rma->win_ptr->rma_puts_accs_received ++;
	pscom_request_free(req);
}


pscom_request_t *MPID_do_recv_rma_put(pscom_connection_t *con, MPID_PSCOM_XHeader_Rma_put_t *xhead_rma)
{
	MPI_Datatype datatype = MPID_PSP_Datatype_decode(xhead_rma->encoded_type);

	pscom_request_t *req = PSCOM_REQUEST_CREATE();
	pscom_request_put_recv_t *rpr = &req->user->type.put_recv;

	MPID_PSP_packed_msg_prepare(xhead_rma->target_buf,
				    xhead_rma->target_count, datatype, &rpr->msg);

	req->xheader_len = sizeof(req->xheader.user.put);
	req->data_len = rpr->msg.msg_sz;
	req->data = rpr->msg.msg;
	req->ops.io_done = rma_put_receive_done;

	rpr->datatype = datatype;

	return req;
}
