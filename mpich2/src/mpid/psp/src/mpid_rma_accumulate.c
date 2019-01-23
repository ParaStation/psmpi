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

// This must be the last include before sysmbols are defined:
#include "mpid_visibility.h"


/*
 *  RMA Accumulate
 */

static
void rma_accumulate_done(pscom_request_t *req)
{
	MPIR_Request *mpid_req = req->user->type.accumulate_send.mpid_req;
	/* This is an pscom.io_done call. Global lock state undefined! */
	MPID_PSP_packed_msg_cleanup(&req->user->type.accumulate_send.msg);
	/* ToDo: this is not threadsave */
	req->user->type.accumulate_send.win_ptr->rma_local_pending_cnt--;
	req->user->type.accumulate_send.win_ptr->rma_local_pending_rank[req->user->type.accumulate_send.target_rank]--;

	if(mpid_req) {
		MPID_PSP_Subrequest_completed(mpid_req);
		MPIR_Request_free(mpid_req);
	} else {
		pscom_request_free(req);
	}
}


static
int MPID_Accumulate_generic(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
			    int target_rank, MPI_Aint target_disp, int target_count,
			    MPI_Datatype target_datatype, MPI_Op op, MPIR_Win *win_ptr,
			    MPIR_Request **request)
{
	int mpi_error = MPI_SUCCESS;
	MPID_PSP_Datatype_info dt_info;
	MPID_PSP_packed_msg_t msg;
	MPID_Win_rank_info *ri = win_ptr->rank_info + target_rank;
	char *target_buf;
#if 0
	fprintf(stderr, "int MPID_Accumulate(origin_addr: %p, origin_count: %d, origin_datatype: %08x,"
		" target_rank: %d, target_disp: %d, target_count: %d, target_datatype: %08x,"
		" op: 0x%x, *win_ptr: %p)\n",
		origin_addr, origin_count, origin_datatype,
		target_rank, target_disp, target_count,
		target_datatype, op, win_ptr);
#endif

	if (unlikely(op == MPI_REPLACE)) {
		/*  MPI_PUT is a special case of MPI_ACCUMULATE, with the operation MPI_REPLACE.
		 |  However, PUT and ACCUMULATE have different constraints on concurrent updates!
		 |  Therefore, in the SHMEM case, the PUT/REPLACE operation must here be locked:
		 */
		if(unlikely(win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED)) {
			MPID_PSP_shm_rma_mutex_lock(win_ptr);
			mpi_error =  MPID_Put_generic(origin_addr, origin_count, origin_datatype,
						target_rank, target_disp, target_count,
						target_datatype, win_ptr, request);
			MPID_PSP_shm_rma_mutex_unlock(win_ptr);
			return mpi_error;
		} else {
			return MPID_Put_generic(origin_addr, origin_count, origin_datatype,
						target_rank, target_disp, target_count,
						target_datatype, win_ptr, request);
		}
	}

	/* Datatype */
	MPID_PSP_Datatype_get_info(target_datatype, &dt_info);

	if(request) {
		*request = MPIR_Request_create(MPIR_REQUEST_KIND__SEND);
		(*request)->comm = win_ptr->comm_ptr;
		MPIR_Comm_add_ref(win_ptr->comm_ptr);
	}

	if (unlikely(target_rank == MPI_PROC_NULL)) {

		goto fn_completed;
	}


	/* Request-based RMA operations are only valid within a passive target epoch! */
	if(request && win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK && win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK_ALL) {
		mpi_error = MPI_ERR_RMA_SYNC;
		goto err_sync_rma;
	}

	/* Check that we are within an access/exposure epoch: */
	if (win_ptr->epoch_state == MPID_PSP_EPOCH_NONE) {
		mpi_error = MPI_ERR_RMA_SYNC;
		goto err_sync_rma;
	}

	/* Track access epoch state: */
	if (win_ptr->epoch_state == MPID_PSP_EPOCH_FENCE_ISSUED) {
		win_ptr->epoch_state = MPID_PSP_EPOCH_FENCE;
	}


	/* Data */
	mpi_error = MPID_PSP_packed_msg_prepare(origin_addr, origin_count, origin_datatype, &msg);
	if (unlikely(mpi_error != MPI_SUCCESS)) goto err_create_packed_msg;

	MPID_PSP_packed_msg_pack(origin_addr, origin_count, origin_datatype, &msg);

	target_buf = (char *) ri->base_addr + ri->disp_unit * target_disp;

	/* If the acc is a local operation, do it here */
	if (target_rank == win_ptr->rank || win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED) {

		if (target_rank != win_ptr->rank) {
			int disp_unit;
			void* base;

			MPID_PSP_shm_rma_get_base(win_ptr, target_rank, &disp_unit, &base);

			assert(ri->disp_unit == disp_unit);
			target_buf = (char *) base + disp_unit * target_disp;

			/* accumulate may be executed concurrently --> locking required! */
			MPID_PSP_shm_rma_mutex_lock(win_ptr);
			MPID_PSP_packed_msg_acc(target_buf, target_count, target_datatype,
						msg.msg, msg.msg_sz, op);
			MPID_PSP_shm_rma_mutex_unlock(win_ptr);

		} else {
			/* This is a local acc, but do locking just in SHMEM case! */
			if(unlikely(win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED)) {

				/* in case of a COMM_SELF clone, mutex_lock()/unlock() will just act as no-ops: */
				MPID_PSP_shm_rma_mutex_lock(win_ptr);
				MPID_PSP_packed_msg_acc(target_buf, target_count, target_datatype,
							msg.msg, msg.msg_sz, op);
				MPID_PSP_shm_rma_mutex_unlock(win_ptr);
			} else {
				/* this is a local operation on non-shared memory: */
				MPID_PSP_packed_msg_acc(target_buf, target_count, target_datatype,
							msg.msg, msg.msg_sz, op);
			}
		}

		MPID_PSP_packed_msg_cleanup(&msg);

		goto fn_completed;
	}

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
		unsigned int	xheader_len	= sizeof(MPID_PSCOM_XHeader_Rma_accumulate_t) + encode_dt_size;
		pscom_request_t *req = pscom_request_create(xheader_len, sizeof(pscom_request_accumulate_send_t));
		MPID_PSCOM_XHeader_Rma_accumulate_t *xheader = &req->xheader.user.accumulate;

		/* encoded datatype too large for xheader? */
		assert(xheader_len < (1<<(8*sizeof(((struct PSCOM_header_net*)0)->xheader_len))));

		req->user->type.accumulate_send.msg = msg;
		req->user->type.accumulate_send.win_ptr = win_ptr;

		MPID_PSP_Datatype_encode(&dt_info, &xheader->encoded_type);

		xheader->common.tag = 0;
		xheader->common.context_id = 0;
		xheader->common.type = MPID_PSP_MSGTYPE_RMA_ACCUMULATE;
		xheader->common._reserved_ = 0;
		xheader->common.src_rank = win_ptr->rank;

		/* xheader->target_disp = target_disp; */
		xheader->target_count = target_count;
		xheader->target_buf = target_buf;
/*		xheader->epoch = ri->epoch_origin; */
		xheader->win_ptr = ri->win_ptr; /* remote win_ptr */
		xheader->op = op; /* ToDo: check: is op a buildin op? */

		req->xheader_len = xheader_len;

		req->data = msg.msg;
		req->data_len = msg.msg_sz;

		req->ops.io_done = rma_accumulate_done;
		req->user->type.accumulate_send.target_rank = target_rank;
		req->connection = ri->con;

		win_ptr->rma_local_pending_cnt++;
		win_ptr->rma_local_pending_rank[target_rank]++;
		win_ptr->rma_puts_accs[target_rank]++;

		if(request) {
			MPIR_Request *mpid_req = *request;
			/* TODO: Use a new and 'acc_send'-dedicated MPID_DEV_Request_create() */
			/*       instead of allocating and overloading a common send request. */
			pscom_request_free(mpid_req->dev.kind.common.pscom_req);
			mpid_req->dev.kind.common.pscom_req = req;
			MPIR_Request_add_ref(mpid_req);
			req->user->type.accumulate_send.mpid_req = mpid_req;
		} else {
			req->user->type.accumulate_send.mpid_req = NULL;
		}

		pscom_post_send(req);
	}

fn_exit:
	return MPI_SUCCESS;
fn_completed:
	if(request) {
		MPIDI_PSP_Request_set_completed(*request);
	}
	return MPI_SUCCESS;
	/* --- */
err_exit:
	if(request) {
		MPIDI_PSP_Request_set_completed(*request);
		MPIR_Request_free(*request);
	}
	return mpi_error;
	/* --- */
err_create_packed_msg:
	goto err_exit;
err_sync_rma:
	goto err_exit;
}


static
void rma_accumulate_receive_done(pscom_request_t *req)
{
	/* This is an pscom.io_done call. Global lock state undefined! */
	MPID_PSCOM_XHeader_Rma_accumulate_t *xhead_rma = &req->xheader.user.accumulate;
	pscom_request_accumulate_recv_t *rpr = &req->user->type.accumulate_recv;
/*
  void *origin_addr		= req->data;
  int origin_count		= req->data_len / sizeof(basic buildin type);
  MPI_Datatype origin_datatype	= basic buildin type;
*/

	void *target_buf		= xhead_rma->target_buf;
	int target_count		= xhead_rma->target_count;
	MPI_Datatype target_datatype	= rpr->datatype;
	MPI_Op op			= xhead_rma->op;

	MPIR_Win *win_ptr = xhead_rma->win_ptr;

	MPID_PSP_packed_msg_acc(target_buf, target_count, target_datatype,
				req->data, req->data_len, op);

	MPID_PSP_Datatype_release(target_datatype);
	/* ToDo: this is not threadsave */
	win_ptr->rma_puts_accs_received ++;
	pscom_request_free(req);
}


pscom_request_t *MPID_do_recv_rma_accumulate(pscom_connection_t *con, pscom_header_net_t *header_net)
{
	MPID_PSCOM_XHeader_Rma_accumulate_t *xhead_rma = &header_net->xheader->user.accumulate;

	MPI_Datatype datatype = MPID_PSP_Datatype_decode(xhead_rma->encoded_type);

	pscom_request_t *req =
		pscom_request_create(sizeof(MPID_PSCOM_XHeader_Rma_accumulate_t),
				     sizeof(pscom_request_accumulate_recv_t) + header_net->data_len);

	pscom_request_accumulate_recv_t *rpr = &req->user->type.accumulate_recv;


	/* Receive the packed_msg into request->user space */
	req->xheader_len = sizeof(MPID_PSCOM_XHeader_Rma_accumulate_t);
	req->data_len = header_net->data_len;
	req->data = &rpr->packed_msg;

	rpr->datatype = datatype;
	req->ops.io_done = rma_accumulate_receive_done;

	return req;
}


int MPID_Accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
		    int target_rank, MPI_Aint target_disp, int target_count,
		    MPI_Datatype target_datatype, MPI_Op op, MPIR_Win *win_ptr)
{
	return MPID_Accumulate_generic(origin_addr, origin_count, origin_datatype,
				       target_rank, target_disp, target_count, target_datatype,
				       op, win_ptr, NULL);
}

int MPID_Raccumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
		     int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype,
		     MPI_Op op, MPIR_Win *win_ptr, MPIR_Request **request)
{
	return MPID_Accumulate_generic((void*)origin_addr, origin_count, origin_datatype,
				       target_rank, target_disp, target_count, target_datatype,
				       op, win_ptr, request);
}


/***********************************************************************************************************
 *   RMA-3.0 Get & Accumulate / Fetch & Op Functions:
 */

static
int MPID_Get_accumulate_generic(const void *origin_addr, int origin_count,
				MPI_Datatype origin_datatype, void *result_addr, int result_count,
				MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
				int target_count, MPI_Datatype target_datatype, MPI_Op op, MPIR_Win *win_ptr,
				MPIR_Request **request)
{
	if (unlikely(target_rank == MPI_PROC_NULL)) {

		if(request) {
			*request = MPIR_Request_create(MPIR_REQUEST_KIND__SEND);
			(*request)->comm = win_ptr->comm_ptr;
			MPIR_Comm_add_ref(win_ptr->comm_ptr);
			MPIDI_PSP_Request_set_completed(*request);
		}

		return MPI_SUCCESS;
	}

	if (unlikely(op == MPI_NO_OP)) {
		return MPID_Get_generic(result_addr, result_count, result_datatype,
					target_rank, target_disp, target_count,
					target_datatype, win_ptr, request);
	}

	if(1) { /* TODO: This implementation is just based on the common Get/Accumulate ops (plus some additional internal locking): */

		MPID_Win_lock_internal(target_rank, win_ptr);

		MPID_Get(result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, win_ptr);

		MPID_Win_wait_local_completion(target_rank, win_ptr);

		MPID_Accumulate_generic((void*)origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win_ptr, request);

		MPID_Win_unlock_internal(target_rank, win_ptr);
	}
	else {
		/* TODO: A dedicated Get_accumulate() implementation goes here... */
		assert(0);
	}

	return MPI_SUCCESS;
}

int MPID_Get_accumulate(const void *origin_addr, int origin_count,
			MPI_Datatype origin_datatype, void *result_addr, int result_count,
			MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
			int target_count, MPI_Datatype target_datatype, MPI_Op op, MPIR_Win *win_ptr)
{
	return MPID_Get_accumulate_generic(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype,
					   target_rank, target_disp, target_count, target_datatype, op, win_ptr, NULL);
}

int MPID_Rget_accumulate(const void *origin_addr, int origin_count,
			 MPI_Datatype origin_datatype, void *result_addr, int result_count,
			 MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
			 int target_count, MPI_Datatype target_datatype, MPI_Op op, MPIR_Win *win_ptr,
			 MPIR_Request **request)
{
	return MPID_Get_accumulate_generic(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype,
					   target_rank, target_disp, target_count, target_datatype, op, win_ptr, request);
}


int MPID_Fetch_and_op(const void *origin_addr, void *result_addr,
		      MPI_Datatype datatype, int target_rank, MPI_Aint target_disp,
		      MPI_Op op, MPIR_Win *win)
{
	if (unlikely(target_rank == MPI_PROC_NULL)) {
		goto fn_exit;
	}

	if(1) { /* TODO: This implementation is just based on Get&Accumulate: */

		return MPID_Get_accumulate(origin_addr, 1, datatype, result_addr, 1, datatype, target_rank, target_disp, 1, datatype, op, win);
	}
	else {
		/* TODO: A dedicated Fetch_and_op() implementation goes here... */
		assert(0);
	}
fn_exit:
	return MPI_SUCCESS;
}

int MPID_Compare_and_swap(const void *origin_addr, const void *compare_addr,
			  void *result_addr, MPI_Datatype datatype, int target_rank,
			  MPI_Aint target_disp, MPIR_Win *win_ptr)
{
	if(1) { /* TODO: This implementation is just based on Get (plus some additional internal locking): */

		if (unlikely(target_rank == MPI_PROC_NULL)) {
			goto fn_exit;
		}

		MPID_Win_lock_internal(target_rank, win_ptr);

		MPID_Get(result_addr, 1, datatype, target_rank, target_disp, 1, datatype, win_ptr);

		MPID_Win_wait_local_completion(target_rank, win_ptr);

		if(MPIR_Compare_equal(compare_addr, result_addr, datatype)) {

			MPID_Put((void*)origin_addr, 1, datatype, target_rank, target_disp, 1, datatype, win_ptr);
		}

		MPID_Win_unlock_internal(target_rank, win_ptr);
	}
	else {
		/* TODO: A dedicated Compare_and_swap() implementation goes here... */
		assert(0);
	}
fn_exit:
	return MPI_SUCCESS;
}
