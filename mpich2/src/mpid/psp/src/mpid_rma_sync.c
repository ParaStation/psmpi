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
#include "mpid_psp_datatype.h"
#include "mpid_psp_packed_msg.h"
#include "mpid_psp_request.h"


#define TAG_POST	11
#define TAG_COMPLETE	12

/*
 * array containing "1"
 */

static unsigned int array_1_size = 0;
static int *array_1 = NULL;

static void cleanup_array_1(void)
{
	if (array_1) {
		MPIU_Free(array_1);
		array_1 = NULL;
		array_1_size = 0;
	}
}

static
int *get_array_1(unsigned int size)
{
	if (size > array_1_size) {
		unsigned int i;
		cleanup_array_1();
		array_1 = MPIU_Malloc(sizeof(*array_1) * size);
		assert(array_1 != NULL);
		array_1_size = size;

		for (i = 0; i < size; ++i) {
			array_1[i] = 1;
		}
	}
	return array_1;
}

/*
 * array containing 0,1,2,3,4....
 */
static unsigned int array_123_size = 0;
static int *array_123 = NULL;

static void cleanup_array_123(void)
{
	if (array_123) {
		MPIU_Free(array_123);
		array_123 = NULL;
		array_123_size = 0;
	}
}

static
int *get_array_123(unsigned int size)
{
	if (size > array_123_size) {
		unsigned int i;
		cleanup_array_123();
		array_123 = MPIU_Malloc(sizeof(*array_123) * size);
		assert(array_123 != NULL);
		array_123_size = size;

		for (i = 0; i < size; ++i) {
			array_123[i] = i;
		}
	}
	return array_123;
}

/*----------------------*/

/* translate group to comm ranks.
 * return int array with ranks of comm of size group_ptr->size.
 * Caller must call MPIU_Free on result after usage. */
static
int *get_group_ranks(MPID_Comm *comm_ptr, MPID_Group *group_ptr)
{
	int grp_sz = group_ptr->size;
	int *arr123 = get_array_123(grp_sz);
	int *res = MPIU_Malloc(sizeof(*res) * grp_sz);
	MPID_Group *win_grp_ptr = NULL;
	int mpi_errno;

	mpi_errno = MPIR_Comm_group_impl(comm_ptr, &win_grp_ptr);
	if (mpi_errno) goto fn_fail;

	/* ToDo: Optimize translate_ranks in "mpi/group/group_translate_ranks.c" */
	mpi_errno = MPIR_Group_translate_ranks_impl(group_ptr, grp_sz,
						    arr123, win_grp_ptr, res);
	if (mpi_errno) goto fn_fail;

fn_exit:
	if (win_grp_ptr) {
		MPIR_Group_free_impl(win_grp_ptr); /* ignore error */
		win_grp_ptr = NULL;
	}
	return res;
	/* --- */
fn_fail:
	{
		int i;
		for (i = 0; i < grp_sz; i++) {
			res[i] = MPI_UNDEFINED;
		}
	}
	goto fn_exit;
}


void MPID_PSP_rma_cleanup(void)
{
	cleanup_array_1();
	cleanup_array_123();
}


static
int _accept_never(pscom_request_t *request,
		  pscom_connection_t *connection,
		  pscom_header_net_t *header_net)
{
	return 0;
}


int MPID_Win_fence(int assert, MPID_Win *win_ptr)
{

	int mpi_errno, comm_size;
	MPID_Comm *comm_ptr;
	int * recvcnts;
	unsigned int total_rma_puts_accs;
	int errflag = 0;
	pscom_request_t *req_any = NULL;

	comm_ptr = win_ptr->comm_ptr;
	comm_size = comm_ptr->local_size;
	recvcnts = get_array_1(comm_size);


	mpi_errno = MPIR_Reduce_scatter_impl(win_ptr->rma_puts_accs,
					     &total_rma_puts_accs, recvcnts,
					     MPI_INT, MPI_SUM, comm_ptr, &errflag);

	if (mpi_errno != MPI_SUCCESS)
		return mpi_errno;

	while ((win_ptr->rma_puts_accs_received != total_rma_puts_accs) ||
	       win_ptr->rma_local_pending_cnt) {

		if (!req_any) {
			/* Post a dummy ANY_SOURCE receive to listen on all
			   connections for incoming RMA messages. */
			req_any = pscom_request_create(0, 0);
			req_any->xheader_len = 0;
			req_any->data_len = 0;
			req_any->connection = NULL;
			req_any->socket = MPIDI_Process.socket; /* ToDo: get socket from comm? */
			req_any->ops.recv_accept = _accept_never;
			pscom_post_recv(req_any);
		}

		MPID_PSP_LOCKFREE_CALL(pscom_wait_any());
	}
	if (req_any) {
		/* cancel and free the any_source dummy request */
		int rc = pscom_cancel_recv(req_any);
		assert(pscom_req_is_done(req_any));
		pscom_request_free(req_any);
	}

	return MPIR_Barrier_impl(comm_ptr, &errflag);
}


#define DEBUG_START_POST 0


int MPID_Win_post(MPID_Group *group_ptr, int assert, MPID_Win *win_ptr)
{
	if (win_ptr->ranks_post) {
		/* Error: win_post already called */
		return MPI_ERR_ARG;
	}

	int *ranks;
	int ranks_sz = group_ptr->size;
	int i;
	int dummy;
	int mpi_errno = MPI_SUCCESS;

	ranks = get_group_ranks(win_ptr->comm_ptr, group_ptr);
	for (i = 0; i < ranks_sz; i++) {
		int rank = ranks[i];
		int rc;
		MPID_Request * sreq = NULL;

		/* Send TAG_POST to MPID_Win_start of rank; */
		rc = MPID_Send(&dummy, 0 , MPI_INT, rank, TAG_POST, win_ptr->comm_ptr,
			       MPID_CONTEXT_INTRA_PT2PT, &sreq);
		if (rc != MPI_SUCCESS) {
			mpi_errno = rc;
		}
	}

	win_ptr->ranks_post = ranks;
	win_ptr->ranks_post_sz = ranks_sz;

	if (DEBUG_START_POST) { /* Debug: */
		printf("%s(group_ptr %p, win_ptr %p) ranks:", __func__, group_ptr, win_ptr);
		for (i = 0; i < ranks_sz; i++) {
			printf(" %d", ranks[i]);
		}
		printf("\n");
	}

	return mpi_errno;
}


int MPID_Win_start(MPID_Group *group_ptr, int assert, MPID_Win *win_ptr)
{
	if (win_ptr->ranks_start) {
		/* Error: win_start already called */
		return MPI_ERR_ARG;
	}
	int *ranks;
	int ranks_sz = group_ptr->size;
	int i;
	int dummy;
	int mpi_errno = MPI_SUCCESS;

	ranks = get_group_ranks(win_ptr->comm_ptr, group_ptr);
	for (i = 0; i < ranks_sz; i++) {
		int rank = ranks[i];
		MPI_Status status;
		int rc;
		MPID_Request * rreq = NULL;

		/* Recv TAG_POST from MPID_Win_post of rank; */
		rc = MPID_Recv(&dummy, 0, MPI_INT, rank, TAG_POST, win_ptr->comm_ptr,
			       MPID_CONTEXT_INTRA_PT2PT, &status, &rreq);
		if (rc != MPI_SUCCESS) {
			mpi_errno = rc;
		}
	}

	win_ptr->ranks_start = ranks;
	win_ptr->ranks_start_sz = ranks_sz;

	if (DEBUG_START_POST) { /* Debug: */
		printf("%s(group_ptr %p, win_ptr %p) ranks:", __func__, group_ptr, win_ptr);
		for (i = 0; i < ranks_sz; i++) {
			printf(" %d", ranks[i]);
		}
		printf("\n");
	}

	return mpi_errno;
}


int MPID_Win_complete(MPID_Win *win_ptr)
{
	if (!win_ptr->ranks_start) {
		return MPI_ERR_ARG;
	}

	/* Wait for local rma opperations */
	while (win_ptr->rma_local_pending_cnt) {
		MPID_PSP_LOCKFREE_CALL(pscom_wait_any());
	}

	int *ranks = win_ptr->ranks_start;
	int ranks_sz = win_ptr->ranks_start_sz;
	int i;
	int dummy;
	int mpi_errno = MPI_SUCCESS;

	for (i = 0; i < ranks_sz; i++) {
		int rank = ranks[i];
		int rc;
		MPID_Request *sreq = NULL;

		/* Send TAG_COMPLETE to MPID_Win_wait of rank */
		rc = MPID_Send(&dummy, 0 , MPI_INT, rank, TAG_COMPLETE, win_ptr->comm_ptr,
			       MPID_CONTEXT_INTRA_PT2PT, &sreq);
		if (rc != MPI_SUCCESS) {
			mpi_errno = rc;
		}
	}

	if (DEBUG_START_POST) { /* Debug: */
		printf("%s(win_ptr %p) ranks:", __func__, win_ptr);
		for (i = 0; i < ranks_sz; i++) {
			printf(" %d", ranks[i]);
		}
		printf("\n");
	}

	win_ptr->ranks_start = NULL;
	win_ptr->ranks_start_sz = 0;
	MPIU_Free(ranks);

	return mpi_errno;
}


int MPID_Win_wait(MPID_Win *win_ptr)
{
	if (!win_ptr->ranks_post) {
		return MPI_ERR_ARG;
	}

	int *ranks = win_ptr->ranks_post;
	int ranks_sz = win_ptr->ranks_post_sz;
	int i;
	int dummy;
	int mpi_errno = MPI_SUCCESS;

	for (i = 0; i < ranks_sz; i++) {
		int rank = ranks[i];
		MPI_Status status;
		int rc;
		MPID_Request *rreq = NULL;

		/* Recv TAG_COMPLETE from MPID_Win_complete of rank */
		rc = MPID_Recv(&dummy, 0, MPI_INT, rank, TAG_COMPLETE, win_ptr->comm_ptr,
			       MPID_CONTEXT_INTRA_PT2PT, &status, &rreq);
		if (rc != MPI_SUCCESS) {
			/* Set mpi_errno, but stay in the loop and receive all other TAG_COMPLETE's */
			mpi_errno = rc;
		}
	}

	if (DEBUG_START_POST) { /* Debug: */
		printf("%s(win_ptr %p) ranks:", __func__, win_ptr);
		for (i = 0; i < ranks_sz; i++) {
			printf(" %d", ranks[i]);
		}
		printf("\n");
	}

	win_ptr->ranks_post = NULL;
	win_ptr->ranks_post_sz = 0;
	MPIU_Free(ranks);

	return mpi_errno;
}


int MPID_Win_test(MPID_Win *win_ptr, int *flag)
{
	if (!win_ptr->ranks_post) {
		return MPI_ERR_ARG;
	}

	int *ranks = win_ptr->ranks_post;
	int ranks_sz = win_ptr->ranks_post_sz;
	int i;
	int mpi_errno = MPI_SUCCESS;
	int ret_flag = 1;

	for (i = 0; i < ranks_sz; i++) {
		int rank = ranks[i];
		MPI_Status status;
		int rc;
		int aflag;

		/* Recv TAG_COMPLETE from MPID_Win_complete of rank */
		rc = MPID_Iprobe(rank, TAG_COMPLETE, win_ptr->comm_ptr, MPID_CONTEXT_INTRA_PT2PT, &aflag, &status);

		if (rc != MPI_SUCCESS) {
			mpi_errno = rc;
			ret_flag = 0;
			break;
		}
		if (!aflag) {
			ret_flag = 0;
			break;
		}
	}

	if ((mpi_errno == MPI_SUCCESS) && ret_flag) {
		mpi_errno = MPID_Win_wait(win_ptr);
		assert(ret_flag == 1);
	}
	*flag = ret_flag;

	return mpi_errno;
}


static
void MPID_PSP_SendRmaCtrl(MPID_Win *win_ptr, MPID_Comm *comm, pscom_connection_t *con,
			  int dest_rank, enum MPID_PSP_MSGTYPE msgtype)
{
	MPID_PSCOM_XHeader_Rma_lock_t xhead;

	MPID_Win_rank_info *ri = win_ptr->rank_info + dest_rank;

	xhead.common.tag = 0;
	xhead.common.context_id = comm->context_id;
	xhead.common.type = msgtype;
	xhead.common._reserved_ = 0;
	xhead.common.src_rank = comm->rank;
	xhead.win_ptr = ri->win_ptr;

	pscom_send(con, &xhead, sizeof(xhead), NULL, 0);
}


static inline
int do_trylock(MPID_Win *win_ptr, pscom_request_t *req)
{
	int exclusive = req->user->type.rma_lock.exclusive;

	if (!win_ptr->lock_cnt ||
	    (!win_ptr->lock_exclusive && !exclusive)) {
		/* unlocked or shared locked */

		/* lock */
		win_ptr->lock_cnt++;
		win_ptr->lock_exclusive = exclusive;

		return 1;
	} else {
		return 0;
	}
}


static
void do_lock(MPID_Win *win_ptr, pscom_request_t *req)
{
	if (do_trylock(win_ptr, req)) {
		/* window locked, send ack */
		pscom_post_send(req);
	} else {
		/* schedule this lock */
		list_add_tail(&req->user->type.rma_lock.next, &win_ptr->lock_list);
	}
}


static
void do_unlock(MPID_Win *win_ptr, pscom_request_t *req)
{
	/* send answer */
	pscom_post_send(req);
	req = NULL;

	win_ptr->lock_cnt--; /* unlock */

	if (!list_empty(&win_ptr->lock_list)) {
		pscom_request_rma_lock_t *rma_lock =
			list_entry(win_ptr->lock_list.next, pscom_request_rma_lock_t, next);

		pscom_request_t *lreq = rma_lock->req;

		list_del(&rma_lock->next);

		if (do_trylock(win_ptr, lreq)) {
			/* window locked, send ack */
			pscom_post_send(lreq);
		} else {
			/* reschedule this lock */
			list_add(&rma_lock->next, &win_ptr->lock_list);
		}
	}
}


static
void MPID_do_recv_rma_lock_req(pscom_request_t *req, int exclusive)
{
	/* This is an pscom callback. Global lock state undefined! */
	MPID_PSCOM_XHeader_Rma_lock_t *xhead_lock = &req->xheader.user.rma_lock;

	MPID_Win *win_ptr = xhead_lock->win_ptr;

	/* reuse orignal header, but overwrite type,src_rank and xheader_len: */
	xhead_lock->common.type = MPID_PSP_MSGTYPE_RMA_LOCK_ANSWER;
	xhead_lock->common.src_rank = MPI_ANY_SOURCE;

	req->xheader_len = sizeof(xhead_lock->common);

	req->user->type.rma_lock.exclusive = exclusive;
	req->user->type.rma_lock.req = req;

	req->ops.io_done = pscom_request_free;

	do_lock(win_ptr, req);
}


void MPID_do_recv_rma_lock_exclusive_req(pscom_request_t *req)
{
	/* This is an pscom callback. Global lock state undefined! */
	MPID_do_recv_rma_lock_req(req, 1);
}


void MPID_do_recv_rma_lock_shared_req(pscom_request_t *req)
{
	/* This is an pscom callback. Global lock state undefined! */
	MPID_do_recv_rma_lock_req(req, 0);
}


void MPID_do_recv_rma_unlock_req(pscom_request_t *req)
{
	/* This is an pscom callback. Global lock state undefined! */
	MPID_PSCOM_XHeader_Rma_lock_t *xhead_lock = &req->xheader.user.rma_lock;

	MPID_Win *win_ptr = xhead_lock->win_ptr;

	/* reuse orignal header, but overwrite type,src_rank and xheader_len: */
	xhead_lock->common.type = MPID_PSP_MSGTYPE_RMA_UNLOCK_ANSWER;
	xhead_lock->common.src_rank = MPI_ANY_SOURCE;

	req->xheader_len = sizeof(xhead_lock->common);

	req->ops.io_done = pscom_request_free;

	do_unlock(win_ptr, req);
}


int MPID_Win_lock(int lock_type, int dest, int assert, MPID_Win *win_ptr)
{
	MPID_Comm *comm;
	pscom_connection_t *con;
	enum MPID_PSP_MSGTYPE msgt;

	if (lock_type == MPI_LOCK_EXCLUSIVE) {
		msgt = MPID_PSP_MSGTYPE_RMA_LOCK_EXCLUSIVE_REQUEST;
	} else if (lock_type == MPI_LOCK_SHARED) {
		msgt = MPID_PSP_MSGTYPE_RMA_LOCK_SHARED_REQUEST;
	} else {
		return MPI_ERR_ARG;
	}

	comm = win_ptr->comm_ptr;
	con = MPID_PSCOM_rank2connection(comm, dest);

	MPID_PSP_SendRmaCtrl(win_ptr, comm, con, dest, msgt);
	MPID_PSP_RecvCtrl(0/*tag*/, comm->context_id, MPI_ANY_SOURCE, con, MPID_PSP_MSGTYPE_RMA_LOCK_ANSWER);

	return MPI_SUCCESS;
}


int MPID_Win_unlock(int dest, MPID_Win *win_ptr)
{
	MPID_Comm *comm;
	pscom_connection_t *con;

	comm = win_ptr->comm_ptr;
	con = MPID_PSCOM_rank2connection(comm, dest);

	MPID_PSP_SendRmaCtrl(win_ptr, comm, con, dest, MPID_PSP_MSGTYPE_RMA_UNLOCK_REQUEST);
	MPID_PSP_RecvCtrl(0/*tag*/, comm->context_id, MPI_ANY_SOURCE, con, MPID_PSP_MSGTYPE_RMA_UNLOCK_ANSWER);

	return MPI_SUCCESS;
}
