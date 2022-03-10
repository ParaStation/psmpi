/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2022 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"
#include "mpid_psp_datatype.h"
#include "mpid_psp_packed_msg.h"
#include "mpid_psp_request.h"

/*
 * array containing "1"
 */

static unsigned int array_1_size = 0;
static int *array_1 = NULL;

static void cleanup_array_1(void)
{
	if (array_1) {
		MPL_free(array_1);
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
		array_1 = MPL_malloc(sizeof(*array_1) * size, MPL_MEM_OTHER);
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
		MPL_free(array_123);
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
		array_123 = MPL_malloc(sizeof(*array_123) * size, MPL_MEM_OTHER);
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
 * Caller must call MPL_free on result after usage. */
static
int *get_group_ranks(MPIR_Comm *comm_ptr, MPIR_Group *group_ptr)
{
	int grp_sz = group_ptr->size;
	int *arr123 = get_array_123(grp_sz);
	int *res = MPL_malloc(sizeof(*res) * grp_sz, MPL_MEM_OTHER);
	MPIR_Group *win_grp_ptr = NULL;
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

	MPID_PSP_rma_pscom_sockets_cleanup();
}



int MPID_Win_fence(int assert, MPIR_Win *win_ptr)
{
	int mpi_errno, comm_size;
	MPIR_Comm *comm_ptr;
	int * recvcnts;
	unsigned int total_rma_puts_accs;
	MPIR_Errflag_t errflag = 0;

	if(win_ptr->epoch_state != MPID_PSP_EPOCH_NONE && win_ptr->epoch_state != MPID_PSP_EPOCH_FENCE && win_ptr->epoch_state != MPID_PSP_EPOCH_FENCE_ISSUED) {
		return MPI_ERR_RMA_SYNC;
	}

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

		MPID_PSP_LOCKFREE_CALL(pscom_wait_any());
	}

	if(win_ptr->epoch_state == MPID_PSP_EPOCH_NONE) {
		win_ptr->epoch_state = MPID_PSP_EPOCH_FENCE_ISSUED;
	} else {
		if(win_ptr->epoch_state == MPID_PSP_EPOCH_FENCE) {
			win_ptr->epoch_state = MPID_PSP_EPOCH_FENCE_ISSUED;
		} else {
			assert(win_ptr->epoch_state == MPID_PSP_EPOCH_FENCE_ISSUED);
		}
	}

	return MPIR_Barrier_impl(comm_ptr, &errflag);
}


#define DEBUG_START_POST 0


int MPID_Win_post(MPIR_Group *group_ptr, int assert, MPIR_Win *win_ptr)
{
	int *ranks;
	int ranks_sz = group_ptr->size;
	int i;
	int mpi_errno = MPI_SUCCESS;

#if 0
	if (win_ptr->ranks_post) {
		/* Error: win_post already called */
		return MPI_ERR_ARG;
	}
#else
	if(win_ptr->epoch_state != MPID_PSP_EPOCH_NONE &&
	   win_ptr->epoch_state != MPID_PSP_EPOCH_FENCE_ISSUED &&
	   win_ptr->epoch_state != MPID_PSP_EPOCH_START) {
		return MPI_ERR_RMA_SYNC;
	}

	/* Track access epoch state */
	if (win_ptr->epoch_state == MPID_PSP_EPOCH_START) {
		win_ptr->epoch_state = MPID_PSP_EPOCH_PSCW;
	}
	else {
		win_ptr->epoch_state = MPID_PSP_EPOCH_POST;
	}
#endif

	ranks = get_group_ranks(win_ptr->comm_ptr, group_ptr);
	for (i = 0; i < ranks_sz; i++) {
		int rank = ranks[i];
		pscom_connection_t *con = MPID_PSCOM_rank2connection(win_ptr->comm_ptr, rank);

		/* Send tag POST to MPID_Win_start of rank: */
		MPIDI_PSP_SendCtrl(MPIDI_PSP_CTRL_TAG__WIN__POST, win_ptr->comm_ptr->context_id + MPIR_CONTEXT_INTRA_PT2PT, win_ptr->comm_ptr->rank, con, MPID_PSP_MSGTYPE_RMA_SYNC);
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


int MPID_Win_start(MPIR_Group *group_ptr, int assert, MPIR_Win *win_ptr)
{
	int *ranks;
	int ranks_sz = group_ptr->size;
	int i;
	int mpi_errno = MPI_SUCCESS;

#if 0
	if (win_ptr->ranks_start) {
		/* Error: win_start already called */
		return MPI_ERR_ARG;
	}
#else
	if(win_ptr->epoch_state != MPID_PSP_EPOCH_NONE &&
	   win_ptr->epoch_state != MPID_PSP_EPOCH_FENCE_ISSUED &&
	   win_ptr->epoch_state != MPID_PSP_EPOCH_POST) {
		return MPI_ERR_RMA_SYNC;
	}

	/* Track access epoch state */
	if (win_ptr->epoch_state == MPID_PSP_EPOCH_POST) {
		win_ptr->epoch_state = MPID_PSP_EPOCH_PSCW;
	}
	else {
		win_ptr->epoch_state = MPID_PSP_EPOCH_START;
	}
#endif

	ranks = get_group_ranks(win_ptr->comm_ptr, group_ptr);
	for (i = 0; i < ranks_sz; i++) {
		int rank = ranks[i];
		pscom_connection_t *con = MPID_PSCOM_rank2connection(win_ptr->comm_ptr, rank);

		/* Recv tag POST from MPID_Win_post of rank: */
		MPIDI_PSP_RecvCtrl(MPIDI_PSP_CTRL_TAG__WIN__POST, win_ptr->comm_ptr->recvcontext_id + MPIR_CONTEXT_INTRA_PT2PT, rank, con, MPID_PSP_MSGTYPE_RMA_SYNC);
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


int MPID_Win_complete(MPIR_Win *win_ptr)
{
	int *ranks = win_ptr->ranks_start;
	int ranks_sz = win_ptr->ranks_start_sz;
	int i;
	int mpi_errno = MPI_SUCCESS;

#if 0
	if (!win_ptr->ranks_start) {
		return MPI_ERR_ARG;
	}
#else
	if(win_ptr->epoch_state != MPID_PSP_EPOCH_PSCW && win_ptr->epoch_state != MPID_PSP_EPOCH_START) {
		return MPI_ERR_RMA_SYNC;
	}

	/* Track access epoch state */
	if (win_ptr->epoch_state == MPID_PSP_EPOCH_PSCW) {
		win_ptr->epoch_state = MPID_PSP_EPOCH_POST;
	}
	else {
		win_ptr->epoch_state = MPID_PSP_EPOCH_NONE;
	}
#endif

	/* Wait for local rma opperations */
	while (win_ptr->rma_local_pending_cnt) {
		MPID_PSP_LOCKFREE_CALL(pscom_wait_any());
	}

	for (i = 0; i < ranks_sz; i++) {
		int rank = ranks[i];
		pscom_connection_t *con = MPID_PSCOM_rank2connection(win_ptr->comm_ptr, rank);

		/* Send tag COMPLETE to MPID_Win_wait of rank: */
		MPIDI_PSP_SendCtrl(MPIDI_PSP_CTRL_TAG__WIN__COMPLETE, win_ptr->comm_ptr->context_id + MPIR_CONTEXT_INTRA_PT2PT, win_ptr->comm_ptr->rank, con, MPID_PSP_MSGTYPE_RMA_SYNC);
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
	MPL_free(ranks);

	return mpi_errno;
}


int MPID_Win_wait(MPIR_Win *win_ptr)
{
	int *ranks = win_ptr->ranks_post;
	int ranks_sz = win_ptr->ranks_post_sz;
	int i;
	int mpi_errno = MPI_SUCCESS;

#if 0
	if (!win_ptr->ranks_post) {
		return MPI_ERR_ARG;
	}
#else
	if(win_ptr->epoch_state != MPID_PSP_EPOCH_PSCW && win_ptr->epoch_state != MPID_PSP_EPOCH_POST) {
		return MPI_ERR_RMA_SYNC;
	}

	/* Track access epoch state */
	if (win_ptr->epoch_state == MPID_PSP_EPOCH_PSCW) {
		win_ptr->epoch_state = MPID_PSP_EPOCH_START;
	}
	else {
		win_ptr->epoch_state = MPID_PSP_EPOCH_NONE;
	}
#endif

	for (i = 0; i < ranks_sz; i++) {
		int rank = ranks[i];
		pscom_connection_t *con = MPID_PSCOM_rank2connection(win_ptr->comm_ptr, rank);

		MPIDI_PSP_Win_wait_passive_completion(rank, win_ptr);

		/* Recv tag COMPLETE from MPID_Win_complete of rank: */
		MPIDI_PSP_RecvCtrl(MPIDI_PSP_CTRL_TAG__WIN__COMPLETE, win_ptr->comm_ptr->recvcontext_id + MPIR_CONTEXT_INTRA_PT2PT, rank, con, MPID_PSP_MSGTYPE_RMA_SYNC);
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
	MPL_free(ranks);

	return mpi_errno;
}


int MPID_Win_test(MPIR_Win *win_ptr, int *flag)
{
	int *ranks = win_ptr->ranks_post;
	int ranks_sz = win_ptr->ranks_post_sz;
	int i;
	int mpi_errno = MPI_SUCCESS;
	int ret_flag = 1;

#if 0
	if (!win_ptr->ranks_post) {
		return MPI_ERR_ARG;
	}
#else
	if(win_ptr->epoch_state != MPID_PSP_EPOCH_PSCW && win_ptr->epoch_state != MPID_PSP_EPOCH_POST) {
		return MPI_ERR_RMA_SYNC;
	}
#endif

	for (i = 0; i < ranks_sz; i++) {
		int rank = ranks[i];
		int aflag;
		pscom_connection_t *con = MPID_PSCOM_rank2connection(win_ptr->comm_ptr, rank);

		/* Probe for COMPLETE from MPID_Win_complete of rank: */
		MPIDI_PSP_IprobeCtrl(MPIDI_PSP_CTRL_TAG__WIN__COMPLETE, win_ptr->comm_ptr->recvcontext_id + MPIR_CONTEXT_INTRA_PT2PT, rank, con, MPID_PSP_MSGTYPE_RMA_SYNC, &aflag);

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


static inline
int do_trylock(MPIR_Win *win_ptr, int exclusive)
{
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
void do_lock(MPIR_Win *win_ptr, pscom_request_t *req)
{
	if (do_trylock(win_ptr, req->user->type.rma_lock.exclusive)) {
		/* window locked, send ack */
		pscom_post_send(req);
	} else {
		/* schedule this lock */
		list_add_tail(&req->user->type.rma_lock.next, &win_ptr->lock_list);
	}
}


static
void do_unlock(MPIR_Win *win_ptr)
{
	win_ptr->lock_cnt--; /* unlock */

	if (!list_empty(&win_ptr->lock_list)) {
		pscom_request_rma_lock_t *rma_lock =
			list_entry(win_ptr->lock_list.next, pscom_request_rma_lock_t, next);

		pscom_request_t *lreq = rma_lock->req;

		list_del(&rma_lock->next);

		if (do_trylock(win_ptr, lreq->user->type.rma_lock.exclusive)) {
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

	MPIR_Win *win_ptr = xhead_lock->win_ptr;

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

	MPIR_Win *win_ptr = xhead_lock->win_ptr;

	MPIDI_PSP_Win_wait_passive_completion(xhead_lock->common.src_rank, win_ptr);

	/* reuse orignal header, but overwrite type,src_rank and xheader_len: */
	xhead_lock->common.type = MPID_PSP_MSGTYPE_RMA_UNLOCK_ANSWER;
	xhead_lock->common.src_rank = MPI_ANY_SOURCE;

	req->xheader_len = sizeof(xhead_lock->common);

	req->ops.io_done = pscom_request_free;

	/* send answer */
	pscom_post_send(req);

	do_unlock(win_ptr);
}


int MPID_Win_lock(int lock_type, int dest, int assert, MPIR_Win *win_ptr)
{
	int exclusive;
	MPIR_Comm *comm;
	pscom_connection_t *con;
	enum MPID_PSP_MSGTYPE msgt;

	if (unlikely(dest == MPI_PROC_NULL)) {
		return MPI_SUCCESS;
	}

	if(win_ptr->remote_lock_state[dest] != MPID_PSP_LOCK_UNLOCKED) {
		return MPI_ERR_RMA_SYNC;
	}

	if(win_ptr->epoch_state != MPID_PSP_EPOCH_NONE &&
	   win_ptr->epoch_state != MPID_PSP_EPOCH_FENCE_ISSUED &&
	   win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK) {
		return MPI_ERR_RMA_SYNC;
	}

	if (lock_type == MPI_LOCK_EXCLUSIVE) {
		msgt = MPID_PSP_MSGTYPE_RMA_LOCK_EXCLUSIVE_REQUEST;
		exclusive = 1;
	} else if (lock_type == MPI_LOCK_SHARED) {
		msgt = MPID_PSP_MSGTYPE_RMA_LOCK_SHARED_REQUEST;
		exclusive = 0;
	} else {
		return MPI_ERR_ARG;
	}

	if(dest == win_ptr->rank) {
		/* avoid starvation / foster progress: */
		pscom_test_any();

		if(do_trylock(win_ptr, exclusive)) {
			/* This is a shortcut for _local_ locks! */
			goto fn_exit;
		}
	}

	comm = win_ptr->comm_ptr;
	con = MPID_PSCOM_rank2connection(comm, dest);

	MPIDI_PSP_SendRmaCtrl(win_ptr, comm, con, dest, msgt);
	MPIDI_PSP_RecvCtrl(0/*tag*/, comm->recvcontext_id, MPI_ANY_SOURCE, con, MPID_PSP_MSGTYPE_RMA_LOCK_ANSWER);

fn_exit:
	win_ptr->remote_lock_state[dest] = MPID_PSP_LOCK_LOCKED;

	/* Track access epoch state */
	win_ptr->epoch_state = MPID_PSP_EPOCH_LOCK;
	win_ptr->epoch_lock_count++;

	return MPI_SUCCESS;
}


int MPID_Win_unlock(int dest, MPIR_Win *win_ptr)
{
	MPIR_Comm *comm;
	pscom_connection_t *con;

	if (unlikely(dest == MPI_PROC_NULL)) {
		return MPI_SUCCESS;
	}

	if(win_ptr->remote_lock_state[dest] == MPID_PSP_LOCK_UNLOCKED) {
		return MPI_ERR_RMA_SYNC;
	}

	if(win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK && win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK_ALL) {
		return MPI_ERR_RMA_SYNC;
	}

	if(dest == win_ptr->rank) {
		/* This is a shortcut for _local_ locks! */
		do_unlock(win_ptr);
		goto fn_exit;
	}

	comm = win_ptr->comm_ptr;
	con = MPID_PSCOM_rank2connection(comm, dest);

	/* wait until all pending RMA requests are processed*/
	while (win_ptr->rma_local_pending_cnt) {
		pscom_test_any();
	}

	MPIDI_PSP_SendRmaCtrl(win_ptr, comm, con, dest, MPID_PSP_MSGTYPE_RMA_UNLOCK_REQUEST);
	MPIDI_PSP_RecvCtrl(0/*tag*/, comm->recvcontext_id, MPI_ANY_SOURCE, con, MPID_PSP_MSGTYPE_RMA_UNLOCK_ANSWER);

fn_exit:
	win_ptr->remote_lock_state[dest] = MPID_PSP_LOCK_UNLOCKED;

	/* Track access epoch state */
	win_ptr->epoch_lock_count--;
	if (win_ptr->epoch_lock_count == 0) {
		win_ptr->epoch_state = MPID_PSP_EPOCH_NONE;
	}

	return MPI_SUCCESS;
}


/***********************************************************************************************************
 *   RMA-3.0 Functions:
 */


int MPID_Win_sync(MPIR_Win *win_ptr)
{
	/* Flush and Sync can be called only within passive target epochs! */
	if(win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK && win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK_ALL) {
		return MPI_ERR_RMA_SYNC;
	}

	if (win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED) {
		MPL_atomic_read_write_barrier();
	}

        return MPI_SUCCESS;
}


int MPID_Win_lock_all(int assert, MPIR_Win *win_ptr)
{
	int mpi_error;

	if(1) { /* TODO: This is just a simple straightforward implementation! */

		int i;
		for(i=0; i<win_ptr->comm_ptr->local_size; i++) {
			if(win_ptr->remote_lock_state[i] != MPID_PSP_LOCK_UNLOCKED) {
				/* Window is already locked! */
				return MPI_ERR_RMA_SYNC;
			}
			mpi_error = MPID_Win_lock(MPI_LOCK_SHARED, i, 0, win_ptr);
			if(mpi_error) {
				return mpi_error;
			}
			win_ptr->remote_lock_state[i] = MPID_PSP_LOCK_LOCKED_ALL;
		}
	}
	else {
		/* TODO: A more sophisticated implementation goes here... */
		assert(0);
	}

	return MPI_SUCCESS;
}

int MPID_Win_unlock_all(MPIR_Win *win_ptr)
{
	int mpi_error;

	if(1) { /* TODO: This is just a simple straightforward implementation! */

		int i;
		for(i=0; i<win_ptr->comm_ptr->local_size; i++) {
			if(win_ptr->remote_lock_state[i] != MPID_PSP_LOCK_LOCKED_ALL) {
				/* Window wasn't locked by Win_lock_all()! */
				return MPI_ERR_RMA_SYNC;
			}
			mpi_error = MPID_Win_unlock(i, win_ptr);
			if(mpi_error) {
				return mpi_error;
			}
			win_ptr->remote_lock_state[i] = MPID_PSP_LOCK_UNLOCKED;
		}
	}
	else {
		/* TODO: A more sophisticated implementation goes here... */
		assert(0);
	}

	return MPI_SUCCESS;
}


/***********************************************************************************************************
 *   RMA-3.0 Flush Routines:
 */

void MPID_do_recv_rma_flush_req(pscom_request_t *req)
{
	/* This is an pscom callback. Global lock state undefined! */
	MPID_PSCOM_XHeader_Rma_lock_t *xhead_lock = &req->xheader.user.rma_lock;
	MPIR_Win *win_ptr = xhead_lock->win_ptr;

	MPIDI_PSP_Win_wait_passive_completion(xhead_lock->common.src_rank, win_ptr);

	/* reuse orignal header, but overwrite type,src_rank and xheader_len: */
	xhead_lock->common.type = MPID_PSP_MSGTYPE_RMA_FLUSH_ANSWER;
	xhead_lock->common.src_rank = MPI_ANY_SOURCE;

	req->xheader_len = sizeof(xhead_lock->common);

	req->user->type.rma_lock.req = req;

	req->ops.io_done = pscom_request_free;

	pscom_post_send(req);
}

int MPID_Win_flush(int dest, MPIR_Win *win_ptr)
{
	MPIR_Comm *comm;
	pscom_connection_t *con;

	/* Flush and Sync can be called only within passive target epochs! */
	if(win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK && win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK_ALL) {
		return MPI_ERR_RMA_SYNC;
	}

	MPID_Win_flush_local(dest, win_ptr);

	if(dest == win_ptr->rank) {
		/* avoid starvation / foster progress: */
		pscom_test_any();

	} else {

		comm = win_ptr->comm_ptr;
		con = MPID_PSCOM_rank2connection(comm, dest);

		MPIDI_PSP_SendRmaCtrl(win_ptr, comm, con, dest, MPID_PSP_MSGTYPE_RMA_FLUSH_REQUEST);
		MPIDI_PSP_RecvCtrl(0/*tag*/, comm->recvcontext_id, MPI_ANY_SOURCE, con, MPID_PSP_MSGTYPE_RMA_FLUSH_ANSWER);
	}

	return MPI_SUCCESS;
}

int MPID_Win_flush_all(MPIR_Win *win_ptr)
{
	int i;
	MPIR_Comm *comm;
	pscom_connection_t *con;

	/* Flush and Sync can be called only within passive target epochs! */
	if(win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK && win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK_ALL) {
		return MPI_ERR_RMA_SYNC;
	}

	MPID_Win_flush_local_all(win_ptr);

	comm = win_ptr->comm_ptr;

	for(i=0; i<win_ptr->comm_ptr->local_size; i++) {

		if(i != win_ptr->rank) {
			con = MPID_PSCOM_rank2connection(comm, i);
			MPIDI_PSP_SendRmaCtrl(win_ptr, comm, con, i, MPID_PSP_MSGTYPE_RMA_FLUSH_REQUEST);
		}
	}

	for(i=0; i<win_ptr->comm_ptr->local_size; i++) {

		if(i != win_ptr->rank) {
			MPIDI_PSP_RecvCtrl(0/*tag*/, comm->recvcontext_id, MPI_ANY_SOURCE, NULL, MPID_PSP_MSGTYPE_RMA_FLUSH_ANSWER);
		}
	}

	return MPI_SUCCESS;
}

int MPID_Win_wait_local_completion(int rank, MPIR_Win *win_ptr)
{
	if (win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED) {
		MPL_atomic_read_write_barrier();
	}

	while (win_ptr->rma_local_pending_rank[rank]) {

		MPID_PSP_LOCKFREE_CALL(pscom_wait_any());
	}

	return MPI_SUCCESS;
}

int MPIDI_PSP_Win_wait_passive_completion(int rank, MPIR_Win *win_ptr)
{
	while (win_ptr->rma_passive_pending_rank[rank]) {

		MPID_PSP_LOCKFREE_CALL(pscom_wait_any());
	}

	return MPI_SUCCESS;
}

int MPID_Win_flush_local(int rank, MPIR_Win *win_ptr)
{
	/* Flush and Sync can be called only within passive target epochs! */
	if(win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK && win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK_ALL) {
		return MPI_ERR_RMA_SYNC;
	}

	MPID_Win_wait_local_completion(rank, win_ptr);

	return MPI_SUCCESS;
}

int MPID_Win_flush_local_all(MPIR_Win *win_ptr)
{
	/* Flush and Sync can be called only within passive target epochs! */
	if(win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK && win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK_ALL) {
		return MPI_ERR_RMA_SYNC;
	}

	if (win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED) {
		MPL_atomic_read_write_barrier();
	}

	while (win_ptr->rma_local_pending_cnt) {

		MPID_PSP_LOCKFREE_CALL(pscom_wait_any());
	}

	return MPI_SUCCESS;
}


/***********************************************************************************************************
 *   RMA-3.0 Auxiliary Routines for internal locking needed by Fetch&Op & Co.:
 */

static
int do_trylock_internal(MPIR_Win *win_ptr)
{
	if (!win_ptr->lock_internal) {
		/* lock */
		win_ptr->lock_internal = 1;

		return 1;
	} else {
		return 0;
	}
}

static
void do_lock_internal(MPIR_Win *win_ptr, pscom_request_t *req)
{
	if (do_trylock_internal(win_ptr)) {
		/* window locked, send ack */
		pscom_post_send(req);
	} else {
		/* schedule this lock */
		list_add_tail(&req->user->type.rma_lock.next, &win_ptr->lock_list_internal);
	}
}

static
void do_unlock_internal(MPIR_Win *win_ptr)
{
	win_ptr->lock_internal = 0; /* unlock */

	if (!list_empty(&win_ptr->lock_list_internal)) {
		pscom_request_rma_lock_t *rma_lock =
			list_entry(win_ptr->lock_list_internal.next, pscom_request_rma_lock_t, next);

		pscom_request_t *lreq = rma_lock->req;

		list_del(&rma_lock->next);

		if (do_trylock_internal(win_ptr)) {
			/* window locked, send ack */
			pscom_post_send(lreq);
		} else {
			/* reschedule this lock */
			list_add(&rma_lock->next, &win_ptr->lock_list_internal);
		}
	}
}

void MPID_do_recv_rma_lock_internal_req(pscom_request_t *req)
{
	/* This is an pscom callback. Global lock state undefined! */
	MPID_PSCOM_XHeader_Rma_lock_t *xhead_lock = &req->xheader.user.rma_lock;

	MPIR_Win *win_ptr = xhead_lock->win_ptr;

	/* reuse orignal header, but overwrite type,src_rank and xheader_len: */
	xhead_lock->common.type = MPID_PSP_MSGTYPE_RMA_INTERNAL_LOCK_ANSWER;
	xhead_lock->common.src_rank = MPI_ANY_SOURCE;

	req->xheader_len = sizeof(xhead_lock->common);

	req->user->type.rma_lock.req = req;

	req->ops.io_done = pscom_request_free;

	do_lock_internal(win_ptr, req);
}

void MPID_do_recv_rma_unlock_internal_req(pscom_request_t *req)
{
	/* This is an pscom callback. Global lock state undefined! */
	MPID_PSCOM_XHeader_Rma_lock_t *xhead_lock = &req->xheader.user.rma_lock;

	MPIR_Win *win_ptr = xhead_lock->win_ptr;

	MPIDI_PSP_Win_wait_passive_completion(xhead_lock->common.src_rank, win_ptr);

	/* reuse orignal header, but overwrite type,src_rank and xheader_len: */
	xhead_lock->common.type = MPID_PSP_MSGTYPE_RMA_INTERNAL_UNLOCK_ANSWER;
	xhead_lock->common.src_rank = MPI_ANY_SOURCE;

	req->xheader_len = sizeof(xhead_lock->common);

	req->ops.io_done = pscom_request_free;

	/* send answer */
	pscom_post_send(req);

	do_unlock_internal(win_ptr);
}

int MPID_Win_lock_internal(int dest, MPIR_Win *win_ptr)
{
        MPIR_Comm *comm;
        pscom_connection_t *con;

        if (unlikely(dest == MPI_PROC_NULL)) {
		return MPI_SUCCESS;
        }

        if(dest == win_ptr->rank) {
                /* avoid starvation / foster progress: */
                pscom_test_any();

		if(do_trylock_internal(win_ptr)) {
			/* This is a shortcut for _local_ locks! */
			return MPI_SUCCESS;
		}
        }

        comm = win_ptr->comm_ptr;
        con = MPID_PSCOM_rank2connection(comm, dest);

        MPIDI_PSP_SendRmaCtrl(win_ptr, comm, con, dest, MPID_PSP_MSGTYPE_RMA_INTERNAL_LOCK_REQUEST);
        MPIDI_PSP_RecvCtrl(0/*tag*/, comm->recvcontext_id, MPI_ANY_SOURCE, con, MPID_PSP_MSGTYPE_RMA_INTERNAL_LOCK_ANSWER);

        return MPI_SUCCESS;
}

int MPID_Win_unlock_internal(int dest, MPIR_Win *win_ptr)
{
        MPIR_Comm *comm;
        pscom_connection_t *con;

        if (unlikely(dest == MPI_PROC_NULL)) {
		return MPI_SUCCESS;
        }

	if(dest == win_ptr->rank) {
		/* This is a shortcut for _local_ locks! */
		do_unlock_internal(win_ptr);
		return MPI_SUCCESS;
	}

        comm = win_ptr->comm_ptr;
	con = MPID_PSCOM_rank2connection(comm, dest);

		/* wait until all pending RMA requests are processed*/
		while (win_ptr->rma_local_pending_cnt) {
			pscom_test_any();
		}

        MPIDI_PSP_SendRmaCtrl(win_ptr, comm, con, dest, MPID_PSP_MSGTYPE_RMA_INTERNAL_UNLOCK_REQUEST);
        MPIDI_PSP_RecvCtrl(0/*tag*/, comm->recvcontext_id, MPI_ANY_SOURCE, con, MPID_PSP_MSGTYPE_RMA_INTERNAL_UNLOCK_ANSWER);

        return MPI_SUCCESS;
}
