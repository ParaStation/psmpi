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
#include "mpid_debug.h"

#ifndef MPID_REQUEST_PREALLOC
#define MPID_REQUEST_PREALLOC 8
#endif

MPID_Request MPID_Request_direct[MPID_REQUEST_PREALLOC];/* = {{0}}; */

MPIU_Object_alloc_t MPID_Request_mem = {
	0, 0, 0, 0, MPID_REQUEST, sizeof(MPID_Request), MPID_Request_direct,
	MPID_REQUEST_PREALLOC };


/* ToDo: disable: */
#define DEBUG_COUNT_REQUESTS(code) code
#if !defined(DEBUG_COUNT_REQUESTS)
#define DEBUG_COUNT_REQUESTS(code)
#endif

DEBUG_COUNT_REQUESTS(static unsigned int request_alloc_count = 0;)


static inline
MPID_Request *MPID_PSP_Request_alloc(void)
{
	MPID_Request * req;

	/* ToDo: Try MPIU_Malloc(); */
	req = MPIU_Handle_obj_alloc(&MPID_Request_mem);

	DEBUG_COUNT_REQUESTS(request_alloc_count++);
	return req;
}

static inline
void MPID_PSP_Request_free(MPID_Request *req)
{
	MPIU_Handle_obj_free(&MPID_Request_mem, req);

	DEBUG_COUNT_REQUESTS(request_alloc_count--);
}


static inline
void MPID_PSP_Request_init(MPID_Request *req)
{
	struct MPID_DEV_Request_common *creq;
	pscom_request_t *preq;

	if (!req) return;

	/* MPID_Request_construct(req); no need?*/
	MPIU_Object_set_ref(req, 1);

	req->kind = MPID_REQUEST_UNDEFINED;
	req->cc = 1;
	req->cc_ptr = &req->cc;
	req->status.MPI_SOURCE = MPI_UNDEFINED;
	req->status.MPI_TAG = MPI_UNDEFINED;
	req->status.MPI_ERROR = MPI_SUCCESS;
	/* combined MPIR_STATUS_SET_COUNT and MPIR_STATUS_SET_CANCEL_BIT: */
	req->status.count_lo = 0;
	req->status.count_hi_and_cancelled = 0;
	req->comm = NULL;

	req->errflag = MPIR_ERR_NONE;

	creq = &req->dev.kind.common;
	preq = creq->pscom_req;

	preq->connection = NULL;
	preq->socket = NULL;
	preq->ops.recv_accept = NULL;
	preq->ops.io_done = NULL;
	preq->xheader_len = 0;
	preq->data_len = 0;
	preq->data = 0;


/*	req->dev.datatype_ptr = NULL;
	req->dev.cancel_pending = FALSE;
	req->dev.target_win_handle = MPI_WIN_NULL;
	req->dev.source_win_handle = MPI_WIN_NULL;
	req->dev.single_op_opt = 0;
	req->dev.lock_queue_entry = NULL;
	req->dev.dtype_info = NULL;
	req->dev.dataloop = NULL;
	req->dev.rdma_iov_count = 0;
	req->dev.rdma_iov_offset = 0;
*/
}


static MPID_Request *prep_req_queue = NULL;
static unsigned int req_queue_cnt = 0;
static const unsigned int max_req_queue_cnt = 50;

#ifndef MPICH_IS_THREADED

static inline
void prep_req_lock(void)
{}

static inline
void prep_req_unlock(void)
{}

#else /* MPICH_IS_THREADED */

#include <pthread.h>
static pthread_mutex_t prep_req_mutex = PTHREAD_MUTEX_INITIALIZER;

static inline
void prep_req_lock(void)
{
	int res_mutex_lock;
	res_mutex_lock = pthread_mutex_lock(&prep_req_mutex);
	assert(res_mutex_lock == 0);
}


static inline
void prep_req_unlock(void)
{
	int res_mutex_unlock;
	res_mutex_unlock = pthread_mutex_unlock(&prep_req_mutex);
	assert(res_mutex_unlock == 0);
}

#endif


static inline
void prep_req_enqueue(MPID_Request *req)
{
	MPID_PSP_Request_init(req);

	prep_req_lock(); {
		req->partner_request = prep_req_queue;
		prep_req_queue = req;

		req_queue_cnt++;
	} prep_req_unlock();
}


static inline
MPID_Request *prep_req_dequeue(void)
{
	MPID_Request *req;
	prep_req_lock(); {
		req = prep_req_queue;

		if (req) {
			prep_req_queue = req->partner_request;
			req->partner_request = NULL;

			req_queue_cnt--;
		}
	} prep_req_unlock();
	return req;
}


void MPID_req_queue_cleanup(void)
{
	MPID_Request *req;

	Dprintf("Requestqueue queue_len: %d", req_queue_cnt);

	while ((req = prep_req_dequeue())) {
		struct MPID_DEV_Request_common *creq = &req->dev.kind.common;
		pscom_request_free(creq->pscom_req);
		creq->pscom_req = NULL;
		MPID_PSP_Request_free(req);
	}

	DEBUG_COUNT_REQUESTS(
		if (request_alloc_count && (mpid_psp_debug_level > 0)) {
			fprintf(stderr, "mpid_psp: Warning: request_alloc_count after %s(): %u (rank %d)\n",
				__func__, request_alloc_count, MPIDI_Process.my_pg_rank);
		});
}


static inline
MPID_Request * MPID_PSP_Request_create()
{
	MPID_Request * req = prep_req_dequeue();

	if (!req) {
		struct MPID_DEV_Request_common *creq;

		req = MPID_PSP_Request_alloc();

		creq = &req->dev.kind.common;
		creq->pscom_req = PSCOM_REQUEST_CREATE();
		creq->pscom_req->user->type.sr.mpid_req = req;

		MPID_PSP_Request_init(req);
	}

	Dprintf("create request req=%p", req);
	return req;
}


void MPID_PSP_Request_destroy(MPID_Request *req)
{
	struct MPID_DEV_Request_common *creq = &req->dev.kind.common;

	assert(creq->pscom_req &&
	       (creq->pscom_req->state & PSCOM_REQ_STATE_DONE));
	Dprintf("destroy request req=%p type=%d", req, req->kind);
	if (req_queue_cnt < max_req_queue_cnt) {
		/* reuse request */
		prep_req_enqueue(req);
	} else {
		pscom_request_free(creq->pscom_req);
		creq->pscom_req = NULL;
		MPID_PSP_Request_free(req);
	}
}


void MPID_Request_release(MPID_Request * req)
{
	Dprintf("release request req=%p type=%d", req, req->kind);
	MPID_DEV_Request_release_ref(req, req->kind);

}


/**********************************************************
 * MPID_PSP_Requests
 */


/*
 * struct MPID_DEV_Request_common
 */
static inline
MPID_Request *MPID_DEV_Request_common_create(MPID_Comm *comm, MPID_Request_kind_t kind)
{
	MPID_Request * req = MPID_PSP_Request_create();
	/* struct MPID_DEV_Request_common *creq = &req->dev.kind.common; */
	req->comm = comm;
	req->kind = kind;

	return req;
}


static inline
void MPID_DEV_Request_common_destroy(MPID_Request *req)
{
	assert(MPID_Request_is_complete(req) || \
	       req->kind == MPID_PREQUEST_RECV || \
	       req->kind == MPID_PREQUEST_SEND || \
	       req->kind == MPID_REQUEST_MPROBE);
	req->kind = MPID_REQUEST_UNDEFINED;
	MPID_PSP_Request_destroy(req);
}


/*
 * struct MPID_DEV_Request_recv
 */
MPID_Request *MPID_DEV_Request_recv_create(MPID_Comm *comm)
{
	MPID_Request * req = MPID_DEV_Request_common_create(comm, MPID_REQUEST_RECV);
	/* struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv; */
/*
	rreq->noncontig.tmpbuf = NULL;
	ncreq->datatype_ptr = NULL;
*/
	return req;
}


void MPID_DEV_Request_recv_destroy(MPID_Request *req)
{
	/* struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv; */

/*
	if (rreq->noncontig.tmpbuf) {
		MPIU_Free(rreq->noncontig.tmpbuf);
		rreq->noncontig.tmpbuf = NULL;
	}
	if (ncreq->datatype_ptr) {
		MPID_Datatype_release(ncreq->datatype_ptr);
		ncreq->datatype_ptr = NULL;
	}
*/
	MPID_DEV_Request_common_destroy(req);
}


/*
 * struct MPID_DEV_Request_send
 */
MPID_Request *MPID_DEV_Request_send_create(MPID_Comm *comm)
{
	MPID_Request * req = MPID_DEV_Request_common_create(comm, MPID_REQUEST_SEND);
	struct MPID_DEV_Request_send *sreq = &req->dev.kind.send;

	sreq->msg.tmp_buf = NULL;

	return req;
}


void MPID_DEV_Request_send_destroy(MPID_Request *req)
{
	assert(req->dev.kind.send.msg.tmp_buf == NULL);

	MPID_DEV_Request_common_destroy(req);
}


/*
 * struct MPID_DEV_Request_multi multi;
 */

/*
MPID_Request *MPID_PSP_Request_Multi_create(void)
{
	MPID_Request * req;

	req = MPID_PSP_Request_create();

	req->comm = NULL;
	req->kind = MPID_PSP_REQUEST_MULTI;

	return req;
}
*/


/*
 * struct MPID_DEV_Request_precv  Persistent recv
 */
MPID_Request *MPID_DEV_Request_persistent_create(MPID_Comm *comm, MPID_Request_kind_t type)
{
	/* type should be MPID_PREQUEST_RECV or MPID_PREQUEST_SEND */
	MPID_Request * req = MPID_DEV_Request_common_create(comm, type);
	struct MPID_DEV_Request_persistent *preq = &req->dev.kind.persistent;
	preq->datatype = 0;
	preq->comm = NULL;

	return req;
}


void MPID_DEV_Request_persistent_destroy(MPID_Request *req)
{
	struct MPID_DEV_Request_persistent *preq = &req->dev.kind.persistent;
	if (preq->datatype) {
		MPID_PSP_Datatype_release(preq->datatype);
	}
	if (preq->comm) {
		MPIR_Comm_release(preq->comm);
	}
	MPID_DEV_Request_common_destroy(req);
}


void MPID_DEV_Request_ureq_destroy(MPID_Request *req)
{
	if (req->greq_fns != NULL) {
		MPIU_Free(req->greq_fns);
	}
	MPID_DEV_Request_common_destroy(req);
}


void MPID_DEV_Request_coll_destroy(MPID_Request *req)
{
	if (req->comm) {
		MPIR_Comm_release(req->comm);
	}

	MPID_DEV_Request_common_destroy(req);
}


/******************************************************************/


/*@
  MPID_Request_create - Create and return a bare request

  Return value:
  A pointer to a new request object.

  Notes:
  This routine is intended for use by 'MPI_Grequest_start' only.  Note that 
  once a request is created with this routine, any progress engine must assume 
  that an outside function can complete a request with 
  'MPID_Request_complete'.

  The request object returned by this routine should be initialized such that
  ref_count is one and handle contains a valid handle referring to the object.
  @*/

MPID_Request * MPID_Request_create(void)
{
	MPID_Request * req;

	Dprintf("");

	req = MPID_PSP_Request_create();

	req->comm = NULL;
	req->kind = MPID_REQUEST_UNDEFINED;

	return req;
}


/*@
  MPID_Request_complete - Complete a request

  Input Parameter:
. request - request to complete

  Notes:
  This routine is called to decrement the completion count of a
  request object.  If the completion count of the request object has
  reached zero, the reference count for the object will be
  decremented.
  @*/

int MPID_Request_complete(MPID_Request *req)
{
	if(MPID_PSP_Subrequest_completed(req)) {
		MPID_Request_release(req);
	}

	return MPI_SUCCESS;
}
