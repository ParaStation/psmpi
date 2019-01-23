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
#include "mpid_psp_request.h"
#include "mpid_psp_datatype.h"

// This must be the last include before sysmbols are defined:
#include "mpid_visibility.h"


static
int MPID_PSP_persistent_init(const void *buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
			     MPIR_Comm *comm, int context_offset, MPIR_Request **request,
			     int (*call)(const void * buf, MPI_Aint count, MPI_Datatype datatype, int rank,
					 int tag, struct MPIR_Comm * comm, int context_offset, MPIR_Request ** request),
			     MPIR_Request_kind_t type)
{
	MPIR_Request *req;
	struct MPID_DEV_Request_persistent *preq;
/*
	printf("#%d ps--- %s() called\n", MPIDI_Process.my_pg_rank, __func__);
	printf("#%d buf %p, count %d, datatype 0x%0x, rank %d, tag %d, comm %p, off %d\n",
	       MPIDI_Process.my_pg_rank, buf, count, datatype, rank, tag, comm, context_offset);
	printf("#%d ctx.id %d ctx.rank %d, ctx.name %s\n",
	       MPIDI_Process.my_pg_rank, comm->context_id, comm->rank, comm->name);
*/
	req = MPIR_Request_create(type);
	if (unlikely(!req)) goto err_request_recv_create;
	req->comm = comm;
	MPIR_Comm_add_ref(comm);

	req->u.persist.real_request = NULL;
	MPIDI_PSP_Request_set_completed(req); /* an inactive persistent request is a completed request. */

	preq = &req->dev.kind.persistent;

	preq->buf = (void *)buf;
	preq->count = count;
	preq->datatype = datatype;
	MPID_PSP_Datatype_add_ref(preq->datatype);

	preq->rank = rank;
	preq->tag = tag;
	preq->comm = comm;
//	MPIR_Comm_add_ref(comm);

	preq->context_offset = context_offset;

	preq->call = call;

	*request = req;

	return MPI_SUCCESS;
	/* --- */
err_request_recv_create:
	return MPI_ERR_NO_MEM;
}


static
int MPID_PSP_Bsend(const void * buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
		   MPIR_Comm * comm, int context_offset, MPIR_Request ** request)
{
	/* See src/mpid/ch3/src/mpid_startall.c:105   "MPID_Startall(): case MPIDI_REQUEST_TYPE_BSEND:"*/
	MPI_Request sreq_handle;
	int rc;

	// TODO: check THREADPRIV API!

	{
		rc = MPIR_Ibsend_impl((void *)buf, count, datatype, rank,
				      tag, comm, &sreq_handle);
		if (rc == MPI_SUCCESS)
		{
			MPIR_Request *r;
			MPIR_Request_get_ptr(sreq_handle, r);
			*request = r;
		}
	}
	return rc;
}


int MPID_Recv_init(void *buf, int count, MPI_Datatype datatype, int rank, int tag,
		   MPIR_Comm *comm, int context_offset, MPIR_Request **request)
{
	return MPID_PSP_persistent_init(buf, count, datatype, rank, tag, comm,
					context_offset, request,
					(int (*)(const void *, MPI_Aint, MPI_Datatype, int, int, struct MPIR_Comm *, int, MPIR_Request **))MPID_Irecv,
					MPIR_REQUEST_KIND__PREQUEST_RECV);
}


int MPID_Rsend_init(const void * buf, int count, MPI_Datatype datatype,
		    int rank, int tag, MPIR_Comm * comm, int context_offset,
		    MPIR_Request ** request)
{
	return MPID_PSP_persistent_init(buf, count, datatype, rank, tag, comm,
					context_offset, request, MPID_Irsend, MPIR_REQUEST_KIND__PREQUEST_SEND);
}


int MPID_Bsend_init(const void * buf, int count, MPI_Datatype datatype,
		    int rank, int tag, MPIR_Comm * comm, int context_offset,
		    MPIR_Request ** request)
{
	return MPID_PSP_persistent_init(buf, count, datatype, rank, tag, comm,
					context_offset, request, MPID_PSP_Bsend, MPIR_REQUEST_KIND__PREQUEST_SEND);
}


int MPID_Send_init(const void * buf, int count, MPI_Datatype datatype,
		   int rank, int tag, MPIR_Comm * comm, int context_offset,
		   MPIR_Request ** request)
{
	return MPID_PSP_persistent_init(buf, count, datatype, rank, tag, comm,
					context_offset, request, MPID_Isend, MPIR_REQUEST_KIND__PREQUEST_SEND);
}


int MPID_Ssend_init(const void * buf, int count, MPI_Datatype datatype,
		    int rank, int tag, MPIR_Comm * comm, int context_offset,
		    MPIR_Request ** request)
{
	return MPID_PSP_persistent_init(buf, count, datatype, rank, tag, comm,
					context_offset, request, MPID_Issend, MPIR_REQUEST_KIND__PREQUEST_SEND);
}


static
int MPID_Start(MPIR_Request *req)
{
	int mpi_errno = MPI_SUCCESS;

	struct MPID_DEV_Request_persistent *preq = &req->dev.kind.persistent;
/*
	printf("#%d ps--- %s() called\n", MPIDI_Process.my_pg_rank, __func__);
*/
	assert(req->u.persist.real_request == NULL); /* assure inactive persistent request! */

	mpi_errno = preq->call(preq->buf, preq->count, preq->datatype, preq->rank,
			       preq->tag, preq->comm, preq->context_offset,
			       &req->u.persist.real_request);

	if (req->u.persist.real_request) {
		/* Use cc_ptr from partner request.
		   MPIR_Request_complete() in pt2pt/mpir_request.c will reset it to
		   req->cc = 0;
		   req->cc_ptr = &req->cc;
		   req->u.persist.real_request = NULL;
		   when done.
		 */
		req->cc_ptr = req->u.persist.real_request->cc_ptr;
	}

	return mpi_errno;
}


int MPID_Startall(int count, MPIR_Request * requests[])
{
	int mpi_errno = MPI_SUCCESS;

	while (count) {
		mpi_errno = MPID_Start(*requests);
		if (mpi_errno != MPI_SUCCESS)
			break;

		requests ++;
		count --;
	}

	return mpi_errno;
}
