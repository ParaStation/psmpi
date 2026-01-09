/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"
#include "mpid_psp_request.h"
#include "mpid_psp_datatype.h"


static
int MPID_PSP_Bsend(const void *buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
                   MPIR_Comm * comm, int context_offset, MPIR_Request ** request);

int MPID_PSP_persistent_init(const void *buf, MPI_Aint count, MPI_Datatype datatype, int rank,
                             int tag, MPIR_Comm * comm, int context_offset, MPIR_Request ** request,
                             int (*call) (const void *buf, MPI_Aint count, MPI_Datatype datatype,
                                          int rank, int tag, struct MPIR_Comm * comm,
                                          int context_offset, MPIR_Request ** request),
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
    if (unlikely(!req))
        goto err_request_recv_create;
    req->comm = comm;
    MPIR_Comm_add_ref(comm);
    MPIR_Comm_save_inactive_request(comm, req);

    req->u.persist.real_request = NULL;
    MPIDI_PSP_Request_set_completed(req);       /* an inactive persistent request is a completed request. */

    preq = &req->dev.kind.persistent;

    preq->buf = (void *) buf;
    preq->count = count;
    preq->datatype = datatype;
    MPID_PSP_Datatype_add_ref(preq->datatype);

    preq->rank = rank;
    preq->tag = tag;
    preq->comm = comm;
//      MPIR_Comm_add_ref(comm);

    preq->context_offset = context_offset;

    preq->call = call;

    *request = req;

    return MPI_SUCCESS;
    /* --- */
  err_request_recv_create:
    return MPI_ERR_NO_MEM;
}


int MPID_PSP_persistent_start(MPIR_Request * req)
{
    int mpi_errno = MPI_SUCCESS;
    struct MPID_DEV_Request_persistent *preq;
    preq = &req->dev.kind.persistent;
/*
	printf("#%d ps--- %s() called\n", MPIDI_Process.my_pg_rank, __func__);
*/
    MPIR_Assert(req->u.persist.real_request == NULL);   /* assure inactive persistent request! */

    mpi_errno = preq->call(preq->buf, preq->count, preq->datatype, preq->rank,
                           preq->tag, preq->comm, preq->context_offset,
                           &req->u.persist.real_request);

    if (req->u.persist.real_request) {
        /* Use cc_ptr from partner request.
         * MPIR_Request_complete() in pt2pt/mpir_request.c will reset it to
         * req->cc = 0;
         * req->cc_ptr = &req->cc;
         * req->u.persist.real_request = NULL;
         * when done.
         */
        req->cc_ptr = req->u.persist.real_request->cc_ptr;
    }

    /* bsend is local-complete -> set completion counter to 0 */
    if ((mpi_errno == MPI_SUCCESS) && (preq->call == MPID_PSP_Bsend)) {
        req->status.MPI_ERROR = MPI_SUCCESS;
        req->cc_ptr = &req->cc;
        MPIR_cc_set(req->cc_ptr, 0);
    }

    return mpi_errno;
}


static
int MPID_PSP_Bsend(const void *buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
                   MPIR_Comm * comm, int context_offset, MPIR_Request ** request)
{
    /* See src/mpid/ch3/src/mpid_startall.c:105   "MPID_Startall(): case MPIDI_REQUEST_TYPE_BSEND:" */
    int mpi_errno = MPI_SUCCESS;

    // TODO: check THREADPRIV API!

    mpi_errno = MPIR_Bsend_isend((void *) buf, count, datatype, rank, tag, comm, request);
    if (mpi_errno)
        goto fn_fail;

  fn_exit:
    return mpi_errno;

  fn_fail:
    *request = NULL;
    mpi_errno = MPIR_Err_return_comm(comm, __func__, mpi_errno);
    goto fn_exit;
}


int MPID_Recv_init(void *buf, int count, MPI_Datatype datatype, int rank, int tag,
                   MPIR_Comm * comm, int context_offset, MPIR_Request ** request)
{
    return MPID_PSP_persistent_init(buf, count, datatype, rank, tag, comm,
                                    context_offset, request, (int (*)
                                                              (const void *, MPI_Aint, MPI_Datatype,
                                                               int, int, struct MPIR_Comm *, int,
                                                               MPIR_Request **)) MPID_Irecv,
                                    MPIR_REQUEST_KIND__PREQUEST_RECV);
}


int MPID_Rsend_init(const void *buf, int count, MPI_Datatype datatype,
                    int rank, int tag, MPIR_Comm * comm, int context_offset,
                    MPIR_Request ** request)
{
    return MPID_PSP_persistent_init(buf, count, datatype, rank, tag, comm,
                                    context_offset, request, MPID_Irsend,
                                    MPIR_REQUEST_KIND__PREQUEST_SEND);
}


int MPID_Bsend_init(const void *buf, int count, MPI_Datatype datatype,
                    int rank, int tag, MPIR_Comm * comm, int context_offset,
                    MPIR_Request ** request)
{
    return MPID_PSP_persistent_init(buf, count, datatype, rank, tag, comm,
                                    context_offset, request, MPID_PSP_Bsend,
                                    MPIR_REQUEST_KIND__PREQUEST_SEND);
}


int MPID_Send_init(const void *buf, int count, MPI_Datatype datatype,
                   int rank, int tag, MPIR_Comm * comm, int context_offset, MPIR_Request ** request)
{
    return MPID_PSP_persistent_init(buf, count, datatype, rank, tag, comm,
                                    context_offset, request, MPIDI_PSP_Isend,
                                    MPIR_REQUEST_KIND__PREQUEST_SEND);
}


int MPID_Ssend_init(const void *buf, int count, MPI_Datatype datatype,
                    int rank, int tag, MPIR_Comm * comm, int context_offset,
                    MPIR_Request ** request)
{
    return MPID_PSP_persistent_init(buf, count, datatype, rank, tag, comm,
                                    context_offset, request, MPIDI_PSP_Issend,
                                    MPIR_REQUEST_KIND__PREQUEST_SEND);
}
