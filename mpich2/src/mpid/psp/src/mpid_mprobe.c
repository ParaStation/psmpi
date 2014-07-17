/*
 * ParaStation
 *
 * Copyright (C) 2014 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Carsten Clauss <clauss@par-tec.com>
 */

/*
 |  This file is to be included at the end of mpid_irecv.c to get access to
 |  the static functions prepare_recvreq, prepare_source, prepare_cleanup,
 |  set_probe_status and prepare_data defined there.
 */

/*
 |  MPI_(I)Mrecv() provides a mechanism to receive a specific message that was matched
 |  by MPI_(I)mprobe() regardless of other intervening probe or receive operations.
 |  For doing so, a successful (I)mprobe() call changes the message type in the header
 |  to MPROBE_RESERVED_REQUEST and adds a dedicated tag (mprobe_tag), which is actually
 |  just the pointer to the respective message header, in order to guarantee that no
 |  other recv call can fetch this message afterwards.
 */

static
int cb_accept_data_mprobe(pscom_request_t *request, pscom_connection_t *connection, pscom_header_net_t *header_net)
{
	MPID_Request *req = request->user->type.sr.mpid_req;
	struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv;
	MPID_PSCOM_XHeader_t *xhead = &header_net->xheader->user.common;

	if( (xhead->type <= MPID_PSP_MSGTYPE_DATA_REQUEST_ACK) &&
	    ((xhead->tag == rreq->tag) || (rreq->tag == MPI_ANY_TAG)) &&
	    (xhead->context_id == rreq->context_id) ) {

		rreq->mprobe_tag = xhead;

		/* Reserve this message for mrecv by changing the message type */
		if (xhead->type == MPID_PSP_MSGTYPE_DATA) {
			xhead->type = MPID_PSP_MSGTYPE_MPROBE_RESERVED_REQUEST;
		} else {
			xhead->type = MPID_PSP_MSGTYPE_MPROBE_RESERVED_REQUEST_ACK;
		}

		return 1;
	}

	return  ((xhead->type == MPID_PSP_MSGTYPE_MPROBE_RESERVED_REQUEST) ||
		 (xhead->type == MPID_PSP_MSGTYPE_MPROBE_RESERVED_REQUEST_ACK)) &&
		(rreq->mprobe_tag == xhead);
}

static
int cb_accept_data_mrecv(pscom_request_t *request, pscom_connection_t *connection, pscom_header_net_t *header_net)
{
	MPID_Request *req = request->user->type.sr.mpid_req;
	struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv;
	MPID_PSCOM_XHeader_t *xhead = &header_net->xheader->user.common;

	return  ((xhead->type == MPID_PSP_MSGTYPE_MPROBE_RESERVED_REQUEST) ||
		 (xhead->type == MPID_PSP_MSGTYPE_MPROBE_RESERVED_REQUEST_ACK)) &&
		(rreq->mprobe_tag == xhead);
}

static
void prepare_mprobereq(MPID_Request *req, int tag, MPID_Comm * comm, int context_offset)
{
	struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv;
	pscom_request_t *preq = rreq->common.pscom_req;

	prepare_recvreq(req, tag, comm, context_offset);
	preq->ops.recv_accept = cb_accept_data_mprobe;
}

static
void mreceive_done(pscom_request_t *request)
{
	MPID_PSCOM_XHeader_t *xhead = &request->xheader.user.common;

	/* Restore original message type: */
	if (xhead->type == MPID_PSP_MSGTYPE_MPROBE_RESERVED_REQUEST) {
		xhead->type = MPID_PSP_MSGTYPE_DATA;
	} else {
		xhead->type = MPID_PSP_MSGTYPE_DATA_REQUEST_ACK;
	}

	receive_done(request);
}

static
void prepare_mrecvreq(MPID_Request *req)
{
	struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv;
	pscom_request_t *preq = rreq->common.pscom_req;

	/* convert the mprobe request (aka MPI_Message) back into a recv request: */
	req->kind = MPID_REQUEST_RECV;

	preq->ops.recv_accept = cb_accept_data_mrecv;
}

static
void prepare_mrecv_cleanup(MPID_Request *req, void * buf, int count, MPI_Datatype datatype)
{
	struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv;
	pscom_request_t *preq = rreq->common.pscom_req;

	prepare_cleanup(req, buf, count, datatype);
	preq->ops.io_done = mreceive_done;
}


int MPID_Imrecv(void *buf, int count, MPI_Datatype datatype, MPID_Request *message, MPID_Request **request)
{
	MPID_Request *req;
	int rank;

	if (message == NULL) {
		req = MPID_DEV_Request_recv_create(NULL);
		MPIR_Status_set_procnull(&req->status);
		_MPID_Request_set_completed(req);
		*request = req;
		return MPI_SUCCESS;
	}

	/* recycle the message request: */
	req = message;

	rank = req->status.MPI_SOURCE;

	prepare_mrecvreq(req);

	if (rank >= 0 || rank == MPI_ANY_SOURCE) {

		prepare_data(req, buf, count, datatype);

		prepare_mrecv_cleanup(req, buf, count, datatype);

		MPID_PSP_Request_enqueue(req);

		pscom_post_recv(req->dev.kind.recv.common.pscom_req);

	} else {
		assert(rank == MPI_PROC_NULL);
		/* MPIR_Status_set_procnull(&req->status); already called in MPID_Mprobe or MPID_Improbe */
		_MPID_Request_set_completed(req);
	}

	*request = req;

	return MPI_SUCCESS;
}

int MPID_Mrecv(void *buf, int count, MPI_Datatype datatype,  MPID_Request *message, MPI_Status *status)
{
	int mpi_errno;
	MPID_Request *request;

	if (message == NULL) {
		MPIR_Status_set_procnull(status);

		return MPI_SUCCESS;
	}

	mpi_errno = MPID_Imrecv(buf, count, datatype, message, &request);

	if (mpi_errno == MPI_SUCCESS) {
		mpi_errno = MPID_PSP_Wait(request);
	}

	MPIR_Request_extract_status(request, status);
	MPID_DEV_Request_release_ref(request, MPID_REQUEST_RECV);

	return mpi_errno;
}

int MPID_Mprobe(int rank, int tag, MPID_Comm *comm, int context_offset, MPID_Request **message, MPI_Status *status)
{
	pscom_connection_t *con;
	pscom_socket_t *sock;
	MPID_Request *req;

	*message = NULL;

	con = MPID_PSCOM_rank2connection(comm, rank);
	sock = comm->pscom_socket;

	if (con || (rank == MPI_ANY_SOURCE)) {

		req = MPID_DEV_Request_recv_create(comm);
		if (unlikely(!req)) goto err_request_recv_create;

		prepare_mprobereq(req, tag, comm, context_offset);
		prepare_source(req, con, sock);

		MPID_PSP_LOCKFREE_CALL(pscom_probe(req->dev.kind.recv.common.pscom_req));

		set_probe_status(req->dev.kind.recv.common.pscom_req, status);
		/* Save the status */
		set_probe_status(req->dev.kind.recv.common.pscom_req, &req->status);

		/* convert the recv request into an mprobe request (aka MPI_Message) and return it: */
		req->kind = MPID_REQUEST_MPROBE;
		*message = req;

	} else switch (rank) {
		case MPI_PROC_NULL:
			MPIR_Status_set_procnull(status);
			break;
		case MPI_ROOT:
		default:
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

int MPID_Improbe(int rank, int tag, MPID_Comm *comm, int context_offset, int *flag, MPID_Request **message, MPI_Status *status)
{
	pscom_connection_t *con;
	pscom_socket_t *sock;
	MPID_Request *req;

	*message = NULL;

	con = MPID_PSCOM_rank2connection(comm, rank);
	sock = comm->pscom_socket;

	if (con || (rank == MPI_ANY_SOURCE)) {

		req = MPID_DEV_Request_recv_create(comm);
		if (unlikely(!req)) goto err_request_recv_create;

		prepare_mprobereq(req, tag, comm, context_offset);

		prepare_source(req, con, sock);

		*flag = pscom_iprobe(req->dev.kind.recv.common.pscom_req);
		if (*flag) {
			set_probe_status(req->dev.kind.recv.common.pscom_req, status);
			/* Save the status */
			set_probe_status(req->dev.kind.recv.common.pscom_req, &req->status);
			/* convert the recv request into an mprobe request (aka MPI_Message): */
			req->kind = MPID_REQUEST_MPROBE;
		} else {
			/* No matching message found. Release the request. */
			MPID_PSP_Subrequest_completed(req);
			MPID_DEV_Request_release_ref(req, MPID_REQUEST_RECV);
			req = NULL;
		}

		*message = req;

	} else switch (rank) {
		case MPI_PROC_NULL:
			MPIR_Status_set_procnull(status);
			*flag = 1;
			break;
		case MPI_ROOT:
		default:
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
