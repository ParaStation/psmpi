

#ifndef _MPID_IRECV_H_
#define _MPID_IRECV_H_

static inline
void prepare_comreq(MPID_Request *req, int tag, MPID_Comm * comm, int context_offset)
{
	struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv;
	pscom_request_t *preq = rreq->common.pscom_req;
       
	req->comm = comm;
	rreq->tag = tag;
	rreq->context_id = comm->recvcontext_id + context_offset;
	rreq->mprobe_req = NULL;
	preq->ops.recv_accept = cb_accept_data;
	preq->xheader_len = sizeof(MPID_PSCOM_XHeader_Send_t);
}



static inline
void prepare_data(MPID_Request *req, void * buf, int count, MPI_Datatype datatype)
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
	fprintf(stderr, "MPIU_Malloc() failed\n");
	exit(1);
}


static inline
void prepare_cleanup(MPID_Request *req, void * buf, int count, MPI_Datatype datatype)
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


static inline
void prepare_source(MPID_Request *req, pscom_connection_t *con, pscom_socket_t *sock)
{
	struct MPID_DEV_Request_recv *rreq = &req->dev.kind.recv;
	pscom_request_t *preq = rreq->common.pscom_req;

	preq->connection = con;
	preq->socket = sock;
}

static inline
void set_probe_status(pscom_request_t *req, MPI_Status *status)
{
	if (!status || status == MPI_STATUS_IGNORE) return;

	status->count = req->header.data_len;
	status->cancelled = (req->state & PSCOM_REQ_STATE_CANCELED) ? 1 : 0;
	status->MPI_SOURCE = req->xheader.user.common.src_rank;
	status->MPI_TAG    = req->xheader.user.common.tag;
	/* status->MPI_ERROR  = MPI_SUCCESS; */
}

#endif
