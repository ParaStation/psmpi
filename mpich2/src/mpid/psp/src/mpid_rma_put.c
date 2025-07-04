/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2025 ParTec AG, Munich
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
 *  RMA Put
 */

static
void rma_put_done(pscom_request_t * req)
{
    MPIR_Request *mpid_req = req->user->type.put_send.mpid_req;
    assert(pscom_req_successful(req));

    /* This is an pscom.io_done call. Global lock state undefined! */
    MPID_PSP_packed_msg_cleanup(&req->user->type.put_send.msg);
    /* ToDo: this is not threadsafe */
    req->user->type.put_send.win_ptr->rma_local_pending_cnt--;
    req->user->type.put_send.win_ptr->rma_local_pending_rank[req->user->type.
                                                             put_send.target_rank]--;

    if (mpid_req) {
        MPID_PSP_Subrequest_completed(mpid_req);
        MPIR_Request_free(mpid_req);
    } else {
        pscom_request_free(req);
    }
}


int MPIDI_PSP_Put_generic(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
                          int target_rank, MPI_Aint target_disp, int target_count,
                          MPI_Datatype target_datatype, MPIR_Win * win_ptr, MPIR_Request ** request)
{
    int mpi_error = MPI_SUCCESS;
    MPID_PSP_packed_msg_t msg;
    MPID_Win_rank_info *ri = win_ptr->rank_info + target_rank;
    char *target_buf;
#if 0
    fprintf(stderr, "int MPID_Put(origin_addr: %p, origin_count: %d, origin_datatype: %08x,"
            " target_rank: %d, target_disp: %d, target_count: %d, target_datatype: %08x,"
            " *win_ptr: %p)\n",
            origin_addr, origin_count, origin_datatype,
            target_rank, target_disp, target_count, target_datatype, win_ptr);
#endif
    /* Datatype */

    if (request) {
        *request = MPIR_Request_create(MPIR_REQUEST_KIND__RMA);
        (*request)->comm = win_ptr->comm_ptr;
        MPIR_Comm_add_ref(win_ptr->comm_ptr);
    }

    if (unlikely(target_rank == MPI_PROC_NULL)) {

        goto fn_completed;
    }


    /* Request-based RMA operations are only valid within a passive target epoch! */
    if (request && win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK &&
        win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK_ALL) {
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


    /* If the put is a local operation, do it here */
    if (target_rank == win_ptr->rank || win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED) {
        void *base;
        int disp_unit;

        if (win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED) {

            MPID_PSP_shm_rma_get_base(win_ptr, target_rank, &disp_unit, &base);
        } else {
            base = win_ptr->base;
            disp_unit = win_ptr->disp_unit;
        }

        mpi_error = MPIR_Localcopy(origin_addr, origin_count, origin_datatype,
                                   (char *) base + disp_unit * target_disp,
                                   target_count, target_datatype);

        if (mpi_error) {
            goto err_local_copy;
        }

        goto fn_completed;
    }

    /* Data */
    mpi_error = MPID_PSP_packed_msg_prepare(origin_addr, origin_count, origin_datatype, &msg);
    if (unlikely(mpi_error != MPI_SUCCESS))
        goto err_create_packed_msg;

    MPID_PSP_packed_msg_pack(origin_addr, origin_count, origin_datatype, &msg);

    target_buf = (char *) ri->base_addr + ri->disp_unit * target_disp;


    if (0) {    /* ToDo: re-enable pscom buildin rma_write */
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
        unsigned int encode_dt_size = MPID_PSP_Datatype_get_size(target_datatype);
        unsigned int xheader_len = sizeof(MPIDI_PSP_PSCOM_Xheader_rma_put_t) + encode_dt_size;
        pscom_request_t *req =
            pscom_request_create(xheader_len, sizeof(MPIDI_PSP_PSCOM_Request_put_send_t));
        MPIDI_PSP_PSCOM_Xheader_rma_put_t *xheader = &req->xheader.user.put;

        /* TODO:
         * We currently transfer the encoded datatype via the xheader.
         * However, this might exceed the maximum length (65536 Bytes).
         */
        MPIR_Assert(xheader_len <
                    (1 << (8 * sizeof(((struct PSCOM_header_net *) 0)->xheader_len))));

        req->user->type.put_send.msg = msg;
        req->user->type.put_send.win_ptr = win_ptr;

        MPID_PSP_Datatype_encode(target_datatype, &xheader->encoded_type);

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
        req->user->type.put_send.target_rank = target_rank;
        req->connection = ri->con;

        win_ptr->rma_local_pending_cnt++;
        win_ptr->rma_local_pending_rank[target_rank]++;
        win_ptr->rma_puts_accs[target_rank]++;

        if (request) {
            MPIR_Request *mpid_req = *request;
            /* TODO: Use a new and 'put_send'-dedicated MPID_DEV_Request_create() */
            /*       instead of allocating and overloading a common send request. */
            pscom_request_free(mpid_req->dev.kind.common.pscom_req);
            mpid_req->dev.kind.common.pscom_req = req;
            MPIR_Request_add_ref(mpid_req);
            req->user->type.put_send.mpid_req = mpid_req;
        } else {
            req->user->type.put_send.mpid_req = NULL;
        }

        pscom_post_send(req);
    }

    return MPI_SUCCESS;
  fn_completed:
    if (request) {
        MPIDI_PSP_Request_set_completed(*request);
    }
    return MPI_SUCCESS;
    /* --- */
  err_exit:
    if (request) {
        MPIDI_PSP_Request_set_completed(*request);
        MPIR_Request_free(*request);
    }
    return mpi_error;
    /* --- */
  err_create_packed_msg:
    goto err_exit;
  err_local_copy:
    goto err_exit;
  err_sync_rma:
    goto err_exit;
}


static
void rma_put_receive_done(pscom_request_t * req)
{
    /* This is an pscom.io_done call. Global lock state undefined! */
    MPIDI_PSP_PSCOM_Request_put_recv_t *rpr = &req->user->type.put_recv;
    MPIDI_PSP_PSCOM_Xheader_rma_put_t *xhead_rma = &req->xheader.user.put;
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
    xhead_rma->win_ptr->rma_puts_accs_received++;

    xhead_rma->win_ptr->rma_passive_pending_rank[xhead_rma->common.src_rank]--;

    pscom_request_free(req);
}


pscom_request_t *MPID_do_recv_rma_put(pscom_connection_t * con,
                                      MPIDI_PSP_PSCOM_Xheader_rma_put_t * xhead_rma)
{
    MPI_Datatype datatype = MPID_PSP_Datatype_decode(xhead_rma->encoded_type);

    pscom_request_t *req = PSCOM_REQUEST_CREATE();
    MPIDI_PSP_PSCOM_Request_put_recv_t *rpr = &req->user->type.put_recv;

    MPID_PSP_packed_msg_prepare(xhead_rma->target_buf,
                                xhead_rma->target_count, datatype, &rpr->msg);

    req->xheader_len = sizeof(req->xheader.user.put);
    req->data_len = rpr->msg.msg_sz;
    req->data = rpr->msg.msg;
    req->ops.io_done = rma_put_receive_done;

    rpr->datatype = datatype;

    xhead_rma->win_ptr->rma_passive_pending_rank[xhead_rma->common.src_rank]++;

    return req;
}


int MPID_Put(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
             int target_rank, MPI_Aint target_disp, int target_count,
             MPI_Datatype target_datatype, MPIR_Win * win_ptr)
{
    return MPIDI_PSP_Put_generic(origin_addr, origin_count, origin_datatype, target_rank,
                                 target_disp, target_count, target_datatype, win_ptr, NULL);
}

int MPID_Rput(const void *origin_addr, int origin_count,
              MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
              int target_count, MPI_Datatype target_datatype, MPIR_Win * win_ptr,
              MPIR_Request ** request)
{
    return MPIDI_PSP_Put_generic(origin_addr, origin_count, origin_datatype, target_rank,
                                 target_disp, target_count, target_datatype, win_ptr, request);
}
