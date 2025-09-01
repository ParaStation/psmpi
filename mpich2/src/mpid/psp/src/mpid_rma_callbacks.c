/*
 *   add callback functions for RMA one-sided
*/


#include "mpidimpl.h"
#include "mpid_psp_datatype.h"
#include "mpid_psp_packed_msg.h"
#include "mpid_psp_request.h"

/* Callback functions for RMA based on two-sided semantics.
These functions have to be registered globally in pscom.
The origin callbacks will be called when the request is finished locally.
The target callbacks will be called when the RMA operation is finished at the target side.
The callbacks are used for synchronization (operation counter) and MPI operations (e.g., MPI_SUM) at target side
*/

#if MPID_PSP_HAVE_PSCOM_RMA_API
/* origin callbacks */
void MPIDI_PSP_rma_put_origin_cb(pscom_request_t * req)
{
    MPIR_Win *win_ptr = (MPIR_Win *) req->user->pscom_put.win_id;
    win_ptr->rma_local_pending_cnt--;
    win_ptr->rma_local_pending_rank[req->user->pscom_put.target_rank]--;
    MPIR_Request *mpid_req = req->user->pscom_put.mpid_req;
    /* This is an pscom.io_done call. Global lock state undefined! */
    MPID_PSP_packed_msg_cleanup(&req->user->pscom_put.msg);
    /* pscom request has to be freed here */
    if (mpid_req) {
        MPID_PSP_Subrequest_completed(mpid_req);
        MPIR_Request_free(mpid_req);
    } else {
        pscom_request_free(req);
    }
}

void MPIDI_PSP_rma_get_origin_cb(pscom_request_t * req)
{
    MPIR_Win *win_ptr = (MPIR_Win *) req->user->pscom_get.win_id;
    win_ptr->rma_local_pending_cnt--;
    win_ptr->rma_local_pending_rank[req->user->pscom_get.target_rank]--;
    MPIR_Request *mpid_req = req->user->pscom_get.mpid_req;
    /* This is an pscom.io_done call. Global lock state undefined! */
    MPID_PSP_packed_msg_cleanup(&req->user->pscom_get.msg);
    /* pscom request has to be freed here */
    if (mpid_req) {
        MPID_PSP_Subrequest_completed(mpid_req);
        MPIR_Request_free(mpid_req);
    } else {
        pscom_request_free(req);
    }
}

void MPIDI_PSP_rma_acc_origin_cb(pscom_request_t * req)
{
    MPIR_Win *win_ptr = (MPIR_Win *) req->user->pscom_accumulate.win_id;
    win_ptr->rma_local_pending_cnt--;
    win_ptr->rma_local_pending_rank[req->user->pscom_accumulate.target_rank]--;
    assert(win_ptr->rma_pending_accumulates[req->user->pscom_accumulate.target_rank] == 1);
    win_ptr->rma_pending_accumulates[req->user->pscom_accumulate.target_rank] = 0;

    /* This is an pscom.io_done call. Global lock state undefined! */
    MPID_PSP_packed_msg_cleanup(&req->user->pscom_accumulate.msg);
    MPIR_Request *mpid_req = req->user->pscom_accumulate.mpid_req;
    /* pscom request has to be freed here */
    if (mpid_req) {
        MPID_PSP_Subrequest_completed(mpid_req);
        MPIR_Request_free(mpid_req);
    } else {
        pscom_request_free(req);
    }
}

void MPIDI_PSP_rma_get_acc_origin_cb(pscom_request_t * req)
{
    MPIR_Win *win_ptr = (MPIR_Win *) req->user->pscom_get_accumulate.win_id;
    win_ptr->rma_local_pending_cnt--;
    win_ptr->rma_local_pending_rank[req->user->pscom_get_accumulate.target_rank]--;
    /* This is an pscom.io_done call. Global lock state undefined! */
    MPID_PSP_packed_msg_cleanup(&req->user->pscom_get_accumulate.msg);
    MPIR_Request *mpid_req = req->user->pscom_get_accumulate.mpid_req;
    /* pscom request has to be freed here */
    if (mpid_req) {
        MPID_PSP_Subrequest_completed(mpid_req);
        MPIR_Request_free(mpid_req);
    } else {
        pscom_request_free(req);
    }
}

void MPIDI_PSP_rma_fetch_op_origin_cb(pscom_request_t * req)
{
    MPIR_Win *win_ptr = (MPIR_Win *) req->user->pscom_fetch_op.win_id;
    win_ptr->rma_local_pending_cnt--;
    win_ptr->rma_local_pending_rank[req->user->pscom_fetch_op.target_rank]--;
    pscom_request_free(req);
}

void MPIDI_PSP_rma_comp_swap_origin_cb(pscom_request_t * req)
{
    MPIR_Win *win_ptr = (MPIR_Win *) req->user->pscom_compare_swap.win_id;
    win_ptr->rma_local_pending_cnt--;
    win_ptr->rma_local_pending_rank[req->user->pscom_compare_swap.target_rank]--;
    pscom_request_free(req);
}


/* target callbacks */
void MPIDI_PSP_rma_put_target_cb(pscom_request_t * req)
{
    MPIDI_PSCOM_RMA_API_put_t *xhead_rma = &req->xheader.rma_put.user;
    MPIR_Win *win_ptr = (MPIR_Win *) xhead_rma->remote_win;
    win_ptr->rma_puts_accs_received++;
    win_ptr->rma_source_rank_received[xhead_rma->source_rank]--;
}

void MPIDI_PSP_rma_get_target_cb(pscom_request_t * req)
{

}

void MPIDI_PSP_rma_acc_target_cb(pscom_request_t * req)
{
    // RMA accumulate target side callback, recv accumulate request and do MPI ops here
    MPIDI_PSCOM_RMA_API_accumulate_t *xhead_rma = &req->xheader.rma_accumulate.user;
    void *target_buf = req->xheader.rma_accumulate.common.dest;

    MPIR_Win *win_ptr = (MPIR_Win *) xhead_rma->remote_win;
    MPI_Datatype origin_datatype = xhead_rma->origin_datatype;
    int origin_count = xhead_rma->origin_count;
    int target_count = xhead_rma->target_count;
    MPI_Datatype target_datatype = MPID_PSP_Datatype_decode(xhead_rma->encoded_type);
    MPI_Op op = xhead_rma->mpi_op;

    MPIDI_PSP_compute_acc_op((void *) req->data, origin_count, origin_datatype,
                             target_buf, target_count, target_datatype, op, TRUE);

    MPID_PSP_Datatype_release(target_datatype);

    /* ToDo: this is not threadsafe */
    win_ptr->rma_puts_accs_received++;
    win_ptr->rma_source_rank_received[xhead_rma->source_rank]--;
}

void MPIDI_PSP_rma_get_acc_target_cb(pscom_request_t * req)
{
    // RMA get_accumulate target side callback, recv accumulate request and do MPI ops here
    MPIDI_PSCOM_RMA_API_get_accumulate_t *xhead_rma = &req->xheader.rma_get_accumulate.user;
    void *target_buf = req->xheader.rma_get_accumulate.common.src;
    MPIR_Win *win_ptr = (MPIR_Win *) xhead_rma->remote_win;

    MPI_Datatype origin_datatype = xhead_rma->origin_datatype;
    int origin_count = xhead_rma->origin_count;
    int target_count = xhead_rma->target_count;
    MPI_Datatype target_datatype = MPID_PSP_Datatype_decode(xhead_rma->encoded_type);
    MPI_Op op = xhead_rma->mpi_op;

    MPIDI_PSP_compute_acc_op((void *) req->data, origin_count, origin_datatype,
                             target_buf, target_count, target_datatype, op, TRUE);

    MPID_PSP_Datatype_release(target_datatype);

    /* ToDo: this is not threadsafe */
    win_ptr->rma_puts_accs_received++;
    win_ptr->rma_source_rank_received[xhead_rma->source_rank]--;
}

void MPIDI_PSP_rma_fetch_op_target_cb(pscom_request_t * req)
{
    // RMA fetch_op target side callback, recv fetch op request and do MPI ops here
    MPIDI_PSCOM_RMA_API_fetch_op_t *xhead_rma = &req->xheader.rma_fetch_op.user;
    void *target_buf = req->xheader.rma_fetch_op.common.src;
    MPIR_Win *win_ptr = (MPIR_Win *) xhead_rma->remote_win;

    MPI_Datatype datatype = xhead_rma->datatype;
    MPI_Op op = xhead_rma->mpi_op;

    MPIDI_PSP_compute_acc_op((void *) req->data, 1, datatype, target_buf, 1, datatype, op, TRUE);

    /* ToDo: this is not threadsafe */
    win_ptr->rma_puts_accs_received++;
    win_ptr->rma_source_rank_received[xhead_rma->source_rank]--;
}

void MPIDI_PSP_rma_comp_swap_target_cb(pscom_request_t * req)
{
    MPIDI_PSCOM_RMA_API_compare_swap_t *xhead_rma = &req->xheader.rma_compare_swap.user;
    MPIR_Win *win_ptr = (MPIR_Win *) xhead_rma->remote_win;
    win_ptr->rma_puts_accs_received++;
    win_ptr->rma_source_rank_received[xhead_rma->source_rank]--;
}
#endif
