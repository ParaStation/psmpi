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
 *  RMA Accumulate
 */

static
void rma_accumulate_done(pscom_request_t * req)
{
    MPIR_Request *mpid_req = req->user->accumulate_send.mpid_req;
    /* This is an pscom.io_done call. Global lock state undefined! */
    MPID_PSP_packed_msg_cleanup(&req->user->accumulate_send.msg);
    /* ToDo: this is not threadsafe */
    req->user->accumulate_send.win_ptr->rma_local_pending_cnt--;
    req->user->accumulate_send.win_ptr->rma_local_pending_rank[req->user->
                                                               accumulate_send.target_rank]--;

    assert(req->user->accumulate_send.
           win_ptr->rma_pending_accumulates[req->user->accumulate_send.target_rank] == 1);
    req->user->accumulate_send.win_ptr->rma_pending_accumulates[req->user->
                                                                accumulate_send.target_rank] = 0;

    if (mpid_req) {
        MPID_PSP_Subrequest_completed(mpid_req);
        MPIR_Request_free(mpid_req);
    } else {
        pscom_request_free(req);
    }
}


static
int MPIDI_PSP_Accumulate_generic(const void *origin_addr, int origin_count,
                                 MPI_Datatype origin_datatype, int target_rank,
                                 MPI_Aint target_disp, int target_count,
                                 MPI_Datatype target_datatype, MPI_Op op, MPIR_Win * win_ptr,
                                 MPIR_Request ** request)
{
    int mpi_error = MPI_SUCCESS;
    MPID_PSP_packed_msg_t msg;
    MPID_Win_rank_info *ri = win_ptr->rank_info + target_rank;
    uint64_t data_sz, target_data_sz;
    MPIR_Datatype *dt_ptr;
    char *target_buf;
    MPI_Datatype origin_datatype_mapped;
    MPI_Datatype origin_count_mapped;

#if 0
    fprintf(stderr, "int MPID_Accumulate(origin_addr: %p, origin_count: %d, origin_datatype: %08x,"
            " target_rank: %d, target_disp: %d, target_count: %d, target_datatype: %08x,"
            " op: 0x%x, *win_ptr: %p)\n",
            origin_addr, origin_count, origin_datatype,
            target_rank, target_disp, target_count, target_datatype, op, win_ptr);
#endif

    MPIDI_PSP_Datatype_get_size_dt_ptr(origin_count, origin_datatype, data_sz, dt_ptr);
    MPIDI_PSP_Datatype_check_size(target_datatype, target_count, target_data_sz);
    if (data_sz == 0 || target_data_sz == 0) {
        goto fn_immed_completed;
    }

    if (!win_ptr->enable_rma_accumulate_ordering && unlikely(op == MPI_REPLACE)) {
        /*  MPI_PUT is a special case of MPI_ACCUMULATE, with the operation MPI_REPLACE.
         * |  However, PUT and ACCUMULATE have different constraints on concurrent updates!
         * |  Therefore, in the SHMEM case, the PUT/REPLACE operation must here be locked:
         */
        if (unlikely(win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED)) {
            MPID_PSP_shm_rma_mutex_lock(win_ptr);
            mpi_error = MPIDI_PSP_Put_generic(origin_addr, origin_count, origin_datatype,
                                              target_rank, target_disp, target_count,
                                              target_datatype, win_ptr, request);
            MPID_PSP_shm_rma_mutex_unlock(win_ptr);
            return mpi_error;
        } else {
            return MPIDI_PSP_Put_generic(origin_addr, origin_count, origin_datatype,
                                         target_rank, target_disp, target_count,
                                         target_datatype, win_ptr, request);
        }
    }

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
#ifdef MPIDI_PSP_WITH_CUDA_AWARENESS
    if (pscom_is_gpu_mem(origin_addr)) {
        int contig;
        size_t data_sz;
        MPIR_Datatype *dtp;
        MPI_Aint true_lb;

        MPIDI_Datatype_get_info(origin_count, origin_datatype, contig, data_sz, dtp, true_lb);
        mpi_error = MPID_PSP_packed_msg_allocate(data_sz, &msg);

        // Avoid compiler warnings about unused variables:
        (void) contig;
        (void) true_lb;
    } else
#endif
    {
        mpi_error = MPID_PSP_packed_msg_prepare(origin_addr, origin_count, origin_datatype, &msg);
    }

    if (unlikely(mpi_error != MPI_SUCCESS))
        goto err_create_packed_msg;

    /* map origin datatype to its basic_type */
    MPIDI_PSP_Datatype_map_to_basic_type(origin_datatype,
                                         origin_count,
                                         &origin_datatype_mapped, &origin_count_mapped);

    MPID_PSP_packed_msg_pack(origin_addr, origin_count, origin_datatype, &msg);

    target_buf = (char *) ri->base_addr + ri->disp_unit * target_disp;

    /* If the acc is a local operation, do it here */
    if (target_rank == win_ptr->rank || win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED) {
        if (target_rank != win_ptr->rank) {
            int disp_unit;
            void *base;

            MPID_PSP_shm_rma_get_base(win_ptr, target_rank, &disp_unit, &base);

            assert(ri->disp_unit == disp_unit);
            target_buf = (char *) base + disp_unit * target_disp;

            /* accumulate may be executed concurrently --> locking required! */
            MPID_PSP_shm_rma_mutex_lock(win_ptr);
            MPIDI_PSP_compute_acc_op(msg.msg, origin_count_mapped,
                                     origin_datatype_mapped, target_buf,
                                     target_count, target_datatype, op, TRUE);
            MPID_PSP_shm_rma_mutex_unlock(win_ptr);

        } else {
            /* This is a local acc, but do locking just in SHMEM case! */
            if (unlikely(win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED)) {

                /* in case of a COMM_SELF clone, mutex_lock()/unlock() will just act as no-ops: */
                MPID_PSP_shm_rma_mutex_lock(win_ptr);
                MPIDI_PSP_compute_acc_op(msg.msg, origin_count_mapped,
                                         origin_datatype_mapped, target_buf,
                                         target_count, target_datatype, op, TRUE);
                MPID_PSP_shm_rma_mutex_unlock(win_ptr);
            } else {
                /* this is a local operation on non-shared memory: */
                MPIDI_PSP_compute_acc_op(msg.msg, origin_count_mapped,
                                         origin_datatype_mapped, target_buf,
                                         target_count, target_datatype, op, TRUE);
            }
        }

        MPID_PSP_packed_msg_cleanup(&msg);

        goto fn_completed;
    }
#if MPID_PSP_HAVE_PSCOM_RMA_API
    /* check if the data type is contig or not */
    int is_contig = 0;
    MPID_PSP_Datatype_is_contig_and_zero_true_lb(target_datatype, &is_contig);
    if (is_contig) {
        /* Use pscom buildin rma */
        /* encode the target data type, comment: if data is predefined, should we still encode and decode？ ofc encoding and decoding do nothing */
        /* comments: we do not consider enable_rma_accumulate_ordering when using pscom RMA api : probably yes due to the failed test */
        unsigned int encode_dt_size = MPID_PSP_Datatype_get_size(target_datatype);
        pscom_request_t *req =
            pscom_request_create(sizeof(pscom_xheader_rma_accumulate_t) + encode_dt_size,
                                 sizeof(MPIDI_PSP_PSCOM_Request_rma_accumulate_t));
        /* essential data required by pscom */
        req->rma.origin_addr = msg.msg;
        req->rma.target_addr = target_buf;
        req->rma.rkey = win_ptr->pscom_rkey[target_rank];
        req->data_len = msg.msg_sz;
        req->connection = ri->con;
        req->ops.io_done = MPIDI_PSP_rma_acc_origin_cb;

        /* user-defined extended headers */
        req->xheader.rma_accumulate.user.remote_win = (void *) ri->win_ptr;
        req->xheader.rma_accumulate.user.source_rank = win_ptr->rank;
        req->xheader.rma_accumulate.user.origin_datatype = origin_datatype_mapped;
        req->xheader.rma_accumulate.user.origin_count = origin_count_mapped;
        req->xheader.rma_accumulate.user.target_count = target_count;
        req->xheader.rma_accumulate.user.mpi_op = op;

        /* encode the target datatype */
        MPID_PSP_Datatype_encode(target_datatype, &req->xheader.rma_accumulate.user.encoded_type);

        /* local user-defined data for the callback at origin side */
        MPIDI_PSP_PSCOM_Request_rma_accumulate_t *pscom_rma_user = &req->user->pscom_accumulate;
        pscom_rma_user->win_id = (void *) win_ptr;
        pscom_rma_user->target_rank = target_rank;
        pscom_rma_user->msg = msg;

        if (win_ptr->enable_rma_accumulate_ordering) {
            /* wait until a previous accumulated request is finished */
            while (win_ptr->rma_pending_accumulates[target_rank]) {
                pscom_test_any();
            }
        }
        win_ptr->rma_pending_accumulates[target_rank] = 1;

        /* store request and msg, do cleanup when req is finished */
        if (request) {
            MPIR_Request *mpid_req = *request;
            /* TODO: Use a new and 'put_send'-dedicated MPID_DEV_Request_create() */
            /*       instead of allocating and overloading a common send request. */
            pscom_request_free(mpid_req->dev.kind.common.pscom_req);
            mpid_req->dev.kind.common.pscom_req = req;
            MPIR_Request_add_ref(mpid_req);
            pscom_rma_user->mpid_req = mpid_req;
        } else {
            pscom_rma_user->mpid_req = NULL;
        }

        pscom_post_rma_accumulate(req);

        win_ptr->rma_local_pending_cnt++;
        win_ptr->rma_local_pending_rank[target_rank]++;
        win_ptr->rma_puts_accs[target_rank]++;
    } else
#endif
    {
        unsigned int encode_dt_size = MPID_PSP_Datatype_get_size(target_datatype);
        unsigned int xheader_len =
            sizeof(MPIDI_PSP_PSCOM_Xheader_rma_accumulate_t) + encode_dt_size;

        pscom_request_t *req =
            pscom_request_create(xheader_len, sizeof(MPIDI_PSP_PSCOM_Request_accumulate_send_t));
        MPIDI_PSP_PSCOM_Xheader_rma_accumulate_t *xheader = &req->xheader.user.accumulate;

        /* TODO:
         * We currently transfer the encoded datatype via the xheader.
         * However, this might exceed the maximum length (65536 Bytes).
         */
        MPIR_Assert(xheader_len <
                    (1 << (8 * sizeof(((struct PSCOM_header_net *) 0)->xheader_len))));

        req->user->accumulate_send.msg = msg;
        req->user->accumulate_send.win_ptr = win_ptr;

        /* encode the target datatype */
        MPID_PSP_Datatype_encode(target_datatype, &xheader->encoded_type);

        xheader->common.tag = 0;
        xheader->common.context_id = 0;
        xheader->common.type = MPID_PSP_MSGTYPE_RMA_ACCUMULATE;
        xheader->common._reserved_ = 0;
        xheader->common.src_rank = win_ptr->rank;

        /* xheader->target_disp = target_disp; */
        xheader->origin_datatype = origin_datatype_mapped;
        xheader->origin_count = origin_count_mapped;
        xheader->target_count = target_count;
        xheader->target_count = target_count;
        xheader->target_buf = target_buf;
        /*              xheader->epoch = ri->epoch_origin; */
        xheader->win_ptr = ri->win_ptr; /* remote win_ptr */
        xheader->op = op;       /* ToDo: check: is op a buildin op? */

        req->xheader_len = xheader_len;

        req->data = msg.msg;
        req->data_len = msg.msg_sz;

        req->ops.io_done = rma_accumulate_done;
        req->user->accumulate_send.target_rank = target_rank;
        req->connection = ri->con;

        if (win_ptr->enable_rma_accumulate_ordering) {
            /* wait until a previous accumulated request is finished */
            while (win_ptr->rma_pending_accumulates[target_rank]) {
                pscom_test_any();
            }
        }
        win_ptr->rma_pending_accumulates[target_rank] = 1;

        win_ptr->rma_local_pending_cnt++;
        win_ptr->rma_local_pending_rank[target_rank]++;
        win_ptr->rma_puts_accs[target_rank]++;

        if (request) {
            MPIR_Request *mpid_req = *request;
            /* TODO: Use a new and 'acc_send'-dedicated MPID_DEV_Request_create() */
            /*       instead of allocating and overloading a common send request. */
            pscom_request_free(mpid_req->dev.kind.common.pscom_req);
            mpid_req->dev.kind.common.pscom_req = req;
            MPIR_Request_add_ref(mpid_req);
            req->user->accumulate_send.mpid_req = mpid_req;
        } else {
            req->user->accumulate_send.mpid_req = NULL;
        }

        pscom_post_send(req);
    }

    return MPI_SUCCESS;
    /* --- */
  fn_immed_completed:
    if (request && !(*request)) {
        *request = MPIR_Request_create_complete(MPIR_REQUEST_KIND__RMA);
        MPIR_ERR_CHKANDSTMT((*request) == NULL, mpi_error,
                            MPIX_ERR_NOREQ, goto err_exit, "**nomemreq");
    }

    return MPI_SUCCESS;
    /* --- */
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
  err_sync_rma:
    goto err_exit;
}


static
void rma_accumulate_receive_done(pscom_request_t * req)
{
    /* This is an pscom.io_done call. Global lock state undefined! */
    MPIDI_PSP_PSCOM_Xheader_rma_accumulate_t *xhead_rma = &req->xheader.user.accumulate;
    MPIDI_PSP_PSCOM_Request_accumulate_recv_t *rpr = &req->user->accumulate_recv;
/*
  void *origin_addr		= req->data;
  int origin_count		= req->data_len / sizeof(basic buildin type);
  MPI_Datatype origin_datatype	= basic buildin type;
*/

    MPI_Datatype origin_datatype = xhead_rma->origin_datatype;
    int origin_count = xhead_rma->origin_count;
    void *target_buf = xhead_rma->target_buf;
    int target_count = xhead_rma->target_count;
    MPI_Datatype target_datatype = rpr->datatype;
    MPI_Op op = xhead_rma->op;

    MPIR_Win *win_ptr = xhead_rma->win_ptr;

    MPIDI_PSP_compute_acc_op(req->data, origin_count, origin_datatype,
                             target_buf, target_count, target_datatype, op, TRUE);

    MPID_PSP_Datatype_release(target_datatype);
    /* ToDo: this is not threadsafe */
    win_ptr->rma_puts_accs_received++;

    xhead_rma->win_ptr->rma_source_rank_received[xhead_rma->common.src_rank]--;
    xhead_rma->win_ptr->rma_passive_pending_rank[xhead_rma->common.src_rank]--;

    pscom_request_free(req);
}


pscom_request_t *MPID_do_recv_rma_accumulate(pscom_connection_t * con,
                                             pscom_header_net_t * header_net)
{
    MPIDI_PSP_PSCOM_Xheader_rma_accumulate_t *xhead_rma = &header_net->xheader->user.accumulate;

    MPI_Datatype datatype = MPID_PSP_Datatype_decode(xhead_rma->encoded_type);

    pscom_request_t *req = pscom_request_create(sizeof(MPIDI_PSP_PSCOM_Xheader_rma_accumulate_t),
                                                sizeof(MPIDI_PSP_PSCOM_Request_accumulate_recv_t) +
                                                header_net->data_len);

    MPIDI_PSP_PSCOM_Request_accumulate_recv_t *rpr = &req->user->accumulate_recv;


    /* Receive the packed_msg into request->user space */
    req->xheader_len = sizeof(MPIDI_PSP_PSCOM_Xheader_rma_accumulate_t);
    req->data_len = header_net->data_len;
    req->data = &rpr->packed_msg;

    rpr->datatype = datatype;
    req->ops.io_done = rma_accumulate_receive_done;

    xhead_rma->win_ptr->rma_passive_pending_rank[xhead_rma->common.src_rank]++;

    return req;
}


int MPID_Accumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
                    int target_rank, MPI_Aint target_disp, int target_count,
                    MPI_Datatype target_datatype, MPI_Op op, MPIR_Win * win_ptr)
{
    return MPIDI_PSP_Accumulate_generic(origin_addr, origin_count, origin_datatype,
                                        target_rank, target_disp, target_count, target_datatype,
                                        op, win_ptr, NULL);
}

int MPID_Raccumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
                     int target_rank, MPI_Aint target_disp, int target_count,
                     MPI_Datatype target_datatype, MPI_Op op, MPIR_Win * win_ptr,
                     MPIR_Request ** request)
{
    return MPIDI_PSP_Accumulate_generic((void *) origin_addr, origin_count, origin_datatype,
                                        target_rank, target_disp, target_count, target_datatype,
                                        op, win_ptr, request);
}


/***********************************************************************************************************
 *   RMA-3.0 Get & Accumulate / Fetch & Op Functions:
 */

static
int MPIDI_PSP_Get_accumulate_generic(const void *origin_addr, int origin_count,
                                     MPI_Datatype origin_datatype, void *result_addr,
                                     int result_count, MPI_Datatype result_datatype,
                                     int target_rank, MPI_Aint target_disp, int target_count,
                                     MPI_Datatype target_datatype, MPI_Op op, MPIR_Win * win_ptr,
                                     MPIR_Request ** request)
{
    int mpi_error = MPI_SUCCESS;

    if (unlikely(target_rank == MPI_PROC_NULL)) {

        if (request) {
            *request = MPIR_Request_create(MPIR_REQUEST_KIND__RMA);
            (*request)->comm = win_ptr->comm_ptr;
            MPIR_Comm_add_ref(win_ptr->comm_ptr);
            MPIDI_PSP_Request_set_completed(*request);
        }

        return MPI_SUCCESS;
    }

    if (unlikely(op == MPI_NO_OP)) {
        return MPIDI_PSP_Get_generic(result_addr, result_count, result_datatype,
                                     target_rank, target_disp, target_count,
                                     target_datatype, win_ptr, request);
    }
#if MPID_PSP_HAVE_PSCOM_RMA_API
    int is_target_contig = 0;
    int is_origin_contig = 0;
    int is_result_contig = 0;
    MPID_PSP_Datatype_is_contig_and_zero_true_lb(target_datatype, &is_target_contig);
    MPID_PSP_Datatype_is_contig_and_zero_true_lb(origin_datatype, &is_origin_contig);
    MPID_PSP_Datatype_is_contig_and_zero_true_lb(result_datatype, &is_result_contig);

    if (!is_target_contig || !is_origin_contig || !is_result_contig)
#endif
    {   /* TODO: This implementation is just based on the common Get/Accumulate ops (plus some additional internal locking): */

        MPIDI_PSP_Win_lock_internal(target_rank, win_ptr);

        MPID_Get(result_addr, result_count, result_datatype, target_rank, target_disp, target_count,
                 target_datatype, win_ptr);

        MPIDI_PSP_Win_wait_local_completion(target_rank, win_ptr);

        MPIDI_PSP_Accumulate_generic((void *) origin_addr, origin_count, origin_datatype,
                                     target_rank, target_disp, target_count, target_datatype, op,
                                     win_ptr, request);

        MPIDI_PSP_Win_unlock_internal(target_rank, win_ptr);
    }
#if MPID_PSP_HAVE_PSCOM_RMA_API
    else {
        /* A dedicated Get_accumulate() implementation using pscom RMA API */
        /* note: only supports contig data type */
        MPID_PSP_packed_msg_t msg;
        MPID_Win_rank_info *ri = win_ptr->rank_info + target_rank;
        uint64_t data_sz, target_data_sz;
        MPIR_Datatype *dt_ptr;
        char *target_buf;
        MPI_Datatype origin_datatype_mapped;
        MPI_Datatype origin_count_mapped;

        MPIDI_PSP_Datatype_get_size_dt_ptr(origin_count, origin_datatype, data_sz, dt_ptr);
        MPIDI_PSP_Datatype_check_size(target_datatype, target_count, target_data_sz);
        if (data_sz == 0 || target_data_sz == 0) {
            goto fn_immed_completed;
        }

        if (request) {
            *request = MPIR_Request_create(MPIR_REQUEST_KIND__RMA);
            (*request)->comm = win_ptr->comm_ptr;
            MPIR_Comm_add_ref(win_ptr->comm_ptr);
        }

        /* Request-based RMA operations are only valid within a passive target epoch! */
        if (request && win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK &&
            win_ptr->epoch_state != MPID_PSP_EPOCH_LOCK_ALL) {
            mpi_error = MPI_ERR_RMA_SYNC;
            goto err_exit;
        }

        /* Check that we are within an access/exposure epoch: */
        if (win_ptr->epoch_state == MPID_PSP_EPOCH_NONE) {
            mpi_error = MPI_ERR_RMA_SYNC;
            goto err_exit;
        }

        /* Track access epoch state: */
        if (win_ptr->epoch_state == MPID_PSP_EPOCH_FENCE_ISSUED) {
            win_ptr->epoch_state = MPID_PSP_EPOCH_FENCE;
        }
#ifdef MPIDI_PSP_WITH_CUDA_AWARENESS
        if (pscom_is_gpu_mem(origin_addr)) {
            int contig;
            size_t data_sz;
            MPIR_Datatype *dtp;
            MPI_Aint true_lb;

            MPIDI_Datatype_get_info(origin_count, origin_datatype, contig, data_sz, dtp, true_lb);
            mpi_error = MPID_PSP_packed_msg_allocate(data_sz, &msg);

            // Avoid compiler warnings about unused variables:
            (void) contig;
            (void) true_lb;
        } else
#endif
        {
            mpi_error =
                MPID_PSP_packed_msg_prepare(origin_addr, origin_count, origin_datatype, &msg);
        }

        if (unlikely(mpi_error != MPI_SUCCESS))
            goto err_exit;

        /* map origin datatype to its basic_type */
        MPIDI_PSP_Datatype_map_to_basic_type(origin_datatype,
                                             origin_count,
                                             &origin_datatype_mapped, &origin_count_mapped);

        MPID_PSP_packed_msg_pack(origin_addr, origin_count, origin_datatype, &msg);

        target_buf = (char *) ri->base_addr + ri->disp_unit * target_disp;

        /* If the get_acc is a local operation, do it here */
        if (target_rank == win_ptr->rank || win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED) {
            if (target_rank != win_ptr->rank) {
                int disp_unit;
                void *base;

                MPID_PSP_shm_rma_get_base(win_ptr, target_rank, &disp_unit, &base);

                assert(ri->disp_unit == disp_unit);
                target_buf = (char *) base + disp_unit * target_disp;

                /* accumulate may be executed concurrently --> locking required! */
                MPID_PSP_shm_rma_mutex_lock(win_ptr);
                /* local copy target buff back to result addr */
                mpi_error = MPIR_Localcopy(target_buf,
                                           target_count, target_datatype, result_addr,
                                           result_count, result_datatype);
                if (mpi_error)
                    goto err_exit;
                /* do MPI OP from origin addr to the target buff */
                MPIDI_PSP_compute_acc_op(msg.msg, origin_count_mapped,
                                         origin_datatype_mapped, target_buf,
                                         target_count, target_datatype, op, TRUE);
                MPID_PSP_shm_rma_mutex_unlock(win_ptr);

            } else {
                /* This is a local acc, but do locking just in SHMEM case! */
                if (unlikely(win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED)) {
                    /* in case of a COMM_SELF clone, mutex_lock()/unlock() will just act as no-ops: */
                    MPID_PSP_shm_rma_mutex_lock(win_ptr);
                    mpi_error = MPIR_Localcopy(target_buf,
                                               target_count, target_datatype, result_addr,
                                               result_count, result_datatype);
                    if (mpi_error)
                        goto err_exit;
                    MPIDI_PSP_compute_acc_op(msg.msg, origin_count_mapped,
                                             origin_datatype_mapped, target_buf,
                                             target_count, target_datatype, op, TRUE);
                    MPID_PSP_shm_rma_mutex_unlock(win_ptr);
                } else {
                    /* this is a local operation on non-shared memory: */
                    mpi_error = MPIR_Localcopy(target_buf,
                                               target_count, target_datatype, result_addr,
                                               result_count, result_datatype);
                    if (mpi_error)
                        goto err_exit;
                    MPIDI_PSP_compute_acc_op(msg.msg, origin_count_mapped,
                                             origin_datatype_mapped, target_buf,
                                             target_count, target_datatype, op, TRUE);
                }
            }

            MPID_PSP_packed_msg_cleanup(&msg);

            goto fn_completed;
        }

        /* Use pscom buildin rma */
        /* encode the target data type, comment: if data is predefined, should we still encode and decode？ ofc encoding and decoding do nothing
         * if  MPIR_DATATYPE_IS_PREDEFINED(target_datatype) we can only send datatype
         * hardware-accelerated RMA only use predefined MPI data type? */

        unsigned int encode_dt_size = MPID_PSP_Datatype_get_size(target_datatype);
        pscom_request_t *req =
            pscom_request_create(sizeof(pscom_xheader_rma_get_accumulate_t) + encode_dt_size,
                                 sizeof(MPIDI_PSP_PSCOM_Request_rma_get_accumulate_t));
        /* essential data required by pscom */
        req->rma.origin_addr = msg.msg;
        req->rma.target_addr = target_buf;
        req->rma.result_addr = result_addr;
        req->rma.rkey = win_ptr->pscom_rkey[target_rank];
        req->data_len = msg.msg_sz;
        req->connection = ri->con;
        req->ops.io_done = MPIDI_PSP_rma_get_acc_origin_cb;

        /* user-defined extended headers */
        req->xheader.rma_get_accumulate.user.remote_win = (void *) ri->win_ptr;
        req->xheader.rma_get_accumulate.user.source_rank = win_ptr->rank;
        req->xheader.rma_get_accumulate.user.origin_datatype = origin_datatype_mapped;
        req->xheader.rma_get_accumulate.user.origin_count = origin_count_mapped;
        req->xheader.rma_get_accumulate.user.target_count = target_count;
        req->xheader.rma_get_accumulate.user.mpi_op = op;       /* ToDo: check: is op a buildin op? */
        /* encode the target datatype */
        MPID_PSP_Datatype_encode(target_datatype,
                                 &req->xheader.rma_get_accumulate.user.encoded_type);

        /* information for the callback at origin side */
        MPIDI_PSP_PSCOM_Request_rma_get_accumulate_t *pscom_rma_user =
            &req->user->pscom_get_accumulate;
        pscom_rma_user->win_id = (void *) win_ptr;
        pscom_rma_user->target_rank = target_rank;
        pscom_rma_user->msg = msg;
        /* store request and msg, do cleanup when req is finished */
        if (request) {
            MPIR_Request *mpid_req = *request;
            /* TODO: Use a new and 'put_send'-dedicated MPID_DEV_Request_create() */
            /*       instead of allocating and overloading a common send request. */
            pscom_request_free(mpid_req->dev.kind.common.pscom_req);
            mpid_req->dev.kind.common.pscom_req = req;
            MPIR_Request_add_ref(mpid_req);
            pscom_rma_user->mpid_req = mpid_req;
        } else {
            pscom_rma_user->mpid_req = NULL;
        }

        pscom_post_rma_get_accumulate(req);

        win_ptr->rma_local_pending_cnt++;
        win_ptr->rma_local_pending_rank[target_rank]++;
        win_ptr->rma_puts_accs[target_rank]++;
    }
#endif

    return MPI_SUCCESS;
    /* --- */
  fn_immed_completed:
    if (request && !(*request)) {
        *request = MPIR_Request_create_complete(MPIR_REQUEST_KIND__RMA);
        MPIR_ERR_CHKANDSTMT((*request) == NULL, mpi_error,
                            MPIX_ERR_NOREQ, goto err_exit, "**nomemreq");
    }

    return MPI_SUCCESS;
    /* --- */
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
}

int MPID_Get_accumulate(const void *origin_addr, int origin_count,
                        MPI_Datatype origin_datatype, void *result_addr, int result_count,
                        MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
                        int target_count, MPI_Datatype target_datatype, MPI_Op op,
                        MPIR_Win * win_ptr)
{
    return MPIDI_PSP_Get_accumulate_generic(origin_addr, origin_count, origin_datatype, result_addr,
                                            result_count, result_datatype, target_rank, target_disp,
                                            target_count, target_datatype, op, win_ptr, NULL);
}

int MPID_Rget_accumulate(const void *origin_addr, int origin_count,
                         MPI_Datatype origin_datatype, void *result_addr, int result_count,
                         MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
                         int target_count, MPI_Datatype target_datatype, MPI_Op op,
                         MPIR_Win * win_ptr, MPIR_Request ** request)
{
    return MPIDI_PSP_Get_accumulate_generic(origin_addr, origin_count, origin_datatype, result_addr,
                                            result_count, result_datatype, target_rank, target_disp,
                                            target_count, target_datatype, op, win_ptr, request);
}


int MPID_Fetch_and_op(const void *origin_addr, void *result_addr,
                      MPI_Datatype datatype, int target_rank, MPI_Aint target_disp,
                      MPI_Op op, MPIR_Win * win_ptr)
{
    int mpi_error = 0;
    if (unlikely(target_rank == MPI_PROC_NULL)) {
        goto fn_completed;
    }

    if (unlikely(op == MPI_NO_OP)) {
        return MPIDI_PSP_Get_generic(result_addr, 1, datatype,
                                     target_rank, target_disp, 1, datatype, win_ptr, NULL);
    }
#if MPID_PSP_HAVE_PSCOM_RMA_API
    {
        /*A dedicated fetch and op implementation goes here... */
        /* Request-based RMA operations are not supported by fetch and op ! */
        /* The datatype argument must be a predefined datatype */
        MPID_PSP_packed_msg_t msg;
        MPID_Win_rank_info *ri = win_ptr->rank_info + target_rank;
        char *target_buf;

        if (unlikely(target_rank == MPI_PROC_NULL)) {
            goto fn_completed;
        }


        /* Check that we are within an access/exposure epoch: */
        if (win_ptr->epoch_state == MPID_PSP_EPOCH_NONE) {
            mpi_error = MPI_ERR_RMA_SYNC;
            goto err_exit;
        }

        /* Track access epoch state: */
        if (win_ptr->epoch_state == MPID_PSP_EPOCH_FENCE_ISSUED) {
            win_ptr->epoch_state = MPID_PSP_EPOCH_FENCE;
        }

        /* what should we do with #ifdef MPIDI_PSP_WITH_CUDA_AWARENESS */
        mpi_error = MPID_PSP_packed_msg_prepare(origin_addr, 1, datatype, &msg);
        if (unlikely(mpi_error != MPI_SUCCESS))
            goto err_exit;
        /* only pack one */
        MPID_PSP_packed_msg_pack(origin_addr, 1, datatype, &msg);

        target_buf = (char *) ri->base_addr + ri->disp_unit * target_disp;

        /* If the acc is a local operation, do it here */
        /* If the get_acc is a local operation, do it here */
        if (target_rank == win_ptr->rank || win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED) {
            if (target_rank != win_ptr->rank) {
                int disp_unit;
                void *base;

                MPID_PSP_shm_rma_get_base(win_ptr, target_rank, &disp_unit, &base);

                assert(ri->disp_unit == disp_unit);
                target_buf = (char *) base + disp_unit * target_disp;

                /* accumulate may be executed concurrently --> locking required! */
                MPID_PSP_shm_rma_mutex_lock(win_ptr);
                /* local copy target buff back to result addr */
                mpi_error = MPIR_Localcopy(target_buf, 1, datatype, result_addr, 1, datatype);
                if (mpi_error)
                    goto err_exit;
                /* do MPI OP from origin addr to the target buff */
                /* Q: what is TRUE (source_is_packed) used for. here datatype is predefined, can we use FALSE here? */
                MPIDI_PSP_compute_acc_op(msg.msg, 1, datatype, target_buf, 1, datatype, op, TRUE);
                MPID_PSP_shm_rma_mutex_unlock(win_ptr);

            } else {
                /* This is a local acc, but do locking just in SHMEM case! */
                if (unlikely(win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED)) {
                    /* in case of a COMM_SELF clone, mutex_lock()/unlock() will just act as no-ops: */
                    MPID_PSP_shm_rma_mutex_lock(win_ptr);
                    mpi_error = MPIR_Localcopy(target_buf, 1, datatype, result_addr, 1, datatype);
                    if (mpi_error)
                        goto err_exit;

                    MPIDI_PSP_compute_acc_op(msg.msg, 1,
                                             datatype, target_buf, 1, datatype, op, TRUE);
                    MPID_PSP_shm_rma_mutex_unlock(win_ptr);
                } else {
                    /* this is a local operation on non-shared memory: */
                    mpi_error = MPIR_Localcopy(target_buf, 1, datatype, result_addr, 1, datatype);
                    if (mpi_error)
                        goto err_exit;

                    MPIDI_PSP_compute_acc_op(msg.msg, 1,
                                             datatype, target_buf, 1, datatype, op, TRUE);
                }
            }

            MPID_PSP_packed_msg_cleanup(&msg);

            goto fn_completed;
        }

        /* Contig message. Use pscom buildin rma */
        pscom_request_t *req = pscom_request_create(sizeof(pscom_xheader_rma_fetch_op_t),
                                                    sizeof(MPIDI_PSP_PSCOM_Request_rma_fetch_op_t));
        /* essential data required by pscom */
        req->rma.origin_addr = msg.msg;
        req->rma.target_addr = target_buf;
        req->rma.result_addr = result_addr;
        req->rma.rkey = win_ptr->pscom_rkey[target_rank];
        req->data_len = msg.msg_sz;
        req->connection = ri->con;
        req->ops.io_done = MPIDI_PSP_rma_fetch_op_origin_cb;

        /* user-defined extended headers */
        req->xheader.rma_fetch_op.user.remote_win = (void *) ri->win_ptr;
        req->xheader.rma_fetch_op.user.source_rank = win_ptr->rank;
        req->xheader.rma_fetch_op.user.datatype = datatype;
        req->xheader.rma_fetch_op.user.mpi_op = op;     /* ToDo: check: is op a buildin op? */

        /* information for the callback at origin side */
        MPIDI_PSP_PSCOM_Request_rma_fetch_op_t *pscom_rma_user = &req->user->pscom_fetch_op;
        pscom_rma_user->win_id = (void *) win_ptr;
        pscom_rma_user->target_rank = target_rank;

        pscom_post_rma_fetch_op(req);

        win_ptr->rma_local_pending_cnt++;
        win_ptr->rma_local_pending_rank[target_rank]++;
        win_ptr->rma_puts_accs[target_rank]++;
    }
#else
    {   /* This implementation is just based on Get&Accumulate */

        return MPID_Get_accumulate(origin_addr, 1, datatype, result_addr, 1, datatype, target_rank,
                                   target_disp, 1, datatype, op, win_ptr);
    }
#endif

  fn_completed:
    return MPI_SUCCESS;
  err_exit:
    return mpi_error;
}

int MPID_Compare_and_swap(const void *origin_addr, const void *compare_addr,
                          void *result_addr, MPI_Datatype datatype, int target_rank,
                          MPI_Aint target_disp, MPIR_Win * win_ptr)
{
    int mpi_error = 0;

    if (unlikely(target_rank == MPI_PROC_NULL)) {
        goto fn_completed;
    }
#if MPID_PSP_HAVE_PSCOM_RMA_API
    {
        /* A dedicated compare and swap implementation goes here... */
        /* Request-based RMA operations are not supported by compare and swap ! */
        /* The datatype argument must be a predefined datatype */
        MPID_PSP_packed_msg_t msg;
        MPID_Win_rank_info *ri = win_ptr->rank_info + target_rank;
        char *target_buf;

        /* Check that we are within an access/exposure epoch: */
        if (win_ptr->epoch_state == MPID_PSP_EPOCH_NONE) {
            mpi_error = MPI_ERR_RMA_SYNC;
            goto err_exit;
        }

        /* Track access epoch state: */
        if (win_ptr->epoch_state == MPID_PSP_EPOCH_FENCE_ISSUED) {
            win_ptr->epoch_state = MPID_PSP_EPOCH_FENCE;
        }

        /* comment: what should we do with #ifdef MPIDI_PSP_WITH_CUDA_AWARENESS */

        mpi_error = MPID_PSP_packed_msg_prepare(origin_addr, 1, datatype, &msg);
        if (unlikely(mpi_error != MPI_SUCCESS))
            goto err_exit;
        MPID_PSP_packed_msg_pack(origin_addr, 1, datatype, &msg);

        target_buf = (char *) ri->base_addr + ri->disp_unit * target_disp;

        /* If the acc is a local operation, do it here */
        if (target_rank == win_ptr->rank || win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED) {
            if (target_rank != win_ptr->rank) {
                int disp_unit;
                void *base;

                MPID_PSP_shm_rma_get_base(win_ptr, target_rank, &disp_unit, &base);

                assert(ri->disp_unit == disp_unit);
                target_buf = (char *) base + disp_unit * target_disp;

                /* accumulate may be executed concurrently --> locking required! */
                MPID_PSP_shm_rma_mutex_lock(win_ptr);
                /* local copy target buff back to result addr */
                mpi_error = MPIR_Localcopy(target_buf, 1, datatype, result_addr, 1, datatype);
                if (mpi_error)
                    goto err_exit;
                if (MPIR_Compare_equal(compare_addr, result_addr, datatype)) {
                    mpi_error = MPIR_Localcopy(origin_addr, 1, datatype, target_buf, 1, datatype);
                    if (mpi_error)
                        goto err_exit;
                }

                MPID_PSP_shm_rma_mutex_unlock(win_ptr);

            } else {
                /* This is a local acc, but do locking just in SHMEM case! */
                if (unlikely(win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED)) {
                    /* in case of a COMM_SELF clone, mutex_lock()/unlock() will just act as no-ops: */
                    MPID_PSP_shm_rma_mutex_lock(win_ptr);
                    mpi_error = MPIR_Localcopy(target_buf, 1, datatype, result_addr, 1, datatype);
                    if (mpi_error)
                        goto err_exit;
                    if (MPIR_Compare_equal(compare_addr, result_addr, datatype)) {
                        mpi_error = MPIR_Localcopy(origin_addr, 1, datatype,
                                                   target_buf, 1, datatype);
                        if (mpi_error)
                            goto err_exit;
                    }

                    MPID_PSP_shm_rma_mutex_unlock(win_ptr);
                } else {
                    /* this is a local operation on non-shared memory: */
                    mpi_error = MPIR_Localcopy(target_buf, 1, datatype, result_addr, 1, datatype);
                    if (mpi_error)
                        goto err_exit;
                    if (MPIR_Compare_equal(compare_addr, result_addr, datatype)) {
                        mpi_error = MPIR_Localcopy(origin_addr, 1, datatype,
                                                   target_buf, 1, datatype);
                        if (mpi_error)
                            goto err_exit;
                    }
                }
            }

            MPID_PSP_packed_msg_cleanup(&msg);

            goto fn_completed;
        }

        /* Contig message. Use pscom buildin rma */
        pscom_request_t *req = pscom_request_create(sizeof(pscom_xheader_rma_compare_swap_t),
                                                    sizeof
                                                    (MPIDI_PSP_PSCOM_Request_rma_compare_swap_t));
        /* essential data required by pscom */
        req->rma.origin_addr = msg.msg;
        req->rma.target_addr = target_buf;
        req->rma.result_addr = result_addr;
        req->rma.compare_addr = compare_addr;
        req->rma.rkey = win_ptr->pscom_rkey[target_rank];
        req->data_len = msg.msg_sz;
        req->connection = ri->con;
        req->ops.io_done = MPIDI_PSP_rma_comp_swap_origin_cb;

        /* user-defined extended headers */
        req->xheader.rma_compare_swap.user.remote_win = (void *) ri->win_ptr;
        req->xheader.rma_compare_swap.user.source_rank = win_ptr->rank;

        /* information for the callback at origin side */
        MPIDI_PSP_PSCOM_Request_rma_compare_swap_t *pscom_rma_user = &req->user->pscom_compare_swap;
        pscom_rma_user->win_id = (void *) win_ptr;
        pscom_rma_user->target_rank = target_rank;

        pscom_post_rma_compare_swap(req);

        win_ptr->rma_local_pending_cnt++;
        win_ptr->rma_local_pending_rank[target_rank]++;
        win_ptr->rma_puts_accs[target_rank]++;

    }
#else
    {   /* This implementation is just based on Get (plus some additional internal locking): */
#ifdef MPIDI_PSP_WITH_CUDA_AWARENESS
        int result_addr_is_gpu_mem, compare_addr_is_gpu_mem;
        size_t data_sz = 0;
#endif
        void *result_addr_tmp = result_addr;
        void *compare_addr_tmp = (void *) compare_addr;

        MPIDI_PSP_Win_lock_internal(target_rank, win_ptr);

#ifdef MPIDI_PSP_WITH_CUDA_AWARENESS
        /* check whether we need to stage the buffers */
        result_addr_is_gpu_mem = pscom_is_gpu_mem(result_addr);
        compare_addr_is_gpu_mem = pscom_is_gpu_mem(compare_addr);
        if (result_addr_is_gpu_mem || compare_addr_is_gpu_mem) {

            MPIDI_PSP_Datatype_check_size(datatype, 1, data_sz);

            if (result_addr_is_gpu_mem) {
                result_addr_tmp = MPL_malloc(data_sz, MPL_MEM_RMA);
                MPIR_Assert(result_addr_tmp);
                MPIR_Localcopy(result_addr, 1, datatype, result_addr_tmp, 1, datatype);
            }
            if (compare_addr_is_gpu_mem) {
                compare_addr_tmp = MPL_malloc(data_sz, MPL_MEM_RMA);
                MPIR_Assert(compare_addr_tmp);
                MPIR_Localcopy(compare_addr, 1, datatype, compare_addr_tmp, 1, datatype);
            }
        }
#endif

        MPID_Get(result_addr_tmp, 1, datatype, target_rank, target_disp, 1, datatype, win_ptr);

        MPIDI_PSP_Win_wait_local_completion(target_rank, win_ptr);

        if (MPIR_Compare_equal(compare_addr_tmp, result_addr_tmp, datatype)) {

            MPID_Put((void *) origin_addr, 1, datatype, target_rank, target_disp, 1, datatype,
                     win_ptr);
        }

        MPIDI_PSP_Win_unlock_internal(target_rank, win_ptr);

#ifdef MPIDI_PSP_WITH_CUDA_AWARENESS
        /* did we stage any buffers? */
        if (result_addr_is_gpu_mem) {
            MPID_Memcpy(result_addr, result_addr_tmp, data_sz);
            MPL_free(result_addr_tmp);
        }
        if (compare_addr_is_gpu_mem) {
            MPL_free(compare_addr_tmp);
        }
#endif
    }
#endif

  fn_completed:
    return MPI_SUCCESS;
    /* --- */
  err_exit:
    return mpi_error;
}
