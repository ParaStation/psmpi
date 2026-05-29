/*
 * ParaStation
 *
 * Copyright (C) 2022-2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <dlfcn.h>
#include "mpidimpl.h"
#include "mpid_psp_request.h"
#include "mpid_psp_datatype.h"
#include "mpid_psp_packed_msg.h"

#define MPIDI_PSP_PART_REQ_USES_COMPRESSOR(_preq)                       \
    (_preq->compr_req && _preq->compr_req->compressor &&                \
     _preq->compr_req->compressor->inflate_fn &&                        \
     (_preq->compr_req->compressor->inflate_fn !=                       \
      MPIX_COMPRESSOR_CONVERSION_FN_NULL))

/**
 * @brief Check if a partitioned request matches to all given parameters rank, tag and context_id.
 *
 * @param rank rank to match
 * @param tag tag to match
 * @param context_id context ID to match
 * @param req pointer to the partitioned request that shall be matched
 *
 * @return bool true if match is successful, false otherwise
 */
static
bool partitioned_requests_do_match(int rank, int tag, MPIR_Context_id_t context_id,
                                   MPIR_Request * req)
{
    struct MPID_DEV_Request_partitioned *preq = &req->dev.kind.partitioned;

    /* compare (source) rank, message tag and context id */
    return (preq->rank == rank) && (preq->tag == tag) && (preq->context_id == context_id);
}

/**
 * @brief Find a request in a list and remove the request from the list if found.
 *
 * @note Thread safety: This function has to be used from within a lock.
 *
 * @param rank rank to match
 * @param tag tag to match
 * @param context_id context ID to match
 * @param queue pointer to head of list to search in
 *
 * @return MPIR_Request* Returns pointer to the found partitioned request or NULL
 */
static
MPIR_Request *match_and_deq_request(int rank, int tag, MPIR_Context_id_t context_id,
                                    struct list_head *queue)
{
    struct list_head *pos;

    list_for_each(pos, queue) {
        MPIR_Request *r = list_entry(pos, MPIR_Request, dev.kind.partitioned.next);

        if (partitioned_requests_do_match(rank, tag, context_id, r)) {
            /* remove the request from the list and return it */
            list_del(&r->dev.kind.partitioned.next);
            return r;
        }
    }
    return NULL;
}

/**
 * @brief Set the status and check for errors in a matched partitioned request.
 *
 * @note Thread safety: This function has to be used from within a lock.
 *
 * @param req pointer to matched partitioned request
 */
static
void MPID_PSP_part_request_matched(MPIR_Request * req)
{

    struct MPID_DEV_Request_partitioned *preq = &(req->dev.kind.partitioned);
    MPI_Aint sdata_size = preq->sdata_size;

    /* set status for partitioned req */
    MPIR_STATUS_SET_COUNT(req->status, sdata_size);
    req->status.MPI_SOURCE = preq->rank;
    req->status.MPI_TAG = preq->tag;
    req->status.MPI_ERROR = MPI_SUCCESS;

    /* additional check for partitioned pt2pt: require identical buffer size */
    if (req->status.MPI_ERROR == MPI_SUCCESS) {
        MPI_Aint rdata_size;
        MPIR_Datatype_get_size_macro(preq->datatype, rdata_size);
        rdata_size *= preq->count * preq->partitions;
        if (sdata_size != rdata_size) {
            req->status.MPI_ERROR = MPI_ERR_OTHER;
        }
    }
}

/**
 * @brief Call Irecv with sub requests to issue the data receive for partitioned request
 *        and activate completion notification of sub requests.
 *
 * @note Thread safety: This function has to be used from within a lock.
 *
 * @param req Pointer to partitioned request for which data receive stall be issued
 * @return int MPI_SUCCESS on success
 *             MPI error code of Irecv on failure
 *             MPI_ERR_ARG if this function is not called for a partitioned recv request
 */
static
int MPID_part_issue_data_recv(MPIR_Request * req)
{
    struct MPID_DEV_Request_partitioned *preq;
    MPI_Aint elements;
    int mpi_errno = MPI_SUCCESS;

    MPIR_ERR_CHKANDJUMP(req->kind != MPIR_REQUEST_KIND__PART_RECV, mpi_errno, MPI_ERR_ARG, "**arg");

    preq = &(req->dev.kind.partitioned);
    elements = preq->count * preq->part_per_req;
    /* check potential overflow */
    MPIR_Assert(elements < MPIR_AINT_MAX);

    /* call irecv for actual data transfer, one irecv for each partitioned request */
    for (MPI_Aint i = 0; i < preq->requests; i++) {
        MPIR_Request *new_req;
        MPI_Aint offset = i * elements;
        MPI_Aint base;
        MPI_Aint dtype_size = 0;
        MPI_Aint part_buf;
        MPI_Aint count;
        MPI_Datatype dtype;
        void *buffer;

        /* last request could be smaller, take the rest */
        if (i == preq->requests - 1) {
            elements = (preq->count * preq->partitions) - (i * elements);
        }

        MPI_Get_address(preq->buf, &base);
        MPIR_Datatype_get_size_macro(preq->datatype, dtype_size);
        part_buf = MPI_Aint_add(base, dtype_size * offset);

        /* TODO: rely on pscom interface (+xheader) instead of using MPI tags */
        int msg_tag = preq->tag + i + 1;
        MPIR_Assert(msg_tag <= INT_MAX);

        count = elements;
        dtype = preq->datatype;
        buffer = (void *) part_buf;

        if (MPIDI_PSP_PART_REQ_USES_COMPRESSOR(preq)) {
            /* prepare the receiving in the temporary buffer at the right position for this partition */
            MPI_Get_address(preq->compr_req->compr_buffer, &base);
            MPI_Aint part_buf_compr = MPI_Aint_add(base, i * preq->compr_req->compr_part_size);

            /* temporarily adjust `count` and `dtype` for receiving the compressed message */
            count = preq->compr_req->compr_part_size;
            dtype = MPIX_COMPRESSED;
            buffer = (void *) part_buf_compr;
        }

        mpi_errno =
            MPID_Irecv(buffer, count, dtype, preq->rank, msg_tag, req->comm, preq->context_offset,
                       &new_req);
        MPIR_ERR_CHECK(mpi_errno);

        if (MPIDI_PSP_PART_REQ_USES_COMPRESSOR(preq)) {

            struct MPID_DEV_Request_recv *rreq = &new_req->dev.kind.recv;

            /* The assumption is that here the request has been posted but not been completed yet. */
            MPIR_Assert((rreq->common.pscom_req->state & PSCOM_REQ_STATE_POSTED) &&
                        !(rreq->common.pscom_req->state & PSCOM_REQ_STATE_IO_STARTED) &&
                        !(rreq->common.pscom_req->state & PSCOM_REQ_STATE_IO_DONE));

            /* create the compressor-related part of th request */
            rreq->compr_req =
                MPL_calloc(1, sizeof(struct MPIR_Compressor_request_recv), MPL_MEM_OTHER);

            rreq->compr_req->compressor = preq->compr_req->compressor;
            rreq->compr_req->partition = i;
            rreq->compr_req->count = elements;
            rreq->compr_req->datatype = preq->datatype;
            rreq->compr_req->user_buf_ptr = (void *) part_buf;
            rreq->compr_req->compr_buf_ptr = buffer;
            rreq->compr_req->extra_req_state = preq->compr_req->extra_req_state;
        }

        /*
         * Set the completion notification of the new subrequest to the completion counter of
         * the overall partitioned request once the subrequest completes, the completion
         * counter of the partitioned request req gets decremented.
         */
        struct MPID_DEV_Request_common *new_dev_req = &new_req->dev.kind.common;
        new_dev_req->completion_notification = &(req->cc);

        /* TODO: Keep track of sub requests to enable checks per partition in parrived */
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/**
 * @brief Call Isend for a sub-request to issue the data send for partitioned request
 *        and activate completion notification of sub request
 *
 * @note Thread safety: This function has to be used from within a lock.
 *
 * @param req Pointer to partitioned request for which data receive stall be issued
 * @param req_idx index of the send sub-request
 * @return int MPI_SUCCESS on success
 *             MPI error code of Isend on failure
 *             MPI_ERR_ARG if this function is not called for partitioned send request
 */
static
int MPID_part_issue_data_send(MPIR_Request * req, int req_idx)
{
    struct MPID_DEV_Request_partitioned *preq;
    MPI_Aint elements;
    MPIR_Request *new_req;
    MPI_Aint base;
    MPI_Aint dtype_size = 0;
    MPI_Aint part_buf;
    MPI_Aint count;
    MPI_Datatype dtype;
    void *buffer;
    int mpi_errno = MPI_SUCCESS;

    MPIR_ERR_CHKANDJUMP(req->kind != MPIR_REQUEST_KIND__PART_SEND, mpi_errno, MPI_ERR_ARG, "**arg");

    preq = &(req->dev.kind.partitioned);
    elements = preq->count * preq->part_per_req;
    /* check potential overflow */
    MPIR_Assert(elements < MPIR_AINT_MAX);

    /* last request could be smaller, take the rest */
    if (req_idx == preq->requests - 1) {
        elements = (preq->count * preq->partitions) - (req_idx * elements);
    }

    MPI_Get_address(preq->buf, &base);
    MPIR_Datatype_get_size_macro(preq->datatype, dtype_size);
    part_buf = MPI_Aint_add(base, req_idx * elements * dtype_size);

    /* TODO: rely on pscom interface (+xheader) instead of using MPI tags */
    int msg_tag = preq->tag + req_idx + 1;
    MPIR_Assert(msg_tag <= INT_MAX);

    count = elements;
    dtype = preq->datatype;
    buffer = (void *) part_buf;

    if (MPIDI_PSP_PART_REQ_USES_COMPRESSOR(preq)) {
        MPIR_Assert(preq->compr_req->compr_buffer);
        MPI_Get_address(preq->compr_req->compr_buffer, &base);
        MPI_Aint compr_part_addr = MPI_Aint_add(base, req_idx * preq->compr_req->compr_part_size);

        /* INPUT is the size of the partition on user side */
        MPI_Aint size = dtype_size * count;

        int retval =
            preq->compr_req->compressor->deflate_fn((void *) part_buf, req_idx, elements, dtype,
                                                    (void *) compr_part_addr, &size,
                                                    preq->compr_req->extra_req_state);

        if (retval != MPI_SUCCESS) {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
                                             MPIR_ERR_FATAL,
                                             __func__, __LINE__,
                                             MPI_ERR_OTHER,
                                             "**compressorfailed", "**compressorfailed %s",
                                             preq->compr_req->compressor->name);
            goto fn_fail;
        }

        /* OUTPUT is the size of the compressed buffer to send */
        MPIR_Assert(size <= preq->compr_req->compr_part_size);

        count = size;
        dtype = MPIX_COMPRESSED;
        buffer = (void *) compr_part_addr;
    }

    mpi_errno =
        MPID_Isend(buffer, count, dtype, preq->rank, msg_tag, req->comm, preq->context_offset,
                   &new_req);
    MPIR_ERR_CHECK(mpi_errno);

    preq->send_ctr++;

    /*
     * Once isend for last partition is submitted:
     * reset all send side status variables and prepare for clean up or new MPI_Start.
     */
    if (preq->send_ctr == preq->requests) {
        /*
         * free memory for partition status tracking
         * (will be (re-)allocated in next call to MPI_Start)
         */
        MPL_free(preq->part_ready);
        preq->part_ready = NULL;

        /*
         * reset peer request, next data transmission can only start after next CTS message was
         * received (see CTS callback function)
         */
        preq->peer_request = NULL;

        /* reset send counter */
        preq->send_ctr = 0;
    }

    /*
     * set the completion notification of the subrequest to the completion counter of the
     * overall partitioned request once the subrequest completes, the completion counter of
     * the partitioned request req gets decremented.
     */
    struct MPID_DEV_Request_common *new_dev_req = &new_req->dev.kind.common;
    new_dev_req->completion_notification = &(req->cc);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/**
 * @brief Send a sub-request if all partitions that belong to it are marked ready.
 *
 * @note Thread safety: This function has to be used from within a lock.
 *
 * @param sreq pointer to partitioned send request
 * @param req_idx index of send sub-request
 *
 * @return int MPI_SUCESS if either sending is not yet possible or data transmission was issued
 *             successfully, or MPI error code of the data transmission issuing if this fails.
 */
static
int MPID_part_send_if_ready(MPIR_Request * sreq, int req_idx)
{
    int mpi_error = MPI_SUCCESS;
    bool req_ready = true;
    struct MPID_DEV_Request_partitioned *preq;
    preq = &(sreq->dev.kind.partitioned);

    /* CTS not received? sending not yet possible (not an error!) */
    if (!preq->peer_request) {
        goto fn_exit;
    }

    for (int i = 0; i < preq->part_per_req; i++) {
        int base_partition = req_idx * preq->part_per_req;
        if (base_partition + i < preq->partitions) {
            req_ready = req_ready && preq->part_ready[base_partition + i];
        }
    }

    if (req_ready) {
        /* all partitions ready AND CTS received: issue data transmission */
        mpi_error = MPID_part_issue_data_send(sreq, req_idx);
        MPIR_ERR_CHECK(mpi_error);
    }

  fn_exit:
    return mpi_error;
  fn_fail:
    goto fn_exit;
}

/**
 * @brief Send sub-requests if a partition is completely ready.
 *        Checks if send sub-request of one partition is completely ready
 *        to be sent and if yes, tries to issue the send sub-request.
 *
 * @note Thread safety: This function has to be used from within a lock.
 *
 * @param sreq pointer to partitioned send request
 * @param part partition to be checked
 * @return int MPI error code of transmission issuing
 */
static
int MPID_part_check_data_transmission(MPIR_Request * sreq, int part)
{
    int mpi_error = MPI_SUCCESS;
    struct MPID_DEV_Request_partitioned *preq;
    preq = &(sreq->dev.kind.partitioned);

    if (part >= 0) {
        int req_idx = 0;

        /* check if all partitions that belong to the request are ready */
        req_idx = part / preq->part_per_req;    // integer division!!
        mpi_error = MPID_part_send_if_ready(sreq, req_idx);
        MPIR_ERR_CHECK(mpi_error);
    } else {
        /* check for all requests and send ready requests */
        for (int i = 0; i < preq->requests; i++) {
            mpi_error = MPID_part_send_if_ready(sreq, i);
            MPIR_ERR_CHECK(mpi_error);
        }
    }

  fn_exit:
    return mpi_error;
  fn_fail:
    goto fn_exit;
}

/**
 * @brief Check for the correct number of sub-requests.
 *        Checks if partitioned request has certain number of sub-requests
 *        and if not adapts partitioned request accordingly.
 *
 * @note Thread safety: This function has to be used from within a lock.
 *
 * @param preq Pointer to partitioned request
 * @param num_peer_requests number of peer requests to compare to
 */
static
void MPID_part_check_num_requests(struct MPIR_Request *req, int num_peer_requests)
{
    struct MPID_DEV_Request_partitioned *preq = &req->dev.kind.partitioned;

    if (preq->requests != num_peer_requests) {
        /* number of requests not equal, take minimum */
        int min_requests = MIN(preq->requests, num_peer_requests);
        preq->requests = min_requests;

        /* recompute partitions per request accordingly */
        preq->part_per_req = preq->partitions / preq->requests;
        if (preq->partitions % preq->requests > 0) {
            /* if division has a remainder, take one more element per partition to fit */
            preq->part_per_req++;
        }
    }
}

/**
 * @brief Init message callback, called by receiver when send init msg is received.
 *
 * @param request pscom request
 */
void MPID_do_recv_part_send_init(pscom_request_t * request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_PSP_PSCOM_Xheader_part_t *xheader_part = &(request->xheader.user.part);

    /* match request from global posted partitioned receive requests queue */
    MPIR_Request *posted_req = match_and_deq_request(xheader_part->common.src_rank,
                                                     xheader_part->common.tag,
                                                     xheader_part->common.context_id,
                                                     &(MPIDI_Process.part_posted_list));
    if (posted_req) {
        struct MPID_DEV_Request_partitioned *preq = &(posted_req->dev.kind.partitioned);

        /* cancel the matched send init pscom receive request */
        pscom_cancel(preq->pscom_recv_req);
        preq->pscom_recv_req = NULL;

        /* copy infos from header into partitioned receive request */
        preq->sdata_size = xheader_part->sdata_size;
        preq->peer_request = xheader_part->sreq_ptr;

        /* check if peer request has same number of requests */
        MPID_part_check_num_requests(posted_req, xheader_part->requests);

        MPID_PSP_part_request_matched(posted_req);

        if (MPIR_Part_request_is_active(posted_req)) {

            /* set completion counter */
            MPIR_cc_set(posted_req->cc_ptr, preq->requests);

            /* if match successful AND MPI_Start called: send CTS message */
            MPIDI_PSP_SendPartitionedCtrl(preq->tag,
                                          posted_req->comm->context_id,
                                          posted_req->comm->rank,
                                          MPID_PSCOM_rank2connection(posted_req->comm, preq->rank),
                                          preq->sdata_size,
                                          preq->requests,
                                          preq->peer_request,
                                          posted_req, MPID_PSP_MSGTYPE_PART_CLEAR_TO_SEND);
            /* issue sub requests for receive */
            mpi_errno = MPID_part_issue_data_recv(posted_req);
        }

        MPIR_ERR_CHKANDJUMP1((mpi_errno != MPI_SUCCESS), mpi_errno, MPI_ERR_OTHER,
                             "**psp|part_sendinit", "**psp|part_sendinit %d", mpi_errno);

        /* release handshake reference */
        MPIR_Request_free_unsafe(posted_req);

    } else {
        /*
         * create temporary request (will be freed if respective receive request is posted on
         * reicever side)
         */
        MPIR_Request *unexp_req = NULL;
        unexp_req = MPIR_Request_create(MPIR_REQUEST_KIND__PART_RECV);

        /* prepare param based on xheader info */
        unexp_req->dev.kind.partitioned.rank = xheader_part->common.src_rank;
        unexp_req->dev.kind.partitioned.tag = xheader_part->common.tag;
        unexp_req->dev.kind.partitioned.context_id = xheader_part->common.context_id;
        unexp_req->dev.kind.partitioned.sdata_size = xheader_part->sdata_size;
        unexp_req->dev.kind.partitioned.peer_request = xheader_part->sreq_ptr;
        unexp_req->dev.kind.partitioned.requests = xheader_part->requests;

        /* enqueue in global unexpected list */
        list_add_tail(&unexp_req->dev.kind.partitioned.next, &(MPIDI_Process.part_unexp_list));
    }

    pscom_request_free(request);

  fn_exit:
    return;
  fn_fail:
    goto fn_exit;
}

/**
 * @brief Clear-to-send message callback, called by sender when clear-to-send message is received.
 *
 * @param request pscom request
 */
void MPID_do_recv_part_cts(pscom_request_t * request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_PSP_PSCOM_Xheader_part_t *xheader_part = &(request->xheader.user.part);
    MPIR_Request *part_sreq = xheader_part->sreq_ptr;
    MPIR_Assert(part_sreq);

    struct MPID_DEV_Request_partitioned *preq = &part_sreq->dev.kind.partitioned;
    preq->peer_request = xheader_part->rreq_ptr;

    /* cancel the CTS pscom receive request */
    pscom_cancel(preq->pscom_recv_req);
    preq->pscom_recv_req = NULL;

    /* check if peer request has same number of requests */
    MPID_part_check_num_requests(part_sreq, xheader_part->requests);

    if (MPIR_Request_is_active(part_sreq)) {

        /* set completion counter */
        MPIR_cc_set(part_sreq->cc_ptr, preq->requests);

        /*
         * If start was called already for the send request
         * check all subrequests and send the ones that are ready
         */
        mpi_errno = MPID_part_check_data_transmission(part_sreq, -1);
    }

    MPIR_ERR_CHKANDSTMT1((mpi_errno != MPI_SUCCESS), mpi_errno, MPI_ERR_OTHER,
                         part_sreq->status.MPI_ERROR = mpi_errno,
                         "**psp|part_cts", "**psp|part_cts %d", mpi_errno);

    pscom_request_free(request);

    return;
}

/**
 * @brief Mark partition as ready (partition must not be marked ready before).
 *
 * @note Thread safety: This function has to be used from within a lock.
 *
 * @param preq pointer to partitioned send request
 * @param partition partition to be marked ready (counting starts at 0)
 *
 * @return int MPI_ERR_OTHER if status array for partitions is not initialized or if the partition
 *             is already marked as ready; MPI_SUCCESS otherwise
 */
static
int MPID_part_set_ready(struct MPID_DEV_Request_partitioned *preq, int partition)
{
    if (!preq->part_ready) {
        /* err, no memory allocated for status array, probably pready was called before MPI_start */
        return MPI_ERR_OTHER;
    }

    if (preq->part_ready[partition]) {
        /* error, partition is already marked as ready */
        return MPI_ERR_OTHER;
    }

    /* mark the partition as ready */
    preq->part_ready[partition] = true;
    return MPI_SUCCESS;
}


/**
 * @brief Determine the number of requests to be sent/received and the number of partitions per request.
 *        This function can be used to optimize the send/recv granularity based on parameters of the
 *        request.
 *
 * @note Thread safety: This function has to be used from within a lock.
 *
 * @note Depending on the requests determined on the peer partitioned request, the settings
 *       computed here may be overwritten in the init callback (receiver side) or CTS callback (sender
 *       side) since both sides have to submit the same number of requests.
 *
 * @param req pointer to a partitioned request
 */
static
void MPID_part_distribute_partitions_to_requests(MPIR_Request * req)
{
    struct MPID_DEV_Request_partitioned *preq = &req->dev.kind.partitioned;

    /* simple mapping: one request per partition */
    preq->requests = preq->partitions;

    preq->part_per_req = preq->partitions / preq->requests;
}

/**
 * @brief Checks whether the given info object contains any relevant information.
 *        (Currently, this can only be the case with regard to a payload compressor.)
 *
 * @param info pointer to info object
 * @param req pointer to a partitioned request
 *
 * @return int MPI_SUCCESS
 */
static
int MPIDI_PSP_part_check_info(MPIR_Info * info, MPIR_Request * req)
{
    int mpi_errno = MPI_SUCCESS;

    if (!info) {
        return MPI_SUCCESS;
    }

    /* Check if the use of a payload compressor is requested. */
    int info_flag = 0;
    char info_compressor_name[MPI_MAX_INFO_VAL + 1];
    MPIR_Info_get_impl(info, compressor_info_key, MPI_MAX_INFO_VAL, info_compressor_name,
                       &info_flag);
    if (info_flag) {

        struct MPID_DEV_Request_partitioned *preq = &req->dev.kind.partitioned;

        if (preq->part_per_req != 1) {
            /* For this prototype implementation, we only support
             * an equal number of requests and partitions. */
            goto fn_exit;
        }

        MPIR_Compressor *compressor_found = NULL;
        MPIR_Compressor_lookup(info_compressor_name, &compressor_found);

        /* If not found, check if a matching compressor plugin can be loaded. */
        if (!compressor_found) {

            if (!MPIDI_Process.env.enable_compressor_plugins) {
                /* Compressor plugins not activated at runtime: Do nothing! */
                goto fn_exit;
            }

            /* First check if a plugin name was given. */
            char info_compressor_plugin[MPI_MAX_INFO_VAL + 1];
            MPIR_Info_get_impl(info, compressor_info_key_plugin, MPI_MAX_INFO_VAL,
                               info_compressor_plugin, &info_flag);
            if (!info_flag) {
                goto fn_exit;
            }

            /* If so, try to load the shared library and call the register function. */
            void *dlhandle;
            char *dlerror_str;
            MPIX_Compressor_register_plugin_function *compressor_register_fn;

            dlerror();
            dlhandle = dlopen(info_compressor_plugin, RTLD_LAZY);
            dlerror_str = dlerror();
            if (!dlhandle || dlerror_str) {
                goto fn_exit;
            }

            dlerror();
            compressor_register_fn = dlsym(dlhandle, compressor_register_plugin_fn);
            dlerror_str = dlerror();
            if (dlerror_str) {
                goto fn_exit;
            }

            mpi_errno = compressor_register_fn(info_compressor_name, info->handle);
            if (mpi_errno != MPI_SUCCESS) {
                /* No error code generation: A failing `compressor_register_fn` is not to be
                 * treated as an MPI error. This just deactivates the compressor use. */
                goto fn_exit;
            }

            /* ...and then retry the lookup.  */
            MPIR_Compressor_lookup(info_compressor_name, &compressor_found);
            if (!compressor_found) {
                goto fn_exit;
            }
        }

        MPI_Aint size = 0;
        void *buffer = NULL;
        int buf_free = 0;

        MPIR_Assertp(compressor_found);

        /* create the compressor-related part of th request */
        preq->compr_req = MPL_calloc(1, sizeof(struct MPIR_Compressor_request_part), MPL_MEM_OTHER);

        if (compressor_found->req_init_fn &&
            (compressor_found->req_init_fn != MPIX_COMPRESSOR_REQ_INIT_FN_NULL)) {

            void *extra_req_state = &preq->compr_req->extra_req_state;
            mpi_errno =
                compressor_found->req_init_fn(preq->buf, &preq->partitions, &preq->count,
                                              &preq->datatype, info->handle, &buffer, &size,
                                              compressor_found->extra_state, extra_req_state);
            if (mpi_errno != MPI_SUCCESS) {
                /* No error code generation: A failing `req_init_fn` is not to be treated as
                 * an MPI error. This just deactivates the compressor use for this request. */
                goto fn_exit;
            }
        }

        if (!size) {
            /* Either no `req_init_fn` given or `req_init_fn` returned `size = 0`. In both cases,
             * we allocate an auxiliary buffer that is as large as the user buffer, since the
             * assumption is that compression only makes things smaller, but not larger. */
            MPI_Aint dtype_size;
            MPIR_Datatype_get_size_macro(preq->datatype, dtype_size);
            size = dtype_size * preq->count * preq->partitions;
        }

        if (!buffer || (buffer == MPI_BUFFER_AUTOMATIC)) {
            /* Either `buffer` has not been set by the compressor or MPI_BUFFER_AUTOMATIC
             * was chosen. In both cases we allocate the buffer on the MPI side. */
            buffer = MPL_malloc(size, MPL_MEM_BUFFER);
            buf_free = 1;
        }

        preq->compr_req->compr_buffer = buffer;
        preq->compr_req->compr_buf_free = buf_free;
        preq->compr_req->compr_part_size = size / preq->partitions;
        preq->compr_req->compressor = compressor_found;
    }

  fn_exit:
    return mpi_errno;
}

/**
 * @brief Common initialization for partitioned communication requests
 *
 * @note Thread safety: This function has to be used from within a lock.
 *
 * @param buf starting address of send/ recv buffer for all partitions
 * @param partitions number of partitions
 * @param count number of elements per partition
 * @param datatype data type of each element
 * @param rank rank of source or destination
 * @param tag message tag
 * @param comm communicator
 * @param info info argument
 * @param request partitioned communication request (output value of this function)
 * @param type type of partitioned request (*__PART_RECV or *__PART_SEND)
 *
 * @return int  MPI_SUCCESS on success
 *              MPI_ERR_NO_MEM if there was a memory allocation problem
 */
static
int MPID_PSP_part_init_common(const void *buf, int partitions, MPI_Count count,
                              MPI_Datatype datatype, int rank, int tag, MPIR_Comm * comm,
                              MPIR_Info * info, MPIR_Request ** request, MPIR_Request_kind_t type)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Request *req;
    struct MPID_DEV_Request_partitioned *preq;

    req = MPIR_Request_create(type);
    if (unlikely(!req)) {
        return MPI_ERR_NO_MEM;
    }
    req->comm = comm;
    MPIR_Comm_add_ref(comm);
    MPIR_Comm_save_inactive_request(comm, req);

    preq = &req->dev.kind.partitioned;

    preq->buf = (void *) buf;
    preq->count = count;
    preq->partitions = partitions;
    preq->datatype = datatype;
    MPID_PSP_Datatype_add_ref(preq->datatype);

    preq->rank = rank;
    preq->tag = tag;
    preq->context_id = comm->context_id;
    preq->context_offset = 0;

    req->u.part.partitions = partitions;
    MPIR_Part_request_inactivate(req);

    /*
     * Inactive partitioned comm request can be freed by request_free.
     * Completion cntr increases when request becomes active at start.
     */
    MPIDI_PSP_Request_set_completed(req);

    preq->peer_request = NULL;
    preq->part_ready = NULL;
    preq->first_use = 1;
    req->dev.kind.partitioned.send_ctr = 0;

    /* compute and save initial settings for partitioned communication */
    MPID_part_distribute_partitions_to_requests(req);

    mpi_errno = MPIDI_PSP_part_check_info(info, req);
    MPIR_ERR_CHECK(mpi_errno);

    *request = req;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/**
 * @brief Start a partitioned send request.
 *
 * @param req pointer to partitioned send request to be started
 *
 * @return int MPI_SUCCESS on success and MPI_ERR_ARG if req is not a partitioned send request
 */
int MPID_PSP_psend_start(MPIR_Request * req)
{
    struct MPID_DEV_Request_partitioned *preq;
    int mpi_errno = MPI_SUCCESS;

    MPIR_ERR_CHKANDJUMP(req->kind != MPIR_REQUEST_KIND__PART_SEND, mpi_errno, MPI_ERR_ARG, "**arg");

    preq = &req->dev.kind.partitioned;

    /* init status of partitions */
    preq->part_ready = (bool *) MPL_malloc(sizeof(bool) * preq->partitions, MPL_MEM_OTHER);
    for (int i = 0; i < preq->partitions; i++) {
        preq->part_ready[i] = false;    // all partitions' status not ready
    }

    /* init send counter */
    preq->send_ctr = 0;

    if (!preq->first_use) {
        /*
         * If this is not the first time that start is called for this send request we need a new
         * recv for a CTS.
         */
        mpi_errno = MPIDI_PSP_RecvPartitionedCtrl(preq->tag, req->comm->context_id, preq->rank,
                                                  MPID_PSCOM_rank2connection(req->comm, preq->rank),
                                                  MPID_PSP_MSGTYPE_PART_CLEAR_TO_SEND, req);
        MPIR_ERR_CHECK(mpi_errno);
    } else {
        preq->first_use = 0;
    }

    /* activate request */
    MPIR_Part_request_activate(req);

    /* indicate data transfer starts, set completion counter to number of partitioned requests */
    MPIR_cc_set(req->cc_ptr, preq->requests);

    if (preq->peer_request) {
        /* CTS already received, start send for ready partitions */
        mpi_errno = MPID_part_check_data_transmission(req, -1);
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/**
 * @brief Start a partitioned receive request.
 *
 * @param req pointer to partitioned receive request to be started
 *
 * @return int  MPI_SUCCESS on success
 *              MPI_ERR_ARG if req is not a partitioned recv request
 *              MPI error code of data transmission issuing
 */
int MPID_PSP_precv_start(MPIR_Request * req)
{
    int mpi_errno = MPI_SUCCESS;
    struct MPID_DEV_Request_partitioned *preq;

    MPIR_ERR_CHKANDJUMP(req->kind != MPIR_REQUEST_KIND__PART_RECV, mpi_errno, MPI_ERR_ARG, "**arg");

    preq = &req->dev.kind.partitioned;

    /* activate request */
    MPIR_Part_request_activate(req);

    /* indicate data transfer starts, set completion counter to number of partitioned requests */
    MPIR_cc_set(req->cc_ptr, preq->requests);

    if (preq->peer_request) {
        /*
         * If init request is completed for this partitioned receive request
         * (= SEND_INIT received and matched!)
         * send clear to send message and post irecv request.
         */
        MPIDI_PSP_SendPartitionedCtrl(preq->tag,
                                      req->comm->context_id,
                                      req->comm->rank,
                                      MPID_PSCOM_rank2connection(req->comm, preq->rank),
                                      preq->sdata_size,
                                      preq->requests,
                                      preq->peer_request, req, MPID_PSP_MSGTYPE_PART_CLEAR_TO_SEND);

        mpi_errno = MPID_part_issue_data_recv(req);
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/**
 * @brief Initialize a partitioned send request.
 *
 * @param buf starting address of the send data
 * @param partitions number of partitions
 * @param count number of elements per partition
 * @param datatype data type of each element
 * @param dest destination rank
 * @param tag message tag
 * @param comm communicator
 * @param info info object
 * @param request partitioned send request (output of this function)
 *
 * @return int  MPI_SUCCESS on success
 *              MPI_ERR_NO_MEM if there was a memory allocation problem
 *              MPI_ERR_INTERN if there was any other error creating the request
 */
int MPID_Psend_init(const void *buf, int partitions, MPI_Count count, MPI_Datatype datatype,
                    int dest, int tag, MPIR_Comm * comm, MPIR_Info * info, MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    struct MPID_DEV_Request_partitioned *preq;
    MPI_Aint dtype_size = 0;

    /* common inits */
    mpi_errno = MPID_PSP_part_init_common(buf, partitions, count,
                                          datatype, dest, tag,
                                          comm, info, request, MPIR_REQUEST_KIND__PART_SEND);
    MPIR_ERR_CHECK(mpi_errno);

    /* init send data size */
    preq = &((*request)->dev.kind.partitioned);
    MPIR_Datatype_get_size_macro(datatype, dtype_size);
    /* count is per partition */
    preq->sdata_size = dtype_size * count * partitions;

    /* post recv request for CTS (is redone in start function as of 2nd use of this request) */
    mpi_errno = MPIDI_PSP_RecvPartitionedCtrl(preq->tag,
                                              (*request)->comm->context_id,
                                              preq->rank,
                                              MPID_PSCOM_rank2connection((*request)->comm,
                                                                         preq->rank),
                                              MPID_PSP_MSGTYPE_PART_CLEAR_TO_SEND, *request);
    MPIR_ERR_CHECK(mpi_errno);

    /*
     * send msg of type MPID_PSP_MSGTYPE_PART_SEND_INIT
     *
     * NOTE: receive request unknown at this point
     */
    MPIDI_PSP_SendPartitionedCtrl(preq->tag, preq->context_id, (*request)->comm->rank,
                                  MPID_PSCOM_rank2connection((*request)->comm, preq->rank),
                                  preq->sdata_size, preq->requests, (*request), NULL,
                                  MPID_PSP_MSGTYPE_PART_SEND_INIT);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/**
 * @brief Initialize a partitioned receive request.
 *
 * @param buf starting address of the receive data
 * @param partitions number of partitions
 * @param count number of elements per partition
 * @param datatype data type of each element
 * @param source source rank
 * @param tag message tag
 * @param comm communicator
 * @param info info object
 * @param request partitioned receive request (output of this function)
 *
 * @return int  MPI_SUCCESS on success
 *              MPI_ERR_NO_MEM if there was a memory allocation problem
 *              MPI_ERR_INTERN if there was any other error creating the request
 */
int MPID_Precv_init(void *buf, int partitions, MPI_Count count, MPI_Datatype datatype,
                    int source, int tag, MPIR_Comm * comm, MPIR_Info * info,
                    MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    struct MPID_DEV_Request_partitioned *preq;
    MPIR_Request *unexp_req = NULL;

    /* common inits */
    mpi_errno = MPID_PSP_part_init_common(buf, partitions, count,
                                          datatype, source, tag,
                                          comm, info, request, MPIR_REQUEST_KIND__PART_RECV);
    MPIR_ERR_CHECK(mpi_errno);

    /* post receive request for the send init message */
    preq = &((*request)->dev.kind.partitioned);
    mpi_errno = MPIDI_PSP_RecvPartitionedCtrl(preq->tag,
                                              preq->context_id,
                                              preq->rank,
                                              MPID_PSCOM_rank2connection((*request)->comm,
                                                                         preq->rank),
                                              MPID_PSP_MSGTYPE_PART_SEND_INIT, *request);
    MPIR_ERR_CHECK(mpi_errno);

    /*
     * try matching this recv request to unexpected SEND_INIT request from the global
     * partitioned unexpected list
     */
    unexp_req = match_and_deq_request(preq->rank,
                                      preq->tag,
                                      preq->context_id, &(MPIDI_Process.part_unexp_list));

    if (unexp_req) {
        /* cancel the matched send init pscom receive request */
        pscom_cancel(preq->pscom_recv_req);
        preq->pscom_recv_req = NULL;

        /* copy sender info from unexp_req to local part_rreq */
        preq->sdata_size = unexp_req->dev.kind.partitioned.sdata_size;
        preq->peer_request = unexp_req->dev.kind.partitioned.peer_request;

        /* check if peer request has same number of requests */
        MPID_part_check_num_requests((*request), unexp_req->dev.kind.partitioned.requests);

        /* free memory of dequeued unexpected request */
        MPIR_Request_free(unexp_req);

        MPID_PSP_part_request_matched((*request));
    } else {
        /* enqueue new partitioned recv request to global partitioned posted recv requests list */
        list_add_tail(&((*request)->dev.kind.partitioned.next), &(MPIDI_Process.part_posted_list));
        MPIR_Request_add_ref((*request));
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/**
 * @brief Mark a range of partitions as ready to be sent
 *
 * @param partition_low lower bound of partition range
 * @param partition_high upper bound of partition range
 * @param sreq pointer to partitioned send request
 *
 * @return int see MPID_part_set_ready(...) and MPID_part_check_data_transmission(...)
 */
int MPID_Pready_range(int partition_low, int partition_high, MPIR_Request * sreq)
{
    struct MPID_DEV_Request_partitioned *preq;
    preq = &(sreq->dev.kind.partitioned);

    int mpi_error = MPI_SUCCESS;

    for (int part = partition_low; part <= partition_high; part++) {
        mpi_error = MPID_part_set_ready(preq, part);
        MPIR_ERR_CHECK(mpi_error);

        mpi_error = MPID_part_check_data_transmission(sreq, part);
        MPIR_ERR_CHECK(mpi_error);
    }

  fn_exit:
    return mpi_error;
  fn_fail:
    goto fn_exit;
}

/**
 * @brief Mark a list of partitions as ready to be sent.
 *
 * @param length length of the list of partitions
 * @param array_of_partitions array of partitions to be marked ready
 * @param sreq pointer to partitioned send request
 *
 * @return int see MPID_part_set_ready(...) and MPID_part_check_data_transmission(...)
 */
int MPID_Pready_list(int length, const int array_of_partitions[], MPIR_Request * sreq)
{
    struct MPID_DEV_Request_partitioned *preq;
    preq = &(sreq->dev.kind.partitioned);

    int mpi_error = MPI_SUCCESS;

    for (int i = 0; i < length; i++) {
        int part = array_of_partitions[i];

        mpi_error = MPID_part_set_ready(preq, part);
        MPIR_ERR_CHECK(mpi_error);

        mpi_error = MPID_part_check_data_transmission(sreq, part);
        MPIR_ERR_CHECK(mpi_error);
    }

  fn_exit:
    return mpi_error;
  fn_fail:
    goto fn_exit;
}

/**
 * @brief Check if data for a partition has arrived on receiver side.
 *
 * @note This implementation does not check the data arrival per partition,
 *       but it returns the completion status of the partitioned receive request.
 *
 * @param rreq pointer to partitioned receive request
 * @param partition partition to be checked (argument not used)
 * @param flag status of the partition (output of this function)
 *
 * @return int see MPIDI_PSP_Progress_test()
 */
int MPID_Parrived(MPIR_Request * rreq, int partition, int *flag)
{
    int mpi_errno = MPI_SUCCESS;

    /*
     * Do not maintain per-partition completion. Arrived when full data transfer is done.
     * TODO: to be optimized for "real" partitioned communication
     */
    if (!(*flag = MPIR_Request_is_complete(rreq))) {
        /* allow communication progress (needed in case parrived is called in a loop) */
        mpi_errno = MPIDI_PSP_Progress_test();
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
