/*
 * ParaStation
 *
 * Copyright (C) 2022      ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"
#include "mpid_psp_request.h"
#include "mpid_psp_datatype.h"
#include "mpid_psp_packed_msg.h"

/**
 * @brief Check if a partitioned request matches to all given parameters rank, tag and context_id.
 *
 * @param rank rank to match
 * @param tag tag to match
 * @param context_id context ID to match
 * @param req pointer to the partitioned request that shall be matched
 *
 * @return bool true if match is sucessful, false otherwise
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
 * @brief       Call Irecv with sub requests to issue the data receive for partitioned request
 *              and activate completion notification of sub requests
 *
 * @param req   Pointer to partitioned request for which data receive stall be issued
 * @return int  MPI_SUCCESS on success
 *              MPI error code of Irecv on failure
 *              MPI_ERR_ARG if this function is not called for a partitioned recv request
 */
static
int MPID_part_issue_data_recv(MPIR_Request * req)
{
    struct MPID_DEV_Request_partitioned *preq;
    MPI_Aint elements;
    int mpi_errno = MPI_SUCCESS;

    if (req->kind != MPIR_REQUEST_KIND__PART_RECV) {
        mpi_errno = MPI_ERR_ARG;
        goto fn_exit;
    }

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

        mpi_errno = MPID_Irecv((void *) part_buf,
                               elements,
                               preq->datatype,
                               preq->rank, msg_tag, req->comm, preq->context_offset, &new_req);
        if (mpi_errno != MPI_SUCCESS)
            goto fn_exit;

        /*
         * Set the completion notification of the new subrequest to the completion counter of
         * the overall partitioned request once the subrequest completes, the completion
         * counter of the partitioned request req gets decremented.
         */
        new_req->completion_notification = &(req->cc);

        /* TODO: Keep track of sub requests to enable checks per partition in parrived */
    }

  fn_exit:
    return mpi_errno;
}

/**
 * @brief           Call Isend for a sub-request to issue the data send for partitioned request
 *                  and activate completion notification of sub request
 *
 * @param req       Pointer to partitioned request for which data receive stall be issued
 * @param req_idx   index of the send sub-request
 * @return int      MPI_SUCCESS on success
 *                  MPI error code of Isend on failure
 *                  MPI_ERR_ARG if this function is not called for partitioned send request
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
    int mpi_errno = MPI_SUCCESS;

    if (req->kind != MPIR_REQUEST_KIND__PART_SEND) {
        mpi_errno = MPI_ERR_ARG;
        goto fn_exit;
    }

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

    mpi_errno = MPID_Isend((void *) part_buf,
                           elements,
                           preq->datatype,
                           preq->rank, msg_tag, req->comm, preq->context_offset, &new_req);
    if (mpi_errno != MPI_SUCCESS)
        goto fn_exit;

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
    new_req->completion_notification = &(req->cc);

  fn_exit:
    return mpi_errno;
}

/**
 * @brief Send a sub-request if all partitions that belong to it are marked ready.
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
        return mpi_error;
    }

    for (int i = 0; i < preq->part_per_req; i++) {
        int base_partition = req_idx * preq->part_per_req;
        if (base_partition + i < preq->partitions) {
            req_ready = req_ready && preq->part_ready[base_partition + i];
        }
    }

    if (req_ready) {
        /* all partitions ready AND CTS received: issue data transmission */
        MPID_PSP_LOCKFREE_CALL(mpi_error = MPID_part_issue_data_send(sreq, req_idx));
    }
    return mpi_error;
}

/**
 * @brief Send sub-requests if a partition is completely ready.
 *
 * This call checks if send sub-request of one partition is completely ready to be sent and if yes,
 * tries to issue the send sub-request.
 *
 * @param sreq pointer to partitioned send request
 * @param part partition to be checked
 * @return int MPI error code of transmission issueing
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
    } else {
        /* check for all requests and send ready requests */
        for (int i = 0; i < preq->requests; i++) {
            mpi_error = MPID_part_send_if_ready(sreq, i);
            if (mpi_error != MPI_SUCCESS)
                break;
        }
    }
    return mpi_error;
}

/**
 * @brief Check for the correct number of sub-requests.
 *
 * This call checks if partitioned request has certain number of sub-request and if not adapts
 * partitioned request accordingly.
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
 * @param con pointer to pscom connection
 * @param header_net pointer to pscom network side header
 *
 * @return pscom_request_t* always NULL
 */
pscom_request_t *MPID_do_recv_part_send_init(pscom_connection_t * con,
                                             pscom_header_net_t * header_net)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_PSCOM_XHeader_Part_t *xheader_part = &(header_net->xheader->user.part);

    /* match request from global posted partitioned receive requests queue */
    MPIR_Request *posted_req = match_and_deq_request(xheader_part->common.src_rank,
                                                     xheader_part->common.tag,
                                                     xheader_part->common.context_id,
                                                     &(MPIDI_Process.part_posted_list));
    if (posted_req) {
        struct MPID_DEV_Request_partitioned *preq = &(posted_req->dev.kind.partitioned);
        /* copy infos from header into partitioned receive request */
        preq->sdata_size = xheader_part->sdata_size;
        preq->peer_request = xheader_part->sreq_ptr;

        /* check if peer request has same number of requests */
        MPID_part_check_num_requests(posted_req, xheader_part->requests);

        MPID_PSP_part_request_matched(posted_req);

        if (MPIR_Part_request_is_active(posted_req)) {

            /* set completion counter */
            MPIR_cc_set(posted_req->cc_ptr, preq->requests);

            /* if match sucessful AND MPI_Start called: send CTS message */
            MPIDI_PSP_SendPartitionedCtrl(preq->tag,
                                          posted_req->comm->context_id,
                                          posted_req->comm->rank,
                                          MPID_PSCOM_rank2connection(posted_req->comm, preq->rank),
                                          preq->sdata_size,
                                          preq->requests,
                                          preq->peer_request,
                                          posted_req, MPID_PSP_MSGTYPE_PART_CLEAR_TO_SEND);
            /* issue sub requests for receive */
            MPID_PSP_LOCKFREE_CALL(mpi_errno = MPID_part_issue_data_recv(posted_req));
        }

        if (mpi_errno != MPI_SUCCESS)
            goto fn_err_exit;

        /* release handshake reference */
        MPIR_Request_free_unsafe(posted_req);

    } else {
        /*
         * create temporary request (will be freed if repective receive request is posted on
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
        MPIR_Request_add_ref(unexp_req);
    }

  fn_err_exit:
    return NULL;
}

/**
 * @brief Clear-to-send message callback, called by sender when clear-to-send message is received.
 *
 * @param con pointer to pscom connection
 * @param header_net pointer to pscom network side header
 *
 * @return pscom_request_t* always NULL
 */
pscom_request_t *MPID_do_recv_part_cts(pscom_connection_t * con, pscom_header_net_t * header_net)
{
    int mpi_errno = MPI_SUCCESS;
    MPID_PSCOM_XHeader_Part_t *xheader_part = &(header_net->xheader->user.part);
    MPIR_Request *part_sreq = xheader_part->sreq_ptr;
    MPIR_Assert(part_sreq);

    struct MPID_DEV_Request_partitioned *preq = &part_sreq->dev.kind.partitioned;
    preq->peer_request = xheader_part->rreq_ptr;

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

    if (mpi_errno != MPI_SUCCESS) {
        PRINTERROR
            ("MPI errno %d, partitioned communication clear-to-send callback failed at %d: isend failed for ready partitions",
             mpi_errno, __LINE__);
        part_sreq->status.MPI_ERROR = mpi_errno;
    }

    return NULL;
}

/**
 * @brief Mark partition as ready (partition must not be marked ready before).
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
 *
 * This function can be used to optimize the send/recv granularity based on parameters of the
 * request.
 *
 * @note Depending on the requests determined on the peer partitioned request, the settings
 * computed here may be overwritten in the init callback (receiver side) or CTS callback (sender
 * side) since both sides have to submit the same number of requests.
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
 * @brief Common initialization for partitioned communication requests
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
    MPIR_Request *req;
    struct MPID_DEV_Request_partitioned *preq;

    req = MPIR_Request_create(type);
    if (unlikely(!req)) {
        return MPI_ERR_NO_MEM;
    }
    req->comm = comm;
    MPIR_Comm_add_ref(comm);

    preq = &req->dev.kind.partitioned;

    preq->buf = (void *) buf;
    preq->count = count;
    preq->partitions = partitions;
    preq->datatype = datatype;
    MPID_PSP_Datatype_add_ref(preq->datatype);

    preq->rank = rank;
    preq->tag = tag;
    preq->context_id = comm->context_id;
    preq->info = info;
    preq->context_offset = MPIR_CONTEXT_INTRA_PT2PT;

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

    *request = req;

    return MPI_SUCCESS;
}

/**
 * @brief Initialize a partitioned send request.
 *
 * @param buf starting address of send buffer for all partitions
 * @param partitions number of partitions
 * @param count number of elements per partition
 * @param datatype data type of each element
 * @param rank rank of source or destination
 * @param tag message tag
 * @param comm communicator
 * @param info info argument
 * @param request partitioned communication request (output value of this function)
 *
 * @return int  MPI_SUCCESS on success
 *              MPI_ERR_NO_MEM if there was a memory allocation problem
 *              MPI_ERR_INTERN if there was any other error creating the request
 */
static
int MPID_PSP_psend_init(const void *buf, int partitions, MPI_Count count,
                        MPI_Datatype datatype, int rank, int tag, MPIR_Comm * comm,
                        MPIR_Info * info, MPIR_Request ** request)
{
    int mpi_errno;
    struct MPID_DEV_Request_partitioned *preq;
    MPI_Aint dtype_size = 0;

    /* common inits */
    mpi_errno = MPID_PSP_part_init_common(buf, partitions, count,
                                          datatype, rank, tag,
                                          comm, info, request, MPIR_REQUEST_KIND__PART_SEND);
    if (mpi_errno != MPI_SUCCESS) {
        return mpi_errno;
    } else if ((*request) == NULL) {
        return MPI_ERR_INTERN;
    }

    /* init send data size */
    preq = &((*request)->dev.kind.partitioned);
    MPIR_Datatype_get_size_macro(datatype, dtype_size);
    /* count is per partition */
    preq->sdata_size = dtype_size * count * partitions;

    /* post recv request for CTS (is redone in start function as of 2nd use of this request) */
    MPIDI_PSP_RecvPartitionedCtrl(preq->tag,
                                  (*request)->comm->context_id,
                                  preq->rank,
                                  MPID_PSCOM_rank2connection((*request)->comm, preq->rank),
                                  MPID_PSP_MSGTYPE_PART_CLEAR_TO_SEND);

    /*
     * send msg of type MPID_PSP_MSGTYPE_PART_SEND_INIT
     *
     * NOTE: receive request unknown at this point
     */
    MPIDI_PSP_SendPartitionedCtrl(preq->tag, preq->context_id, (*request)->comm->rank,
                                  MPID_PSCOM_rank2connection((*request)->comm, preq->rank),
                                  preq->sdata_size, preq->requests, (*request), NULL,
                                  MPID_PSP_MSGTYPE_PART_SEND_INIT);

    return MPI_SUCCESS;
}

/**
 * @brief Initialize a partitioned receive request.
 *
 * @param buf starting address of recv buffer for all partitions
 * @param partitions number of partitions
 * @param count number of elements per partition
 * @param datatype data type of each element
 * @param rank rank of source or destination
 * @param tag message tag
 * @param comm communicator
 * @param info info argument
 * @param request partitioned communication request (output value of this function)
 *
 * @return int  MPI_SUCCESS on success
 *              MPI_ERR_NO_MEM if there was a memory allocation problem
                MPI_ERR_INTERN if there was any other error creating the request
 */
static
int MPID_PSP_precv_init(const void *buf, int partitions, MPI_Count count,
                        MPI_Datatype datatype, int rank, int tag, MPIR_Comm * comm,
                        MPIR_Info * info, MPIR_Request ** request)
{
    int mpi_errno;
    struct MPID_DEV_Request_partitioned *preq;
    MPIR_Request *unexp_req = NULL;

    /* common inits */
    mpi_errno = MPID_PSP_part_init_common(buf, partitions, count,
                                          datatype, rank, tag,
                                          comm, info, request, MPIR_REQUEST_KIND__PART_RECV);
    if (mpi_errno != MPI_SUCCESS) {
        return mpi_errno;
    } else if ((*request) == NULL) {
        return MPI_ERR_INTERN;
    }

    /* post receive request for the send init message */
    preq = &((*request)->dev.kind.partitioned);
    MPIDI_PSP_RecvPartitionedCtrl(preq->tag,
                                  preq->context_id,
                                  preq->rank,
                                  MPID_PSCOM_rank2connection((*request)->comm, preq->rank),
                                  MPID_PSP_MSGTYPE_PART_SEND_INIT);

    /*
     * try matching this recv request to unexpected SEND_INIT request from the global
     * partitioned unexpected list
     */
    unexp_req = match_and_deq_request(preq->rank,
                                      preq->tag,
                                      preq->context_id, &(MPIDI_Process.part_unexp_list));

    if (unexp_req) {
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

    return MPI_SUCCESS;
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

    if (req->kind != MPIR_REQUEST_KIND__PART_SEND) {
        return MPI_ERR_ARG;
    }

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
        MPIDI_PSP_RecvPartitionedCtrl(preq->tag, req->comm->context_id, preq->rank,
                                      MPID_PSCOM_rank2connection(req->comm, preq->rank),
                                      MPID_PSP_MSGTYPE_PART_CLEAR_TO_SEND);
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
    }

    return mpi_errno;
}

/**
 * @brief Start a partitioned receive request.
 *
 * @param req pointer to partitioned receive request to be started
 *
 * @return int  MPI_SUCCESS on success
 *              MPI_ERR_ARG if req is not a partitioned recv request
 *              MPI error code of data transmission issueing
 */
int MPID_PSP_precv_start(MPIR_Request * req)
{
    int mpi_errno = MPI_SUCCESS;
    struct MPID_DEV_Request_partitioned *preq;

    if (req->kind != MPIR_REQUEST_KIND__PART_RECV) {
        return MPI_ERR_ARG;
    }
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
    }

    return mpi_errno;
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
 * @return int see MPID_PSP_psend_init(...)
 */
int MPID_Psend_init(const void *buf, int partitions, MPI_Count count, MPI_Datatype datatype,
                    int dest, int tag, MPIR_Comm * comm, MPIR_Info * info, MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;

    MPID_PSP_LOCKFREE_CALL(mpi_errno = MPID_PSP_psend_init(buf, partitions, count,
                                                           datatype, dest, tag,
                                                           comm, info, request));

    return mpi_errno;
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
 * @return int see MPID_PSP_precv_init(...)
 */
int MPID_Precv_init(void *buf, int partitions, MPI_Count count, MPI_Datatype datatype,
                    int source, int tag, MPIR_Comm * comm, MPIR_Info * info,
                    MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;

    MPID_PSP_LOCKFREE_CALL(mpi_errno = MPID_PSP_precv_init(buf, partitions, count,
                                                           datatype, source, tag,
                                                           comm, info, request));

    return mpi_errno;
}

/**
 * @brief Mark a range of partitions as ready to be sent
 *
 * This call marks the partitions ranging from (including) partition_low to partition_high as ready.
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
        if (mpi_error != MPI_SUCCESS)
            goto exit_fn;

        mpi_error = MPID_part_check_data_transmission(sreq, part);
        if (mpi_error != MPI_SUCCESS)
            goto exit_fn;
    }

  exit_fn:
    return mpi_error;
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
        if (mpi_error != MPI_SUCCESS)
            goto exit_fn;

        mpi_error = MPID_part_check_data_transmission(sreq, part);
        if (mpi_error != MPI_SUCCESS)
            goto exit_fn;
    }

  exit_fn:
    return mpi_error;
}

/**
 * @brief Check if data for a partition has arrived on receiver side.
 *
 *        Remark: This implementation does not check the data arrival per partition,
 *        but it returns the completion status of the partitioned receive request.
 *
 * @param rreq pointer to partitioned receive request
 * @param partition partition to be checked (argument not used)
 * @param flag status of the partition (output of this function)
 *
 * @return int see MPIDI_PSP_Progress_test()
 */
static
int MPID_PSP_Parrived(MPIR_Request * rreq, int partition, int *flag)
{
    int mpi_errno = MPI_SUCCESS;

    /*
     * Do not maintain per-partition completion. Arrived when full data transfer is done.
     * TODO: to be optimized for "real" partitioned communication
     */
    if (!(*flag = MPIR_Request_is_complete(rreq))) {
        /* allow communication progress (needed in case parrived is called in a loop) */
        mpi_errno = MPIDI_PSP_Progress_test();
    }

    return mpi_errno;
}

/**
 * @brief Check if data for a partition has arrived on receiver side.
 *
 * @param rreq pointer to partitioned receive request
 * @param partition partition to be checked
 * @param flag status of the partition (output of this function)
 *
 * @return int see MPID_PSP_Parrived(...)
 */
int MPID_Parrived(MPIR_Request * rreq, int partition, int *flag)
{
    int mpi_errno = MPI_SUCCESS;

    MPID_PSP_LOCKFREE_CALL(mpi_errno = MPID_PSP_Parrived(rreq, partition, flag));

    return mpi_errno;
}
