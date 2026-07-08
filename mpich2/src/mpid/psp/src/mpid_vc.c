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


static
int MPIDI_VCR_DeleteFromPG(MPIDI_VC_t * vcr);

static
MPIDI_VC_t *new_VCR(MPIDI_PG_t * pg, int pg_rank, pscom_connection_t * con, MPIR_Lpid lpid)
{
    MPIDI_VC_t *vcr = MPL_malloc(sizeof(*vcr), MPL_MEM_OTHER);
    MPIR_Assert(vcr);

    vcr->con = con;
    vcr->lpid = lpid;
    vcr->refcnt = 1;

    vcr->pg = pg;
    vcr->pg_rank = pg_rank;

    if (pg) {
        pg->vcr[pg_rank] = vcr;
        pg->cons[pg_rank] = con;
        pg->lpids[pg_rank] = lpid;

        pg->refcnt++;
    }

    return vcr;
}


static
void VCR_put(MPIDI_VC_t * vcr, int isDisconnect)
{
    vcr->refcnt--;

    if (isDisconnect && (vcr->refcnt == 1)) {

        MPIDI_VCR_DeleteFromPG(vcr);

        if (!MPIDI_Process.env.enable_lazy_disconnect) {
            /* Finally, tear down this connection: */
            pscom_close_connection(vcr->con);
        }

        MPL_free(vcr);
    }
}


static
MPIDI_VC_t *VCR_get(MPIDI_VC_t * vcr)
{
    vcr->refcnt++;
    return vcr;
}

MPIDI_VCRT_t *MPIDI_VCRT_Create(int size)
{
    int i;
    MPIDI_VCRT_t *vcrt;

    MPIR_Assert(size >= 0);

    vcrt = MPL_malloc(sizeof(MPIDI_VCRT_t) + size * sizeof(MPIDI_VC_t), MPL_MEM_OTHER);

    Dprintf("(size=%d), vcrt=%p", size, vcrt);

    MPIR_Assert(vcrt);

    vcrt->refcnt = 1;
    vcrt->size = size;

    for (i = 0; i < size; i++) {
        vcrt->vcr[i] = NULL;
    }

    return vcrt;
}

static
int MPIDI_VCRT_Add_ref(MPIDI_VCRT_t * vcrt)
{
    Dprintf("(vcrt=%p), refcnt=%d", vcrt, vcrt->refcnt);

    vcrt->refcnt++;

    return MPI_SUCCESS;
}

MPIDI_VCRT_t *MPIDI_VCRT_Dup(MPIDI_VCRT_t * vcrt)
{
    MPIDI_VCRT_Add_ref(vcrt);
    return vcrt;
}

static
void MPIDI_VCRT_Destroy(MPIDI_VCRT_t * vcrt, int isDisconnect)
{
    int i;
    if (!vcrt)
        return;

    for (i = 0; i < vcrt->size; i++) {
        MPIDI_VC_t *vcr = vcrt->vcr[i];
        vcrt->vcr[i] = NULL;
        if (vcr)
            VCR_put(vcr, isDisconnect);
    }

    MPL_free(vcrt);
}

int MPIDI_VCRT_Release(MPIDI_VCRT_t * vcrt, int isDisconnect)
{
    if (!vcrt) {
        return MPI_SUCCESS;
    }

    Dprintf("(vcrt=%p), refcnt=%d, isDisconnect=%d", vcrt, vcrt->refcnt, isDisconnect);

    vcrt->refcnt--;

    if (vcrt->refcnt <= 0) {
        MPIR_Assert(vcrt->refcnt == 0);
        MPIDI_VCRT_Destroy(vcrt, isDisconnect);
    }

    return MPI_SUCCESS;
}

/* used in mpid_init.c to set comm_world */
MPIDI_VC_t *MPIDI_VC_Create(MPIDI_PG_t * pg, int pg_rank, pscom_connection_t * con, MPIR_Lpid lpid)
{
    Dprintf("(con=%p, lpid=%" PRIu64 ")", con, lpid);

    return new_VCR(pg, pg_rank, con, lpid);
}

/* Create a duplicate reference to a virtual connection */
MPIDI_VC_t *MPIDI_VC_Dup(MPIDI_VC_t * orig_vcr)
{
    return VCR_get(orig_vcr);
}


static
int MPIDI_VCR_DeleteFromPG(MPIDI_VC_t * vcr)
{
    MPIDI_PG_t *pg = vcr->pg;

    MPIR_Assert(vcr->con == pg->cons[vcr->pg_rank]);

    pg->vcr[vcr->pg_rank] = NULL;

    if (!MPIDI_Process.env.enable_lazy_disconnect) {
        /* For lazy disconnect, we keep this information! */
        pg->lpids[vcr->pg_rank] = MPIDI_PSP_INVALID_LPID;
        pg->cons[vcr->pg_rank] = NULL;
    }

    pg->refcnt--;

    if (pg->refcnt <= 0) {
        /* If this PG has got no more connections, remove it, too! */
        MPIR_Assert(pg->refcnt == 0);
        MPIDI_PG_Destroy(pg);
    }

    vcr->pg_rank = -1;
    vcr->pg = NULL;

    return MPI_SUCCESS;
}

#if 0

static inline int MPIDI_LPID_GetAllInComm(MPIR_Comm * comm_ptr, int local_size,
                                          MPIR_Lpid local_lpids[])
{
    int i;
    int mpi_errno = MPI_SUCCESS;
    MPIR_Assert(comm_ptr->local_size == local_size);
    for (i = 0; i < comm_ptr->local_size; i++) {
        local_lpids[i] = MPIR_comm_rank_to_lpid(comm_ptr, i);
    }
    return mpi_errno;
}

/*@
  MPID_Intercomm_exchange_map - Exchange address mapping for intercomm creation.
 @*/
int MPID_Intercomm_exchange_map(MPIR_Comm * local_comm_ptr, int local_leader,
                                MPIR_Comm * peer_comm_ptr, int remote_leader,
                                int *remote_size, MPIR_Lpid ** remote_lpids, int *is_low_group)
{
    int mpi_errno = MPI_SUCCESS;
    int singlePG;
    int local_size = 0;
    MPIR_Lpid *local_lpids = NULL;
    MPIDI_Gpid *local_gpids = NULL, *remote_gpids = NULL;
    int comm_info[2];
    int cts_tag;
    int coll_attr = MPIR_COLL_ATTR_SYNC;
    MPIR_CHKLMEM_DECL();

    cts_tag = 0 | MPIR_TAG_COLL_BIT;

    if (local_comm_ptr->rank == local_leader) {

        /* First, exchange the group information.  If we were certain
         * that the groups were disjoint, we could exchange possible
         * context ids at the same time, saving one communication.
         * But experience has shown that that is a risky assumption.
         */
        /* Exchange information with my peer.  Use sendrecv */

        local_size = local_comm_ptr->local_size;

        mpi_errno = MPIC_Sendrecv(&local_size, 1, MPIR_INT_INTERNAL,
                                  remote_leader, cts_tag,
                                  remote_size, 1, MPIR_INT_INTERNAL,
                                  remote_leader, cts_tag,
                                  peer_comm_ptr, MPI_STATUS_IGNORE, coll_attr);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        /* With this information, we can now send and receive the
         * global process ids from the peer. */
        MPIR_CHKLMEM_MALLOC(remote_gpids, (*remote_size) * sizeof(MPIDI_Gpid));
        *remote_lpids =
            (MPIR_Lpid *) MPL_malloc((*remote_size) * sizeof(MPIR_Lpid), MPL_MEM_ADDRESS);
        MPIR_CHKLMEM_MALLOC(local_gpids, local_size * sizeof(MPIDI_Gpid));
        MPIR_CHKLMEM_MALLOC(local_lpids, local_size * sizeof(MPIR_Lpid));

        mpi_errno = MPIDI_GPID_GetAllInComm(local_comm_ptr, local_size, local_gpids, &singlePG);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        /* Exchange the lpid arrays */
        mpi_errno = MPIC_Sendrecv(local_gpids, local_size * sizeof(MPIDI_Gpid), MPIR_BYTE_INTERNAL,
                                  remote_leader, cts_tag,
                                  remote_gpids, (*remote_size) * sizeof(MPIDI_Gpid),
                                  MPIR_BYTE_INTERNAL,
                                  remote_leader, cts_tag, peer_comm_ptr,
                                  MPI_STATUS_IGNORE, coll_attr);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        /* Convert the remote gpids to the lpids */
        mpi_errno = MPIDI_GPID_ToLpidArray(*remote_size, remote_gpids, *remote_lpids);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        /* Get our own lpids */
        mpi_errno = MPIDI_LPID_GetAllInComm(local_comm_ptr, local_size, local_lpids);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        /* Make an arbitrary decision about which group of processs is
         * the low group.  The LEADERS do this by comparing the
         * local gpids of the 0th member of the two groups. If these match,
         * they fall back to the rank ID within that gpid */
        if (local_gpids[0].gpid[0] == remote_gpids[0].gpid[0]) {
            (*is_low_group) = local_gpids[0].gpid[1] < remote_gpids[0].gpid[1];
        } else {
            (*is_low_group) = local_gpids[0].gpid[0] < remote_gpids[0].gpid[0];
        }

        /* At this point, we're done with the local lpids; they'll
         * be freed with the other local memory on exit */

    }
    /* End of the first phase of the leader communication */
    /* Leaders can now swap context ids and then broadcast the value
     * to the local group of processes */
    if (local_comm_ptr->rank == local_leader) {
        /* Now, send all of our local processes the remote_lpids,
         * along with the final context id */
        comm_info[0] = *remote_size;
        comm_info[1] = *is_low_group;
        mpi_errno =
            MPIR_Bcast(comm_info, 2, MPIR_INT_INTERNAL, local_leader, local_comm_ptr, coll_attr);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        MPIR_ERR_CHKANDJUMP(MPIR_COLL_ATTR_HAS_ERR(coll_attr), mpi_errno, MPI_ERR_OTHER,
                            "**coll_fail");
        mpi_errno =
            MPIR_Bcast(remote_gpids, (*remote_size) * sizeof(MPIDI_Gpid), MPIR_BYTE_INTERNAL,
                       local_leader, local_comm_ptr, coll_attr);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        MPIR_ERR_CHKANDJUMP(MPIR_COLL_ATTR_HAS_ERR(coll_attr), mpi_errno, MPI_ERR_OTHER,
                            "**coll_fail");
    } else {
        /* we're the other processes */
        mpi_errno =
            MPIR_Bcast(comm_info, 2, MPIR_INT_INTERNAL, local_leader, local_comm_ptr, coll_attr);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        MPIR_ERR_CHKANDJUMP(MPIR_COLL_ATTR_HAS_ERR(coll_attr), mpi_errno, MPI_ERR_OTHER,
                            "**coll_fail");
        *remote_size = comm_info[0];
        MPIR_CHKLMEM_MALLOC(remote_gpids, (*remote_size) * sizeof(MPIDI_Gpid));
        *remote_lpids =
            (MPIR_Lpid *) MPL_malloc((*remote_size) * sizeof(MPIR_Lpid), MPL_MEM_ADDRESS);
        mpi_errno =
            MPIR_Bcast(remote_gpids, (*remote_size) * sizeof(MPIDI_Gpid), MPIR_BYTE_INTERNAL,
                       local_leader, local_comm_ptr, coll_attr);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
        MPIR_ERR_CHKANDJUMP(MPIR_COLL_ATTR_HAS_ERR(coll_attr), mpi_errno, MPI_ERR_OTHER,
                            "**coll_fail");

        /* Extract the context and group sign informatin */
        *is_low_group = comm_info[1];
    }

    /* Finish up by giving the device the opportunity to update
     * any other information among these processes.  Note that the
     * new intercomm has not been set up; in fact, we haven't yet
     * attempted to set up the connection tables.
     *
     * In the case of the ch3 device, this calls MPID_PG_ForwardPGInfo
     * to ensure that all processes have the information about all
     * process groups.  This must be done before the call
     * to MPID_GPID_ToLpidArray, as that call needs to know about
     * all of the process groups.
     */
    MPID_ICCREATE_REMOTECOMM_HOOK(peer_comm_ptr, local_comm_ptr,
                                  *remote_size, (const MPIDI_Gpid *) remote_gpids, local_leader);


    /* Finally, if we are not the local leader, we need to
     * convert the remote gpids to local pids.  This must be done
     * after we allow the device to handle any steps that it needs to
     * take to ensure that all processes contain the necessary process
     * group information */
    if (local_comm_ptr->rank != local_leader) {
        mpi_errno = MPIDI_GPID_ToLpidArray(*remote_size, remote_gpids, *remote_lpids);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    }

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
#endif

void MPID_PSP_comm_set_vcrt(MPIR_Comm * comm, MPIDI_VCRT_t * vcrt)
{
    MPIR_Assert(vcrt);

    comm->vcrt = vcrt;
    comm->vcr = vcrt->vcr;
}

void MPID_PSP_comm_set_local_vcrt(MPIR_Comm * comm, MPIDI_VCRT_t * vcrt)
{
    MPIR_Assert(vcrt);

    comm->local_vcrt = vcrt;
    comm->local_vcr = vcrt->vcr;
}
