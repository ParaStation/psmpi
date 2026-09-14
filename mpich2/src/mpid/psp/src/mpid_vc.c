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
        pg->lpids[vcr->pg_rank] = MPIR_LPID_INVALID;
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

/* Provide a pointer to the vcr for a given lpid (no vcr ref counting!) */
static
int get_vcr_for_lpid(MPIR_Lpid lpid, MPIDI_VC_t ** vcr)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_Assert(lpid != MPIR_LPID_INVALID);

    int world_idx = MPIR_LPID_WORLD_INDEX(lpid);        /* world index to which the process belongs */
    int world_rank = MPIR_LPID_WORLD_RANK(lpid);        /* rank of the process in its world */

    MPIDI_PG_t *pg = NULL;
    mpi_errno = MPIDI_PG_get(world_idx, &pg);
    MPIR_ERR_CHECK(mpi_errno);
    MPIR_Assert(pg != NULL);

    if (!pg->vcr || !pg->vcr[world_rank]) {
        /* Either the connection (table) is not set - we cannot provide the vcr */
        MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**procnotfound", "**procnotfound %d", lpid);
    }

    *vcr = pg->vcr[world_rank];

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Create a new VCRT based on the lpids in a group */
static int create_vcrt_from_group(MPIR_Group * group, struct MPIDI_VCRT **vcrt_out)
{
    int mpi_errno = MPI_SUCCESS;

    if (group->psp_vcrt) {
        *vcrt_out = MPIDI_VCRT_Dup(group->psp_vcrt);
        goto fn_exit;
    }

    struct MPIDI_VCRT *vcrt;
    vcrt = MPIDI_VCRT_Create(group->size);
    MPIR_ERR_CHKANDJUMP1(!vcrt, mpi_errno, MPI_ERR_OTHER, "**dev|vcrt_create",
                         "**dev|vcrt_create %s", "GROUP");

    *vcrt_out = vcrt;

    for (int i = 0; i < group->size; i++) {
        MPIR_Lpid lpid = MPIR_Group_rank_to_lpid(group, i);
        MPIDI_VC_t *vcr = NULL;

        mpi_errno = get_vcr_for_lpid(lpid, &vcr);
        MPIR_ERR_CHECK(mpi_errno);

        vcrt->vcr[i] = MPIDI_VC_Dup(vcr);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_PSP_comm_set_vcrts(MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_VCRT_t *local_vcrt = NULL;
    MPIDI_VCRT_t *remote_vcrt = NULL;   /* inter-comm only */

    MPIR_Assert(comm);

    mpi_errno = create_vcrt_from_group(comm->local_group, &local_vcrt);
    MPIR_ERR_CHECK(mpi_errno);

    comm->local_vcrt = local_vcrt;
    comm->local_vcr = local_vcrt->vcr;

    /* add vcrt to the local comm group */
    if (comm->local_group->psp_vcrt == NULL) {
        comm->local_group->psp_vcrt = MPIDI_VCRT_Dup(comm->local_vcrt);
    }

    if (comm->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
        mpi_errno = create_vcrt_from_group(comm->remote_group, &remote_vcrt);
        MPIR_ERR_CHECK(mpi_errno);

        comm->remote_vcrt = remote_vcrt;
        comm->remote_vcr = remote_vcrt->vcr;

        /* add vcrt to the remote comm group */
        if (comm->remote_group->psp_vcrt == NULL) {
            comm->remote_group->psp_vcrt = MPIDI_VCRT_Dup(comm->remote_vcrt);
        }

        /* setup the vcrt for the local_comm in the intercomm */
        if (comm->local_comm) {
            comm->local_comm->local_vcrt = MPIDI_VCRT_Dup(comm->local_vcrt);
        }
    }

    if (!(comm->attr & MPIR_COMM_ATTR__SUBCOMM)) {
        /* Create subcomm vcrts */
        if (comm->node_comm) {
            MPIDI_PSP_comm_set_vcrts(comm->node_comm);
        }

        if (comm->node_roots_comm) {
            MPIDI_PSP_comm_set_vcrts(comm->node_roots_comm);
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
