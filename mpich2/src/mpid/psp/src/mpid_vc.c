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
