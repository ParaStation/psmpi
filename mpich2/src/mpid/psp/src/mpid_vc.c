/*
 * ParaStation
 *
 * Copyright (C) 2006-2010 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include <assert.h>
#include "mpidimpl.h"

static int ENABLE_REAL_DISCONNECT = 1;
static int ENABLE_LAZY_DISCONNECT = 1;

static
int MPID_VCR_DeleteFromPG(MPID_VC_t *vcr);


static
MPID_VC_t *new_VCR(MPIDI_PG_t * pg, int pg_rank, pscom_connection_t *con, int lpid)
{
	MPID_VC_t *vcr = MPIU_Malloc(sizeof(*vcr));
	assert(vcr);

	vcr->con = con;
	vcr->lpid = lpid;
	vcr->refcnt = 1;

	vcr->pg = pg;
	vcr->pg_rank = pg_rank;

	if(pg) {
		pg->vcr[pg_rank] = vcr;
		pg->cons[pg_rank] = con;
		pg->lpids[pg_rank] = lpid;

		pg->refcnt++;
	}

	return vcr;
}


static
void VCR_put(MPID_VC_t *vcr, int isDisconnect)
{
	vcr->refcnt--;

	if(ENABLE_REAL_DISCONNECT && isDisconnect && (vcr->refcnt == 1)) {

		MPID_VCR_DeleteFromPG(vcr);

		if(!ENABLE_LAZY_DISCONNECT) {
			/* Finally, tear down this connection: */
			pscom_close_connection(vcr->con);
		}

		MPIU_Free(vcr);
	}
}


static
MPID_VC_t *VCR_get(MPID_VC_t *vcr)
{
	vcr->refcnt++;
	return vcr;
}


#define FCNAME "MPID_VCRT_Create"
#define FUNCNAME MPID_VCRT_Create
MPID_VC_t **MPID_VCRT_Create(int size)
{
	int mpi_errno = MPI_SUCCESS;
	MPID_VC_t **vcrt;

	MPIDI_STATE_DECL(MPID_STATE_MPID_VCRT_CREATE);
	MPIDI_FUNC_ENTER(MPID_STATE_MPID_VCRT_CREATE);

	assert(size >= 0);

	vcrt = MPIU_Malloc(size * sizeof(*vcrt));

	Dprintf("(size=%d), vcrt=%p", size, vcrt);

	if (vcrt) {
		int i;
		for (i = 0; i < size; i++) {
			vcrt[i] = NULL;
		}
	} else { /* Error */
		mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL, FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", 0);
	}

	MPIDI_FUNC_EXIT(MPID_STATE_MPID_VCRT_CREATE);
	return vcrt;
}
#undef FUNCNAME
#undef FCNAME


MPID_VC_t **MPID_VCRT_Dup(MPID_VC_t **vcrt, int size)
{
	MPID_VC_t **vcrt_new = MPID_VCRT_Create(size);
	int i;

	for (i = 0; i < size; i++) {
		if (vcrt[i]) {
			vcrt_new[i] = MPID_VC_Dup(vcrt[i]);
		}
	}
	return vcrt_new;
}


static
void MPID_VCRT_Destroy(MPID_VC_t **vcrt, unsigned size)
{
	int i;
	if (!vcrt) return;

	for (i = 0; i < size; i++) {
		MPID_VC_t *vcr = vcrt[i];
		vcrt[i] = NULL;
		if (vcr) VCR_put(vcr, 0);
	}

	MPIU_Free(vcrt);
}

/* used in mpid_init.c to set comm_world */
MPID_VC_t *MPID_VC_Create(MPIDI_PG_t *pg, int pg_rank, pscom_connection_t *con, int lpid)
{
	Dprintf("(con=%p, lpid=%d)", con, lpid);

	return new_VCR(pg, pg_rank, con, lpid);
}

/* Create a duplicate reference to a virtual connection */
MPID_VC_t *MPID_VC_Dup(MPID_VC_t *orig_vcr)
{
	return VCR_get(orig_vcr);
}


static
int MPID_VCR_DeleteFromPG(MPID_VC_t *vcr)
{
	MPIDI_PG_t * pg = vcr->pg;

	assert(vcr->con == pg->cons[vcr->pg_rank]);

	pg->vcr[vcr->pg_rank] = NULL;

	if(!ENABLE_LAZY_DISCONNECT) {
		/* For lazy disconnect, we keep this information! */
		pg->lpids[vcr->pg_rank] = -1;
		pg->cons[vcr->pg_rank] = NULL;
	}

	pg->refcnt--;

	if(pg->refcnt <= 0) {
		/* If this PG has got no more connections, remove it, too! */
		assert(pg->refcnt == 0);
		MPIDI_PG_Destroy(pg);
	}

	vcr->pg_rank = -1;
	vcr->pg = NULL;

	return MPI_SUCCESS;
}


int MPID_Comm_get_lpid(MPID_Comm *comm_ptr, int idx, int * lpid_ptr, MPIU_BOOL is_remote)
{
	if (comm_ptr->comm_kind == MPID_INTRACOMM || is_remote) {
		*lpid_ptr = comm_ptr->vcr[idx]->lpid;
	} else {
		*lpid_ptr = comm_ptr->local_vcr[idx]->lpid;
	}

	return MPI_SUCCESS;
}
