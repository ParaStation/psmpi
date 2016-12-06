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

#ifdef MPID_PSP_USE_SMP_AWARE_COLLOPS
#ifndef MPID_USE_NODE_IDS
#error "PSP: The SMP-aware collops features requires that USE_NODE_IDS is enabled for MPICH!"
#endif
int MPID_Get_node_id(MPID_Comm *comm, int rank, MPID_Node_id_t *id_p)
{
	int i;
	int pg_check_id;

	if(!MPIDI_Process.env.enable_smp_aware_collops ||  !MPIDI_Process.node_id_table) {
		/* Just pretend that each rank lives on its own node: */
		*id_p = rank;
		return 0;
	}

	pg_check_id = comm->vcr[0]->pg->id_num;
	for(i=1; i<comm->local_size; i++) {
		if(comm->vcr[i]->pg->id_num != pg_check_id) {
			/* This communicator spans more than one MPICH Process Group (PG)!
			   As we create the node_id_table on an MPI_COMM_WORLD basis, we
			   have to fallback here to the non smp-aware collops...
			   (FIXME: Are we sure that this will be detected here by all ranks within comm?)
			*/
			*id_p = rank;
			return 0;
		}
	}

	*id_p = MPIDI_Process.node_id_table[comm->vcr[rank]->pg_rank];

	return 0;
}

int MPID_Get_max_node_id(MPID_Comm *comm, MPID_Node_id_t *max_id_p)
{
	*max_id_p = MPIDI_Process.my_pg_size;
	return 0;
}
#endif

/*
typedef struct MPIDIx_VCRT * MPID_VCRT;
typedef struct MPIDIx_VC * MPID_VCR;
*/

struct MPIDIx_VCRT {
	unsigned int refcnt;
	unsigned int size;
	MPID_VCR vcr[0];
};


static
MPID_VCR new_VCR(MPIDI_PG_t * pg, int pg_rank, pscom_connection_t *con, int lpid)
{
	MPID_VCR vcr = MPIU_Malloc(sizeof(*vcr));
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
void VCR_put(MPID_VCR vcr, int isDisconnect)
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
MPID_VCR VCR_get(MPID_VCR vcr)
{
	vcr->refcnt++;
	return vcr;
}


#define FCNAME "MPID_VCRT_Create"
#define FUNCNAME MPID_VCRT_Create
int MPID_VCRT_Create(int size, MPID_VCRT *vcrt_ptr)
{
	int mpi_errno = MPI_SUCCESS;
	struct MPIDIx_VCRT * vcrt;

	MPIDI_STATE_DECL(MPID_STATE_MPID_VCRT_CREATE);
	MPIDI_FUNC_ENTER(MPID_STATE_MPID_VCRT_CREATE);

	assert(size >= 0);

	vcrt = MPIU_Malloc(sizeof(*vcrt) + size * sizeof(vcrt->vcr[0]));

	Dprintf("(size=%d, vcrt_ptr=%p), vcrt=%p", size, vcrt_ptr, vcrt);

	if (vcrt) {
		int i;
		vcrt->refcnt = 1;
		vcrt->size = size;
		*vcrt_ptr = vcrt;
		for (i = 0; i < size; i++) {
			vcrt->vcr[i] = NULL;
		}
	} else { /* Error */
		mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_FATAL, FCNAME, __LINE__, MPI_ERR_OTHER, "**nomem", 0);
	}

	MPIDI_FUNC_EXIT(MPID_STATE_MPID_VCRT_CREATE);
	return mpi_errno;
}
#undef FUNCNAME
#undef FCNAME


static
void MPID_VCRT_Destroy(MPID_VCRT vcrt, int isDisconnect)
{
	int i;
	if (!vcrt) return;

	for (i = 0; i < vcrt->size; i++) {
		MPID_VCR vcr = vcrt->vcr[i];
		vcrt->vcr[i] = NULL;
		if (vcr) VCR_put(vcr, isDisconnect);
	}

	MPIU_Free(vcrt);
}


int MPID_VCRT_Add_ref(MPID_VCRT vcrt)
{
	Dprintf("(vcrt=%p), refcnt=%d", vcrt, vcrt->refcnt);

	vcrt->refcnt++;

	return MPI_SUCCESS;
}

int MPID_VCRT_Release(MPID_VCRT vcrt, int isDisconnect)
{
	Dprintf("(vcrt=%p), refcnt=%d, isDisconnect=%d",
		vcrt, vcrt->refcnt, isDisconnect);

	vcrt->refcnt--;

	if (vcrt->refcnt <= 0) {
		assert(vcrt->refcnt == 0);
		MPID_VCRT_Destroy(vcrt, isDisconnect);
	}

	return MPI_SUCCESS;
}

int MPID_VCRT_Get_ptr(MPID_VCRT vcrt, MPID_VCR **vc_pptr)
{
	Dprintf("(vcrt=%p, vc_pptr=%p)", vcrt, vc_pptr);

	*vc_pptr = vcrt->vcr;
	return MPI_SUCCESS;
}

/* used in mpid_init.c to set comm_world */
int MPID_VCR_Initialize(MPID_VCR *vcr_ptr, MPIDI_PG_t * pg, int pg_rank, pscom_connection_t *con, int lpid)
{
	Dprintf("(vcr_ptr=%p, con=%p, lpid=%d)", vcr_ptr, con, lpid);

	assert(!(*vcr_ptr)); /* vcr must be uninitialized. */
	/* if (*vcr_ptr) VCR_put(*vcr_ptr); */

	*vcr_ptr = new_VCR(pg, pg_rank, con, lpid);

	return MPI_SUCCESS;
}

/*@
  MPID_VCR_Dup - Create a duplicate reference to a virtual connection
  @*/
int MPID_VCR_Dup(MPID_VCR orig_vcr, MPID_VCR * new_vcr)
{
	Dprintf("(orig_vcr=%p, new_vcr=%p)", orig_vcr, new_vcr);

	*new_vcr = VCR_get(orig_vcr);

	return MPI_SUCCESS;
}

/*@
   MPID_VCR_Get_lpid - Get the local process id that corresponds to a
   virtual connection reference.

   Notes:
   The local process ids are described elsewhere.  Basically, they are
   a nonnegative number by which this process can refer to other processes
   to which it is connected.  These are local process ids because different
   processes may use different ids to identify the same target process
  @*/
int MPID_VCR_Get_lpid(MPID_VCR vcr, int * lpid_ptr)
{
	*lpid_ptr = vcr->lpid;

	Dprintf("(vcr=%p, lpid_ptr=%p), lpid=%d", vcr, lpid_ptr, *lpid_ptr);

	return MPI_SUCCESS;
}

int MPID_VCR_DeleteFromPG(MPID_VCR vcr)
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
