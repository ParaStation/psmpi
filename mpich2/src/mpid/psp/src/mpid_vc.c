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
#include "mpl_utlist.h"

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

MPID_VCRT_t *MPID_VCRT_Create(int size)
{
	int i;
	MPID_VCRT_t * vcrt;

	assert(size >= 0);

	vcrt = MPIU_Malloc(sizeof(MPID_VCRT_t) + size * sizeof(MPID_VC_t));

	Dprintf("(size=%d), vcrt=%p", size, vcrt);

	assert(vcrt);

	vcrt->refcnt = 1;
	vcrt->size = size;

	for (i = 0; i < size; i++) {
		vcrt->vcr[i] = NULL;
	}

	return vcrt;
}

static
int MPID_VCRT_Add_ref(MPID_VCRT_t *vcrt)
{
	Dprintf("(vcrt=%p), refcnt=%d", vcrt, vcrt->refcnt);

	vcrt->refcnt++;

	return MPI_SUCCESS;
}

MPID_VCRT_t *MPID_VCRT_Dup(MPID_VCRT_t *vcrt)
{
	MPID_VCRT_Add_ref(vcrt);
	return vcrt;
}

static
void MPID_VCRT_Destroy(MPID_VCRT_t *vcrt, int isDisconnect)
{
	int i;
	if (!vcrt) return;

	for (i = 0; i < vcrt->size; i++) {
		MPID_VC_t *vcr = vcrt->vcr[i];
		vcrt->vcr[i] = NULL;
		if (vcr) VCR_put(vcr, isDisconnect);
	}

	MPIU_Free(vcrt);
}

int MPID_VCRT_Release(MPID_VCRT_t *vcrt, int isDisconnect)
{
	Dprintf("(vcrt=%p), refcnt=%d, isDisconnect=%d", vcrt, vcrt->refcnt, isDisconnect);

	vcrt->refcnt--;

	if (vcrt->refcnt <= 0) {
		assert(vcrt->refcnt == 0);
		MPID_VCRT_Destroy(vcrt, isDisconnect);
	}

	return MPI_SUCCESS;
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


/* Code adopted (and adapted) from CH3: */

static void dup_vcrt(MPID_VCRT_t* src_vcrt, MPID_VCRT_t **dest_vcrt,
		     MPIR_Comm_map_t *mapper, int src_comm_size, int vcrt_size,
                     int vcrt_offset)
{
	int flag, i;

	/* try to find the simple case where the new comm is a simple
	 * duplicate of the previous comm.  in that case, we simply add a
	 * reference to the previous VCRT instead of recreating it. */
	if (mapper->type == MPIR_COMM_MAP_DUP && src_comm_size == vcrt_size) {
		*dest_vcrt = MPID_VCRT_Dup(src_vcrt);
		return;
	}
	else if (mapper->type == MPIR_COMM_MAP_IRREGULAR &&
		 mapper->src_mapping_size == vcrt_size) {
		/* if the mapping array is exactly the same as the original
		 * comm's VC list, there is no need to create a new VCRT.
		 * instead simply point to the original comm's VCRT and bump
		 * up it's reference count */
		flag = 1;
		for (i = 0; i < mapper->src_mapping_size; i++)
			if (mapper->src_mapping[i] != i)
				flag = 0;

		if (flag) {
			*dest_vcrt = MPID_VCRT_Dup(src_vcrt);;
			return;
		}
	}

	/* we are in the more complex case where we need to allocate a new
	 * VCRT */

	if (!vcrt_offset) {
		*dest_vcrt = MPID_VCRT_Create(vcrt_size);
	}

	if (mapper->type == MPIR_COMM_MAP_DUP) {
		for (i = 0; i < src_comm_size; i++)
			(*dest_vcrt)->vcr[i + vcrt_offset] = MPID_VC_Dup(src_vcrt->vcr[i]);
	}
	else {
		for (i = 0; i < mapper->src_mapping_size; i++)
			(*dest_vcrt)->vcr[i + vcrt_offset] = MPID_VC_Dup(src_vcrt->vcr[mapper->src_mapping[i]]);
	}
}

static inline int map_size(MPIR_Comm_map_t map)
{
	if (map.type == MPIR_COMM_MAP_IRREGULAR)
		return map.src_mapping_size;
	else if (map.dir == MPIR_COMM_MAP_DIR_L2L || map.dir == MPIR_COMM_MAP_DIR_L2R)
		return map.src_comm->local_size;
	else
		return map.src_comm->remote_size;
}


void MPID_PSP_comm_add_map(MPID_Comm * comm)
{
	MPIR_Comm_map_t *mapper;
	MPID_Comm *src_comm;
	int vcrt_size, vcrt_offset;

	/* initialize the is_disconnected variable to FALSE.  this will be
	 * set to TRUE if the communicator is freed by an
	 * MPI_COMM_DISCONNECT call. */

	comm->is_disconnected = 0;

	/* do some sanity checks */
	MPL_LL_FOREACH(comm->mapper_head, mapper) {
		if (mapper->src_comm->comm_kind == MPID_INTRACOMM)
			MPIU_Assert(mapper->dir == MPIR_COMM_MAP_DIR_L2L ||
				    mapper->dir == MPIR_COMM_MAP_DIR_L2R);
		if (comm->comm_kind == MPID_INTRACOMM)
			MPIU_Assert(mapper->dir == MPIR_COMM_MAP_DIR_L2L ||
				    mapper->dir == MPIR_COMM_MAP_DIR_R2L);
	}

	/* First, handle all the mappers that contribute to the local part
	 * of the comm */
	vcrt_size = 0;
	MPL_LL_FOREACH(comm->mapper_head, mapper) {
		if (mapper->dir == MPIR_COMM_MAP_DIR_L2R ||
		    mapper->dir == MPIR_COMM_MAP_DIR_R2R)
			continue;

		vcrt_size += map_size(*mapper);
	}

	vcrt_offset = 0;
	MPL_LL_FOREACH(comm->mapper_head, mapper) {
		src_comm = mapper->src_comm;

		if (mapper->dir == MPIR_COMM_MAP_DIR_L2R ||
		    mapper->dir == MPIR_COMM_MAP_DIR_R2R)
			continue;

		if (mapper->dir == MPIR_COMM_MAP_DIR_L2L) {
			if (src_comm->comm_kind == MPID_INTRACOMM && comm->comm_kind == MPID_INTRACOMM) {
				dup_vcrt(src_comm->vcrt, &comm->vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			}
			else if (src_comm->comm_kind == MPID_INTRACOMM && comm->comm_kind == MPID_INTERCOMM) {
				dup_vcrt(src_comm->vcrt, &comm->local_vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_local_vcrt(comm, comm->local_vcrt);
			}
			else if (src_comm->comm_kind == MPID_INTERCOMM && comm->comm_kind == MPID_INTRACOMM) {
				dup_vcrt(src_comm->local_vcrt, &comm->vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			}
			else {
				dup_vcrt(src_comm->local_vcrt, &comm->local_vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_local_vcrt(comm, comm->local_vcrt);
			}
		}
		else {  /* mapper->dir == MPIR_COMM_MAP_DIR_R2L */
			MPIU_Assert(src_comm->comm_kind == MPID_INTERCOMM);
			if (comm->comm_kind == MPID_INTRACOMM) {
				dup_vcrt(src_comm->vcrt, &comm->vcrt, mapper, mapper->src_comm->remote_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			}
			else {
				dup_vcrt(src_comm->vcrt, &comm->local_vcrt, mapper, mapper->src_comm->remote_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->local_vcrt);
			}
		}
		vcrt_offset += map_size(*mapper);
	}

	/* Next, handle all the mappers that contribute to the remote part
	 * of the comm (only valid for intercomms) */
	vcrt_size = 0;
	MPL_LL_FOREACH(comm->mapper_head, mapper) {
		if (mapper->dir == MPIR_COMM_MAP_DIR_L2L ||
		    mapper->dir == MPIR_COMM_MAP_DIR_R2L)
			continue;

		vcrt_size += map_size(*mapper);
	}
	vcrt_offset = 0;
	MPL_LL_FOREACH(comm->mapper_head, mapper) {
		src_comm = mapper->src_comm;

		if (mapper->dir == MPIR_COMM_MAP_DIR_L2L ||
		    mapper->dir == MPIR_COMM_MAP_DIR_R2L)
			continue;

		MPIU_Assert(comm->comm_kind == MPID_INTERCOMM);

		if (mapper->dir == MPIR_COMM_MAP_DIR_L2R) {
			if (src_comm->comm_kind == MPID_INTRACOMM) {
				dup_vcrt(src_comm->vcrt, &comm->vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			}
			else {
				dup_vcrt(src_comm->local_vcrt, &comm->vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			}
		}
		else {  /* mapper->dir == MPIR_COMM_MAP_DIR_R2R */
			MPIU_Assert(src_comm->comm_kind == MPID_INTERCOMM);
			dup_vcrt(src_comm->vcrt, &comm->vcrt, mapper, mapper->src_comm->remote_size, vcrt_size, vcrt_offset);
			MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
		}
		vcrt_offset += map_size(*mapper);
	}

	if (comm->comm_kind == MPID_INTERCOMM) {
		/* setup the vcrt for the local_comm in the intercomm */
		if (comm->local_comm) {
			MPID_VCRT_t *vcrt;
			vcrt = MPID_VCRT_Dup(comm->local_vcrt);
			assert(vcrt);
			MPID_PSP_comm_set_vcrt(comm->local_comm, vcrt);
		}
	}
}
