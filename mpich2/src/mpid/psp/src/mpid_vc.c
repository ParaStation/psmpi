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


MPID_VC_t **MPID_VCRT_Create(int size)
{
	MPID_VC_t **vcrt;

	assert(size >= 0);

	vcrt = MPIU_Malloc(size * sizeof(*vcrt));

	Dprintf("(size=%d), vcrt=%p", size, vcrt);

	if (vcrt) {
		int i;
		for (i = 0; i < size; i++) {
			vcrt[i] = NULL;
		}
	} else { /* Error */
	}

	return vcrt;
}


static
void vcr_copy(MPID_VC_t **vcr_dest, MPID_VC_t **vcr_src, unsigned vcr_size)
{
	unsigned i;
	for (i = 0; i < vcr_size; i++) {
		if (vcr_src[i]) {
			vcr_dest[i] = MPID_VC_Dup(vcr_src[i]);
		}
	}
}


MPID_VC_t **MPID_VCRT_Dup(MPID_VC_t **vcrt, int size)
{
	MPID_VC_t **vcrt_new = MPID_VCRT_Create(size);

	vcr_copy(vcrt_new, vcrt, size);

	return vcrt_new;
}


static
MPID_VC_t **comm_local_vcr(MPID_Comm * comm) {
	if (comm->comm_kind == MPID_INTRACOMM) {
		return comm->vcr;
	} else {
		assert(comm->comm_kind == MPID_INTERCOMM);
		return comm->local_vcr;
	}
}


static
MPID_VC_t **comm_remote_vcr(MPID_Comm * comm) {
	assert(comm->comm_kind == MPID_INTRACOMM);
	return comm->vcr;
}


/*
 * mapper tools
 * This should be implemented in a higher sw layer.
 */
static
unsigned mapper_size(MPIR_Comm_map_t *mapper)
{
	if (mapper->type == MPIR_COMM_MAP_IRREGULAR) {
		return mapper->src_mapping_size;
	} else if (mapper->dir == MPIR_COMM_MAP_DIR_L2L || mapper->dir == MPIR_COMM_MAP_DIR_L2R) {
		return mapper->src_comm->local_size;
	} else {
		assert(mapper->dir == MPIR_COMM_MAP_DIR_R2L || mapper->dir == MPIR_COMM_MAP_DIR_R2R);
		return mapper->src_comm->remote_size;
	}
}


static
MPID_VC_t **mapper_src_vcr(MPIR_Comm_map_t *mapper)
{
	if (mapper->dir == MPIR_COMM_MAP_DIR_L2L || mapper->dir == MPIR_COMM_MAP_DIR_L2R) {
		return comm_local_vcr(mapper->src_comm);
	} else {
		assert(mapper->dir == MPIR_COMM_MAP_DIR_R2L || mapper->dir == MPIR_COMM_MAP_DIR_R2R);
		return comm_remote_vcr(mapper->src_comm);
	}
}


/*
 * mapper_list tools
 * This should be implemented in a higher sw layer.
 */
static
unsigned mapper_list_dest_local_size(MPIR_Comm_map_t *mapper_head)
{
	MPIR_Comm_map_t *mapper;
	unsigned size = 0;
	MPL_LL_FOREACH(mapper_head, mapper) {
		if (mapper->dir == MPIR_COMM_MAP_DIR_L2L ||
		    mapper->dir == MPIR_COMM_MAP_DIR_R2L) {
			size += mapper_size(mapper);
		}
	}
	return size;
}


static
unsigned mapper_list_dest_remote_size(MPIR_Comm_map_t *mapper_head)
{
	MPIR_Comm_map_t *mapper;
	unsigned size = 0;
	MPL_LL_FOREACH(mapper_head, mapper) {
		if (mapper->dir == MPIR_COMM_MAP_DIR_L2R ||
		    mapper->dir == MPIR_COMM_MAP_DIR_R2R) {
			size += mapper_size(mapper);
		}
	}
	return size;
}


static
void mapper_list_map_vcr(MPIR_Comm_map_t *mapper_head,
			 MPID_VC_t **vcr_dest_local, unsigned vcr_dest_local_size_max,
			 MPID_VC_t **vcr_dest_remote, unsigned vcr_dest_remote_size_max)
{
	MPIR_Comm_map_t *mapper;

	MPL_LL_FOREACH(mapper_head, mapper) {
		MPID_VC_t **vcr_src = mapper_src_vcr(mapper);
		unsigned size = mapper_size(mapper);

		switch (mapper->dir) {
		case MPIR_COMM_MAP_DIR_L2L:
		case MPIR_COMM_MAP_DIR_R2L:
			assert(size <= vcr_dest_local_size_max);

			vcr_copy(vcr_dest_local, vcr_src, size);

			vcr_dest_local += size;
			vcr_dest_local_size_max -= size;
			break;
		case MPIR_COMM_MAP_DIR_L2R:
		case MPIR_COMM_MAP_DIR_R2R:
			assert(size <= vcr_dest_remote_size_max);

			vcr_copy(vcr_dest_remote, vcr_src, size);

			vcr_dest_remote += size;
			vcr_dest_remote_size_max -= size;
			break;
		}
	}
	assert(vcr_dest_local_size_max == 0);
	assert(vcr_dest_remote_size_max == 0);
}


static
void MPID_VCRT_Create_mapper(MPID_Comm * comm, MPIR_Comm_map_t *mapper_head)
{
	unsigned vcr_local_size = mapper_list_dest_local_size(mapper_head);
	MPID_VC_t **vcr_local = MPID_VCRT_Create(vcr_local_size);

	unsigned vcr_remote_size = 0;
	MPID_VC_t **vcr_remote = NULL;

	if (comm->comm_kind == MPID_INTRACOMM) {
		// vcr is local, vcr_local is unused
		comm->vcr = vcr_local;
	} else {
		assert(comm->comm_kind == MPID_INTERCOMM);

		vcr_remote_size = mapper_list_dest_remote_size(mapper_head);
		vcr_remote = MPID_VCRT_Create(vcr_remote_size);

		// vcr is remote, vcr_local is local
		comm->vcr = vcr_remote;
		comm->local_vcr = vcr_local;
	}

	mapper_list_map_vcr(mapper_head, vcr_local, vcr_local_size, vcr_remote, vcr_remote_size);
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


void MPID_VCRT_Release(MPID_VC_t **vcrt, unsigned size)
{
	Dprintf("(vcrt=%p), size=%u",
		vcrt, size);

	MPID_VCRT_Destroy(vcrt, size);
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


int MPID_PSP_comm_create_hook(MPID_Comm * comm)
{
	pscom_connection_t *con1st;
	int i;

	if (comm->comm_kind == MPID_INTERCOMM) {
		/* do nothing on Intercomms */
		return MPI_SUCCESS;
	}

	if (comm->mapper_head) {
		MPID_VCRT_Create_mapper(comm, comm->mapper_head);
	}


	comm->group = NULL;

	/* ToDo: Fixme! Hack: Use pscom_socket from the rank 0 connection. This will fail
	   with mixed Intra and Inter communicator connections. */
	con1st = MPID_PSCOM_rank2connection(comm, 0);
	comm->pscom_socket = con1st ? con1st->socket : NULL;

	/* Test if connections from different sockets are used ... */
	for (i = 0; i < comm->local_size; i++) {
		if (comm->pscom_socket && MPID_PSCOM_rank2connection(comm, i) &&
		    (MPID_PSCOM_rank2connection(comm, i)->socket != comm->pscom_socket)) {
			/* ... and disallow the usage of comm->pscom_socket in this case.
			   This will disallow ANY_SOURCE receives on that communicator! */
			comm->pscom_socket = NULL;
			break;
		}
	}

	if (!MPIDI_Process.env.enable_collectives) return MPI_SUCCESS;

	MPID_PSP_group_init(comm);

	/*
	printf("%s (comm:%p(%s, id:%08x, size:%u))\n",
	       __func__, comm, comm->name, comm->context_id, comm->local_size););
	*/
	return MPI_SUCCESS;
}


int MPID_PSP_comm_destroy_hook(MPID_Comm * comm)
{
	MPID_VCRT_Release(comm->vcr, comm->remote_size);
	comm->vcr = NULL;

	if (comm->comm_kind == MPID_INTERCOMM) {
		MPID_VCRT_Release(comm->local_vcr, comm->local_size);
	}

	if (!MPIDI_Process.env.enable_collectives) return MPI_SUCCESS;

	/* ToDo: Use comm Barrier before cleanup! */

	MPID_PSP_group_cleanup(comm);

	return MPI_SUCCESS;
}
