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

// This must be the last include before sysmbols are defined:
#include "mpid_visibility.h"


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

	if(!MPIDI_Process.env.enable_smp_awareness) {
		/* Just pretend that each rank lives on its own node: */
		*id_p = rank;
		return 0;
	}

	assert(MPIDI_Process.node_id_table);

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
	if(!MPIDI_Process.node_id_table) {
		/* Most likely that SMP-awareness has been disabled due to process spawning... */
		return  MPI_ERR_OTHER;
	}

	*max_id_p = MPIDI_Process.my_pg_size;
	return 0;
}
#endif

int MPID_PSP_get_host_hash()
{
       char host_name[MPI_MAX_PROCESSOR_NAME];
       int result_len;
       static int host_hash = 0;

       if(!host_hash) {
               MPID_Get_processor_name(host_name, MPI_MAX_PROCESSOR_NAME, &result_len);
               MPIDI_PG_Convert_id(host_name, &host_hash);
               if(!MPIDI_Process.env.enable_smp_awareness) host_hash += MPIDI_Process.my_pg_rank;
       }
       return host_hash;
}

static
int MPIDI_VCR_DeleteFromPG(MPIDI_VC_t *vcr);

static
MPIDI_VC_t *new_VCR(MPIDI_PG_t * pg, int pg_rank, pscom_connection_t *con, int lpid)
{
	MPIDI_VC_t *vcr = MPIU_Malloc(sizeof(*vcr));
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
void VCR_put(MPIDI_VC_t *vcr, int isDisconnect)
{
	vcr->refcnt--;

	if(ENABLE_REAL_DISCONNECT && isDisconnect && (vcr->refcnt == 1)) {

		MPIDI_VCR_DeleteFromPG(vcr);

		if(!ENABLE_LAZY_DISCONNECT) {
			/* Finally, tear down this connection: */
			pscom_close_connection(vcr->con);
		}

		MPIU_Free(vcr);
	}
}


static
MPIDI_VC_t *VCR_get(MPIDI_VC_t *vcr)
{
	vcr->refcnt++;
	return vcr;
}

MPIDI_VCRT_t *MPIDI_VCRT_Create(int size)
{
	int i;
	MPIDI_VCRT_t * vcrt;

	assert(size >= 0);

	vcrt = MPIU_Malloc(sizeof(MPIDI_VCRT_t) + size * sizeof(MPIDI_VC_t));

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
int MPIDI_VCRT_Add_ref(MPIDI_VCRT_t *vcrt)
{
	Dprintf("(vcrt=%p), refcnt=%d", vcrt, vcrt->refcnt);

	vcrt->refcnt++;

	return MPI_SUCCESS;
}

MPIDI_VCRT_t *MPIDI_VCRT_Dup(MPIDI_VCRT_t *vcrt)
{
	MPIDI_VCRT_Add_ref(vcrt);
	return vcrt;
}

static
void MPIDI_VCRT_Destroy(MPIDI_VCRT_t *vcrt, int isDisconnect)
{
	int i;
	if (!vcrt) return;

	for (i = 0; i < vcrt->size; i++) {
		MPIDI_VC_t *vcr = vcrt->vcr[i];
		vcrt->vcr[i] = NULL;
		if (vcr) VCR_put(vcr, isDisconnect);
	}

	MPIU_Free(vcrt);
}

int MPIDI_VCRT_Release(MPIDI_VCRT_t *vcrt, int isDisconnect)
{
	Dprintf("(vcrt=%p), refcnt=%d, isDisconnect=%d", vcrt, vcrt->refcnt, isDisconnect);

	vcrt->refcnt--;

	if (vcrt->refcnt <= 0) {
		assert(vcrt->refcnt == 0);
		MPIDI_VCRT_Destroy(vcrt, isDisconnect);
	}

	return MPI_SUCCESS;
}

/* used in mpid_init.c to set comm_world */
MPIDI_VC_t *MPIDI_VC_Create(MPIDI_PG_t *pg, int pg_rank, pscom_connection_t *con, int lpid)
{
	Dprintf("(con=%p, lpid=%d)", con, lpid);

	return new_VCR(pg, pg_rank, con, lpid);
}

/* Create a duplicate reference to a virtual connection */
MPIDI_VC_t *MPIDI_VC_Dup(MPIDI_VC_t *orig_vcr)
{
	return VCR_get(orig_vcr);
}


static
int MPIDI_VCR_DeleteFromPG(MPIDI_VC_t *vcr)
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

	if (comm->mapper_head) {
		MPID_PSP_comm_create_mapper(comm);
	}

	comm->group = NULL;

	if (comm->comm_kind == MPID_INTERCOMM) {
		/* do nothing on Intercomms */
		comm->pscom_socket = NULL;
		return MPI_SUCCESS;
	}

	/* Use pscom_socket from the rank 0 connection ... */
	con1st = MPID_PSCOM_rank2connection(comm, 0);
	comm->pscom_socket = con1st ? con1st->socket : NULL;

	/* ... and test if connections from different sockets are used ... */
	for (i = 0; i < comm->local_size; i++) {
		if (comm->pscom_socket && MPID_PSCOM_rank2connection(comm, i) &&
		    (MPID_PSCOM_rank2connection(comm, i)->socket != comm->pscom_socket)) {
			/* ... and disallow the usage of comm->pscom_socket in this case.
			   This will disallow ANY_SOURCE receives on that communicator for older pscoms
			   ... but should be fixed/handled within the pscom layer as of pscom 5.2.0 */
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
	MPIDI_VCRT_Release(comm->vcrt, comm->is_disconnected);
	comm->vcr = NULL;

	if (comm->comm_kind == MPID_INTERCOMM) {
		MPIDI_VCRT_Release(comm->local_vcrt, comm->is_disconnected);
	}

	if (!MPIDI_Process.env.enable_collectives) return MPI_SUCCESS;

	/* ToDo: Use comm Barrier before cleanup! */

	MPID_PSP_group_cleanup(comm);

	return MPI_SUCCESS;
}


/*
 * Mapper Tools
 *
 * Code adopted (and adapted) from CH3.
 *
 * (This should be implemented in a higher SW layer...)
 *
 */

static
void MPID_PSP_mapper_dup_vcrt(MPIDI_VCRT_t* src_vcrt, MPIDI_VCRT_t **dest_vcrt,
			      MPIR_Comm_map_t *mapper, int src_comm_size, int vcrt_size,
			      int vcrt_offset)
{
	int flag, i;

	/* try to find the simple case where the new comm is a simple
	 * duplicate of the previous comm.  in that case, we simply add a
	 * reference to the previous VCRT instead of recreating it. */
	if (mapper->type == MPIR_COMM_MAP_DUP && src_comm_size == vcrt_size) {
		*dest_vcrt = MPIDI_VCRT_Dup(src_vcrt);
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
			*dest_vcrt = MPIDI_VCRT_Dup(src_vcrt);;
			return;
		}
	}

	/* we are in the more complex case where we need to allocate a new
	 * VCRT */

	if (!vcrt_offset) {
		*dest_vcrt = MPIDI_VCRT_Create(vcrt_size);
	}

	if (mapper->type == MPIR_COMM_MAP_DUP) {
		for (i = 0; i < src_comm_size; i++)
			(*dest_vcrt)->vcr[i + vcrt_offset] = MPIDI_VC_Dup(src_vcrt->vcr[i]);
	}
	else {
		for (i = 0; i < mapper->src_mapping_size; i++)
			(*dest_vcrt)->vcr[i + vcrt_offset] = MPIDI_VC_Dup(src_vcrt->vcr[mapper->src_mapping[i]]);
	}
}

static
unsigned MPID_PSP_mapper_size(MPIR_Comm_map_t *mapper)
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


/*
 * mapper_list tools
 * This should be implemented in a higher sw layer.
 */
static
unsigned MPID_PSP_mapper_list_dest_local_size(MPIR_Comm_map_t *mapper_head)
{
	MPIR_Comm_map_t *mapper;
	unsigned size = 0;
	MPL_LL_FOREACH(mapper_head, mapper) {
		if (mapper->dir == MPIR_COMM_MAP_DIR_L2L ||
		    mapper->dir == MPIR_COMM_MAP_DIR_R2L) {
			size += MPID_PSP_mapper_size(mapper);
		}
	}
	return size;
}


static
unsigned MPID_PSP_mapper_list_dest_remote_size(MPIR_Comm_map_t *mapper_head)
{
	MPIR_Comm_map_t *mapper;
	unsigned size = 0;
	MPL_LL_FOREACH(mapper_head, mapper) {
		if (mapper->dir == MPIR_COMM_MAP_DIR_L2R ||
		    mapper->dir == MPIR_COMM_MAP_DIR_R2R) {
			size += MPID_PSP_mapper_size(mapper);
		}
	}
	return size;
}


static
void MPID_PSP_mapper_list_map_local_vcr(MPID_Comm *comm, int vcrt_size)
{
	MPID_Comm *src_comm;
	MPIR_Comm_map_t *mapper;
	int vcrt_offset = 0;

	MPL_LL_FOREACH(comm->mapper_head, mapper) {
		src_comm = mapper->src_comm;

		switch(mapper->dir) {

		case MPIR_COMM_MAP_DIR_L2R:
		case MPIR_COMM_MAP_DIR_R2R:
			break;

		case MPIR_COMM_MAP_DIR_L2L:
			if (src_comm->comm_kind == MPID_INTRACOMM && comm->comm_kind == MPID_INTRACOMM) {
				MPID_PSP_mapper_dup_vcrt(src_comm->vcrt, &comm->vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			}
			else if (src_comm->comm_kind == MPID_INTRACOMM && comm->comm_kind == MPID_INTERCOMM) {
				MPID_PSP_mapper_dup_vcrt(src_comm->vcrt, &comm->local_vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_local_vcrt(comm, comm->local_vcrt);
			}
			else if (src_comm->comm_kind == MPID_INTERCOMM && comm->comm_kind == MPID_INTRACOMM) {
				MPID_PSP_mapper_dup_vcrt(src_comm->local_vcrt, &comm->vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			}
			else {
				MPID_PSP_mapper_dup_vcrt(src_comm->local_vcrt, &comm->local_vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_local_vcrt(comm, comm->local_vcrt);
			}
			vcrt_offset += MPID_PSP_mapper_size(mapper);
			break;

		case MPIR_COMM_MAP_DIR_R2L:
			MPIU_Assert(src_comm->comm_kind == MPID_INTERCOMM);
			if (comm->comm_kind == MPID_INTRACOMM) {
				MPID_PSP_mapper_dup_vcrt(src_comm->vcrt, &comm->vcrt, mapper, mapper->src_comm->remote_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			}
			else {
				MPID_PSP_mapper_dup_vcrt(src_comm->vcrt, &comm->local_vcrt, mapper, mapper->src_comm->remote_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->local_vcrt);
			}
			vcrt_offset += MPID_PSP_mapper_size(mapper);
			break;

		default: assert(0);
		}
	}
}

static
void MPID_PSP_mapper_list_map_remote_vcr(MPID_Comm *comm, int vcrt_size)
{
	MPID_Comm *src_comm;
	MPIR_Comm_map_t *mapper;
	int vcrt_offset = 0;

	MPL_LL_FOREACH(comm->mapper_head, mapper) {
		src_comm = mapper->src_comm;

		switch(mapper->dir) {

		case MPIR_COMM_MAP_DIR_L2L:
		case MPIR_COMM_MAP_DIR_R2L:
			break;

		case MPIR_COMM_MAP_DIR_L2R:
			MPIU_Assert(comm->comm_kind == MPID_INTERCOMM);
			if (src_comm->comm_kind == MPID_INTRACOMM) {
				MPID_PSP_mapper_dup_vcrt(src_comm->vcrt, &comm->vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			}
			else {
				MPID_PSP_mapper_dup_vcrt(src_comm->local_vcrt, &comm->vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			}
			vcrt_offset += MPID_PSP_mapper_size(mapper);
			break;

		case MPIR_COMM_MAP_DIR_R2R:
			MPIU_Assert(comm->comm_kind == MPID_INTERCOMM);
			MPIU_Assert(src_comm->comm_kind == MPID_INTERCOMM);
			MPID_PSP_mapper_dup_vcrt(src_comm->vcrt, &comm->vcrt, mapper, mapper->src_comm->remote_size, vcrt_size, vcrt_offset);
			MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			vcrt_offset +=MPID_PSP_mapper_size(mapper);
			break;

		default: assert(0);
		}
	}
}


void MPID_PSP_comm_create_mapper(MPID_Comm * comm)
{
	int vcrt_size;

	comm->is_disconnected = 0;

	if(!comm->mapper_head) return;

	vcrt_size = MPID_PSP_mapper_list_dest_local_size(comm->mapper_head);
	MPID_PSP_mapper_list_map_local_vcr(comm, vcrt_size);

	vcrt_size = MPID_PSP_mapper_list_dest_remote_size(comm->mapper_head);
	MPID_PSP_mapper_list_map_remote_vcr(comm, vcrt_size);

	if (comm->comm_kind == MPID_INTERCOMM) {
		/* setup the vcrt for the local_comm in the intercomm */
		if (comm->local_comm) {
			MPIDI_VCRT_t *vcrt;
			vcrt = MPIDI_VCRT_Dup(comm->local_vcrt);
			assert(vcrt);
			MPID_PSP_comm_set_vcrt(comm->local_comm, vcrt);
		}
	}
}


void MPID_PSP_comm_set_vcrt(MPID_Comm *comm, MPIDI_VCRT_t *vcrt)
{
       assert(vcrt);

       comm->vcrt = vcrt;
       comm->vcr  = vcrt->vcr;
}

void MPID_PSP_comm_set_local_vcrt(MPID_Comm *comm, MPIDI_VCRT_t *vcrt)
{
       assert(vcrt);

       comm->local_vcrt = vcrt;
       comm->local_vcr  = vcrt->vcr;
}
