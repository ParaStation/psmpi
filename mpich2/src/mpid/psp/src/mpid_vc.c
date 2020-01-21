/*
 * ParaStation
 *
 * Copyright (C) 2006-2020 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include <assert.h>
#include "mpidimpl.h"

// This must be the last include before sysmbols are defined:
#include "mpid_visibility.h"


static
int MPIDI_VCR_DeleteFromPG(MPIDI_VC_t *vcr);

static
MPIDI_VC_t *new_VCR(MPIDI_PG_t * pg, int pg_rank, pscom_connection_t *con, int lpid)
{
	MPIDI_VC_t *vcr = MPL_malloc(sizeof(*vcr), MPL_MEM_OTHER);
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

	if(isDisconnect && (vcr->refcnt == 1)) {

		MPIDI_VCR_DeleteFromPG(vcr);

		if(!MPIDI_Process.env.enable_lazy_disconnect) {
			/* Finally, tear down this connection: */
			pscom_close_connection(vcr->con);
		}

		MPL_free(vcr);
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

	vcrt = MPL_malloc(sizeof(MPIDI_VCRT_t) + size * sizeof(MPIDI_VC_t), MPL_MEM_OTHER);

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

	MPL_free(vcrt);
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

	if(!MPIDI_Process.env.enable_lazy_disconnect) {
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


#undef FUNCNAME
#define FUNCNAME MPIDI_LPID_GetAllInComm
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static inline int MPIDI_LPID_GetAllInComm(MPIR_Comm *comm_ptr, int local_size,
                                          int local_lpids[])
{
    int i;
    int mpi_errno = MPI_SUCCESS;
    MPIR_Assert( comm_ptr->local_size == local_size );
    for (i=0; i<comm_ptr->local_size; i++) {
		mpi_errno |= MPID_Comm_get_lpid( comm_ptr, i, &local_lpids[i], FALSE );
    }
    return mpi_errno;
}

/*@
  MPID_Intercomm_exchange_map - Exchange address mapping for intercomm creation.
 @*/
#undef FUNCNAME
#define FUNCNAME MPID_Intercomm_exchange_map
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPID_Intercomm_exchange_map(MPIR_Comm *local_comm_ptr, int local_leader,
                                MPIR_Comm *peer_comm_ptr, int remote_leader,
                                int *remote_size, int **remote_lpids,
                                int *is_low_group)
{
    int mpi_errno = MPI_SUCCESS;
    int singlePG;
    int local_size,*local_lpids=0;
    MPIDI_Gpid *local_gpids=NULL, *remote_gpids=NULL;
    int comm_info[2];
    int cts_tag;
    MPIR_Errflag_t errflag = MPIR_ERR_NONE;
    MPIR_CHKLMEM_DECL(3);

    cts_tag = 0 | MPIR_TAG_COLL_BIT;

    if (local_comm_ptr->rank == local_leader) {

        /* First, exchange the group information.  If we were certain
           that the groups were disjoint, we could exchange possible
           context ids at the same time, saving one communication.
           But experience has shown that that is a risky assumption.
        */
        /* Exchange information with my peer.  Use sendrecv */

        local_size = local_comm_ptr->local_size;

        mpi_errno = MPIC_Sendrecv( &local_size,  1, MPI_INT,
                                      remote_leader, cts_tag,
                                      remote_size, 1, MPI_INT,
                                      remote_leader, cts_tag,
                                      peer_comm_ptr, MPI_STATUS_IGNORE, &errflag );
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);

        /* With this information, we can now send and receive the
           global process ids from the peer. */
        MPIR_CHKLMEM_MALLOC(remote_gpids,MPIDI_Gpid*,(*remote_size)*sizeof(MPIDI_Gpid), mpi_errno,"remote_gpids", MPL_MEM_DYNAMIC);
        *remote_lpids = (int*) MPL_malloc((*remote_size)*sizeof(int), MPL_MEM_ADDRESS);
        MPIR_CHKLMEM_MALLOC(local_gpids,MPIDI_Gpid*,local_size*sizeof(MPIDI_Gpid), mpi_errno,"local_gpids", MPL_MEM_DYNAMIC);
        MPIR_CHKLMEM_MALLOC(local_lpids,int*,local_size*sizeof(int), mpi_errno,"local_lpids", MPL_MEM_DYNAMIC);

        mpi_errno = MPIDI_GPID_GetAllInComm( local_comm_ptr, local_size, local_gpids, &singlePG );
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);

        /* Exchange the lpid arrays */
        mpi_errno = MPIC_Sendrecv( local_gpids, local_size*sizeof(MPIDI_Gpid), MPI_BYTE,
                                      remote_leader, cts_tag,
                                      remote_gpids, (*remote_size)*sizeof(MPIDI_Gpid), MPI_BYTE,
                                      remote_leader, cts_tag, peer_comm_ptr,
                                      MPI_STATUS_IGNORE, &errflag );
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);

        /* Convert the remote gpids to the lpids */
        mpi_errno = MPIDI_GPID_ToLpidArray( *remote_size, remote_gpids, *remote_lpids );
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);

        /* Get our own lpids */
        mpi_errno = MPIDI_LPID_GetAllInComm( local_comm_ptr, local_size, local_lpids );
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);

        /* Make an arbitrary decision about which group of processs is
		   the low group.  The LEADERS do this by comparing the
		   local gpids of the 0th member of the two groups. If these match,
		   they fall back to the rank ID within that gpid */
		if (local_gpids[0].gpid[0] == remote_gpids[0].gpid[0]) {
			(*is_low_group) = local_gpids[0].gpid[1] < remote_gpids[0].gpid[1];
		} else {
			(*is_low_group) = local_gpids[0].gpid[0] < remote_gpids[0].gpid[0];
		}

        /* At this point, we're done with the local lpids; they'll
           be freed with the other local memory on exit */

    } /* End of the first phase of the leader communication */
    /* Leaders can now swap context ids and then broadcast the value
       to the local group of processes */
    if (local_comm_ptr->rank == local_leader) {
        /* Now, send all of our local processes the remote_lpids,
           along with the final context id */
        comm_info[0] = *remote_size;
        comm_info[1] = *is_low_group;
        mpi_errno = MPIR_Bcast( comm_info, 2, MPI_INT, local_leader, local_comm_ptr, &errflag );
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");
        mpi_errno = MPIR_Bcast( remote_gpids, (*remote_size)*sizeof(MPIDI_Gpid), MPI_BYTE, local_leader,
                                     local_comm_ptr, &errflag );
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");
    }
    else
    {
        /* we're the other processes */
        mpi_errno = MPIR_Bcast( comm_info, 2, MPI_INT, local_leader, local_comm_ptr, &errflag );
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");
        *remote_size = comm_info[0];
        MPIR_CHKLMEM_MALLOC(remote_gpids,MPIDI_Gpid*,(*remote_size)*sizeof(MPIDI_Gpid), mpi_errno,"remote_gpids", MPL_MEM_DYNAMIC);
        *remote_lpids = (int*) MPL_malloc((*remote_size)*sizeof(int), MPL_MEM_ADDRESS);
        mpi_errno = MPIR_Bcast( remote_gpids, (*remote_size)*sizeof(MPIDI_Gpid), MPI_BYTE, local_leader,
                                     local_comm_ptr, &errflag );
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        MPIR_ERR_CHKANDJUMP(errflag, mpi_errno, MPI_ERR_OTHER, "**coll_fail");

        /* Extract the context and group sign informatin */
        *is_low_group     = comm_info[1];
    }

    /* Finish up by giving the device the opportunity to update
       any other infomration among these processes.  Note that the
       new intercomm has not been set up; in fact, we haven't yet
       attempted to set up the connection tables.

       In the case of the ch3 device, this calls MPID_PG_ForwardPGInfo
       to ensure that all processes have the information about all
       process groups.  This must be done before the call
       to MPID_GPID_ToLpidArray, as that call needs to know about
       all of the process groups.
    */
    MPID_ICCREATE_REMOTECOMM_HOOK( peer_comm_ptr, local_comm_ptr,
                            *remote_size, (const MPIDI_Gpid*)remote_gpids, local_leader );


    /* Finally, if we are not the local leader, we need to
       convert the remote gpids to local pids.  This must be done
       after we allow the device to handle any steps that it needs to
       take to ensure that all processes contain the necessary process
       group information */
    if (local_comm_ptr->rank != local_leader) {
        mpi_errno = MPIDI_GPID_ToLpidArray( *remote_size, remote_gpids, *remote_lpids );
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }

fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
fn_fail:
    goto fn_exit;
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
	if (mapper->type == MPIR_COMM_MAP_TYPE__DUP && src_comm_size == vcrt_size) {
		*dest_vcrt = MPIDI_VCRT_Dup(src_vcrt);
		return;
	}
	else if (mapper->type == MPIR_COMM_MAP_TYPE__IRREGULAR &&
		 mapper->src_mapping_size == vcrt_size) {
		/* if the mapping array is exactly the same as the original
		 * comm's VC list, there is no need to create a new VCRT.
		 * instead simply point to the original comm's VCRT and bump
		 * up it's reference count */
		flag = 1;
		for (i = 0; i < mapper->src_mapping_size; i++) {
			if (mapper->src_mapping[i] != i) {
				flag = 0;
				break;
			}
		}

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

	if (mapper->type == MPIR_COMM_MAP_TYPE__DUP) {
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
	if (mapper->type == MPIR_COMM_MAP_TYPE__IRREGULAR) {
		return mapper->src_mapping_size;
	} else if (mapper->dir == MPIR_COMM_MAP_DIR__L2L || mapper->dir == MPIR_COMM_MAP_DIR__L2R) {
		return mapper->src_comm->local_size;
	} else {
		assert(mapper->dir == MPIR_COMM_MAP_DIR__R2L || mapper->dir == MPIR_COMM_MAP_DIR__R2R);
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
	LL_FOREACH(mapper_head, mapper) {
		if (mapper->dir == MPIR_COMM_MAP_DIR__L2L ||
		    mapper->dir == MPIR_COMM_MAP_DIR__R2L) {
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
	LL_FOREACH(mapper_head, mapper) {
		if (mapper->dir == MPIR_COMM_MAP_DIR__L2R ||
		    mapper->dir == MPIR_COMM_MAP_DIR__R2R) {
			size += MPID_PSP_mapper_size(mapper);
		}
	}
	return size;
}


static
void MPID_PSP_mapper_list_map_local_vcr(MPIR_Comm *comm, int vcrt_size)
{
	MPIR_Comm *src_comm;
	MPIR_Comm_map_t *mapper;
	int vcrt_offset = 0;

	LL_FOREACH(comm->mapper_head, mapper) {
		src_comm = mapper->src_comm;

		switch(mapper->dir) {

		case MPIR_COMM_MAP_DIR__L2R:
		case MPIR_COMM_MAP_DIR__R2R:
			break;

		case MPIR_COMM_MAP_DIR__L2L:
			if (src_comm->comm_kind == MPIR_COMM_KIND__INTRACOMM && comm->comm_kind == MPIR_COMM_KIND__INTRACOMM) {
				MPID_PSP_mapper_dup_vcrt(src_comm->vcrt, &comm->vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			}
			else if (src_comm->comm_kind == MPIR_COMM_KIND__INTRACOMM && comm->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
				MPID_PSP_mapper_dup_vcrt(src_comm->vcrt, &comm->local_vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_local_vcrt(comm, comm->local_vcrt);
			}
			else if (src_comm->comm_kind == MPIR_COMM_KIND__INTERCOMM && comm->comm_kind == MPIR_COMM_KIND__INTRACOMM) {
				MPID_PSP_mapper_dup_vcrt(src_comm->local_vcrt, &comm->vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			}
			else {
				MPID_PSP_mapper_dup_vcrt(src_comm->local_vcrt, &comm->local_vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_local_vcrt(comm, comm->local_vcrt);
			}
			vcrt_offset += MPID_PSP_mapper_size(mapper);
			break;

		case MPIR_COMM_MAP_DIR__R2L:
			MPIR_Assert(src_comm->comm_kind == MPIR_COMM_KIND__INTERCOMM);
			if (comm->comm_kind == MPIR_COMM_KIND__INTRACOMM) {
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
void MPID_PSP_mapper_list_map_remote_vcr(MPIR_Comm *comm, int vcrt_size)
{
	MPIR_Comm *src_comm;
	MPIR_Comm_map_t *mapper;
	int vcrt_offset = 0;

	LL_FOREACH(comm->mapper_head, mapper) {
		src_comm = mapper->src_comm;

		switch(mapper->dir) {

		case MPIR_COMM_MAP_DIR__L2L:
		case MPIR_COMM_MAP_DIR__R2L:
			break;

		case MPIR_COMM_MAP_DIR__L2R:
			MPIR_Assert(comm->comm_kind == MPIR_COMM_KIND__INTERCOMM);
			if (src_comm->comm_kind == MPIR_COMM_KIND__INTRACOMM) {
				MPID_PSP_mapper_dup_vcrt(src_comm->vcrt, &comm->vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			}
			else {
				MPID_PSP_mapper_dup_vcrt(src_comm->local_vcrt, &comm->vcrt, mapper, mapper->src_comm->local_size, vcrt_size, vcrt_offset);
				MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			}
			vcrt_offset += MPID_PSP_mapper_size(mapper);
			break;

		case MPIR_COMM_MAP_DIR__R2R:
			MPIR_Assert(comm->comm_kind == MPIR_COMM_KIND__INTERCOMM);
			MPIR_Assert(src_comm->comm_kind == MPIR_COMM_KIND__INTERCOMM);
			MPID_PSP_mapper_dup_vcrt(src_comm->vcrt, &comm->vcrt, mapper, mapper->src_comm->remote_size, vcrt_size, vcrt_offset);
			MPID_PSP_comm_set_vcrt(comm, comm->vcrt);
			vcrt_offset +=MPID_PSP_mapper_size(mapper);
			break;

		default: assert(0);
		}
	}
}


void MPID_PSP_comm_create_mapper(MPIR_Comm * comm)
{
	int vcrt_size;

	comm->is_disconnected = 0;

	if(!comm->mapper_head) return;

	vcrt_size = MPID_PSP_mapper_list_dest_local_size(comm->mapper_head);
	MPID_PSP_mapper_list_map_local_vcr(comm, vcrt_size);

	vcrt_size = MPID_PSP_mapper_list_dest_remote_size(comm->mapper_head);
	MPID_PSP_mapper_list_map_remote_vcr(comm, vcrt_size);

	if (comm->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
		/* setup the vcrt for the local_comm in the intercomm */
		if (comm->local_comm) {
			MPIDI_VCRT_t *vcrt;
			vcrt = MPIDI_VCRT_Dup(comm->local_vcrt);
			assert(vcrt);
			MPID_PSP_comm_set_vcrt(comm->local_comm, vcrt);
		}
	}
}


void MPID_PSP_comm_set_vcrt(MPIR_Comm *comm, MPIDI_VCRT_t *vcrt)
{
       assert(vcrt);

       comm->vcrt = vcrt;
       comm->vcr  = vcrt->vcr;
}

void MPID_PSP_comm_set_local_vcrt(MPIR_Comm *comm, MPIDI_VCRT_t *vcrt)
{
       assert(vcrt);

       comm->local_vcrt = vcrt;
       comm->local_vcr  = vcrt->vcr;
}
