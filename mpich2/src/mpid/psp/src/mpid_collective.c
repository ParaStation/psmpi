/*
 * ParaStation
 *
 * Copyright (C) 2008-2010 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include "mpid_collective.h"
#include "mpid_psp_packed_msg.h"
#include <assert.h>

#if 1
#define D(cmd)
#else
#define D(cmd) cmd
#endif

/* define USE_POST_BCAST to use pscom_post_bcast instead of pscom_bcast.
 *  1) pscom_group_bcast.c: #define USE_ASYNCHRONOUS_BCAST 1
 *      pscom_post_bcast is 0.15usec faster (imb, sharedmem, x86_64...), because it
 *      use the preallocated request comm_ptr->bcast_request.
 *  2) pscom_group_bcast.c: #undef USE_ASYNCHRONOUS_BCAST
 *      pscom_bcast is 0.05usec faster, because pscom_post_bcast use pscom_bcast()
*/
/* #define USE_POST_BCAST 1 */


/* not declared static because it is called in ch3_comm_connect/accept */
static
int MPID_PSP_Barrier(MPID_Comm *comm_ptr, int *errflag)
{
	if (comm_ptr->group) {
		pscom_barrier(comm_ptr->group);
		return MPI_SUCCESS;
	} else {
		/* Fallback to MPIch default Barrier */
		return MPIR_Barrier(comm_ptr, errflag);
	}
}


static
int MPID_PSP_Bcast_send(void *buffer, int count, MPI_Datatype datatype, int root,
			MPID_Comm *comm_ptr)
{
	MPID_PSP_packed_msg_t msg;
	int ret;
#ifdef USE_POST_BCAST
	pscom_request_t *req;
#endif

	ret = MPID_PSP_packed_msg_prepare(buffer, count, datatype, &msg);
	if (unlikely(ret != MPI_SUCCESS)) goto err_create_packed_msg;

	MPID_PSP_packed_msg_pack(buffer, count, datatype, &msg);

#ifdef USE_POST_BCAST
	req = comm_ptr->bcast_request;
	assert(req);

	req->xheader.bcast.bcast_root = root;
	req->data = msg.msg;
	req->data_len = msg.msg_sz;

	pscom_post_bcast(req);
	MPID_PSP_LOCKFREE_CALL(pscom_wait(req));

#else
	MPID_PSP_LOCKFREE_CALL(
		pscom_bcast(comm_ptr->group, root,
			    NULL, 0, msg.msg, msg.msg_sz));
#endif

	MPID_PSP_packed_msg_cleanup(&msg);
	D(printf("pscom_usage %s\n", __func__);)

	return MPI_SUCCESS;
	/* --- */
err_create_packed_msg:
	return ret;
}


static
int MPID_PSP_Bcast_recv(void *buffer, int count, MPI_Datatype datatype, int root,
			MPID_Comm *comm_ptr)
{
	MPID_PSP_packed_msg_t msg;
	int ret;
#ifdef USE_POST_BCAST
	pscom_request_t *req;
#endif

	ret = MPID_PSP_packed_msg_prepare(buffer, count, datatype, &msg);
	if (unlikely(ret != MPI_SUCCESS)) goto err_create_packed_msg;

#ifdef USE_POST_BCAST
	req = comm_ptr->bcast_request;
	assert(req);

	req->xheader.bcast.bcast_root = root;
	req->data = msg.msg;
	req->data_len = msg.msg_sz;

	pscom_post_bcast(req);
	MPID_PSP_LOCKFREE_CALL(pscom_wait(req));
#else
	MPID_PSP_LOCKFREE_CALL(
		pscom_bcast(comm_ptr->group, root,
			    NULL, 0, msg.msg, msg.msg_sz));
#endif

	MPID_PSP_packed_msg_unpack(buffer, count, datatype,
				   &msg, msg.msg_sz);

	MPID_PSP_packed_msg_cleanup(&msg);

	D(printf("pscom_usage %s\n", __func__);)

	return MPI_SUCCESS;
	/* --- */
err_create_packed_msg:
	return ret;
}


/*
MPI_Bcast - Broadcasts a message from the process with rank "root" to
	    all other processes of the communicator
*/
static
int MPID_PSP_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
		   MPID_Comm *comm_ptr, int *errflag)
{
	int mpi_errno;

	D(printf("%s(buffer:%p, count:%u, root:%u, comm:%p(%s, rank:%2u, id:%x, rid:%x, size:%u))\n",
		 __func__, buffer, count, root, comm_ptr, comm_ptr->name,
		 comm_ptr->rank, comm_ptr->context_id, comm_ptr->recvcontext_id,
		 comm_ptr->local_size);
		)

	if (!comm_ptr->group) {
		/* Fallback to MPIch default Bcast */
		mpi_errno = MPIR_Bcast(buffer, count, datatype, root, comm_ptr, errflag);
		return mpi_errno;
	}

	if (root == comm_ptr->rank) {
		/* I am the root */
		return MPID_PSP_Bcast_send(buffer, count, datatype, root, comm_ptr);
	} else {
		return MPID_PSP_Bcast_recv(buffer, count, datatype, root, comm_ptr);
	}
}


static
void group_init(MPID_Comm *comm_ptr)
{
	unsigned comm_size = comm_ptr->local_size;
	unsigned rank;
	pscom_group_t *group;
	pscom_connection_t **connections;

	pscom_socket_t *sock;
	unsigned group_id = comm_ptr->context_id;
	pscom_request_t *req;

	connections = MPIU_Malloc(comm_size * sizeof(*connections));
	assert(connections);

	for (rank = 0; rank < comm_size; rank++) {
		connections[rank] = MPID_PSCOM_rank2connection(comm_ptr, rank);
		assert(connections[rank]);
	}

	sock = comm_ptr->pscom_socket;

	if (sock) {
		/* define pscom group */
		group = pscom_group_open(sock,
					 group_id, comm_ptr->rank,
					 comm_size, connections);
		assert(group);
	} else {
		group = NULL;
	}
	assert(!comm_ptr->group);
	comm_ptr->group = group;

	/* prepare the bcast_request */
	req =  pscom_request_create(sizeof(req->xheader.bcast), 0);

	req->xheader_len = sizeof(req->xheader.bcast);
	req->xheader.bcast.group_id = group_id;
	req->xheader.bcast.bcast_root = -1;

	req->data_len = 0;
	req->data = NULL;
	req->socket = sock;

	comm_ptr->bcast_request = req;

	MPIU_Free(connections); connections = NULL;
}


static
void group_cleanup(MPID_Comm *comm_ptr)
{
	if (comm_ptr->group) {
		pscom_group_close(comm_ptr->group);
		comm_ptr->group = NULL;
	}
	if (comm_ptr->bcast_request) {
		pscom_request_free(comm_ptr->bcast_request);
		comm_ptr->bcast_request = NULL;
	}
}


static
MPID_Collops mpid_psp_collective_functions = {
	~0,    /* ref_count */
	&MPID_PSP_Barrier, /* Barrier */
	&MPID_PSP_Bcast, /* Bcast */
	NULL, /* Gather */
	NULL, /* Gatherv */
	NULL, /* Scatter */
	NULL, /* Scatterv */
	NULL, /* Allgather */
	NULL, /* Allgatherv */
	NULL, /* Alltoall */
	NULL, /* Alltoallv */
	NULL, /* Alltoallw */
	NULL, /* Reduce */
	NULL, /* Allreduce */
	NULL, /* Reduce_scatter */
	NULL, /* Scan */
	NULL, /* Exscan */
	NULL  /* Reduce_scatter_block */
};


int MPID_PSP_comm_create_hook(MPID_Comm * comm)
{
	pscom_connection_t *con1st;
	int i;
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

	comm->coll_fns = &mpid_psp_collective_functions;

	group_init(comm);

	D(printf("%s (comm:%p(%s, id:%08x, size:%u))\n",
		 __func__, comm, comm->name, comm->context_id, comm->local_size););
	return MPI_SUCCESS;
}


int MPID_PSP_comm_destroy_hook(MPID_Comm * comm)
{
	if (!MPIDI_Process.env.enable_collectives) return MPI_SUCCESS;

	/* ToDo: Use comm Barrier before cleanup! */

	group_cleanup(comm);

	D(printf("%s\n", __func__););
	return MPI_SUCCESS;
}
