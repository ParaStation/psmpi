/*
 * ParaStation
 *
 * Copyright (C) 2008-2019 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include "mpid_coll.h"
#include "mpid_psp_packed_msg.h"
#include <assert.h>

// This must be the last include before sysmbols are defined:
#include "mpid_visibility.h"


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
int MPID_PSP_Barrier(MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag)
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
			MPIR_Comm *comm_ptr)
{
	MPID_PSP_packed_msg_t msg;
	int ret, buffered;
#ifdef USE_POST_BCAST
	pscom_request_t *req;
#endif

#ifdef MPID_PSP_WITH_CUDA_AWARENESS
	buffered = pscom_is_gpu_mem(buffer);
	ret = MPID_PSP_packed_msg_prepare(buffer, count, datatype, &msg, buffered);
#else
	ret = MPID_PSP_packed_msg_prepare(buffer, count, datatype, &msg, 0);
#endif

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
			MPIR_Comm *comm_ptr)
{
	MPID_PSP_packed_msg_t msg;
	int ret, buffered;
#ifdef USE_POST_BCAST
	pscom_request_t *req;
#endif

#ifdef MPID_PSP_WITH_CUDA_AWARENESS
	buffered = pscom_is_gpu_mem(buffer);
	ret = MPID_PSP_packed_msg_prepare(buffer, count, datatype, &msg, buffered);
#else
	ret = MPID_PSP_packed_msg_prepare(buffer, count, datatype, &msg, 0);
#endif

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
		   MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag)
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

void MPID_PSP_group_init(MPIR_Comm *comm_ptr)
{
	unsigned comm_size = comm_ptr->local_size;
	unsigned rank;
	pscom_group_t *group;
	pscom_connection_t **connections;

	pscom_socket_t *sock;
	unsigned group_id = comm_ptr->context_id;
	pscom_request_t *req;

// TODO: use new collective interface
//	comm_ptr->coll_fns = &mpid_psp_collective_functions;

	connections = MPL_malloc(comm_size * sizeof(*connections), MPL_MEM_BUFFER);
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

	MPL_free(connections); connections = NULL;
}


void MPID_PSP_group_cleanup(MPIR_Comm *comm_ptr)
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

////////////////////////////////////////////////////////////////////////////////////////////
#ifdef MPID_PSP_WITH_CUDA_AWARENESS
////////////////////////////////////////////////////////////////////////////////////////////

int MPID_PSP_Reduce_for_cuda(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
			     MPI_Op op, int root, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
	int             rc;
	int	        contig;
	unsigned int    data_sz = 0;
	MPIR_Datatype  *dtp;
	MPI_Aint        true_lb;
	char *tmp_sendbuf = NULL;
	char *tmp_recvbuf = NULL;
	char *sendptr = (char*)sendbuf;
	char *recvptr = (char*)recvbuf;

	if(pscom_is_gpu_mem(sendbuf)) {

		MPIDI_Datatype_get_info(count, datatype, contig, data_sz, dtp, true_lb);

		tmp_sendbuf = MPL_malloc(data_sz, MPL_MEM_BUFFER);
		MPID_Memcpy(tmp_sendbuf, sendbuf, data_sz);
		sendptr = tmp_sendbuf;
	}

	if( (comm_ptr->rank == root) && (pscom_is_gpu_mem(recvbuf)) ) {

		if(!tmp_sendbuf) {
			MPIDI_Datatype_get_info(count, datatype, contig, data_sz, dtp, true_lb);
		}

		tmp_recvbuf = MPL_malloc(data_sz, MPL_MEM_BUFFER);
		MPID_Memcpy(tmp_recvbuf, recvbuf, data_sz);
		recvptr = tmp_recvbuf;
	}

	rc = MPIR_Reduce_impl(sendptr, recvptr, count, datatype, op, root, comm_ptr, errflag);

	if(tmp_sendbuf) {
		MPL_free(tmp_sendbuf);
	}

	if(tmp_recvbuf) {
		MPID_Memcpy(recvbuf, tmp_recvbuf, data_sz);
		MPL_free(tmp_recvbuf);
	}

	return rc;
}

int MPID_PSP_Allreduce_for_cuda(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
				MPI_Op op, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
	int             rc;
	int	        contig;
	unsigned int    data_sz = 0;
	MPIR_Datatype  *dtp;
	MPI_Aint        true_lb;
	char *tmp_sendbuf = NULL;
	char *tmp_recvbuf = NULL;
	char *sendptr = (char*)sendbuf;
	char *recvptr = (char*)recvbuf;

	if(pscom_is_gpu_mem(sendbuf)) {

		MPIDI_Datatype_get_info(count, datatype, contig, data_sz, dtp, true_lb);

		tmp_sendbuf = MPL_malloc(data_sz, MPL_MEM_BUFFER);
		MPID_Memcpy(tmp_sendbuf, sendbuf, data_sz);
		sendptr = tmp_sendbuf;
	}

	if(pscom_is_gpu_mem(recvbuf)) {

		if(!tmp_sendbuf) {
			MPIDI_Datatype_get_info(count, datatype, contig, data_sz, dtp, true_lb);
		}

		tmp_recvbuf = MPL_malloc(data_sz, MPL_MEM_BUFFER);
		MPID_Memcpy(tmp_recvbuf, recvbuf, data_sz);
		recvptr = tmp_recvbuf;
	}

	rc = MPIR_Allreduce_impl(sendptr, recvptr, count, datatype, op, comm_ptr, errflag);

	if(tmp_sendbuf) {
		MPL_free(tmp_sendbuf);
	}

	if(tmp_recvbuf) {
		MPID_Memcpy(recvbuf, tmp_recvbuf, data_sz);
		MPL_free(tmp_recvbuf);
	}

	return rc;
}

int MPID_PSP_Reduce_scatter_for_cuda(const void *sendbuf, void *recvbuf, const int recvcounts[],
				     MPI_Datatype datatype, MPI_Op op, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
	int             rc, i, j;
	int	        contig;
	unsigned int    sdata_sz = 0;
	unsigned int    rdata_sz = 0;
	MPIR_Datatype  *dtp;
	MPI_Aint        true_lb;
	char *tmp_sendbuf = NULL;
	char *tmp_recvbuf = NULL;
	char *sendptr = (char*)sendbuf;
	char *recvptr = (char*)recvbuf;

	if(pscom_is_gpu_mem(sendbuf)) {

		for(i=0,j=0; i< comm_ptr->local_size; i++) j += recvcounts[i];
		MPIDI_Datatype_get_info(j, datatype, contig, sdata_sz, dtp, true_lb);

		tmp_sendbuf = MPL_malloc(sdata_sz, MPL_MEM_BUFFER);
		MPID_Memcpy(tmp_sendbuf, sendbuf, sdata_sz);
		sendptr = tmp_sendbuf;
	}

	if(pscom_is_gpu_mem(recvbuf)) {

		MPIDI_Datatype_get_info(recvcounts[comm_ptr->rank], datatype, contig, rdata_sz, dtp, true_lb);

		if(sendbuf == MPI_IN_PLACE) {

			for(i=0,j=0; i< comm_ptr->local_size; i++) j += recvcounts[i];
			MPIDI_Datatype_get_info(j, datatype, contig, sdata_sz, dtp, true_lb);

			tmp_recvbuf = MPL_malloc(sdata_sz, MPL_MEM_BUFFER);
			MPID_Memcpy(tmp_recvbuf, recvbuf, sdata_sz);

		} else {

			tmp_recvbuf = MPL_malloc(rdata_sz, MPL_MEM_BUFFER);
			MPID_Memcpy(tmp_recvbuf, recvbuf, rdata_sz);
		}

		recvptr = tmp_recvbuf;
	}

	rc = MPIR_Reduce_scatter_impl(sendptr, recvptr, recvcounts, datatype, op, comm_ptr, errflag);

	if(tmp_sendbuf) {
		MPL_free(tmp_sendbuf);
	}

	if(tmp_recvbuf) {
		MPID_Memcpy(recvbuf, tmp_recvbuf, rdata_sz);
		MPL_free(tmp_recvbuf);
	}

	return rc;
}


int MPID_PSP_Reduce_scatter_block_for_cuda(const void *sendbuf, void *recvbuf,  int recvcount,
					   MPI_Datatype datatype, MPI_Op op, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
	int             rc;
	int	        contig;
	unsigned int    data_sz = 0;
	MPIR_Datatype  *dtp;
	MPI_Aint        true_lb;
	char *tmp_sendbuf = NULL;
	char *tmp_recvbuf = NULL;
	char *sendptr = (char*)sendbuf;
	char *recvptr = (char*)recvbuf;

	if(pscom_is_gpu_mem(sendbuf)) {

		MPIDI_Datatype_get_info(recvcount, datatype, contig, data_sz, dtp, true_lb);

		tmp_sendbuf = MPL_malloc(data_sz * comm_ptr->local_size, MPL_MEM_BUFFER);
		MPID_Memcpy(tmp_sendbuf, sendbuf, data_sz * comm_ptr->local_size);
		sendptr = tmp_sendbuf;
	}

	if(pscom_is_gpu_mem(recvbuf)) {

		if(!tmp_sendbuf) {
			MPIDI_Datatype_get_info(recvcount, datatype, contig, data_sz, dtp, true_lb);
		}

		if(sendbuf == MPI_IN_PLACE) {
			data_sz *= comm_ptr->local_size;
		}

		tmp_recvbuf = MPL_malloc(data_sz, MPL_MEM_BUFFER);
		MPID_Memcpy(tmp_recvbuf, recvbuf, data_sz);
		recvptr = tmp_recvbuf;
	}

	rc = MPIR_Reduce_scatter_block_impl(sendptr, recvptr, recvcount, datatype, op, comm_ptr, errflag);

	if(tmp_sendbuf) {
		MPL_free(tmp_sendbuf);
	}

	if(tmp_recvbuf) {
		MPID_Memcpy(recvbuf, tmp_recvbuf, data_sz);
		MPL_free(tmp_recvbuf);
	}

	return rc;
}

static
int MPID_PSP_Scan_generic_for_cuda(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
				   MPI_Op op, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag, int ex_flag)
{
	int             rc;
	int	        contig;
	unsigned int    data_sz = 0;
	MPIR_Datatype  *dtp;
	MPI_Aint        true_lb;
	char *tmp_sendbuf = NULL;
	char *tmp_recvbuf = NULL;
	char *sendptr = (char*)sendbuf;
	char *recvptr = (char*)recvbuf;

	if(pscom_is_gpu_mem(sendbuf)) {

		MPIDI_Datatype_get_info(count, datatype, contig, data_sz, dtp, true_lb);

		tmp_sendbuf = MPL_malloc(data_sz, MPL_MEM_BUFFER);
		MPID_Memcpy(tmp_sendbuf, sendbuf, data_sz);
		sendptr = tmp_sendbuf;
	}

	if(pscom_is_gpu_mem(recvbuf)) {

		if(!tmp_sendbuf) {
			MPIDI_Datatype_get_info(count, datatype, contig, data_sz, dtp, true_lb);
		}

		tmp_recvbuf = MPL_malloc(data_sz, MPL_MEM_BUFFER);
		MPID_Memcpy(tmp_recvbuf, recvbuf, data_sz);
		recvptr = tmp_recvbuf;
	}

	if(!ex_flag){
		rc = MPIR_Scan_impl(sendptr, recvptr, count, datatype, op, comm_ptr, errflag);
	} else {
		rc = MPIR_Exscan_impl(sendptr, recvptr, count, datatype, op, comm_ptr, errflag);
	}

	if(tmp_sendbuf) {
		MPL_free(tmp_sendbuf);
	}

	if(tmp_recvbuf) {
		MPID_Memcpy(recvbuf, tmp_recvbuf, data_sz);
		MPL_free(tmp_recvbuf);
	}

	return rc;
}

int MPID_PSP_Scan_for_cuda(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
			   MPI_Op op, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
	return MPID_PSP_Scan_generic_for_cuda(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag, 0);
}

int MPID_PSP_Exscan_for_cuda(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
			   MPI_Op op, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag)
{
	return MPID_PSP_Scan_generic_for_cuda(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag, 1);
}

int MPID_PSP_Reduce_local_for_cuda(const void *inbuf, void *inoutbuf, int count, MPI_Datatype datatype, MPI_Op op)
{
	int             rc;
	int	        contig;
	unsigned int    data_sz = 0;
	MPIR_Datatype  *dtp;
	MPI_Aint        true_lb;
	char *tmp_inbuf = NULL;
	char *tmp_inoutbuf = NULL;
	char *inptr = (char*)inbuf;
	char *inoutptr = (char*)inoutbuf;

	if(pscom_is_gpu_mem(inbuf)) {

		MPIDI_Datatype_get_info(count, datatype, contig, data_sz, dtp, true_lb);

		tmp_inbuf = MPL_malloc(data_sz, MPL_MEM_BUFFER);
		MPID_Memcpy(tmp_inbuf, inbuf, data_sz);
		inptr = tmp_inbuf;
	}

	if(pscom_is_gpu_mem(inoutbuf)) {

		if(!tmp_inbuf) {
			MPIDI_Datatype_get_info(count, datatype, contig, data_sz, dtp, true_lb);
		}

		tmp_inoutbuf = MPL_malloc(data_sz, MPL_MEM_BUFFER);
		MPID_Memcpy(tmp_inoutbuf, inoutbuf, data_sz);
		inoutptr = tmp_inoutbuf;
	}

	rc = MPIR_Reduce_local_impl(inptr, inoutptr, count, datatype, op);

	if(tmp_inbuf) {
		MPL_free(tmp_inbuf);
	}

	if(tmp_inoutbuf) {
		MPID_Memcpy(inoutbuf, tmp_inoutbuf, data_sz);
		MPL_free(tmp_inoutbuf);
	}

	return rc;
}
////////////////////////////////////////////////////////////////////////////////////////////
#endif
////////////////////////////////////////////////////////////////////////////////////////////
