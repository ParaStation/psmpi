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

#define _GNU_SOURCE 1
#include <pthread.h>
#include <sys/types.h>
#include <sys/shm.h>

#include "mpidimpl.h"
#include "mpid_psp_datatype.h"
#include "mpid_psp_packed_msg.h"
#include "mpid_psp_request.h"

static
int _accept_never(pscom_request_t *request,
		  pscom_connection_t *connection,
		  pscom_header_net_t *header_net)
{
	return 0;
}

static
int is_comm_self_clone(MPID_Comm * comm_ptr)
{
	MPID_Comm *comm_self_ptr = NULL;
	MPID_Group *group_comm, *group_self;
	int result;

	MPID_Comm_get_ptr( MPI_COMM_SELF, comm_self_ptr );

	MPIR_Comm_group_impl(comm_ptr, &group_comm);

	MPIR_Comm_group_impl(comm_self_ptr, &group_self);

	MPIR_Group_compare_impl(group_comm, group_self, &result);

	MPIR_Group_free_impl(group_comm);

	MPIR_Group_free_impl(group_self);

	if (result == MPI_IDENT) {
		return 1;
	}

	return 0;
}


static
void MPID_PSP_rma_init(pscom_socket_t *socket, pscom_request_t **req)
{
	assert(socket != NULL);

	pscom_request_t *dummy_req = *req;

	MPID_enable_receive_dispach(socket);

	if (!dummy_req) {
		/* Post a dummy ANY_SOURCE receive to listen on all
		   connections for incoming RMA messages. */
		dummy_req = pscom_request_create(0, 0);
		dummy_req->xheader_len = 0;
		dummy_req->data_len = 0;
		dummy_req->connection = NULL;
		dummy_req->socket = socket; /* ToDo: get socket from comm? */
		dummy_req->ops.recv_accept = _accept_never;
		pscom_post_recv(dummy_req);

		*req = dummy_req;
	}
}

static
void MPID_PSP_rma_fini(pscom_socket_t *socket, pscom_request_t **req)
{
	pscom_request_t *dummy_req = *req;

	if (dummy_req) {
		/* cancel and free the any_source dummy request */
		pscom_cancel_recv(dummy_req);
		assert(pscom_req_is_done(dummy_req));
		pscom_request_free(dummy_req);
		dummy_req = NULL;
		*req = dummy_req;
	}
}


/*
 * Handling of pscom dummy any source receives for all rma using pscom_socket_t's.
 */

struct rma_pscom_socket {
	struct list_head next;
	unsigned ref_cnt;
	pscom_socket_t *pscom_socket;
	pscom_request_t *req;
};


static
struct list_head rma_pscom_sockets = LIST_HEAD_INIT(rma_pscom_sockets);


static
struct rma_pscom_socket *rma_pscom_sockets_find(pscom_socket_t *socket) {
	struct list_head *pos;
	list_for_each(pos, &rma_pscom_sockets) {
		struct rma_pscom_socket *rma_sock  = list_entry(pos, struct rma_pscom_socket, next);
		if (rma_sock->pscom_socket == socket) return rma_sock;
	}
	return NULL;
}


static
struct rma_pscom_socket *rma_pscom_sockets_create(unsigned ref_cnt, pscom_socket_t *socket) {
	struct rma_pscom_socket *rma_sock = (struct rma_pscom_socket *) MPIU_Malloc(sizeof(*rma_sock));

	rma_sock->ref_cnt = ref_cnt;
	rma_sock->pscom_socket = socket;
	rma_sock->req = NULL;

	list_add(&rma_sock->next, &rma_pscom_sockets);

	/* Initialize RMA for this pscom socket */
	MPID_PSP_rma_init(socket, &rma_sock->req);

	return rma_sock;
}


static
void rma_pscom_sockets_destroy(struct rma_pscom_socket *rma_sock) {
	/* Cleanup RMA for this pscom socket */
	MPID_PSP_rma_fini(rma_sock->pscom_socket, &rma_sock->req);

	list_del(&rma_sock->next);

	MPIU_Free(rma_sock);
}


/* Add reference to socket. Return ref_cnt */
static
unsigned rma_pscom_socket_add_ref(pscom_socket_t *socket) {
	struct rma_pscom_socket *rma_sock = rma_pscom_sockets_find(socket);
	if (!rma_sock) rma_sock = rma_pscom_sockets_create(0, socket);

	return ++rma_sock->ref_cnt;
}


/* Delete a reference to socket. Return ref_cnt */
static
unsigned rma_pscom_socket_del_ref(pscom_socket_t *socket) {
	struct rma_pscom_socket *rma_sock = rma_pscom_sockets_find(socket);
	assert(rma_sock != NULL);
	if (--rma_sock->ref_cnt == 0) {
		rma_pscom_sockets_destroy(rma_sock);
		return 0;
	} else {
		return rma_sock->ref_cnt;
	}
}


void MPID_PSP_rma_pscom_sockets_cleanup(void) {
	struct list_head *pos, *next;
	list_for_each_safe(pos, next, &rma_pscom_sockets) {
		struct rma_pscom_socket *rma_sock  = list_entry(pos, struct rma_pscom_socket, next);
		rma_pscom_sockets_destroy(rma_sock);
	}
}


static
void MPID_PSP_rma_check_init(MPID_Comm *comm)
{
	int i;
	for (i = 0; i < comm->local_size; i++) {
		rma_pscom_socket_add_ref(MPID_PSCOM_rank2connection(comm, i)->socket);
	}
}


static
void MPID_PSP_rma_check_fini(MPID_Comm *comm)
{
	int i;
	for (i = 0; i < comm->local_size; i++) {
		rma_pscom_socket_del_ref(MPID_PSCOM_rank2connection(comm, i)->socket);
	}
}


typedef struct MPID_Wincreate_msg
{
	void *base;
	int disp_unit;
	MPID_Win *win_ptr;
} MPID_Wincreate_msg;


#define FUNCNAME MPID_Win_create
#define FCNAME "MPID_Win_create"
int MPID_Win_create(void *base, MPI_Aint size, int disp_unit, MPID_Info *info,
		    MPID_Comm *comm_ptr, MPID_Win **_win_ptr)
{
	/* from MPIDI_Win_create() */
	int mpi_errno = MPI_SUCCESS, i, comm_size, rank;
	int errflag = 0;
	MPID_Wincreate_msg *tmp_buf;
	MPID_Win *win_ptr;

	MPIU_CHKPMEM_DECL(7);
	MPIU_CHKLMEM_DECL(1);

	MPID_PSP_rma_check_init(comm_ptr);

	comm_size = comm_ptr->local_size;
	rank = comm_ptr->rank;

	win_ptr = (MPID_Win *)MPIU_Handle_obj_alloc( &MPID_Win_mem );
	MPIU_ERR_CHKANDJUMP(!win_ptr, mpi_errno, MPI_ERR_OTHER, "**nomem");

	(*_win_ptr) = win_ptr;

	win_ptr->base = base;
	win_ptr->size = size;
	win_ptr->disp_unit = disp_unit;
	win_ptr->start_group_ptr = NULL;
	win_ptr->attributes = NULL;
	win_ptr->create_flavor = MPI_WIN_FLAVOR_CREATE;
	win_ptr->model = MPI_WIN_SEPARATE;
/*
	win_ptr->rma_ops_list = NULL;
	win_ptr->lock_granted = 0;
	win_ptr->current_lock_type = MPID_LOCK_NONE;
	win_ptr->shared_lock_ref_cnt = 0;
	win_ptr->lock_queue = NULL;
	win_ptr->my_counter = 0;
	win_ptr->my_pt_rma_puts_accs = 0;
*/
	mpi_errno = MPIR_Comm_dup_impl(comm_ptr, &win_ptr->comm_ptr);

	if (mpi_errno) { MPIU_ERR_POP(mpi_errno); }

	/* allocate memory for the base addresses, disp_units, and
	   completion counters of all processes */
	MPIU_CHKPMEM_MALLOC(win_ptr->rank_info, MPID_Win_rank_info *, comm_size * sizeof(MPID_Win_rank_info),
			    mpi_errno, "win_ptr->rank_info");

	MPIU_CHKPMEM_MALLOC(win_ptr->rma_puts_accs, unsigned int *, comm_size * sizeof(unsigned int),
			    mpi_errno, "win_ptr->rma_puts_accs");

	MPIU_CHKPMEM_MALLOC(win_ptr->rma_local_pending_rank, int *, comm_size * sizeof(int),
			    mpi_errno, "win_ptr->rma_local_pending_rank");

	MPIU_CHKPMEM_MALLOC(win_ptr->remote_lock_state, enum MPID_PSP_Win_lock_state *, comm_size * sizeof(int),
			    mpi_errno, "win_ptr->remote_lock_state");

	win_ptr->rank = rank;
	win_ptr->rma_puts_accs_received	= 0;
	win_ptr->rma_local_pending_cnt = 0;
	win_ptr->ranks_start = NULL;
	win_ptr->ranks_start_sz = 0;
	win_ptr->ranks_post = NULL;
	win_ptr->ranks_post_sz = 0;
	INIT_LIST_HEAD(&win_ptr->lock_list);
	win_ptr->lock_exclusive = 0;
	win_ptr->lock_cnt = 0;
	INIT_LIST_HEAD(&win_ptr->lock_list_internal);
	win_ptr->lock_internal = 0;
	win_ptr->epoch_state = MPID_PSP_EPOCH_NONE;
	win_ptr->epoch_lock_count = 0;


	/* get the addresses of the windows, window objects, and completion counters
	   of all processes.  allocate temp. buffer for communication */
	MPIU_CHKLMEM_MALLOC(tmp_buf, MPID_Wincreate_msg *,
			    comm_size * sizeof(MPID_Wincreate_msg),
			    mpi_errno, "tmp_buf");

	/* ToDo: get (comm_size - 1) refs to *win_ptr!!! */
	/* FIXME: This needs to be fixed for heterogeneous systems */

	tmp_buf[rank].base = base;
	tmp_buf[rank].disp_unit = disp_unit;
	tmp_buf[rank].win_ptr = win_ptr;

	mpi_errno = MPIR_Allgather_impl(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
					tmp_buf, sizeof(MPID_Wincreate_msg), MPI_BYTE,
					comm_ptr, &errflag); /* ToDo: errflag usage! */
	if (mpi_errno) { MPIU_ERR_POP(mpi_errno); }

	for (i=0; i<comm_size; i++) {
		MPID_Win_rank_info *ri = win_ptr->rank_info + i;
		MPID_Wincreate_msg *ti = tmp_buf + i;

		win_ptr->rma_puts_accs[i] = 0;
/*		ri->epoch_origin = i    * 1000000 + rank * 1000 + 1000000000; */
/*		ri->epoch_target = rank * 1000000 +    i * 1000 + 1000000000; */

		ri->con = MPID_PSCOM_rank2connection(comm_ptr, i);

		ri->base_addr = ti->base;
		ri->disp_unit = ti->disp_unit;
		ri->win_ptr = ti->win_ptr;

		win_ptr->rma_local_pending_rank[i] = 0;
		win_ptr->remote_lock_state[i] = MPID_PSP_LOCK_UNLOCKED;
	}

	/* ToDo: post psport_recv request. */
fn_exit:
	MPIU_CHKLMEM_FREEALL();

	return mpi_errno;
	/* --BEGIN ERROR HANDLING-- */
fn_fail:
	MPIU_CHKPMEM_REAP();
	goto fn_exit;
	/* --END ERROR HANDLING-- */
}
#undef FCNAME
#undef FUNCNAME



#define FUNCNAME MPID_Win_free
#define FCNAME "MPID_Win_free"
int MPID_Win_free(MPID_Win **_win_ptr)
{
	int mpi_errno = MPI_SUCCESS /*, total_pt_rma_puts_accs, i, *recvcnts, comm_size */;
	/* MPID_Comm *comm_ptr; */
	MPID_Win *win_ptr = *_win_ptr;
#if 0
	MPIU_CHKLMEM_DECL(1);
#endif

	/* Check that we are _not_ within an access/exposure epoch: */
	if(win_ptr->epoch_state != MPID_PSP_EPOCH_NONE && win_ptr->epoch_state != MPID_PSP_EPOCH_FENCE && win_ptr->epoch_state != MPID_PSP_EPOCH_FENCE_ISSUED) {
		mpi_errno = MPI_ERR_RMA_SYNC;
		goto fn_fail;
	}


	/* ToDo: cancel psport_recv request. */

	MPID_Win_fence(0, win_ptr);

#if 0
	/* set up the recvcnts array for the reduce scatter to check if all
	   passive target rma operations are done */
	MPID_Comm_get_ptr(win_ptr->comm, comm_ptr);
	comm_size = comm_ptr->local_size;

	MPIU_CHKLMEM_MALLOC(recvcnts, int *, comm_size*sizeof(int), mpi_errno, "recvcnts");
	for (i=0; i<comm_size; i++)  recvcnts[i] = 1;

	mpi_errno = MPIR_Reduce_scatter_impl(win_ptr->pt_rma_puts_accs,
					     &total_pt_rma_puts_accs, recvcnts,
					     MPI_INT, MPI_SUM, win_ptr->comm);
	if (mpi_errno) { MPIU_ERR_POP(mpi_errno); }
#endif
#if 0
	if (total_pt_rma_puts_accs != win_ptr->my_pt_rma_puts_accs) {
		MPID_Progress_state progress_state;

		/* poke the progress engine until the two are equal */
		MPID_Progress_start(&progress_state);

		while (total_pt_rma_puts_accs != win_ptr->my_pt_rma_puts_accs) {
			mpi_errno = MPID_Progress_wait(&progress_state);
			/* --BEGIN ERROR HANDLING-- */
			if (mpi_errno != MPI_SUCCESS) {
				MPID_Progress_end(&progress_state);
				mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_FATAL, FCNAME, __LINE__, MPI_ERR_OTHER,
								 "**fail", "**fail %s", "making progress on the rma messages failed");
				goto fn_exit;
			}
			/* --END ERROR HANDLING-- */
		}
		MPID_Progress_end(&progress_state);
	}
#endif
	/* check whether this was the last active window */
	MPID_PSP_rma_check_fini(win_ptr->comm_ptr);

	MPIU_Free(win_ptr->rank_info);

	MPIU_Free(win_ptr->rma_puts_accs);

	MPIU_Free(win_ptr->rma_local_pending_rank);

	MPIU_Free(win_ptr->remote_lock_state);

	/* Free the attached buffer for windows created with MPI_Win_allocate() */
	if (win_ptr->create_flavor == MPI_WIN_FLAVOR_ALLOCATE) {
		if(win_ptr->size > 0) {
			MPIU_Free(win_ptr->base);
		}

	} else if(win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED) {
		if( is_comm_self_clone(win_ptr->comm_ptr) && (win_ptr->size > 0) ) {
			MPIU_Free(win_ptr->base);
		}
	}

	MPIR_Comm_free_impl(win_ptr->comm_ptr);

	/* check whether refcount needs to be decremented here as in group_free */
	MPIU_Handle_obj_free(&MPID_Win_mem, win_ptr);
	(*_win_ptr) = win_ptr;
fn_exit:
#if 0
	MPIU_CHKLMEM_FREEALL();
#endif

	return mpi_errno;
	/* --BEGIN ERROR HANDLING-- */
fn_fail:
	goto fn_exit;
	/* --END ERROR HANDLING-- */
}
#undef FCNAME
#undef FUNCNAME


/*
 * MPID_Alloc_mem - Allocate memory suitable for passive target RMA operations
 */
void *MPID_Alloc_mem(size_t size, MPID_Info *info)
{
	return MPIU_Malloc(size);
}


/*
 * MPID_Free_mem - Frees memory allocated with 'MPID_Alloc_mem'
 */
int MPID_Free_mem(void *ptr)
{
	MPIU_Free(ptr);
	return MPI_SUCCESS;
}


/***********************************************************************************************************
 *   RMA-3.0 Win_create()/allocate() Functions:
 */

int MPID_Win_attach(MPID_Win *win_ptr, void *base, MPI_Aint size)
{
	if(win_ptr->create_flavor != MPI_WIN_FLAVOR_DYNAMIC) {
		return MPI_ERR_RMA_FLAVOR;
	}

	/* no op, all of memory is exposed */

	return MPI_SUCCESS;;
}

int MPID_Win_detach(MPID_Win *win_ptr, const void *base)
{
	if(win_ptr->create_flavor != MPI_WIN_FLAVOR_DYNAMIC) {
		return MPI_ERR_RMA_FLAVOR;
	}

	/* no op, all of memory is exposed */

	return MPI_SUCCESS;;
}


int MPID_Win_create_dynamic(MPID_Info *info, MPID_Comm *comm_ptr, MPID_Win **win_ptr)
{
	int mpi_errno = MPI_SUCCESS;

	mpi_errno = MPID_Win_create(MPI_BOTTOM, 0, 1, info, comm_ptr, win_ptr);

	(*win_ptr)->create_flavor = MPI_WIN_FLAVOR_DYNAMIC;

	return mpi_errno;
}

#define FUNCNAME MPID_Win_allocate
#define FCNAME "MPID_Win_allocate"
int MPID_Win_allocate(MPI_Aint size, int disp_unit, MPID_Info *info,
		      MPID_Comm *comm_ptr, void *base_ptr, MPID_Win **win_ptr)
{
	void **base_pp = (void **) base_ptr;
	int mpi_errno = MPI_SUCCESS;
	MPIU_CHKPMEM_DECL(1);

	if (size > 0) {
		MPIU_CHKPMEM_MALLOC(*base_pp, void *, size, mpi_errno, "(*win_ptr)->base");
	} else if (size == 0) {
		*base_pp = NULL;
	} else {
		mpi_errno = MPI_ERR_SIZE;
		goto fn_fail;
	}

	mpi_errno = MPID_Win_create(*base_pp, size, disp_unit, info, comm_ptr, win_ptr);

	(*win_ptr)->create_flavor = MPI_WIN_FLAVOR_ALLOCATE;

fn_exit:
	return mpi_errno;
	/* --BEGIN ERROR HANDLING-- */
fn_fail:
	MPIU_CHKPMEM_REAP();
	goto fn_exit;
	/* --END ERROR HANDLING-- */
}
#undef FCNAME
#undef FUNCNAME


/***********************************************************************************************************
 *   RMA-3.0 SHMEM Functions:
 */

extern struct MPID_CommOps  *MPID_Comm_fns;
struct MPID_CommOps MPID_PSP_Comm_fns;

typedef struct _MPID_PSP_shm_attr_t
{
	int local_size;
	void **ptr_buf;
	int *disp_buf;
	int *shmid_buf;
	MPI_Aint *size_buf;
	pthread_mutex_t* lock;
} MPID_PSP_shm_attr_t;

static int MPID_PSP_shm_attr_delete_fn(MPI_Win, int, void*, void*);
static int MPID_PSP_split_type(MPID_Comm*, int, int, MPID_Info*, MPID_Comm**);
static void MPID_PSP_shm_rma_set_attr(MPID_Win*, MPID_PSP_shm_attr_t*);
static void MPID_PSP_shm_rma_get_attr(MPID_Win*, MPID_PSP_shm_attr_t**);

void MPID_PSP_shm_rma_init(void)
{
	MPID_Comm_fns = &MPID_PSP_Comm_fns;
	MPID_Comm_fns->split_type = MPID_PSP_split_type;
	MPIR_Comm_create_keyval_impl(MPI_COMM_DUP_FN, MPID_PSP_shm_attr_delete_fn, &MPIDI_Process.shm_attr_key, NULL);
}

void MPID_PSP_shm_rma_mutex_lock(MPID_Win *win_ptr)
{
	MPID_PSP_shm_attr_t* attr = NULL;
	MPID_PSP_shm_rma_get_attr(win_ptr, &attr);
	if(attr) pthread_mutex_lock(attr->lock);
}

void MPID_PSP_shm_rma_mutex_unlock(MPID_Win *win_ptr)
{
	MPID_PSP_shm_attr_t* attr = NULL;
	MPID_PSP_shm_rma_get_attr(win_ptr, &attr);
	if(attr) pthread_mutex_unlock(attr->lock);
}

extern MPIU_Object_alloc_t MPID_Keyval_mem;
extern MPIU_Object_alloc_t MPID_Attr_mem;
extern MPID_Keyval MPID_Keyval_direct[];

static
void MPID_PSP_shm_rma_set_attr(MPID_Win *win_ptr, MPID_PSP_shm_attr_t *attr)
{
	MPID_Keyval *keyval_ptr = NULL;
	MPID_Attribute *new_p;

	/* store shmem region related information as an attribute of win: */
	MPID_Keyval_get_ptr(MPIDI_Process.shm_attr_key, keyval_ptr);
	MPIR_Keyval_add_ref(keyval_ptr);

	new_p = (MPID_Attribute *)MPIU_Handle_obj_alloc(&MPID_Attr_mem);
	MPIU_Object_set_ref(new_p, 0);

	new_p->attrType      = MPIR_ATTR_PTR;
	new_p->keyval	     = keyval_ptr;
	new_p->pre_sentinal  = 0;
	new_p->value	     = (MPID_AttrVal_t)(void*)attr;
	new_p->post_sentinal = 0;
	new_p->next	     = win_ptr->attributes;
	win_ptr->attributes = new_p;
}

static
void MPID_PSP_shm_rma_get_attr(MPID_Win *win_ptr, MPID_PSP_shm_attr_t **attr)
{
	MPID_Attribute *p = win_ptr->attributes;

	assert(win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED);

	*attr = NULL;

	/* retrieve the stored information about the shared region: */
	while (p) {
		if (p->keyval->handle == MPIDI_Process.shm_attr_key) {
			assert(p->value != NULL);
			/* found attribute! */
			*attr = (MPID_PSP_shm_attr_t*)p->value;
			break;
		}
		p = p->next;
	}
}

void MPID_PSP_shm_rma_get_base(MPID_Win *win_ptr, int rank, int *disp, void **base)
{
	MPID_PSP_shm_attr_t *attr = NULL;

	assert(win_ptr->create_flavor == MPI_WIN_FLAVOR_SHARED);

	/* retrieve the stored information about the shared region: */
	MPID_PSP_shm_rma_get_attr(win_ptr, &attr);

	if(attr) {
		/* Attribute found: */
		*base = attr->ptr_buf[rank];
		*disp = attr->disp_buf[rank];
	} else {
		/* The related communicator must be an MPI_COMM_SELF clone! */
		assert(is_comm_self_clone(win_ptr->comm_ptr));
		assert(rank == win_ptr->rank);
		*base = win_ptr->base;
		*disp = win_ptr->disp_unit;
	}
}

static
int get_my_shmem_split_color(MPID_Comm * comm_ptr)
{
	int i, color = MPI_UNDEFINED;

	for(i=0; i<comm_ptr->local_size; i++) {
		if( (comm_ptr->vcr[i]->con->type == PSCOM_CON_TYPE_SHM) || (comm_ptr->rank == i) ) {
			color = i;
			break;
		}
	}

	return color;
}

static
int MPID_PSP_split_type(MPID_Comm * comm_ptr, int split_type, int key,
			MPID_Info * info_ptr, MPID_Comm ** newcomm_ptr)
{
	int mpi_errno = MPI_SUCCESS;

	if(split_type == MPI_COMM_TYPE_SHARED) {
		int color;

		color = get_my_shmem_split_color(comm_ptr);

		mpi_errno = MPIR_Comm_split_impl(comm_ptr, color, key, newcomm_ptr);

		if(mpi_errno == MPI_SUCCESS) {
			mpi_errno = MPIR_Comm_set_attr_impl(*newcomm_ptr, MPIDI_Process.shm_attr_key, NULL, MPIR_ATTR_PTR);
		}

	} else {
		mpi_errno = MPIR_Comm_split_impl(comm_ptr,  MPI_UNDEFINED, key, newcomm_ptr);
	}

	return mpi_errno;
}

static
int MPID_PSP_shm_attr_delete_fn(MPI_Win win, int keyval, void *attribute_val, void *extra_state)
{
	int i;
	MPID_PSP_shm_attr_t *attr = (MPID_PSP_shm_attr_t*)attribute_val;
	MPID_Win *win_ptr = NULL;

	if(attr) {

		for(i=0; i<attr->local_size; i++) {
			if(attr->shmid_buf[i] != -1) {
				shmdt(attr->ptr_buf[i]);
			}
		}

		MPIU_Free(attr->ptr_buf);
		MPIU_Free(attr->size_buf);
		MPIU_Free(attr->disp_buf);
		MPIU_Free(attr->shmid_buf);

		MPID_Win_get_ptr(win, win_ptr);
		assert(win_ptr);
		assert(win_ptr->comm_ptr);
		if(win_ptr->comm_ptr->rank == 0) {
			pthread_mutex_destroy(attr->lock);
		}
		shmdt(attr->lock);

		MPIU_Free(attr);
	}

	return MPI_SUCCESS;
}

static
int MPID_PSP_Win_allocate_shmget(MPI_Aint size, int disp_unit, MPID_Info *info,
				 MPID_Comm *comm_ptr, void *base_ptr, MPID_Win **win_ptr,
				 MPID_PSP_shm_attr_t *attr)
{
	int i;
	int contig = 1;
	int shmid = -1;
	void *shm = NULL;
	void **base_pp = (void **) base_ptr;
	int errflag = 0;
	int mpi_errno = MPI_SUCCESS;
	MPI_Aint total_size = 0;

	void **ptr_buf;
	int *disp_buf;
	int *shmid_buf;
	MPI_Aint *size_buf;

	if(size < 0) {
		mpi_errno = MPI_ERR_SIZE;
		goto fn_fail;
	}

	disp_buf  = MPIU_Malloc(comm_ptr->local_size * sizeof(int));
	shmid_buf = MPIU_Malloc(comm_ptr->local_size * sizeof(int));
	ptr_buf   = MPIU_Malloc(comm_ptr->local_size * sizeof(void*));
	size_buf  = MPIU_Malloc(comm_ptr->local_size * sizeof(MPI_Aint));

	/* Initialize shmid_buf[] */
	for (i = 0; i < comm_ptr->local_size; i++) shmid_buf[i] = -1;

	mpi_errno = MPIR_Allgather_impl(&size, 1, MPI_AINT, size_buf, 1, MPI_AINT, comm_ptr, &errflag);
	if (mpi_errno) {
		goto fn_fail;
	}

	mpi_errno = MPIR_Allgather_impl(&disp_unit, 1, MPI_INT, disp_buf, 1, MPI_INT, comm_ptr, &errflag);
	if (mpi_errno) {
		goto fn_fail;
	}

	/* check whether the memory to be allocated should be contigous: */
	while(info) {
		if( info->key && (strcmp(info->key, "alloc_shared_noncontig") == 0) && (strcmp(info->value, "true") == 0) ) {
			contig = 0;
			break;
		}
		info = info->next;
	}

	if(contig) { /* In this case, the shared memory must be contigous across all procs.
		      | That means that the first address of the memory segment of proc i
		      | must be consecutive with the last address of that of proc i-1.
		      | Therefore, we have the proc with local rank==0 allocate the whole
		      | shared memory and let the other procs then attach to it.
		     */

		mpi_errno = MPIR_Allgather_impl(&size, 1, MPI_AINT, size_buf, 1, MPI_AINT, comm_ptr, &errflag);
		if (mpi_errno) {
			goto fn_fail;
		}

		for(i=0; i<comm_ptr->local_size; i++) total_size += size_buf[i];

		if(total_size > 0) {

			if(comm_ptr->rank == 0) {

				/* create one big SHMEM segment: */
				shmid = shmget(0, total_size, IPC_CREAT | 0777);
				if (shmid == -1) goto err_shmget;

				shm = shmat(shmid, 0, 0);
				if (((long)shm == -1) || !shm) goto err_shmat;

				shmctl(shmid, IPC_RMID, NULL);
			}

			mpi_errno = MPIR_Bcast_impl(&shmid, 1, MPI_INT, 0, comm_ptr, &errflag);
			if (mpi_errno) {
				goto fn_fail;
			}
			shmid_buf[0] = shmid;

			if(comm_ptr->rank != 0) {
				shm = shmat(shmid_buf[0], 0, 0);
				if (((long)shm == -1) || !shm) goto err_shmat;
			}

			ptr_buf[0] = shm;

			for(i=1; i<comm_ptr->local_size; i++) {
				ptr_buf[i] = (char*)ptr_buf[i-1] + size_buf[i-1];
			}

			*base_pp = ptr_buf[comm_ptr->rank];

		} else {
			*base_pp = NULL;
		}

	} else { /* Here, non-contigous memory is allowed. Therefore, we let each
		  | proc allocate its own local shared segment and then have all
		  | procs connect to all the other remote segments.
		 */

		if (size > 0) {

			/* create local SHMEM segment: */
			shmid = shmget(0, size, IPC_CREAT | 0777);
			if (shmid == -1) goto err_shmget;

			shm = shmat(shmid, 0, 0);
			if (((long)shm == -1) || !shm) goto err_shmat;

			shmctl(shmid, IPC_RMID, NULL);

			*base_pp = shm;

		} else {
			*base_pp = NULL;
		}

		/* attach to remote segments, too: */

		mpi_errno = MPIR_Allgather_impl(&shmid, 1, MPI_INT, shmid_buf, 1, MPI_INT, comm_ptr, &errflag);
		if (mpi_errno) {
			goto fn_fail;
		}

		ptr_buf[comm_ptr->rank] = *base_pp;

		for(i=0; i<comm_ptr->local_size; i++) {
			if(i != comm_ptr->rank) {
				if(shmid_buf[i] != -1)  {
					ptr_buf[i] = shmat(shmid_buf[i], 0, 0);
					if (((long)ptr_buf[i] == -1) || !ptr_buf[i]) goto err_shmat;
				} else {
					ptr_buf[i] = NULL;
				}
			}
		}
	}

	mpi_errno = MPID_Win_create(*base_pp, size, disp_unit, info, comm_ptr, win_ptr);

	if (mpi_errno) {
		goto fn_fail;
	}

	attr->ptr_buf = ptr_buf;
	attr->size_buf = size_buf;
	attr->disp_buf = disp_buf;
	attr->shmid_buf = shmid_buf;
	attr->local_size = comm_ptr->local_size;

	/* create mutex for atomic/accumulate ops: */

	if(comm_ptr->rank == 0) {

		pthread_mutexattr_t mutex_attr;
		int rval;

		shmid = shmget(0, sizeof(pthread_mutex_t), IPC_CREAT | 0777);

		if (shmid == -1) goto err_shmget;

		attr->lock = shmat(shmid, 0, 0);
		if (((long)attr->lock == -1) || !attr->lock) goto err_shmat;

		shmctl(shmid, IPC_RMID, NULL);

		pthread_mutexattr_init(&mutex_attr);
		rval = pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
		if (rval) goto err_setpshared;

		pthread_mutex_init(attr->lock, &mutex_attr);
		pthread_mutexattr_destroy(&mutex_attr);
	}

	mpi_errno = MPIR_Bcast_impl(&shmid, 1, MPI_INT, 0, comm_ptr, &errflag);
	if (mpi_errno) {
			goto fn_fail;
	}

	if(comm_ptr->rank != 0) {

		attr->lock = shmat(shmid, 0, 0);
		if (((long)attr->lock == -1) || !attr->lock) goto err_shmat;
	}

	mpi_errno = MPIR_Barrier_impl(comm_ptr, &errflag);
	if (mpi_errno) {
			goto fn_fail;
	}

fn_exit:
	return mpi_errno;
	/* --BEGIN ERROR HANDLING-- */
err_shmget:
	mpi_errno = MPI_ERR_RMA_SHARED;
	goto fn_exit;
err_setpshared:
err_shmat:
	shmctl(shmid, IPC_RMID, NULL);
	mpi_errno = MPI_ERR_RMA_SHARED;
	goto fn_exit;
fn_fail:
	goto fn_exit;
	/* --END ERROR HANDLING-- */
}

int MPID_Win_allocate_shared(MPI_Aint size, int disp_unit, MPID_Info *info_ptr, MPID_Comm *comm_ptr,
			     void *base_ptr, MPID_Win **win_ptr)
{
	void **base_pp = (void **) base_ptr;
	int mpi_errno = MPI_SUCCESS;

	if(!is_comm_self_clone(comm_ptr)) {

		/* The related communicator is NOT a MPI_COMM_SELF clone... */

		int flag = 0;
		MPID_Attribute *p = comm_ptr->attributes;

		/* Check for the MPI_COMM_TYPE_SHARED attribute in comm: */
		while (p) {
			if (p->keyval->handle == MPIDI_Process.shm_attr_key) {
				assert(p->value == NULL);
				/* found attribute! */
				flag = 1;
				break;
			}
			p = p->next;
		}

		if(flag) { /* SHMEM attribute found: */
			MPID_PSP_shm_attr_t *shm_attr;

			shm_attr = MPIU_Malloc(sizeof(MPID_PSP_shm_attr_t));

			mpi_errno = MPID_PSP_Win_allocate_shmget(size, disp_unit, info_ptr, comm_ptr, base_ptr, win_ptr, shm_attr);

			if(mpi_errno) {
				goto fn_fail;
			}

			/* store shmem region related information as an attribute of win: */
			MPID_PSP_shm_rma_set_attr(*win_ptr, shm_attr);

			(*win_ptr)->base = *base_pp;
			(*win_ptr)->size = size;
			(*win_ptr)->disp_unit = disp_unit;

		} else {
			/* This communicator cannot be used with MPID_Win_allocate_shared()! */
			return MPI_ERR_RMA_SHARED;
		}
	} else {
		mpi_errno = MPID_Win_allocate(size, disp_unit, info_ptr, comm_ptr, base_ptr, win_ptr);

		if(mpi_errno) {
			goto fn_fail;
		}
	}

	(*win_ptr)->create_flavor = MPI_WIN_FLAVOR_SHARED;
	(*win_ptr)->model = MPI_WIN_UNIFIED;

fn_exit:
	return mpi_errno;
	/* --BEGIN ERROR HANDLING-- */
fn_fail:
	goto fn_exit;
	/* --END ERROR HANDLING-- */
}

int MPID_Win_shared_query(MPID_Win *win_ptr, int rank, MPI_Aint *size, int *disp_unit, void *base_ptr)
{
	void **base_pp = (void **) base_ptr;

	if(win_ptr->create_flavor != MPI_WIN_FLAVOR_SHARED) {
		return MPI_ERR_RMA_FLAVOR;
	}

	if( !is_comm_self_clone(win_ptr->comm_ptr) ) {

		MPID_PSP_shm_attr_t *attr = NULL;

		/* retrieve the stored information about the shared region: */
		MPID_PSP_shm_rma_get_attr(win_ptr, &attr);

		assert(attr != NULL);

		if(rank == MPI_PROC_NULL) {

			/* The standard says: When rank is MPI_PROC_NULL, the pointer, disp_unit, and size returned are the
			 | pointer, disp_unit, and size of the memory segment belonging the lowest rank that specified
			 | size > 0. If all processes in the group attached to the window specified size = 0, then the
			 | call returns size = 0 and a baseptr as if MPI_ALLOC_MEM was called with size = 0.
			 */

			int i;
			for(i=0; i<win_ptr->comm_ptr->local_size; i++) {
				if(attr->size_buf[i] > 0) break;
			}
			if(i == win_ptr->comm_ptr->local_size) {
				*base_pp = NULL;
				*size = 0;
				*disp_unit = 0;
			} else {
				*size = attr->size_buf[i];
				*base_pp = attr->ptr_buf[i];
				*disp_unit = attr->disp_buf[i];
			}
		} else {
			*size = attr->size_buf[rank];
			*base_pp = attr->ptr_buf[rank];
			*disp_unit = attr->disp_buf[rank];
		}

	} else {
		/* trivial: 'SHMEM' for a COMM_SELF clone is just the local memory: */
		*base_pp   = win_ptr->base;
		*size      = win_ptr->size;
		*disp_unit = win_ptr->disp_unit;
	}

	return MPI_SUCCESS;
}
