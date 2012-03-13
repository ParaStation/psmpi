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

#include "mpidimpl.h"
#include "mpid_psp_datatype.h"
#include "mpid_psp_packed_msg.h"
#include "mpid_psp_request.h"

static
void MPID_PSP_rma_init(void)
{
	assert(MPIDI_Process.socket != NULL);

	MPID_enable_receive_dispach();
}


static
void MPID_PSP_rma_check_init(void)
{
	static int initialized = 0;
	if (!initialized) {
		MPID_PSP_rma_init();
		initialized = 1;
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

	MPID_PSP_rma_check_init();

	comm_size = comm_ptr->local_size;
	rank = comm_ptr->rank;

	win_ptr = (MPID_Win *)MPIU_Handle_obj_alloc( &MPID_Win_mem );
	MPIU_ERR_CHKANDJUMP(!win_ptr, mpi_errno, MPI_ERR_OTHER, "**nomem");

	(*_win_ptr) = win_ptr;

	win_ptr->fence_cnt = 0;
	win_ptr->base = base;
	win_ptr->size = size;
	win_ptr->disp_unit = disp_unit;
	win_ptr->start_group_ptr = NULL;
	win_ptr->start_assert = 0;
	win_ptr->attributes = NULL;
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
	int mpi_errno=MPI_SUCCESS, total_pt_rma_puts_accs, i, *recvcnts, comm_size;
	MPID_Comm *comm_ptr;
	MPID_Win *win_ptr = *_win_ptr;

	MPIU_CHKLMEM_DECL(1);


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
	MPIR_Comm_free_impl(win_ptr->comm_ptr);

	MPIU_Free(win_ptr->rank_info);

	MPIU_Free(win_ptr->rma_puts_accs);

	/* check whether refcount needs to be decremented here as in group_free */
	MPIU_Handle_obj_free(&MPID_Win_mem, win_ptr);
	(*_win_ptr) = win_ptr;

fn_exit:
	MPIU_CHKLMEM_FREEALL();

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
