/*
 * ParaStation
 *
 * Copyright (C) 2006-2019 ParTec Cluster Competence Center GmbH, Munich
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

#if 0
void PREPEND_PREFIX(Segment_manipulate)(struct DLOOP_Segment *segp,
					DLOOP_Offset first,
					DLOOP_Offset *lastp,
					int (*piecefn) (DLOOP_Offset *blocks_p,
							DLOOP_Type el_type,
							DLOOP_Offset rel_off,
							void *bufp,
							void *v_paramp),
					int (*vectorfn) (DLOOP_Offset *blocks_p,
							 int count,
							 int blklen,
							 DLOOP_Offset stride,
							 DLOOP_Type el_type,
							 DLOOP_Offset rel_off,
							 void *bufp,
							 void *v_paramp),
					int (*blkidxfn) (DLOOP_Offset *blocks_p,
							 int count,
							 int blklen,
							 DLOOP_Offset *offsetarray,
							 DLOOP_Type el_type,
							 DLOOP_Offset rel_off,
							 void *bufp,
							 void *v_paramp),
					int (*indexfn) (DLOOP_Offset *blocks_p,
							int count,
							int *blockarray,
							DLOOP_Offset *offsetarray,
							DLOOP_Type el_type,
							DLOOP_Offset rel_off,
							void *bufp,
							void *v_paramp),
					DLOOP_Offset (*sizefn) (DLOOP_Type el_type),
					void *pieceparams);
#endif

struct acc_params {
	void		  *msg;
	size_t		   msg_sz;
	MPI_User_function *uop;
	MPI_Op             op;
	MPI_Datatype       dtype;
};


static
int MPIR_Segment_contig_acc(DLOOP_Offset *blocks_p,
			    DLOOP_Type el_type,
			    DLOOP_Offset rel_off,
			    void *bufp,
			    void *v_paramp)
{
	int el_size;
	DLOOP_Offset size;
	int count = *blocks_p;
	struct acc_params *acc_params = v_paramp;
/*	printf("%s() with type %08x\n", __func__, el_type); */

	if( (acc_params->op == MPI_MINLOC || acc_params->op == MPI_MAXLOC) && el_type == MPI_BYTE ) {
		/* Corner case of composed but built-in datatypes that degraded to MPI_BYTE when treated as derived:
		   Contrary to the allowed datatypes for MPI_Accumulate, these are composed of different predefined
		   types (e.g. MPI_DOUBLE_INT) and can only be used meaningfully with MPI_MINLOC/MAXLOC as ops... */
		el_type = acc_params->dtype;
		size  = count;
		count = 1;
	} else {
		/* Usual case of common built-in datatypes... */
		el_size = MPIR_Datatype_get_basic_size(el_type);
		size = *blocks_p * (DLOOP_Offset) el_size;
	}

	/* replace would be: memcpy((char *) bufp + rel_off, paramp->u.unpack.unpack_buffer, size); */

	acc_params->uop(acc_params->msg,
			(char *) bufp + rel_off /* target */,
			&count /* count */,
			&el_type);

	acc_params->msg = (char *) acc_params->msg + size;

	return 0;
}

static
int MPIR_Segment_vector_acc(DLOOP_Offset *blocks_p,
			    DLOOP_Count count,
			    DLOOP_Count blklen,
			    DLOOP_Offset stride,
			    DLOOP_Type el_type,
			    DLOOP_Offset rel_off,
			    DLOOP_Buffer bufp,
			    void *v_paramp)
{
	int i;
	int el_size;
	DLOOP_Offset size;
	struct acc_params *acc_params = v_paramp;

	if( (acc_params->op == MPI_MINLOC || acc_params->op == MPI_MAXLOC) && el_type == MPI_BYTE ) {
		/* Corner case of composed but built-in datatypes that degraded to MPI_BYTE when treated as derived:
		   (see also MPIR_Segment_contig_acc) */
		el_type = acc_params->dtype;
		size   = blklen;
		blklen = 1;
	} else {

		el_size = MPIR_Datatype_get_basic_size(el_type);
		size = blklen * (DLOOP_Offset) el_size;
	}

	for(i=0; i<count; i++) {

		acc_params->uop(acc_params->msg,
				(char *) bufp + rel_off + i * stride /* target */,
				(int*) &blklen /* count */,
				&el_type);

		acc_params->msg = (char *) acc_params->msg + size;
	}

	return 0;
}


/* tm: see src/mpi/coll/opreplace.c
static
void MPIR_REPLACE(void *invec, void *inoutvec, int *Len, MPI_Datatype *type)
{
	memcpy(inoutvec, invec, *Len);
}
*/

void MPID_PSP_packed_msg_acc(const void *target_addr, int target_count, MPI_Datatype datatype,
			     void *msg, size_t msg_sz, MPI_Op op)
{
	MPIR_Segment segment;
	DLOOP_Offset last = msg_sz;
	struct acc_params acc_params;

	void *acc_addr = (void*)target_addr;

#ifdef MPIDI_PSP_WITH_CUDA_AWARENESS
	size_t target_sz = 0;

	/* is target_addr within device memory? */
	if (pscom_is_gpu_mem(target_addr)) {
		int contig;
		MPIR_Datatype *dtp;
		MPI_Aint true_lb;

		MPIDI_Datatype_get_info(target_count, datatype,
			contig, target_sz,
			dtp, true_lb);

		acc_addr = MPL_malloc(target_sz, MPL_MEM_OTHER);
		MPID_Memcpy(acc_addr, target_addr, target_sz);

		// Avoid compiler warnings about unused variables:
		(void)contig;
		(void)true_lb;
	}
#endif

	MPIR_Segment_init(acc_addr, target_count, datatype, &segment);

	acc_params.op = op;
	MPID_PSP_Datatype_get_basic_type(datatype, acc_params.dtype);

	acc_params.msg = msg;
	acc_params.msg_sz = msg_sz;
	/* get the function by indexing into the op table */
	acc_params.uop = MPIR_OP_HDL_TO_FN(op);

	if (!acc_params.uop) return; /* Todo: report error */

	MPIR_Segment_manipulate(&segment, /* first */0, &last,
				MPIR_Segment_contig_acc,
				MPIR_Segment_vector_acc,
				/* ToDo: implement MPIR_Segment_{blkidx,index}_acc! */
				NULL /* MPIR_Segment_blkidx_acc */,
				NULL /* MPIR_Segment_index_acc */,
				NULL /* sizefn */,
				&acc_params);

#ifdef MPIDI_PSP_WITH_CUDA_AWARENESS
	/* do we need to unstage the buffer? */
	if (acc_addr != target_addr) {
		MPID_Memcpy((void*)target_addr, acc_addr, target_sz);
		MPL_free(acc_addr);
	}
#endif
}
