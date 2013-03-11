/*
 * ParaStation
 *
 * Copyright (C) 2006,2007 ParTec Cluster Competence Center GmbH, Munich
 *
 * All rights reserved.
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
	void		*msg;
	unsigned int	msg_sz;
	MPI_User_function *uop;
};


static
int MPID_Segment_contig_acc(DLOOP_Offset *blocks_p,
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

	el_size = MPID_Datatype_get_basic_size(el_type);
	size = *blocks_p * (DLOOP_Offset) el_size;

	/* replace would be: memcpy((char *) bufp + rel_off, paramp->u.unpack.unpack_buffer, size); */

	acc_params->uop(acc_params->msg,
			(char *) bufp + rel_off /* target */,
			&count /* count */,
			&el_type);

	acc_params->msg = (char *) acc_params->msg + size;

	return 0;
}


static
void MPIR_REPLACE(void *invec, void *inoutvec, int *Len, MPI_Datatype *type)
{
	memcpy(inoutvec, invec, *Len);
}


static inline
MPI_User_function *get_op(MPI_Op op)
{
	unsigned int idx = (op - MPI_MAX);

	if (idx < MPI_MAXLOC - MPI_MAX) {
		return MPIR_Op_table[idx];
	} else if (op == MPI_REPLACE) {
		return MPIR_REPLACE;
	} else {
		return NULL;
	}
}


void MPID_PSP_packed_msg_acc(const void *target_addr, int target_count, MPI_Datatype datatype,
			     void *msg, unsigned int msg_sz, MPI_Op op)
{
	MPID_Segment segment;
	DLOOP_Offset last = msg_sz;
	struct acc_params acc_params;

	MPID_Segment_init(target_addr, target_count, datatype, &segment, 0);

	acc_params.msg = msg;
	acc_params.msg_sz = msg_sz;
	/* get the function by indexing into the op table */
	acc_params.uop = get_op(op);

	if (!acc_params.uop) return; /* Todo: report error */

	MPID_Segment_manipulate(&segment, /* first */0, &last,
				MPID_Segment_contig_acc,
				/* ToDo: implement MPID_Segment_{vectpr,blkidx,index}_acc! */
				NULL /* MPID_Segment_vector_acc */,
				NULL /* MPID_Segment_blkidx_acc */,
				NULL /* MPID_Segment_index_acc */,
				NULL /* sizefn */,
				&acc_params);
}
