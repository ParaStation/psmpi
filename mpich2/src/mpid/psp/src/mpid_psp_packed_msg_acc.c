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

int MPIDI_PSP_compute_acc_op(void *origin_addr, int origin_count,
			      MPI_Datatype origin_datatype, void *target_addr,
			      int target_count, MPI_Datatype target_datatype,
			      MPI_Op op, int packed_source_buf)
{
	int mpi_errno = MPI_SUCCESS;
	MPI_User_function *uop = NULL;
	MPI_Aint origin_datatype_size = 0, origin_datatype_extent = 0;
	int is_empty_source = (op == MPI_NO_OP)? TRUE : FALSE;

	if (is_empty_source == FALSE) {
		MPIR_Assert(MPIR_DATATYPE_IS_PREDEFINED(origin_datatype));
		MPIR_Datatype_get_size_macro(origin_datatype, origin_datatype_size);
		MPIR_Datatype_get_extent_macro(origin_datatype, origin_datatype_extent);
	}

	/* we only support buildin operations */
	if ((HANDLE_IS_BUILTIN(op)) &&
	    ((*MPIR_OP_HDL_TO_DTYPE_FN(op))(origin_datatype) == MPI_SUCCESS)) {
		/* get the function by indexing into the op table */
		uop = MPIR_OP_HDL_TO_FN(op);
	} else {
		mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE,
						 __func__, __LINE__, MPI_ERR_OP,
						 "**opnotpredefined", "**opnotpredefined %d", op);
		goto fn_exit;
	}

#ifdef MPIDI_PSP_WITH_CUDA_AWARENESS
	/* is target_addr within device memory? */
	void *in_targetbuf = target_addr;
	void *host_targetbuf = NULL;
	MPL_pointer_attr_t attr;
	if (pscom_is_gpu_mem(target_addr)) {
		MPI_Aint extent, true_extent;
		MPI_Aint true_lb;

		MPIR_Datatype_get_extent_macro(target_datatype, extent);
		MPIR_Type_get_true_extent_impl(target_datatype, &true_lb, &true_extent);
		extent = MPL_MAX(extent, true_extent);

		host_targetbuf = MPL_malloc(extent * target_count, MPL_MEM_RMA);
		MPIR_Assert(host_targetbuf);
		MPIR_Localcopy(target_addr, target_count, target_datatype,
			       host_targetbuf, target_count, target_datatype);
		target_addr = host_targetbuf;
	}
#endif /* MPIDI_PSP_WITH_CUDA_AWARENESS */

	/* directly apply op if target dtp is predefined dtp OR source buffer is empty */
	if ((is_empty_source == TRUE) || HANDLE_IS_BUILTIN(target_datatype)) {
		(*uop)(origin_addr, target_addr, &origin_count, &origin_datatype);
	} else {
		/* derived datatype */
		struct iovec *typerep_vec;
		int i, count;
		MPI_Aint vec_len, type_extent, type_size, src_type_stride;
		MPI_Datatype type;
		MPIR_Datatype *dtp;
		MPI_Aint curr_len;
		void *curr_loc;
		int accumulated_count;

		MPIR_Datatype_get_ptr(target_datatype, dtp);
		MPIR_Assert(dtp != NULL);
		vec_len = dtp->typerep.num_contig_blocks * target_count + 1;
		/* +1 needed because Rob says so */
		typerep_vec = (struct iovec *)
		    MPL_malloc(vec_len * sizeof(struct iovec), MPL_MEM_RMA);
		if (!typerep_vec) {
			mpi_errno =
			    MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,
						 MPI_ERR_OTHER, "**nomem", 0);
			goto fn_exit;
		}

		MPI_Aint actual_iov_len, actual_iov_bytes;
		MPIR_Typerep_to_iov(NULL, target_count, target_datatype, 0,
				    typerep_vec, vec_len,
				    origin_count * origin_datatype_size,
				    &actual_iov_len, &actual_iov_bytes);
		vec_len = actual_iov_len;

		type = dtp->basic_type;
		MPIR_Assert(type != MPI_DATATYPE_NULL);

		MPIR_Assert(type == origin_datatype);
		type_size = origin_datatype_size;
		type_extent = origin_datatype_extent;
		/* If the source buffer has been packed by the caller, the distance between
         * two elements can be smaller than extent. E.g., predefined pairtype may
         * have larger extent than size.*/
		/* when predefined pairtype have larger extent than size, we'll end up
         * missaligned access. Memcpy the source to workaround the alignment issue.
         */
		char *src_ptr = NULL;
		if (packed_source_buf) {
			src_type_stride = origin_datatype_size;
			if (origin_datatype_size < origin_datatype_extent) {
				src_ptr = MPL_malloc(origin_datatype_extent, MPL_MEM_OTHER);
			}
		} else {
			src_type_stride = origin_datatype_extent;
		}

		i = 0;
		curr_loc = typerep_vec[0].iov_base;
		curr_len = typerep_vec[0].iov_len;
		accumulated_count = 0;
		while (i != vec_len) {
			if (curr_len < type_size) {
				MPIR_Assert(i != vec_len);
				i++;
				curr_len += typerep_vec[i].iov_len;
				continue;
			}

			MPIR_Assign_trunc(count, curr_len / type_size, int);

			if (src_ptr) {
				MPI_Aint unpacked_size;
				MPIR_Typerep_unpack((char *)origin_addr + src_type_stride * accumulated_count,
						    origin_datatype_size, src_ptr, 1, origin_datatype, 0, &unpacked_size);
				(*uop)(src_ptr, (char *)target_addr + MPIR_Ptr_to_aint(curr_loc), &count, &type);
			} else {
				(*uop)((char *)origin_addr + src_type_stride * accumulated_count,
				       (char *)target_addr + MPIR_Ptr_to_aint(curr_loc), &count, &type);
			}

			if (curr_len % type_size == 0) {
				i++;
				if (i != vec_len) {
					curr_loc = typerep_vec[i].iov_base;
					curr_len = typerep_vec[i].iov_len;
				}
			} else {
				curr_loc = (void *)((char *)curr_loc + type_extent * count);
				curr_len -= type_size * count;
			}

			accumulated_count += count;
		}

		MPL_free(src_ptr);
		MPL_free(typerep_vec);
	}

#ifdef MPIDI_PSP_WITH_CUDA_AWARENESS
	if (host_targetbuf) {
		target_addr = in_targetbuf;
		MPIR_Localcopy(host_targetbuf, target_count, target_datatype,
			       target_addr, target_count, target_datatype);
		MPL_free(host_targetbuf);
	}
#endif /*  MPIDI_PSP_WITH_CUDA_AWARENESS */

fn_exit:
	/* TODO: Error handling */
	assert (mpi_errno == MPI_SUCCESS);
	return mpi_errno;
}
