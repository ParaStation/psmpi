/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021      ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#ifndef _MPID_PSP_PACKED_MSG_H_
#define _MPID_PSP_PACKED_MSG_H_

#include "mpid_psp_datatype.h"


static inline
int MPID_PSP_packed_msg_allocate(size_t data_sz, MPID_PSP_packed_msg_t *msg)
{
	/* non-contiguous data (or GPU memory) */
	char *tmp_buf = MPL_malloc(data_sz, MPL_MEM_OTHER);

	msg->msg = tmp_buf;
	msg->msg_sz = data_sz;
	msg->tmp_buf = tmp_buf;

	if (unlikely(!tmp_buf && data_sz)) { /* Error: No mem */
		msg->msg_sz = 0;
		return MPI_ERR_NO_MEM;
	}

	return MPI_SUCCESS;
}

/* May return MPI_ERR_NO_MEM.
   cleanup with packed_msg_cleanup */
static inline
int MPID_PSP_packed_msg_prepare(const void *addr, int count, MPI_Datatype datatype,
				MPID_PSP_packed_msg_t *msg)
{
	int	contig;
	size_t data_sz;
	MPIR_Datatype *dtp;
	MPI_Aint true_lb;
	int ret = MPI_SUCCESS;

	MPIDI_Datatype_get_info(count, datatype,
				contig, data_sz,
				dtp, true_lb);

	if (contig || !data_sz) {
		msg->msg = (char *)addr + true_lb;
		msg->msg_sz = data_sz;
		msg->tmp_buf = NULL;
	} else {
		MPI_Aint packsize;
		MPIR_Pack_size_impl(count, datatype, &packsize);
		ret = MPID_PSP_packed_msg_allocate(packsize, msg);
	}

/*	printf("Packed src:(%d) %s\n", origin_data_sz, pscom_dumpstr(msg->msg, pscom_min(origin_data_sz, 64)));
	fflush(stdout); */

	return ret;
}


static inline
int MPID_PSP_packed_msg_need_pack(const MPID_PSP_packed_msg_t *msg)
{
	return !!msg->tmp_buf;
}


/* create a packed_msg at msg->msg of size msg->msg_sz.
 * prepare msg with packed_msg_prepare()
 */
static inline
void MPID_PSP_packed_msg_pack(const void *src_addr, int src_count, MPI_Datatype src_datatype,
			      const MPID_PSP_packed_msg_t *msg)
{
	int res;
	if (msg->tmp_buf) {
		MPI_Aint actual_pack_bytes;

		res = MPIR_Typerep_pack(src_addr, src_count, src_datatype, 0,
					msg->msg, msg->msg_sz, &actual_pack_bytes);

		assert(actual_pack_bytes == msg->msg_sz);
		assert(res == MPI_SUCCESS);
	}
}


static inline
int MPID_PSP_packed_msg_need_unpack(const MPID_PSP_packed_msg_t *msg)
{
	return !!msg->tmp_buf;
}


/* unpack a packed_msg at msg->msg of size min(msg->msg_sz, data_len).
 * prepare msg with packed_msg_prepare()
 */
static inline
int MPID_PSP_packed_msg_unpack(void *addr, int count, MPI_Datatype datatype,
                               const MPID_PSP_packed_msg_t *msg, size_t data_len)
{
	int res = MPI_SUCCESS;
	MPI_Aint actual_unpack_bytes;

	if (msg->tmp_buf) {

		res = MPIR_Typerep_unpack(msg->tmp_buf,
				pscom_min(msg->msg_sz, data_len),
				addr, count, datatype, 0, &actual_unpack_bytes);
		/* From ch3u_handle_recv_pkt.c:
                   "If the data can't be unpacked, the we have a
                    mismatch between the datatype and the amount of
                    data received.  Throw away received data."
		*/
		if (actual_unpack_bytes != pscom_min(msg->msg_sz, data_len)) {
			res = MPI_ERR_TYPE;
		}
	}

	return res;
}


static inline
void MPID_PSP_packed_msg_cleanup(MPID_PSP_packed_msg_t *msg)
{
	if (msg->tmp_buf) {
		MPL_free(msg->tmp_buf);
		msg->tmp_buf = NULL;
	}
}


static inline
void MPID_PSP_packed_msg_cleanup_datatype(MPID_PSP_packed_msg_t *msg, MPI_Datatype datatype)
{
	MPID_PSP_Datatype_release(datatype);

	if (msg->tmp_buf) {
		MPL_free(msg->tmp_buf);
		msg->tmp_buf = NULL;
	}
}


#endif /* _MPID_PSP_PACKED_MSG_H_ */
