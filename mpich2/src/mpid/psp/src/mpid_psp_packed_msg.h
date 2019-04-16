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

#ifndef _MPID_PSP_PACKED_MSG_H_
#define _MPID_PSP_PACKED_MSG_H_

#include "mpid_psp_datatype.h"


/* May return MPI_ERR_NO_MEM.
   cleanup with packed_msg_cleanup */
static inline
int MPID_PSP_packed_msg_prepare(const void *addr, int count, MPI_Datatype datatype,
				MPID_PSP_packed_msg_t *msg, int buffered)
{
	int		contig;
	size_t		data_sz;
	MPIR_Datatype 	*dtp;
	MPI_Aint	true_lb;

	MPIDI_Datatype_get_info(count, datatype,
				contig, data_sz,
				dtp, true_lb);

	if (!buffered && (contig || !data_sz) ) {
		msg->msg = (char *)addr + true_lb;
		msg->msg_sz = data_sz;
		msg->tmp_buf = NULL;
	} else {
		/* non-contiguous data (or GPU memory) */
		char *tmp_buf = MPL_malloc(data_sz, MPL_MEM_OTHER);

		msg->msg = tmp_buf;
		msg->msg_sz = data_sz;
		msg->tmp_buf = tmp_buf;

		if (unlikely(!tmp_buf && data_sz)) { /* Error: No mem */
			msg->msg_sz = 0;
			return MPI_ERR_NO_MEM;
		}

	}

/*	printf("Packed src:(%d) %s\n", origin_data_sz, pscom_dumpstr(msg->msg, pscom_min(origin_data_sz, 64)));
	fflush(stdout); */

	return MPI_SUCCESS;
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
	if (msg->tmp_buf) {
		MPIR_Segment segment;
		DLOOP_Offset last = msg->msg_sz;

		MPIR_Segment_init(src_addr, src_count, src_datatype, &segment);
		MPIR_Segment_pack(&segment, /* first */0, &last, msg->tmp_buf);
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
int MPID_PSP_packed_msg_unpack(const void *addr, int count, MPI_Datatype datatype,
                               const MPID_PSP_packed_msg_t *msg, size_t data_len)
{
	if (msg->tmp_buf) {
		MPIR_Segment segment;
		DLOOP_Offset last  = pscom_min(msg->msg_sz, data_len);

		MPIR_Segment_init(addr, count, datatype, &segment);
		MPIR_Segment_unpack(&segment, /* first */0, &last, msg->tmp_buf);
		/* From ch3u_handle_recv_pkt.c:
		   "If the data can't be unpacked, then we have a mismatch between
		   the datatype and the amount of data received."
		   (see also Segment_manipulate() in mpid/common/datatype/dataloop/segment.c)
		   For a matching signature, 'last' should still point to the end of the
		   dataloop stream after unpacking it:
		*/
		if( last != pscom_min(msg->msg_sz, data_len) ) {
			return MPI_ERR_TYPE;
		}
	}
	return MPI_SUCCESS;
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
