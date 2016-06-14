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

#ifndef _MPIDPOST_H_
#define _MPIDPOST_H_


#include "mpid_datatype.h"

int MPID_PG_ForwardPGInfo( MPID_Comm *peer_ptr, MPID_Comm *comm_ptr,
			   int nPGids, int gpids[],
			   int root, int remote_leader, int cts_tag,
			   pscom_connection_t *con, char *all_ports[], pscom_socket_t *pscom_socket );

int MPID_GPID_Get(MPID_Comm *comm_ptr, int rank, int gpid[]);

#define MPID_REQUEST_SET_COMPLETED(req_)	\
{						\
    MPID_cc_set((req_)->cc_ptr, 0);             \
}


#endif /* _MPIDPOST_H_ */
