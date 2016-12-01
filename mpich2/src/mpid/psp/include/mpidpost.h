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

struct MPID_Comm;
int MPID_GPID_GetAllInComm(struct MPID_Comm *comm_ptr, int local_size,
			   MPID_Gpid local_gpids[], int *singlePG);
int MPID_GPID_ToLpidArray(int size, MPID_Gpid gpid[], int lpid[]);
int MPID_Create_intercomm_from_lpids(struct MPID_Comm *newcomm_ptr,
			   int size, const int lpids[]);
int MPID_PG_ForwardPGInfo( MPID_Comm *peer_ptr, MPID_Comm *comm_ptr,
			   int nPGids, const MPID_Gpid gpids[],
			   int root, int remote_leader, int cts_tag,
			   pscom_connection_t *con, char *all_ports, pscom_socket_t *pscom_socket );

int MPID_GPID_Get(MPID_Comm *comm_ptr, int rank, MPID_Gpid gpid[]);

#define MPID_REQUEST_SET_COMPLETED(req_)	\
{						\
    MPID_cc_set((req_)->cc_ptr, 0);             \
}

#define MPID_ICCREATE_REMOTECOMM_HOOK(peer_comm_ptr, local_comm_ptr, remote_size, remote_gpids, local_leader) \
  MPID_PG_ForwardPGInfo(peer_comm_ptr, local_comm_ptr, remote_size, remote_gpids, local_leader, remote_leader, cts_tag, NULL, NULL, NULL)

#endif /* _MPIDPOST_H_ */
