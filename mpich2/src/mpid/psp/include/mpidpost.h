/*
 * ParaStation
 *
 * Copyright (C) 2006-2010 ParTec Cluster Competence Center GmbH, Munich
 *
 * All rights reserved.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#ifndef _MPIDPOST_H_
#define _MPIDPOST_H_


#include "mpid_datatype.h"

int MPID_PG_ForwardPGInfo( MPID_Comm *peer_ptr, MPID_Comm *comm_ptr,
			   int nPGids, int gpids[],
			   int root );
int MPID_GPID_Get(MPID_Comm *comm_ptr, int rank, int gpid[]);

#endif /* _MPIDPOST_H_ */
