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

#ifndef _MPID_PSP_REQUEST_H_
#define _MPID_PSP_REQUEST_H_

#include <assert.h>

static inline
void MPIDI_PSP_Request_set_completed(MPIR_Request *req)
{
	*(req->cc_ptr) = 0;
}

static inline
void MPID_PSP_Subrequest_add(MPIR_Request *req)
{
	/* ToDo: should be explicit atomic */
	(*(req->cc_ptr))++;
}

static inline
int MPID_PSP_Subrequest_completed(MPIR_Request *req)
{
	/* ToDo: should be explicit atomic */
	(*(req->cc_ptr))--;
	return ((*(req->cc_ptr)) == 0);
}
#endif /* _MPID_PSP_REQUEST_H_ */
