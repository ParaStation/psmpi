/*
 * ParaStation
 *
 * Copyright (C) 2006-2010 ParTec Cluster Competence Center GmbH, Munich
 *
 * All rights reserved.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#ifndef _MPID_PSP_REQUEST_H_
#define _MPID_PSP_REQUEST_H_

#include <assert.h>

/*
 * struct MPID_Request
 */

void MPID_PSP_Request_destroy(MPID_Request *req);

MPID_Request *MPID_DEV_Request_recv_create(MPID_Comm *comm);
void MPID_DEV_Request_recv_destroy(MPID_Request *req);

MPID_Request *MPID_DEV_Request_send_create(MPID_Comm *comm);
void MPID_DEV_Request_send_destroy(MPID_Request *req);

MPID_Request *MPID_DEV_Request_persistent_create(MPID_Comm *comm, MPID_Request_kind_t type);
void MPID_DEV_Request_persistent_destroy(MPID_Request *req);

/* void MPID_DEV_Request_multi_destroy(MPID_Request *req); */


static inline
void MPID_DEV_Request_add_ref(MPID_Request *req)
{
	MPIU_Object_add_ref(req);
}


static inline
void MPID_DEV_Request_release_ref(MPID_Request *req, MPID_Request_kind_t kind)
{
	int ref_count;

	MPIU_Object_release_ref(req, &ref_count);

	if (ref_count == 0) {
		// assert(kind == req->kind);

		switch (kind) {
		case MPID_REQUEST_RECV:
			MPID_DEV_Request_recv_destroy(req);
			break;
		case MPID_REQUEST_SEND:
			MPID_DEV_Request_send_destroy(req);
			break;
		case MPID_PREQUEST_RECV:
			MPID_DEV_Request_persistent_destroy(req);
			break;
		case MPID_PREQUEST_SEND:
			MPID_DEV_Request_persistent_destroy(req);
			break;
/*
		case MPID_REQUEST_MULTI:
			MPID_DEV_Request_multi_destroy(req);
			break;
*/
		case MPID_UREQUEST:
			MPID_PSP_Request_destroy(req);
			break;
		case MPID_REQUEST_UNDEFINED:
		case MPID_LAST_REQUEST_KIND:
			assert(0);
			break;
		}
	}
}


static inline
int MPID_Request_is_completed(MPID_Request *req)
{
	return !*(req->cc_ptr);
}

static inline
void _MPID_Request_set_completed(MPID_Request *req)
{
	*(req->cc_ptr) = 0;
}


static inline
void MPID_PSP_Subrequest_add(MPID_Request *req)
{
	/* ToDo: should be explicit atomic */
	(*(req->cc_ptr))++;
}


static inline
void MPID_PSP_Subrequest_completed(MPID_Request *req)
{
	/* ToDo: should be explicit atomic */
	(*(req->cc_ptr))--;
}


static inline
void MPID_PSP_Request_enqueue(MPID_Request *req)
{
	MPID_DEV_Request_add_ref(req);
}


static inline
void MPID_PSP_Request_dequeue(MPID_Request *req, MPID_Request_kind_t kind)
{
	MPID_DEV_Request_release_ref(req, kind);
}

#endif /* _MPID_PSP_REQUEST_H_ */
