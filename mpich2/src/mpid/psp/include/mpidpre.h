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

#ifndef _MPIDPRE_H_
#define _MPIDPRE_H_

#include "list.h"
#include <stdint.h>
#include "mpid_sched_pre.h"

#include "mpid_thread.h"

/* We simply use the fallback timer functionality and do not define
 * our own */
#include "mpid_timers_fallback.h"


#define MPID_CONTEXT_SUBCOMM_WIDTH (0)
#define MPID_CONTEXT_DYNAMIC_PROC_WIDTH (0)

/*********************************************
 * PSCOM Network header
 */

/* pscom network header common (send, recv...) */
typedef struct MPID_PSCOM_XHeader {
	int32_t		tag;
	uint16_t	context_id;
	uint8_t		type; /* one of MPID_PSP_MSGTYPE */
	uint8_t		_reserved_;
	int32_t		src_rank;
} MPID_PSCOM_XHeader_t;


/* pscom network header send/recv */
typedef struct MPID_PSCOM_XHeader_Send {
	MPID_PSCOM_XHeader_t common;
} MPID_PSCOM_XHeader_Send_t;


/* pscom network header RMA put */
typedef struct MPID_PSCOM_XHeader_Rma_put {
	MPID_PSCOM_XHeader_t common;

/*	MPI_Aint	target_disp; */
	int		target_count;
	char		*target_buf;
/*	unsigned int	epoch; */
	struct MPID_Win *win_ptr; /* win_ptr of target (receiver, passive site) */

	long		encoded_type[0];
} MPID_PSCOM_XHeader_Rma_put_t;


/* pscom network header RMA get request */
typedef struct MPID_PSCOM_XHeader_Rma_get_req {
	MPID_PSCOM_XHeader_t common;
/*	MPI_Aint	target_disp; */
	int		target_count;
	char		*target_buf;
/*	unsigned int	epoch; */
	struct MPID_Win *win_ptr; /* win_ptr of target (receiver, passive site) */

	long		encoded_type[0];
} MPID_PSCOM_XHeader_Rma_get_req_t;


/* pscom network header RMA Accumulate */
typedef struct MPID_PSCOM_XHeader_Rma_accumulate {
	MPID_PSCOM_XHeader_t common;
/*	MPI_Aint	target_disp; */
	int		target_count;
	char		*target_buf;
/*	unsigned int	epoch; */
	struct MPID_Win *win_ptr; /* win_ptr of target (receiver, passive site) */
	MPI_Op		op;
	long		encoded_type[0];
} MPID_PSCOM_XHeader_Rma_accumulate_t;


/* pscom network header RMA get request */
typedef struct MPID_PSCOM_XHeader_Rma_get_answer {
	MPID_PSCOM_XHeader_t common;
} MPID_PSCOM_XHeader_Rma_get_answer_t;

/* pscom network header RMA lock/unlock */
typedef struct MPID_PSCOM_XHeader_Rma_lock {
	MPID_PSCOM_XHeader_t common;
	struct MPID_Win	*win_ptr;
} MPID_PSCOM_XHeader_Rma_lock_t;


#define PSCOM_XHEADER_USER_TYPE union pscom_xheader_user
union pscom_xheader_user
{
	MPID_PSCOM_XHeader_t		common;
	MPID_PSCOM_XHeader_Send_t	send;
	MPID_PSCOM_XHeader_Rma_put_t	put;
	MPID_PSCOM_XHeader_Rma_get_req_t	get_req;
	MPID_PSCOM_XHeader_Rma_get_answer_t	get_answer;
	MPID_PSCOM_XHeader_Rma_accumulate_t	accumulate;
	MPID_PSCOM_XHeader_Rma_lock_t	rma_lock;
};


typedef struct pscom_request_sr {
	struct MPID_Request *mpid_req;
} pscom_request_sr_t;


typedef struct MPID_PSP_packed_msg {
	char		*msg;
	unsigned int	msg_sz;
	char		*tmp_buf;
} MPID_PSP_packed_msg_t;


typedef struct pscom_request_put_send {
	struct MPID_Request     *mpid_req;
	MPID_PSP_packed_msg_t	msg;
	struct MPID_Win		*win_ptr;
	int                     target_rank;
} pscom_request_put_send_t;


typedef struct pscom_request_put_recv {
	MPI_Datatype		datatype;
	MPID_PSP_packed_msg_t	msg;
/*	MPID_Win *win_ptr; */
} pscom_request_put_recv_t;


typedef struct pscom_request_accumulate_send {
	struct MPID_Request     *mpid_req;
	MPID_PSP_packed_msg_t	msg;
	struct MPID_Win		*win_ptr;
	int                     target_rank;
} pscom_request_accumulate_send_t;


typedef struct pscom_request_accumulate_recv {
	MPI_Datatype		datatype;
	char			packed_msg[0];
} pscom_request_accumulate_recv_t;


typedef struct pscom_request_get_answer_recv {
	struct MPID_Request     *mpid_req;
	void			*origin_addr;
	int			origin_count;
	MPI_Datatype		origin_datatype;
	MPID_PSP_packed_msg_t	msg;
	struct MPID_Win		*win_ptr;
	int                     target_rank;
} pscom_request_get_answer_recv_t;


typedef struct pscom_request_get_answer_send {
	MPID_PSP_packed_msg_t	msg;
	MPI_Datatype		datatype;
} pscom_request_get_answer_send_t;


typedef struct pscom_request_rma_lock {
	struct list_head next;
	int		exclusive;	/* boolean exclusive or shared lock */
	struct PSCOM_request *req;
} pscom_request_rma_lock_t;


struct PSCOM_req_user
{
	union {
		pscom_request_sr_t sr; /* send and receive */
		pscom_request_put_recv_t	put_recv; /* receive of non contig rma_put */
		pscom_request_put_send_t	put_send;
		pscom_request_accumulate_send_t	accumulate_send;
		pscom_request_accumulate_recv_t accumulate_recv;
		pscom_request_get_answer_send_t get_answer_send;
		pscom_request_get_answer_recv_t get_answer_recv;
		pscom_request_rma_lock_t	rma_lock;
	} type;
};

#include "pscom.h"


typedef size_t MPIDI_msg_sz_t;

#define MPID_PROGRESS_STATE_DECL int foo;

/*
#define MPID_DEV_REQUEST_KIND_DECL		\
	MPID_PSP_REQUEST_MULTI
*/

struct MPID_Datatype;
/* mpidpost.h will include "mpid_datatype.h" */

#define MPIDI_TAG_UB (0x7fffffff)

enum MPID_PSP_MSGTYPE {
	MPID_PSP_MSGTYPE_DATA,			/* Data message */
	MPID_PSP_MSGTYPE_DATA_REQUEST_ACK,	/* Data message and request DATA_ACK acknowledge */
	MPID_PSP_MSGTYPE_DATA_ACK,		/* Acknowledge of DATA_REQUEST_ACK */
	MPID_PSP_MSGTYPE_CANCEL_DATA_ACK,	/* Acknowledge of CANCEL_DATA_REQUEST_ACK */
	MPID_PSP_MSGTYPE_CANCEL_DATA_REQUEST_ACK, /* Cancel an already send DATA message. Request CANCEL_DATA_ACK. */

	/* One Sided communication: */
	MPID_PSP_MSGTYPE_RMA_PUT,
	MPID_PSP_MSGTYPE_RMA_GET_REQ,
	MPID_PSP_MSGTYPE_RMA_GET_ANSWER,
	MPID_PSP_MSGTYPE_RMA_ACCUMULATE,

	MPID_PSP_MSGTYPE_RMA_LOCK_SHARED_REQUEST,
	MPID_PSP_MSGTYPE_RMA_LOCK_EXCLUSIVE_REQUEST,
	MPID_PSP_MSGTYPE_RMA_LOCK_ANSWER,

	MPID_PSP_MSGTYPE_RMA_UNLOCK_REQUEST,
	MPID_PSP_MSGTYPE_RMA_UNLOCK_ANSWER,

	MPID_PSP_MSGTYPE_DATA_CANCELLED,          /* Data message that should be cancelled */
	MPID_PSP_MSGTYPE_MPROBE_RESERVED_REQUEST, /* Message that has been reserved by mprobe */
	MPID_PSP_MSGTYPE_MPROBE_RESERVED_REQUEST_ACK, /* Message that has been reserved by mprobe with ACK request */

	MPID_PSP_MSGTYPE_RMA_FLUSH_REQUEST,
	MPID_PSP_MSGTYPE_RMA_FLUSH_ANSWER,

	MPID_PSP_MSGTYPE_RMA_INTERNAL_LOCK_REQUEST,
	MPID_PSP_MSGTYPE_RMA_INTERNAL_LOCK_ANSWER,

	MPID_PSP_MSGTYPE_RMA_INTERNAL_UNLOCK_REQUEST,
	MPID_PSP_MSGTYPE_RMA_INTERNAL_UNLOCK_ANSWER
};

enum MPID_PSP_Win_lock_state {
	MPID_PSP_LOCK_UNLOCKED = 0,
	MPID_PSP_LOCK_LOCKED,
	MPID_PSP_LOCK_LOCKED_ALL
};

enum MPID_PSP_Win_epoch_states {
	MPID_PSP_EPOCH_NONE = 0,
	MPID_PSP_EPOCH_FENCE,
	MPID_PSP_EPOCH_FENCE_ISSUED,
	MPID_PSP_EPOCH_POST,
	MPID_PSP_EPOCH_START,
	MPID_PSP_EPOCH_PSCW, /* Both post and start have been called. */
	MPID_PSP_EPOCH_LOCK,
	MPID_PSP_EPOCH_LOCK_ALL
};


/*********************************************
 * MPID_PSP Requests
 */

struct MPID_DEV_Request_common
{
	pscom_request_t	*pscom_req;
};

struct MPID_DEV_Request_recv
{
	struct MPID_DEV_Request_common common;

	int32_t		tag;
	uint16_t	context_id;

	MPID_PSP_packed_msg_t msg;

	/* for non-contiguous receive requests only: */
	const char	*addr;
	int		count;
	MPI_Datatype	datatype;

	/* for mrecv only. Reference to the mprobed request. */
	void *mprobe_tag;
};


struct MPID_DEV_Request_send
{
	struct MPID_DEV_Request_common common;

	MPID_PSP_packed_msg_t msg;

	/* for non-contiguous persistent send requests only: */
	const char	*addr;
	int		count;
	MPI_Datatype	datatype;

	/* for persistent send request only: */
	int		rank;
};


struct MPID_DEV_Request_multi
{
	struct MPID_DEV_Request_common common;

	struct list_head requests;
};


struct MPID_DEV_Request_persistent
{
	struct MPID_DEV_Request_common common;

	void		*buf;
	int		count;
	MPI_Datatype	datatype;
	int		rank;
	int		tag;
	struct MPID_Comm	*comm;
	int		context_offset;

	int (*call)(const void * buf, MPI_Aint count, MPI_Datatype datatype, int rank,
		    int tag, struct MPID_Comm * comm, int context_offset, struct MPID_Request ** request);
};


struct MPID_Comm;
struct MPID_Request;
struct MPI_Status;


/* Extend struct MPID_Request (mpidimpl.h) */
#define MPID_DEV_REQUEST_DECL struct MPID_DEV_Request dev;

struct MPID_DEV_Request
{
	union {
		struct MPID_DEV_Request_common common;
		struct MPID_DEV_Request_recv  recv;
		struct MPID_DEV_Request_send  send;
		struct MPID_DEV_Request_multi multi;
		struct MPID_DEV_Request_persistent persistent; /* Persistent send/recv */
		/* mprobe requests using also MPID_DEV_Request_recv recv */
	} kind;
};


/*
 * RMA
 */

#if 0

#define MPID_DEV_WIN_DECL						\
	volatile int my_counter;  /* completion counter for operations	\
				     targeting this window */		\
	void **base_addrs;     /* array of base addresses of the windows of \
				  all processes */			\
	int *disp_units;      /* array of displacement units of all windows */ \
	MPI_Win *all_win_handles;    /* array of handles to the window objects \
					of all processes */		\
	MPIDI_RMA_ops *rma_ops_list; /* list of outstanding RMA requests */ \
	volatile int lock_granted;  /* flag to indicate whether lock has \
				       been granted to this process (as source) for \
				       passive target rma */		\
	volatile int current_lock_type;   /* current lock type on this window (as target) \
					   * (none, shared, exclusive) */ \
	volatile int shared_lock_ref_cnt;				\
	struct MPIDI_Win_lock_queue volatile *lock_queue;  /* list of unsatisfied locks */ \
									\
	int *pt_rma_puts_accs;  /* array containing the no. of passive target \
				   puts/accums issued from this process to other \
				   processes. */			\
	volatile int my_pt_rma_puts_accs;  /* no. of passive target puts/accums	\
					      that this process has	\
					      completed as target */	\

#endif

typedef struct MPID_Win_rank_info
{
	void *base_addr;     /* base address of the window */
	int disp_unit;      /* displacement unit of window */

	struct MPID_Win *win_ptr;    /* window object */

/*	unsigned int epoch_origin; * access epoch */
/*	unsigned int epoch_target; * exposure epoch */

	pscom_connection_t *con;
} MPID_Win_rank_info;


#define MPID_DEV_WIN_DECL						\
	struct MPID_Win_rank_info *rank_info;				\
	int rank;							\
	unsigned int *rma_puts_accs;					\
	unsigned int rma_puts_accs_received;				\
	unsigned int rma_local_pending_cnt;	/* pending io counter */ \
	unsigned int *rma_local_pending_rank;   /* pending io counter per rank */ \
	MPID_Group *start_group_ptr; /* group passed in MPI_Win_start */ \
	int *ranks_start;		/* ranks of last MPID_Win_start call */	\
	unsigned int ranks_start_sz;					\
	int *ranks_post;		/* ranks of last MPID_Win_post call */ \
	unsigned int ranks_post_sz;					\
	struct list_head lock_list; /* list root of pscom_request_rma_lock_t.next */\
	struct list_head lock_list_internal;				\
	pscom_request_t *lock_tail;					\
	int		lock_exclusive;	/* boolean exclusive or shared lock */ \
	unsigned int	lock_cnt;	/* shared lock holder */ \
	unsigned int    lock_internal;  /* lock for internal purpose (atomic ops) */ \
	enum MPID_PSP_Win_lock_state *remote_lock_state; /* array to remember the locked remote ranks */ \
	enum MPID_PSP_Win_epoch_states epoch_state; /* this is for error detection */ \
	int epoch_lock_count;  /* number of pending locks (for error detection, too) */

typedef struct MPIDI_VCRT MPIDI_VCRT_t;
typedef struct MPIDI_VC MPIDI_VC_t;

typedef struct MPIDI_VCON MPIDI_VCON;

#define MPID_DEV_COMM_DECL						\
	pscom_socket_t	*pscom_socket;					\
	pscom_group_t	*group;						\
	pscom_request_t *bcast_request;					\
	int              is_disconnected;				\
	MPIDI_VCRT_t	*vcrt; /* virtual connection reference table */ \
	MPIDI_VC_t	**vcr; /* alias to the array of virtual connections in vcrt  */	\
	MPIDI_VCRT_t	*local_vcrt; /* local virtual connection reference table */ \
	MPIDI_VC_t	**local_vcr; /* alias to the array of local virtual connections in local vcrt */


/* Somewhere in the middle of the GCC 2.96 development cycle, we implemented
   a mechanism by which the user can annotate likely branch directions and
   expect the blocks to be reordered appropriately.  Define __builtin_expect
   to nothing for earlier compilers.  */
#if (!defined(__GNUC__)) || (__GNUC__ == 2 && __GNUC_MINOR__ < 96)
#define __builtin_expect(x, expected_value) (x)
#endif

/*
#define likely(x)	__builtin_expect((x),1)
#define unlikely(x)	__builtin_expect((x),0)
*/
void MPID_PSP_rma_cleanup(void);
void MPID_PSP_rma_pscom_sockets_cleanup(void);

int MPID_PSP_comm_create_hook(struct MPID_Comm * comm);
int MPID_PSP_comm_destroy_hook(struct MPID_Comm * comm);

#define HAVE_DEV_COMM_HOOK
#define MPID_Dev_comm_create_hook(comm) MPID_PSP_comm_create_hook(comm)
#define MPID_Dev_comm_destroy_hook(comm) MPID_PSP_comm_destroy_hook(comm)

/* Progress hooks. */
#define MPID_Progress_register_hook(fn_, id_) MPI_SUCCESS
#define MPID_Progress_deregister_hook(id_)
#define MPID_Progress_activate_hook(id_)
#define MPID_Progress_deactivate_hook(id_)


/* Tell Intercomm create and friends that the GPID routines have been
   implemented */
#define HAVE_GPID_ROUTINES

#define MPID_DEV_GPID_DECL int gpid[2];

#endif /* _MPIDPRE_H_ */
