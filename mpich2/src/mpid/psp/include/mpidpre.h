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

#ifndef _MPIDPRE_H_
#define _MPIDPRE_H_

#include "list.h"
#include <stdint.h>

#include "mpid_thread.h"
#include "mpid_sched.h"
#include "mpid_cuda_aware.h"

/* MPIDI_PSP_WITH_SESSION_STATISTICS is set if psmpi is configured with --with-session-statistics */
#ifdef MPIDI_PSP_WITH_SESSION_STATISTICS

#define MPID_PSP_HISTOGRAM
/* When MPID_PSP_HISTOGRAM is defined and PSP_HISTOGRAM=1 is set, some statistics
 * about the distribution of message sizes will be gathered during the run by all processes
 * and eventually accumulated and printed by world rank 0 within the MPI_Finalize call. */

#ifdef HAVE_LIBHCOLL
#define MPID_PSP_HCOLL_STATS
/* When MPID_PSP_HCOLL_STATS is defined and HCOLL is enabled and PSP_HCOLL_STATS=1 is set,
 * MPI_Finalize also prints some information about the usage of HCOLL collectives. */
#endif

#endif /* MPIDI_PSP_WITH_SESSION_STATISTICS */

/* MPIDI_PSP_WITH_TOPOLOGY_AWARENESS is set if psmpi is configured with --with-topology-awareness */
#ifdef MPIDI_PSP_WITH_TOPOLOGY_AWARENESS

#define MPID_PSP_MSA_AWARENESS
/* When MPID_PSP_MSA_AWARNESS is defined, the MPI_INFO_ENV object contains a key/value pair
 * indicating the module affiliation of the querying rank. The info key is "msa_module_id".
 */

#define MPID_PSP_TOPOLOGY_AWARE_COLLOPS
/* When MPID_PSP_TOPOLOGY_AWARE_COLLOPS is defined, the additional functions MPID_Get_badge()
 * and MPID_Get_max_badge() have to provide topology information (in terms of node IDs for
 * SMP islands) for identifying SMP nodes and/or MSA modules for applying hierarchy-aware
 * communication topologies for collective MPI operations within the upper MPICH layer.
 */
#endif

#ifdef HAVE_LIBHCOLL
#include "hcoll/api/hcoll_dte.h"
typedef struct {
    hcoll_datatype_t hcoll_datatype;
    int foo; /* Shut up the compiler */
} MPIDI_Devdt_t;
#define MPID_DEV_DATATYPE_DECL   MPIDI_Devdt_t   dev;
#endif

typedef struct {
    int gpid[2];
} MPIDI_Gpid;

/* TODO: dummy typedef taken from ch4 device */
typedef struct {
    int progress_count;
} MPID_Progress_state;

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
	struct MPIR_Win *win_ptr; /* win_ptr of target (receiver, passive site) */

	long		encoded_type[0];
} MPID_PSCOM_XHeader_Rma_put_t;

/* pscom network header RMA get memory locations */
typedef struct MPID_PSCOM_XHeader_Rma_get_mem_locations {
	void 		*origin_addr;
	char		*target_buf;
} MPID_PSCOM_XHeader_Rma_get_mem_locations_t;

/* pscom network header RMA get request */
typedef struct MPID_PSCOM_XHeader_Rma_get_req {
	MPID_PSCOM_XHeader_t common;
/*	MPI_Aint	target_disp; */
	MPID_PSCOM_XHeader_Rma_get_mem_locations_t mem_locations;
	int		target_count;
/*	unsigned int	epoch; */
	struct MPIR_Win *win_ptr; /* win_ptr of target (receiver, passive site) */

	long		encoded_type[0];
} MPID_PSCOM_XHeader_Rma_get_req_t;


/* pscom network header RMA Accumulate */
typedef struct MPID_PSCOM_XHeader_Rma_accumulate {
	MPID_PSCOM_XHeader_t common;
/*	MPI_Aint	target_disp; */
	int		target_count;
	char		*target_buf;
/*	unsigned int	epoch; */
	struct MPIR_Win *win_ptr; /* win_ptr of target (receiver, passive site) */
	MPI_Op		op;
	long		encoded_type[0];
} MPID_PSCOM_XHeader_Rma_accumulate_t;


/* pscom network header RMA get answer */
typedef struct MPID_PSCOM_XHeader_Rma_get_answer {
	MPID_PSCOM_XHeader_t common;
	MPID_PSCOM_XHeader_Rma_get_mem_locations_t mem_locations;
} MPID_PSCOM_XHeader_Rma_get_answer_t;

/* pscom network header RMA lock/unlock */
typedef struct MPID_PSCOM_XHeader_Rma_lock {
	MPID_PSCOM_XHeader_t common;
	struct MPIR_Win	*win_ptr;
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
	struct MPIR_Request *mpid_req;
} pscom_request_sr_t;


typedef struct MPID_PSP_packed_msg {
	char		*msg;
	size_t		msg_sz;
	char		*tmp_buf;
} MPID_PSP_packed_msg_t;


typedef struct pscom_request_put_send {
	struct MPIR_Request     *mpid_req;
	MPID_PSP_packed_msg_t	msg;
	struct MPIR_Win		*win_ptr;
	int                     target_rank;
} pscom_request_put_send_t;


typedef struct pscom_request_put_recv {
	MPI_Datatype		datatype;
	MPID_PSP_packed_msg_t	msg;
/*	MPIR_Win *win_ptr; */
} pscom_request_put_recv_t;


typedef struct pscom_request_accumulate_send {
	struct MPIR_Request     *mpid_req;
	MPID_PSP_packed_msg_t	msg;
	struct MPIR_Win		*win_ptr;
	int                     target_rank;
} pscom_request_accumulate_send_t;


typedef struct pscom_request_accumulate_recv {
	MPI_Datatype		datatype;
	char			packed_msg[0];
} pscom_request_accumulate_recv_t;


typedef struct pscom_request_get_answer_recv {
	struct MPIR_Request     *mpid_req;
	void			*origin_addr;
	char 			*target_buf;
	int			origin_count;
	MPI_Datatype		origin_datatype;
	MPID_PSP_packed_msg_t	msg;
	struct MPIR_Win		*win_ptr;
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

#ifdef MPIDI_PSP_WITH_CUDA_AWARENESS
#define PSCOM_CUDA_AWARENESS
#endif

#include "pscom.h"

/*
 * If MPICH is configured using the device-specific memcpy() we rely on the
 * pscom's CUDA-awareness. Otherwise, just fall back to normal memcpy().
 * TODO: can we safely use CHECK_MEMCPY here?
 */
#if defined(MPIR_USE_DEVICE_MEMCPY) && defined(MPIDI_PSP_WITH_CUDA_AWARENESS)
#define MPID_Memcpy(dst, src, len)         \
	do {                                   \
		pscom_memcpy((dst), (src), (len)); \
	} while (0)
#else
#define MPID_Memcpy(dst, src, len)              \
	do {                                        \
		CHECK_MEMCPY((dst),(src),(len));        \
		memcpy((dst), (src), (len));            \
	} while (0)
#endif



typedef size_t MPIDI_msg_sz_t;

#define MPID_PROGRESS_STATE_DECL int foo;

/*
#define MPID_DEV_REQUEST_KIND_DECL		\
	MPID_PSP_REQUEST_MULTI
*/

struct MPIR_Datatype ;
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

	MPID_PSP_MSGTYPE_RMA_SYNC,

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
	MPID_PSP_MSGTYPE_RMA_INTERNAL_UNLOCK_ANSWER,

	MPID_PSP_MSGTYPE_FINALIZE_TOKEN
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
	char		*addr;
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
	struct MPIR_Comm	*comm;
	int		context_offset;

	int (*call)(const void * buf, MPI_Aint count, MPI_Datatype datatype, int rank,
		    int tag, struct MPIR_Comm * comm, int context_offset, struct MPIR_Request ** request);
};


struct MPI_Status;


/* Extend struct MPIR_Request (mpidimpl.h) */
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

	struct MPIR_Win *win_ptr;    /* window object */

/*	unsigned int epoch_origin; * access epoch */
/*	unsigned int epoch_target; * exposure epoch */

	pscom_connection_t *con;
} MPID_Win_rank_info;


#define MPID_DEV_WIN_DECL						\
	struct MPID_Win_rank_info *rank_info;				\
	int rank;							\
	int rma_accumulate_ordering; /* flag whether accumulate needs strict ordering */ \
	int *rma_pending_accumulates; /* flags for pending accumulates */ \
	unsigned int *rma_puts_accs;					\
	unsigned int rma_puts_accs_received;				\
	unsigned int rma_local_pending_cnt;	/* pending io counter */ \
	unsigned int *rma_local_pending_rank;   /* pending io counter per rank */ \
	MPIR_Group *start_group_ptr; /* group passed in MPI_Win_start */ \
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

/* Just for HCOLL integration: */
typedef struct MPIDI_CH3I_comm
{
	struct MPIDI_VCRT *vcrt; /* virtual connecton reference table */
}
MPIDI_CH3I_comm_t;

#define MPID_DEV_COMM_DECL						\
	pscom_socket_t	*pscom_socket;					\
	pscom_group_t	*group;						\
	pscom_request_t *bcast_request;					\
	int              is_disconnected;				\
	int              is_checked_as_host_local;			\
	union {								\
		MPIDI_VCRT_t	*vcrt; /* virtual connection reference table */ \
		MPIDI_CH3I_comm_t dev;					\
	};								\
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

int MPIDI_PSP_Comm_commit_pre_hook(MPIR_Comm * comm);
int MPIDI_PSP_Comm_commit_post_hook(MPIR_Comm *comm);
int MPIDI_PSP_Comm_destroy_hook(MPIR_Comm * comm);

#define HAVE_DEV_COMM_HOOK
#define MPID_Comm_commit_pre_hook(comm_) MPIDI_PSP_Comm_commit_pre_hook(comm_)
#define MPID_Comm_commit_post_hook(comm_) MPIDI_PSP_Comm_commit_post_hook(comm_)
#define MPID_Comm_free_hook(comm_) MPIDI_PSP_Comm_destroy_hook(comm_)

/* Progress hooks. */
#define MPID_Progress_register_hook(fn_, id_) MPI_SUCCESS
#define MPID_Progress_deregister_hook(id_)
#define MPID_Progress_activate_hook(id_)
#define MPID_Progress_deactivate_hook(id_)


/* Tell Intercomm create and friends that the GPID routines have been
   implemented */
#define HAVE_GPID_ROUTINES

#define MPID_DEV_GPID_DECL int gpid[2];

int MPID_Init(int required, int *provided);

int MPID_InitCompleted( void );

int MPID_Finalize(void);
#define MPID_CS_finalize() do {} while (0)
int MPID_Abort( MPIR_Comm *comm, int mpi_errno, int exit_code, const char *error_msg );

int MPID_Open_port(MPIR_Info *, char *);
int MPID_Close_port(const char *);

int MPID_Comm_accept(const char *, MPIR_Info *, int, MPIR_Comm *, MPIR_Comm **);

int MPID_Comm_connect(const char *, MPIR_Info *, int, MPIR_Comm *, MPIR_Comm **);

int MPID_Comm_disconnect(MPIR_Comm *);

int MPID_Comm_spawn_multiple(int, char *[], char **[], const int [], MPIR_Info* [],
                             int, MPIR_Comm *, MPIR_Comm **, int []);

int MPID_Comm_failure_ack(MPIR_Comm *comm);

int MPID_Comm_failure_get_acked(MPIR_Comm *comm, MPIR_Group **failed_group_ptr);

int MPID_Comm_get_all_failed_procs(MPIR_Comm *comm_ptr, MPIR_Group **failed_group, int tag);

int MPID_Comm_revoke(MPIR_Comm *comm, int is_remote);

int MPID_Send( const void *buf, MPI_Aint count, MPI_Datatype datatype,
	       int dest, int tag, MPIR_Comm *comm, int context_offset,
	       MPIR_Request **request );

int MPID_Send_coll( const void *buf, MPI_Aint count, MPI_Datatype datatype,
                    int dest, int tag, MPIR_Comm *comm, int context_offset,
                    MPIR_Request **request, MPIR_Errflag_t * errflag );

int MPID_Rsend( const void *buf, int count, MPI_Datatype datatype,
		int dest, int tag, MPIR_Comm *comm, int context_offset,
		MPIR_Request **request );

int MPID_Ssend( const void *buf, MPI_Aint count, MPI_Datatype datatype,
		int dest, int tag, MPIR_Comm *comm, int context_offset,
		MPIR_Request **request );

int MPID_tBsend( const void *buf, int count, MPI_Datatype datatype,
		 int dest, int tag, MPIR_Comm *comm, int context_offset );
/* see mpidpost.h
int MPID_Isend( const void *buf, MPI_Aint count, MPI_Datatype datatype,
		int dest, int tag, MPIR_Comm *comm, int context_offset,
		MPIR_Request **request );
*/
int MPID_Isend_coll( const void *buf, MPI_Aint count, MPI_Datatype datatype,
                     int dest, int tag, MPIR_Comm *comm, int context_offset,
                     MPIR_Request **request, MPIR_Errflag_t * errflag );

int MPID_Irsend( const void *buf, MPI_Aint count, MPI_Datatype datatype,
		 int dest, int tag, MPIR_Comm *comm, int context_offset,
		 MPIR_Request **request );
/* see mpidpost.h
int MPID_Issend( const void *buf, MPI_Aint count, MPI_Datatype datatype,
		 int dest, int tag, MPIR_Comm *comm, int context_offset,
		 MPIR_Request **request );
*/
int MPID_Recv( void *buf, MPI_Aint count, MPI_Datatype datatype,
	       int source, int tag, MPIR_Comm *comm, int context_offset,
	       MPI_Status *status, MPIR_Request **request );

/* see mpidpost.h
int MPID_Irecv( void *buf, MPI_Aint count, MPI_Datatype datatype,
		int source, int tag, MPIR_Comm *comm, int context_offset,
		MPIR_Request **request );
*/
int MPID_Send_init( const void *buf, int count, MPI_Datatype datatype,
		    int dest, int tag, MPIR_Comm *comm, int context_offset,
		    MPIR_Request **request );

int MPID_Bsend_init(const void *, int, MPI_Datatype, int, int, MPIR_Comm *,
		   int, MPIR_Request **);
int MPID_Rsend_init( const void *buf, int count, MPI_Datatype datatype,
		     int dest, int tag, MPIR_Comm *comm, int context_offset,
		     MPIR_Request **request );
int MPID_Ssend_init( const void *buf, int count, MPI_Datatype datatype,
		     int dest, int tag, MPIR_Comm *comm, int context_offset,
		     MPIR_Request **request );

int MPID_Recv_init( void *buf, int count, MPI_Datatype datatype,
		    int source, int tag, MPIR_Comm *comm, int context_offset,
		    MPIR_Request **request );

int MPID_Startall(int count, MPIR_Request *requests[]);

int MPID_Probe(int, int, MPIR_Comm *, int, MPI_Status *);
int MPID_Iprobe(int, int, MPIR_Comm *, int, int *, MPI_Status *);

int MPID_Mprobe(int source, int tag, MPIR_Comm *comm, int context_offset,
                MPIR_Request **message, MPI_Status *status);

int MPID_Improbe(int source, int tag, MPIR_Comm *comm, int context_offset,
                 int *flag, MPIR_Request **message, MPI_Status *status);
/* see mpidpost.h
int MPID_Imrecv(void *buf, int count, MPI_Datatype datatype,
                MPIR_Request *message, MPIR_Request **rreqp);
*/
int MPID_Mrecv(void *buf, int count, MPI_Datatype datatype,
               MPIR_Request *message, MPI_Status *status, MPIR_Request **rreq);

int MPID_Cancel_send(MPIR_Request *);
int MPID_Cancel_recv(MPIR_Request *);

MPI_Aint MPID_Aint_add(MPI_Aint base, MPI_Aint disp);

MPI_Aint MPID_Aint_diff(MPI_Aint addr1, MPI_Aint addr2);

int MPID_Win_create(void *, MPI_Aint, int, MPIR_Info *, MPIR_Comm *,
                    MPIR_Win **);
int MPID_Win_free(MPIR_Win **);

int MPID_Put(const void *, int, MPI_Datatype, int, MPI_Aint, int,
             MPI_Datatype, MPIR_Win *);
int MPID_Get(void *, int, MPI_Datatype, int, MPI_Aint, int,
             MPI_Datatype, MPIR_Win *);
int MPID_Accumulate(const void *, int, MPI_Datatype, int, MPI_Aint, int,
                    MPI_Datatype, MPI_Op, MPIR_Win *);

int MPID_Win_fence(int, MPIR_Win *);
int MPID_Win_post(MPIR_Group *group_ptr, int assert, MPIR_Win *win_ptr);
int MPID_Win_start(MPIR_Group *group_ptr, int assert, MPIR_Win *win_ptr);
int MPID_Win_test(MPIR_Win *win_ptr, int *flag);
int MPID_Win_wait(MPIR_Win *win_ptr);
int MPID_Win_complete(MPIR_Win *win_ptr);

int MPID_Win_lock(int lock_type, int dest, int assert, MPIR_Win *win_ptr);
int MPID_Win_unlock(int dest, MPIR_Win *win_ptr);

int MPID_Win_allocate(MPI_Aint size, int disp_unit, MPIR_Info *info,
                      MPIR_Comm *comm, void *baseptr, MPIR_Win **win);
int MPID_Win_allocate_shared(MPI_Aint size, int disp_unit, MPIR_Info *info_ptr, MPIR_Comm *comm_ptr,
                             void *base_ptr, MPIR_Win **win_ptr);
int MPID_Win_shared_query(MPIR_Win *win, int rank, MPI_Aint *size, int *disp_unit,
                          void *baseptr);
int MPID_Win_create_dynamic(MPIR_Info *info, MPIR_Comm *comm, MPIR_Win **win);
int MPID_Win_attach(MPIR_Win *win, void *base, MPI_Aint size);
int MPID_Win_detach(MPIR_Win *win, const void *base);
int MPID_Win_get_info(MPIR_Win *win, MPIR_Info **info_used);
int MPID_Win_set_info(MPIR_Win *win, MPIR_Info *info);

int MPID_Get_accumulate(const void *origin_addr, int origin_count,
                        MPI_Datatype origin_datatype, void *result_addr, int result_count,
                        MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
                        int target_count, MPI_Datatype target_datatype, MPI_Op op, MPIR_Win *win);
int MPID_Fetch_and_op(const void *origin_addr, void *result_addr,
                      MPI_Datatype datatype, int target_rank, MPI_Aint target_disp,
                      MPI_Op op, MPIR_Win *win);
int MPID_Compare_and_swap(const void *origin_addr, const void *compare_addr,
                          void *result_addr, MPI_Datatype datatype, int target_rank,
                          MPI_Aint target_disp, MPIR_Win *win);
int MPID_Rput(const void *origin_addr, int origin_count,
              MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
              int target_count, MPI_Datatype target_datatype, MPIR_Win *win,
              MPIR_Request **request);
int MPID_Rget(void *origin_addr, int origin_count,
              MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
              int target_count, MPI_Datatype target_datatype, MPIR_Win *win,
              MPIR_Request **request);
int MPID_Raccumulate(const void *origin_addr, int origin_count,
                     MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
                     int target_count, MPI_Datatype target_datatype, MPI_Op op, MPIR_Win *win,
                     MPIR_Request **request);
int MPID_Rget_accumulate(const void *origin_addr, int origin_count,
                         MPI_Datatype origin_datatype, void *result_addr, int result_count,
                         MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp,
                         int target_count, MPI_Datatype target_datatype, MPI_Op op, MPIR_Win *win,
                         MPIR_Request **request);

int MPID_Win_lock_all(int assert, MPIR_Win *win);
int MPID_Win_unlock_all(MPIR_Win *win);
int MPID_Win_flush(int rank, MPIR_Win *win);
int MPID_Win_flush_all(MPIR_Win *win);
int MPID_Win_flush_local(int rank, MPIR_Win *win);
int MPID_Win_flush_local_all(MPIR_Win *win);
int MPID_Win_sync(MPIR_Win *win);

void MPID_Progress_start(MPID_Progress_state * state);
int MPID_Progress_wait(MPID_Progress_state * state);
void MPID_Progress_end(MPID_Progress_state * stae);
int MPID_Progress_poke(void);

int MPID_Get_processor_name( char *name, int namelen, int *resultlen);
int MPID_Get_universe_size(int  * universe_size);
int MPID_Comm_get_lpid(MPIR_Comm *comm_ptr, int idx, int * lpid_ptr, bool is_remote);

void MPID_Request_create_hook(MPIR_Request *);
void MPID_Request_free_hook(MPIR_Request *);
void MPID_Request_destroy_hook(MPIR_Request *);
int MPID_Request_complete(MPIR_Request *);

void *MPID_Alloc_mem( size_t size, MPIR_Info *info );
int MPID_Free_mem( void *ptr );

/* Prototypes and definitions for the node ID code.  This is used to support
   hierarchical collectives in a (mostly) device-independent way. */
int MPID_Get_node_id(MPIR_Comm *comm, int rank, int *id_p);
int MPID_Get_max_node_id(MPIR_Comm *comm, int *max_id_p);
/* The PSP layer extends this by multi-level hierarchies and provides the
   following additional functions for this: */
int MPID_Get_badge(MPIR_Comm *comm, int rank, int *badge_p);
int MPID_Get_max_badge(MPIR_Comm *comm, int *max_badge_p);

int MPID_Type_commit_hook(MPIR_Datatype * type);
int MPID_Type_free_hook(MPIR_Datatype * type);
int MPID_Op_commit_hook(MPIR_Op * op);
int MPID_Op_free_hook(MPIR_Op * op);

#ifdef MPIDI_PSP_WITH_CUDA_AWARENESS
int MPID_PSP_Reduce_for_cuda(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
			     MPI_Op op, int root, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag);
int MPID_PSP_Allreduce_for_cuda(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
				MPI_Op op, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag);
int MPID_PSP_Reduce_scatter_for_cuda(const void *sendbuf, void *recvbuf, const int recvcounts[],
				     MPI_Datatype datatype, MPI_Op op, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag);
int MPID_PSP_Reduce_scatter_block_for_cuda(const void *sendbuf, void *recvbuf,  int recvcount,
					   MPI_Datatype datatype, MPI_Op op, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag);
int MPID_PSP_Scan_for_cuda(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
			   MPI_Op op, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag);
int MPID_PSP_Exscan_for_cuda(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
			   MPI_Op op, MPIR_Comm *comm_ptr, MPIR_Errflag_t *errflag);
#endif


#ifdef MPIDI_PSP_WITH_SESSION_STATISTICS
typedef enum {
	mpidi_psp_stats_collops_enum__bcast,
	mpidi_psp_stats_collops_enum__barrier,
	mpidi_psp_stats_collops_enum__reduce,
	mpidi_psp_stats_collops_enum__allreduce,
	mpidi_psp_stats_collops_enum__allgather,
	mpidi_psp_stats_collops_enum__alltoall,
	mpidi_psp_stats_collops_enum__alltoallv,
	mpidi_psp_stats_collops_enum__MAX
} MPIDI_PSP_stats_collops_enum_t;
#endif

#endif /* _MPIDPRE_H_ */
