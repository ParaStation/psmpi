/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2023 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#ifndef _MPIDIMPL_H_
#define _MPIDIMPL_H_

#include "mpiimpl.h"
#include "list.h"
#include "mpid_sched.h"

void MPID_PSP_shm_rma_init(void);
void MPID_PSP_shm_rma_get_base(MPIR_Win *win_ptr, int rank, int *disp, void **base);
void MPID_PSP_shm_rma_mutex_init(MPIR_Win *win_ptr);
void MPID_PSP_shm_rma_mutex_lock(MPIR_Win *win_ptr);
void MPID_PSP_shm_rma_mutex_unlock(MPIR_Win *win_ptr);
void MPID_PSP_shm_rma_mutex_destroy(MPIR_Win *win_ptr);

#define PRINTERROR(fmt, args...) fprintf(stderr, "Error:" fmt "\n" ,##args)

#define PSCOM_PORT_MAXLEN 64 /* "xxx.xxx.xxx.xxx:xxxxx@01234567____" */
typedef char pscom_port_str_t[PSCOM_PORT_MAXLEN];

pscom_port_str_t *MPID_PSP_open_all_ports(int root, MPIR_Comm *comm, MPIR_Comm *intercomm);

#ifdef MPID_PSP_MSA_AWARENESS
typedef struct MPIDI_PSP_topo_level MPIDI_PSP_topo_level_t;
struct MPIDI_PSP_topo_level {
	struct MPIDI_PG *pg;
	struct MPIDI_PSP_topo_level *next;
	int degree;
	int max_badge;
	int badges_are_global; // FIX ME: Do we want to have an array for this?
	int *badge_table;
};
#define MPIDI_PSP_TOPO_BADGE__UNKNOWN(level) (MPIDI_PSP_get_max_badge_by_level(level) + 1)
#define MPIDI_PSP_TOPO_BADGE__NULL -1
#define MPIDI_PSP_TOPO_LEVEL__MODULES 4096
#define MPIDI_PSP_TOPO_LEVEL__NODES   1024
#else
typedef void MPIDI_PSP_topo_level_t;
#endif

#define MPIDI_PSP_INVALID_LPID ((uint64_t)-1)

typedef struct MPIDI_PG MPIDI_PG_t;
struct MPIDI_PG {
	struct MPIDI_PG * next;
	int refcnt;
	int size;
	int id_num;
	MPIDI_VC_t **vcr;
	uint64_t * lpids;
#ifdef MPID_PSP_MSA_AWARENESS
	struct MPIDI_PSP_topo_level *topo_levels;
#endif
	pscom_connection_t **cons;

};


struct MPIDI_VC {
	pscom_connection_t *con;
	uint64_t lpid;
	int pg_rank;
	MPIDI_PG_t * pg;
	int refcnt;
};

struct MPIDI_VCRT {
	int size;
	int refcnt;
	union {
		struct MPIDI_VC* vcr[0];
		struct MPIDI_VC* vcr_table[0];
	};
};


MPIDI_VCRT_t *MPIDI_VCRT_Create(int size);
MPIDI_VCRT_t *MPIDI_VCRT_Dup(MPIDI_VCRT_t *vcrt);
int MPIDI_VCRT_Release(MPIDI_VCRT_t *vcrt, int isDisconnect);

MPIDI_VC_t *MPIDI_VC_Dup(MPIDI_VC_t *orig_vcr);
MPIDI_VC_t *MPIDI_VC_Create(MPIDI_PG_t * pg, int pg_rank, pscom_connection_t *con, uint64_t lpid);

int MPID_PSP_get_host_hash(void);
int MPID_PSP_split_type(MPIR_Comm * comm_ptr, int split_type, int key, MPIR_Info * info_ptr, MPIR_Comm ** newcomm_ptr);

int MPID_PSP_comm_init(int has_parent);
void MPID_PSP_comm_set_vcrt(MPIR_Comm *comm, MPIDI_VCRT_t *vcrt);
void MPID_PSP_comm_set_local_vcrt(MPIR_Comm *comm, MPIDI_VCRT_t *vcrt);
void MPID_PSP_comm_create_mapper(MPIR_Comm * comm);

int MPIDI_PG_Create(int pg_size, int pg_id_num, MPIDI_PSP_topo_level_t *level, MPIDI_PG_t ** pg_ptr);
MPIDI_PG_t* MPIDI_PG_Destroy(MPIDI_PG_t * pg_ptr);
void MPIDI_PG_Convert_id(char *pg_id_name, int *pg_id_num);

typedef struct MPIDI_Process
{
	pscom_socket_t *socket;

	pscom_connection_t **grank2con;

	int		my_pg_rank;
	unsigned int	my_pg_size;
	unsigned int 	singleton_but_no_pm;

	char *pg_id_name;
	uint64_t next_lpid;
	MPIDI_PG_t * my_pg;
#ifdef MPID_PSP_MSA_AWARENESS
	MPIDI_PSP_topo_level_t *topo_levels;
#endif
	int shm_attr_key;

	int smp_node_id;
	int msa_module_id;
	uint8_t use_world_model;

	struct {
		unsigned debug_level;
		unsigned debug_version;
		unsigned enable_collectives;
		unsigned enable_ondemand;
		unsigned enable_ondemand_spawn;
		unsigned enable_smp_awareness;
		unsigned enable_msa_awareness;
		unsigned enable_smp_aware_collops;
#ifdef MPID_PSP_MSA_AWARE_COLLOPS
		unsigned enable_msa_aware_collops;
#endif
#ifdef HAVE_HCOLL
		unsigned enable_hcoll;
#endif
#ifdef MPID_PSP_HISTOGRAM
		unsigned enable_histogram;
#endif
#ifdef MPID_PSP_HCOLL_STATS
		unsigned enable_hcoll_stats;
#endif
		unsigned enable_lazy_disconnect;
		struct {
			int enable_rma_accumulate_ordering;
			int enable_explicit_wait_on_passive_side;
		} rma;
		int hard_abort;
		struct {
			int barrier;
			int timeout;
			int shutdown;
			int exit;
		} finalize;
	} env;

#ifdef MPIDI_PSP_WITH_SESSION_STATISTICS
       struct {
#ifdef MPID_PSP_HISTOGRAM
               struct {
                       char* con_type_str;
                       int   con_type_int;
                       int max_size;
                       int min_size;
                       int step_width;
                       int points;
                       unsigned int* limit;
                       unsigned long long int* count;
	       } histo;
#endif
#ifdef MPID_PSP_HCOLL_STATS
               struct {
                       unsigned long long int counter[mpidi_psp_stats_collops_enum__MAX];
               } hcoll;
#endif
       } stats;
#endif /* MPIDI_PSP_WITH_SESSION_STATISTICS */

	/* 	Partitioned communication lists used on receiver side
		TODO: the following two lists can be optimized
		by saving the requests structured per source rank
		instead of one global list for all source ranks
	*/
	struct list_head part_unexp_list; /* list of unexpected receives (stores received SEND_INIT that can not be matched to partitioned receive request yet)*/
	struct list_head part_posted_list; /* list of posted receive request that could not be matched to SEND_INIT yet*/

} MPIDI_Process_t;

extern MPIDI_Process_t MPIDI_Process;

#ifdef MPID_PSP_MSA_AWARENESS
int MPIDI_PSP_topo_init(void);
int MPIDI_PSP_check_pg_for_level(int degree, MPIDI_PG_t *pg, MPIDI_PSP_topo_level_t **level);
#endif

/* The following two functions are callbacks that are added in MPID_Init() via
 * MPIR_Add_finalize() to the set of finalize hooks that are then called during
 * MPII_Finalize(). Both of them require that the built-in comms are still valid
 * and have thus to be applied with a priority > MPIR_FINALIZE_CALLBACK_PRIO.
 */
int MPIDI_PSP_finalize_print_stats_cb(void *param);
int MPIDI_PSP_finalize_add_barrier_cb(void *param);

int MPIDI_PSP_Isend(const void *buf, MPI_Aint count, MPI_Datatype datatype,
		    int dest, int tag, MPIR_Comm *comm, int context_offset,
		    MPIR_Request **request);
int MPIDI_PSP_Issend(const void * buf, MPI_Aint count, MPI_Datatype datatype,
		     int rank, int tag, MPIR_Comm * comm, int context_offset,
		     MPIR_Request ** request);
int MPIDI_PSP_Irecv(void * buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
		    MPIR_Comm * comm, int context_offset, MPIR_Request ** request);
int MPIDI_PSP_Imrecv(void *buf, int count, MPI_Datatype datatype, MPIR_Request *message,
		     MPIR_Request **request);

void MPID_PSP_RecvAck(MPIR_Request *send_req);
/* persistent receives */
int MPID_PSP_Recv_start(MPIR_Request *request);
/*
int MPID_Recv_init(void * buf, int count, MPI_Datatype datatype, int rank, int tag,
		   MPIR_Comm * comm, int context_offset, MPIR_Request ** request);
*/

/*init persistent request*/
int MPID_PSP_persistent_init(const void *buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
			     MPIR_Comm *comm, int context_offset, MPIR_Request **request,
			     int (*call)(const void * buf, MPI_Aint count, MPI_Datatype datatype, int rank,
					 int tag, struct MPIR_Comm * comm, int context_offset, MPIR_Request ** request),
			     MPIR_Request_kind_t type);

/*start persistent request*/
int MPID_PSP_persistent_start(MPIR_Request *req);

/*start partitioned receive request*/
int MPID_PSP_precv_start(MPIR_Request * req);
/*start partitioned send request*/
int MPID_PSP_psend_start(MPIR_Request * req);
/*callbacks for partitioned communication*/
pscom_request_t * MPID_do_recv_part_send_init(pscom_connection_t *con, pscom_header_net_t *header_net);
pscom_request_t * MPID_do_recv_part_cts(pscom_connection_t *con, pscom_header_net_t *header_net);


/* Control messages */
#define MPIDI_PSP_CTRL_TAG__WIN__POST	  11
#define MPIDI_PSP_CTRL_TAG__WIN__COMPLETE 12
#define MPIDI_PSP_CTRL_TAG__WARMUP__PING  15
#define MPIDI_PSP_CTRL_TAG__WARMUP__PONG  17
void MPIDI_PSP_SendCtrl(int tag, int context_id, int src_rank, pscom_connection_t *con, enum MPID_PSP_MSGTYPE msgtype);
void MPIDI_PSP_RecvCtrl(int tag, int context_id, int src_rank, pscom_connection_t *con, enum MPID_PSP_MSGTYPE msgtype);
void MPIDI_PSP_IprobeCtrl(int tag, int context_id, int src_rank, pscom_connection_t *con, enum MPID_PSP_MSGTYPE msgtype, int *flag);
void MPIDI_PSP_SendRmaCtrl(MPIR_Win *win_ptr, MPIR_Comm *comm, pscom_connection_t *con, int dest_rank, enum MPID_PSP_MSGTYPE msgtype);
void MPIDI_PSP_SendPartitionedCtrl(int tag, int context_id, int src_rank, pscom_connection_t *con, MPI_Aint sdata_size, int requests, MPIR_Request * sreq, MPIR_Request * rreq, enum MPID_PSP_MSGTYPE msgtype);
void MPIDI_PSP_RecvPartitionedCtrl(int tag, int context_id, int src_rank,	pscom_connection_t *con, enum MPID_PSP_MSGTYPE msgtype);

/* from mpid_rma_put.c: */
pscom_request_t *MPID_do_recv_rma_put(pscom_connection_t *con, MPID_PSCOM_XHeader_Rma_put_t *xhead_rma);
/* from mpid_rma_get.c: */
pscom_request_t *MPID_do_recv_rma_get_req(pscom_connection_t *connection, MPID_PSCOM_XHeader_Rma_get_req_t *xhead_get);
/* from mpid_rma_accumulate.c: */
pscom_request_t *MPID_do_recv_rma_accumulate(pscom_connection_t *con,
					     pscom_header_net_t *header_net);
/* from mpid_rma_sync.c: */
void MPID_do_recv_rma_lock_exclusive_req(pscom_request_t *req);
void MPID_do_recv_rma_lock_shared_req(pscom_request_t *req);
void MPID_do_recv_rma_unlock_req(pscom_request_t *req);
void MPID_do_recv_rma_lock_internal_req(pscom_request_t *req);
void MPID_do_recv_rma_unlock_internal_req(pscom_request_t *req);
void MPID_do_recv_rma_flush_req(pscom_request_t *req);

void MPID_enable_receive_dispach(pscom_socket_t *socket);

int MPIDI_PSP_compute_acc_op(void *origin_addr, int origin_cnt,
			      MPI_Datatype origin_datatype, void *target_addr,
			      int target_count, MPI_Datatype target_datatype,
			      MPI_Op op, int packed_source_buf);

/* return connection_t for rank, NULL on error */
pscom_connection_t *MPID_PSCOM_rank2connection(MPIR_Comm *comm, int rank);

int MPIDI_PSP_Wait(MPIR_Request *request);

int MPIDI_PSP_Put_generic(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
			  int target_rank, MPI_Aint target_disp, int target_count,
			  MPI_Datatype target_datatype, MPIR_Win *win_ptr, MPIR_Request **request);
int MPIDI_PSP_Get_generic(void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
			  int target_rank, MPI_Aint target_disp, int target_count,
			  MPI_Datatype target_datatype, MPIR_Win *win_ptr, MPIR_Request **request);

int MPIDI_PSP_Win_lock_internal(int dest, MPIR_Win *win_ptr);
int MPIDI_PSP_Win_unlock_internal(int dest, MPIR_Win *win_ptr);
int MPIDI_PSP_Win_wait_local_completion(int rank, MPIR_Win *win_ptr);
int MPIDI_PSP_Win_wait_passive_completion(int rank, MPIR_Win *win_ptr);

void mpid_debug_init(void);
const char *mpid_msgtype_str(enum MPID_PSP_MSGTYPE msg_type);

#define Dprintf(fmt, arg...)

#if !defined(Dprintf)

#define Dprintf(fmt, arg...) do {					\
	printf("#psp %d %s: " fmt "\n", MPIDI_Process.my_pg_rank,	\
	       __func__,##arg);						\
} while (0)

#endif

static inline
int MPIDI_PSP_env_get_int(const char *env_name, int _default)
{
	char *val = getenv(env_name);
	return val ? atoi(val) : _default;
}

#ifndef MPICH_IS_THREADED
#define MPID_PSP_LOCKFREE_CALL(code) code;
#else
#define MPID_PSP_LOCKFREE_CALL(code) do {				\
	MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);	\
	code;								\
	MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);	\
} while (0);
#endif


int MPID_PSP_GetParentPort(char **parent_port);


/*----------------------
  BEGIN DATATYPE SECTION (from mpid/ch4/src/ch4_impl.h)
  ----------------------*/
#define MPIDI_Datatype_get_info(count_, datatype_,              \
                                dt_contig_out_, data_sz_out_,   \
                                dt_ptr_, dt_true_lb_)           \
    do {                                                        \
        if (HANDLE_IS_BUILTIN(datatype_)) {                     \
            (dt_ptr_)        = NULL;                            \
            (dt_contig_out_) = TRUE;                            \
            (dt_true_lb_)    = 0;                               \
            (data_sz_out_)   = (size_t)(count_) *               \
                MPIR_Datatype_get_basic_size(datatype_);        \
        } else {                                                \
            MPIR_Datatype_get_ptr((datatype_), (dt_ptr_));      \
            if (dt_ptr_)                                        \
            {                                                   \
                (dt_contig_out_) = (dt_ptr_)->is_contig;        \
                (dt_true_lb_)    = (dt_ptr_)->true_lb;          \
                (data_sz_out_)   = (size_t)(count_) *           \
                    (dt_ptr_)->size;                            \
            }                                                   \
            else                                                \
            {                                                   \
                (dt_contig_out_) = 1;                           \
                (dt_true_lb_)    = 0;                           \
                (data_sz_out_)   = 0;                           \
            }                                                   \
        }                                                       \
    } while (0)
/*--------------------
  END DATATYPE SECTION
  --------------------*/


#endif /* _MPIDIMPL_H_ */
