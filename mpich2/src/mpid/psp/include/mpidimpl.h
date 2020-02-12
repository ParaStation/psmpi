/*
 * ParaStation
 *
 * Copyright (C) 2006-2020 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
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

typedef struct MPIDI_PSP_topo_level MPIDI_PSP_topo_level_t;
struct MPIDI_PSP_topo_level {
	struct MPIDI_PG *pg;
	struct MPIDI_PSP_topo_level *next;
	int degree;
	int max_badge;
	int badges_are_global;
	int *badge_table;
};
#define MPIDI_PSP_TOPO_LEVEL__MODULES 4096
#define MPIDI_PSP_TOPO_LEVEL__NODES   1024

typedef struct MPIDI_PG MPIDI_PG_t;
struct MPIDI_PG {
	struct MPIDI_PG * next;
	int refcnt;
	int size;
	int id_num;
	MPIDI_VC_t **vcr;
	int * lpids;
	struct MPIDI_PSP_topo_level *topo_level;
	pscom_connection_t **cons;

};


struct MPIDI_VC {
	pscom_connection_t *con;
	int lpid;
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
MPIDI_VC_t *MPIDI_VC_Create(MPIDI_PG_t * pg, int pg_rank, pscom_connection_t *con, int lpid);

int MPID_PSP_get_host_hash(void);
int MPID_PSP_split_type(MPIR_Comm * comm_ptr, int split_type, int key, MPIR_Info * info_ptr, MPIR_Comm ** newcomm_ptr);

void MPID_PSP_comm_init(void);
void MPID_PSP_comm_set_vcrt(MPIR_Comm *comm, MPIDI_VCRT_t *vcrt);
void MPID_PSP_comm_set_local_vcrt(MPIR_Comm *comm, MPIDI_VCRT_t *vcrt);
void MPID_PSP_comm_create_mapper(MPIR_Comm * comm);

int MPIDI_PG_Create(int pg_size, int pg_id_num, MPIDI_PG_t ** pg_ptr);
MPIDI_PG_t* MPIDI_PG_Destroy(MPIDI_PG_t * pg_ptr);
void MPIDI_PG_Convert_id(char *pg_id_name, int *pg_id_num);

typedef struct MPIDI_Process
{
	/* pscom_socket_t *socket; // moved To comm_ptr->pscom_socket */

	pscom_connection_t **grank2con;

	int		my_pg_rank;
	unsigned int	my_pg_size;
	unsigned int 	singleton_but_no_pm;

	char *pg_id_name;
	int next_lpid;
	MPIDI_PG_t * my_pg;

	int shm_attr_key;

	int msa_module_id;

	int* node_id_table;
	int  node_id_max;
	int  my_node_id;

	struct {
		unsigned enable_collectives;
		unsigned enable_ondemand;
		unsigned enable_ondemand_spawn;
		unsigned enable_smp_awareness;
		unsigned enable_msa_awareness;
#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
		unsigned enable_smp_aware_collops;
		unsigned enable_msa_aware_collops;
#endif
#ifdef HAVE_LIBHCOLL
		unsigned enable_hcoll;
#endif
#ifdef MPID_PSP_CREATE_HISTOGRAM
		unsigned enable_histogram;
#endif
		unsigned enable_lazy_disconnect;
	} env;
#ifdef MPID_PSP_CREATE_HISTOGRAM
	struct {
		int max_size;
		int min_size;
		int step_width;
		int points;
		unsigned int* limit;
		unsigned long long int* count;
	} histo;
#endif
} MPIDI_Process_t;

extern MPIDI_Process_t MPIDI_Process;


void MPID_PSP_RecvAck(MPIR_Request *send_req);
/* persistent receives */
int MPID_PSP_Recv_start(MPIR_Request *request);
/*
int MPID_Recv_init(void * buf, int count, MPI_Datatype datatype, int rank, int tag,
		   MPIR_Comm * comm, int context_offset, MPIR_Request ** request);
*/

/* Control messages */
void MPID_PSP_SendCtrl(int tag, int context_id, int src_rank, pscom_connection_t *con, enum MPID_PSP_MSGTYPE msgtype);
void MPID_PSP_RecvCtrl(int tag, int context_id, int src_rank, pscom_connection_t *con, enum MPID_PSP_MSGTYPE msgtype);

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

void MPID_PSP_packed_msg_acc(const void *target_addr, int target_count, MPI_Datatype datatype,
			     void *msg, size_t msg_sz, MPI_Op op);

/* return connection_t for rank, NULL on error */
pscom_connection_t *MPID_PSCOM_rank2connection(MPIR_Comm *comm, int rank);

int MPIDI_PSP_Wait(MPIR_Request *request);

int MPID_Put_generic(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
		     int target_rank, MPI_Aint target_disp, int target_count,
		     MPI_Datatype target_datatype, MPIR_Win *win_ptr, MPIR_Request **request);
int MPID_Get_generic(void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
		     int target_rank, MPI_Aint target_disp, int target_count,
		     MPI_Datatype target_datatype, MPIR_Win *win_ptr, MPIR_Request **request);

int MPID_Win_lock_internal(int dest, MPIR_Win *win_ptr);
int MPID_Win_unlock_internal(int dest, MPIR_Win *win_ptr);
int MPID_Win_wait_local_completion(int rank, MPIR_Win *win_ptr);
void MPID_PSP_group_init(MPIR_Comm *comm_ptr);
void MPID_PSP_group_cleanup(MPIR_Comm *comm_ptr);

void mpid_debug_init(void);
const char *mpid_msgtype_str(enum MPID_PSP_MSGTYPE msg_type);

#define Dprintf(fmt, arg...)

#if !defined(Dprintf)

#define Dprintf(fmt, arg...) do {					\
	printf("#psp %d %s: " fmt "\n", MPIDI_Process.my_pg_rank,	\
	       __func__,##arg);						\
} while (0)

#endif

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
  BEGIN DATATYPE SECTION (from mpid/ch3/include/mpidimpl.h)
  ----------------------*/
#define MPIDI_Datatype_get_info(count_, datatype_, dt_contig_out_, data_sz_out_, dt_ptr_, dt_true_lb_) \
{									\
    if (HANDLE_GET_KIND(datatype_) == HANDLE_KIND_BUILTIN)		\
    {									\
	(dt_ptr_) = NULL;						\
	(dt_contig_out_) = TRUE;					\
	(dt_true_lb_)    = 0;                                           \
	(data_sz_out_) = (count_) * MPIR_Datatype_get_basic_size(datatype_); \
	/* printf("%s() : basic datatype: dt_contig=%d, dt_sz=%d, data_sz=%d\n", \
	       __func__, (dt_contig_out_),				\
	       MPIR_Datatype_get_basic_size(datatype_), (data_sz_out_));*/ \
    } else {								\
	MPIR_Datatype_get_ptr((datatype_), (dt_ptr_));			\
	(dt_contig_out_) = (dt_ptr_)->is_contig;			\
	(data_sz_out_) = (count_) * (dt_ptr_)->size;			\
	(dt_true_lb_)    = (dt_ptr_)->true_lb;				\
	/* printf("%s() : user defined datatype: dt_contig=%d, dt_sz=%d, data_sz=%d\n", \
	       __func__, (dt_contig_out_),				\
	       (dt_ptr_)->size, (data_sz_out_));*/			\
    }									\
}

/*--------------------
  END DATATYPE SECTION
  --------------------*/


#endif /* _MPIDIMPL_H_ */
