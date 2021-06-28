/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */
/* automatically generated
 *   by:   ./maint/extractcvars
 *   at:   Wed May 26 20:52:04 2021 UTC
 *
 * DO NOT EDIT!!!
 */

#include "mpiimpl.h"

/* Actual storage for cvars */
int MPIR_CVAR_ALLGATHER_SHORT_MSG_SIZE;
int MPIR_CVAR_ALLGATHER_LONG_MSG_SIZE;
int MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM;
int MPIR_CVAR_ALLGATHER_INTER_ALGORITHM;
int MPIR_CVAR_ALLGATHER_DEVICE_COLLECTIVE;
int MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE;
int MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM;
int MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM;
int MPIR_CVAR_ALLGATHERV_DEVICE_COLLECTIVE;
int MPIR_CVAR_ALLREDUCE_SHORT_MSG_SIZE;
int MPIR_CVAR_MAX_SMP_ALLREDUCE_MSG_SIZE;
int MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM;
int MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM;
int MPIR_CVAR_ALLREDUCE_DEVICE_COLLECTIVE;
int MPIR_CVAR_ALLTOALL_SHORT_MSG_SIZE;
int MPIR_CVAR_ALLTOALL_MEDIUM_MSG_SIZE;
int MPIR_CVAR_ALLTOALL_THROTTLE;
int MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM;
int MPIR_CVAR_ALLTOALL_INTER_ALGORITHM;
int MPIR_CVAR_ALLTOALL_DEVICE_COLLECTIVE;
int MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM;
int MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM;
int MPIR_CVAR_ALLTOALLV_DEVICE_COLLECTIVE;
int MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM;
int MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM;
int MPIR_CVAR_ALLTOALLW_DEVICE_COLLECTIVE;
int MPIR_CVAR_BARRIER_INTRA_ALGORITHM;
int MPIR_CVAR_BARRIER_INTER_ALGORITHM;
int MPIR_CVAR_BARRIER_DEVICE_COLLECTIVE;
int MPIR_CVAR_BCAST_MIN_PROCS;
int MPIR_CVAR_BCAST_SHORT_MSG_SIZE;
int MPIR_CVAR_BCAST_LONG_MSG_SIZE;
int MPIR_CVAR_MAX_SMP_BCAST_MSG_SIZE;
int MPIR_CVAR_BCAST_INTRA_ALGORITHM;
int MPIR_CVAR_BCAST_INTER_ALGORITHM;
int MPIR_CVAR_BCAST_DEVICE_COLLECTIVE;
int MPIR_CVAR_EXSCAN_INTRA_ALGORITHM;
int MPIR_CVAR_EXSCAN_DEVICE_COLLECTIVE;
int MPIR_CVAR_GATHER_INTER_SHORT_MSG_SIZE;
int MPIR_CVAR_GATHER_INTRA_ALGORITHM;
int MPIR_CVAR_GATHER_INTER_ALGORITHM;
int MPIR_CVAR_GATHER_DEVICE_COLLECTIVE;
int MPIR_CVAR_GATHER_VSMALL_MSG_SIZE;
int MPIR_CVAR_GATHERV_INTRA_ALGORITHM;
int MPIR_CVAR_GATHERV_INTER_ALGORITHM;
int MPIR_CVAR_GATHERV_DEVICE_COLLECTIVE;
int MPIR_CVAR_GATHERV_INTER_SSEND_MIN_PROCS;
int MPIR_CVAR_IALLGATHER_RECEXCH_KVAL;
int MPIR_CVAR_IALLGATHER_BRUCKS_KVAL;
int MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM;
int MPIR_CVAR_IALLGATHER_INTER_ALGORITHM;
int MPIR_CVAR_IALLGATHER_DEVICE_COLLECTIVE;
int MPIR_CVAR_IALLGATHERV_RECEXCH_KVAL;
int MPIR_CVAR_IALLGATHERV_BRUCKS_KVAL;
int MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM;
int MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM;
int MPIR_CVAR_IALLGATHERV_DEVICE_COLLECTIVE;
int MPIR_CVAR_IALLREDUCE_TREE_KVAL;
const char * MPIR_CVAR_IALLREDUCE_TREE_TYPE;
int MPIR_CVAR_IALLREDUCE_TREE_PIPELINE_CHUNK_SIZE;
int MPIR_CVAR_IALLREDUCE_TREE_BUFFER_PER_CHILD;
int MPIR_CVAR_IALLREDUCE_RECEXCH_KVAL;
int MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM;
int MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM;
int MPIR_CVAR_IALLREDUCE_DEVICE_COLLECTIVE;
int MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM;
int MPIR_CVAR_IALLTOALL_INTER_ALGORITHM;
int MPIR_CVAR_IALLTOALL_DEVICE_COLLECTIVE;
int MPIR_CVAR_IALLTOALL_BRUCKS_KVAL;
int MPIR_CVAR_IALLTOALL_BRUCKS_BUFFER_PER_NBR;
int MPIR_CVAR_IALLTOALL_SCATTERED_OUTSTANDING_TASKS;
int MPIR_CVAR_IALLTOALL_SCATTERED_BATCH_SIZE;
int MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM;
int MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM;
int MPIR_CVAR_IALLTOALLV_DEVICE_COLLECTIVE;
int MPIR_CVAR_IALLTOALLV_SCATTERED_OUTSTANDING_TASKS;
int MPIR_CVAR_IALLTOALLV_SCATTERED_BATCH_SIZE;
int MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM;
int MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM;
int MPIR_CVAR_IALLTOALLW_DEVICE_COLLECTIVE;
int MPIR_CVAR_IBARRIER_RECEXCH_KVAL;
int MPIR_CVAR_IBARRIER_INTRA_ALGORITHM;
int MPIR_CVAR_IBARRIER_INTER_ALGORITHM;
int MPIR_CVAR_IBARRIER_DEVICE_COLLECTIVE;
int MPIR_CVAR_IBCAST_TREE_KVAL;
const char * MPIR_CVAR_IBCAST_TREE_TYPE;
int MPIR_CVAR_IBCAST_TREE_PIPELINE_CHUNK_SIZE;
int MPIR_CVAR_IBCAST_RING_CHUNK_SIZE;
int MPIR_CVAR_IBCAST_INTRA_ALGORITHM;
int MPIR_CVAR_IBCAST_SCATTERV_KVAL;
int MPIR_CVAR_IBCAST_ALLGATHERV_RECEXCH_KVAL;
int MPIR_CVAR_IBCAST_INTER_ALGORITHM;
int MPIR_CVAR_IBCAST_DEVICE_COLLECTIVE;
int MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM;
int MPIR_CVAR_IEXSCAN_DEVICE_COLLECTIVE;
int MPIR_CVAR_IGATHER_INTRA_ALGORITHM;
int MPIR_CVAR_IGATHER_TREE_KVAL;
int MPIR_CVAR_IGATHER_INTER_ALGORITHM;
int MPIR_CVAR_IGATHER_DEVICE_COLLECTIVE;
int MPIR_CVAR_IGATHERV_INTRA_ALGORITHM;
int MPIR_CVAR_IGATHERV_INTER_ALGORITHM;
int MPIR_CVAR_IGATHERV_DEVICE_COLLECTIVE;
int MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM;
int MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM;
int MPIR_CVAR_INEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE;
int MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM;
int MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM;
int MPIR_CVAR_INEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE;
int MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM;
int MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM;
int MPIR_CVAR_INEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE;
int MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM;
int MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM;
int MPIR_CVAR_INEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE;
int MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM;
int MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM;
int MPIR_CVAR_INEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE;
int MPIR_CVAR_IREDUCE_TREE_KVAL;
const char * MPIR_CVAR_IREDUCE_TREE_TYPE;
int MPIR_CVAR_IREDUCE_TREE_PIPELINE_CHUNK_SIZE;
int MPIR_CVAR_IREDUCE_RING_CHUNK_SIZE;
int MPIR_CVAR_IREDUCE_TREE_BUFFER_PER_CHILD;
int MPIR_CVAR_IREDUCE_INTRA_ALGORITHM;
int MPIR_CVAR_IREDUCE_INTER_ALGORITHM;
int MPIR_CVAR_IREDUCE_DEVICE_COLLECTIVE;
int MPIR_CVAR_IREDUCE_SCATTER_RECEXCH_KVAL;
int MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM;
int MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM;
int MPIR_CVAR_IREDUCE_SCATTER_DEVICE_COLLECTIVE;
int MPIR_CVAR_IREDUCE_SCATTER_BLOCK_RECEXCH_KVAL;
int MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM;
int MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM;
int MPIR_CVAR_IREDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE;
int MPIR_CVAR_ISCAN_INTRA_ALGORITHM;
int MPIR_CVAR_ISCAN_DEVICE_COLLECTIVE;
int MPIR_CVAR_ISCATTER_INTRA_ALGORITHM;
int MPIR_CVAR_ISCATTER_TREE_KVAL;
int MPIR_CVAR_ISCATTER_INTER_ALGORITHM;
int MPIR_CVAR_ISCATTER_DEVICE_COLLECTIVE;
int MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM;
int MPIR_CVAR_ISCATTERV_INTER_ALGORITHM;
int MPIR_CVAR_ISCATTERV_DEVICE_COLLECTIVE;
int MPIR_CVAR_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM;
int MPIR_CVAR_NEIGHBOR_ALLGATHER_INTER_ALGORITHM;
int MPIR_CVAR_NEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE;
int MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM;
int MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM;
int MPIR_CVAR_NEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE;
int MPIR_CVAR_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM;
int MPIR_CVAR_NEIGHBOR_ALLTOALL_INTER_ALGORITHM;
int MPIR_CVAR_NEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE;
int MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM;
int MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM;
int MPIR_CVAR_NEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE;
int MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM;
int MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM;
int MPIR_CVAR_NEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE;
int MPIR_CVAR_REDUCE_SHORT_MSG_SIZE;
int MPIR_CVAR_MAX_SMP_REDUCE_MSG_SIZE;
int MPIR_CVAR_REDUCE_INTRA_ALGORITHM;
int MPIR_CVAR_REDUCE_INTER_ALGORITHM;
int MPIR_CVAR_REDUCE_DEVICE_COLLECTIVE;
int MPIR_CVAR_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE;
int MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM;
int MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM;
int MPIR_CVAR_REDUCE_SCATTER_DEVICE_COLLECTIVE;
int MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM;
int MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM;
int MPIR_CVAR_REDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE;
int MPIR_CVAR_SCAN_INTRA_ALGORITHM;
int MPIR_CVAR_SCAN_DEVICE_COLLECTIVE;
int MPIR_CVAR_SCATTER_INTER_SHORT_MSG_SIZE;
int MPIR_CVAR_SCATTER_INTRA_ALGORITHM;
int MPIR_CVAR_SCATTER_INTER_ALGORITHM;
int MPIR_CVAR_SCATTER_DEVICE_COLLECTIVE;
int MPIR_CVAR_SCATTERV_INTRA_ALGORITHM;
int MPIR_CVAR_SCATTERV_INTER_ALGORITHM;
int MPIR_CVAR_SCATTERV_DEVICE_COLLECTIVE;
int MPIR_CVAR_DEVICE_COLLECTIVES;
int MPIR_CVAR_COLLECTIVE_FALLBACK;
const char * MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE;
int MPIR_CVAR_PROGRESS_MAX_COLLS;
int MPIR_CVAR_COMM_SPLIT_USE_QSORT;
int MPIR_CVAR_CTXID_EAGER_SIZE;
int MPIR_CVAR_DATALOOP_FAST_SEEK;
int MPIR_CVAR_PROCTABLE_SIZE;
int MPIR_CVAR_PROCTABLE_PRINT;
int MPIR_CVAR_PRINT_ERROR_STACK;
int MPIR_CVAR_CHOP_ERROR_STACK;
int MPIR_CVAR_SUPPRESS_ABORT_MESSAGE;
const char * MPIR_CVAR_DEFAULT_THREAD_LEVEL;
int MPIR_CVAR_ASYNC_PROGRESS;
int MPIR_CVAR_DEBUG_HOLD;
int MPIR_CVAR_ERROR_CHECKING;
int MPIR_CVAR_MEMDUMP;
int MPIR_CVAR_MEM_CATEGORY_INFORMATION;
int MPIR_CVAR_DIMS_VERBOSE;
const char * MPIR_CVAR_NAMESERV_FILE_PUBDIR;
int MPIR_CVAR_ABORT_ON_LEAKED_HANDLES;
const char * MPIR_CVAR_NETLOC_NODE_FILE;
int MPIR_CVAR_NOLOCAL;
int MPIR_CVAR_ODD_EVEN_CLIQUES;
int MPIR_CVAR_NUM_CLIQUES;
int MPIR_CVAR_COLL_ALIAS_CHECK;
int MPIR_CVAR_ENABLE_GPU;
int MPIR_CVAR_REQUEST_POLL_FREQ;
int MPIR_CVAR_REQUEST_BATCH_SIZE;
int MPIR_CVAR_POLLS_BEFORE_YIELD;
const char * MPIR_CVAR_OFI_USE_PROVIDER;
const char * MPIR_CVAR_CH3_INTERFACE_HOSTNAME;
MPIR_T_cvar_range_value_t MPIR_CVAR_CH3_PORT_RANGE;
const char * MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE;
int MPIR_CVAR_NEMESIS_TCP_HOST_LOOKUP_RETRIES;
int MPIR_CVAR_NEMESIS_ENABLE_CKPOINT;
int MPIR_CVAR_NEMESIS_SHM_EAGER_MAX_SZ;
int MPIR_CVAR_NEMESIS_SHM_READY_EAGER_MAX_SZ;
int MPIR_CVAR_ENABLE_FT;
const char * MPIR_CVAR_NEMESIS_NETMOD;
int MPIR_CVAR_CH3_ENABLE_HCOLL;
int MPIR_CVAR_CH3_COMM_CONNECT_TIMEOUT;
int MPIR_CVAR_CH3_RMA_OP_PIGGYBACK_LOCK_DATA_SIZE;
int MPIR_CVAR_CH3_RMA_ACTIVE_REQ_THRESHOLD;
int MPIR_CVAR_CH3_RMA_POKE_PROGRESS_REQ_THRESHOLD;
int MPIR_CVAR_CH3_RMA_SCALABLE_FENCE_PROCESS_NUM;
int MPIR_CVAR_CH3_RMA_DELAY_ISSUING_FOR_PIGGYBACKING;
int MPIR_CVAR_CH3_RMA_SLOTS_SIZE;
int MPIR_CVAR_CH3_RMA_TARGET_LOCK_DATA_BYTES;
int MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE;
int MPIR_CVAR_CH3_PG_VERBOSE;
int MPIR_CVAR_CH3_RMA_OP_WIN_POOL_SIZE;
int MPIR_CVAR_CH3_RMA_OP_GLOBAL_POOL_SIZE;
int MPIR_CVAR_CH3_RMA_TARGET_WIN_POOL_SIZE;
int MPIR_CVAR_CH3_RMA_TARGET_GLOBAL_POOL_SIZE;
int MPIR_CVAR_CH3_RMA_TARGET_LOCK_ENTRY_WIN_POOL_SIZE;
int MPIR_CVAR_CH4_AM_PACK_BUFFER_SIZE;
int MPIR_CVAR_CH4_NUM_AM_PACK_BUFFERS_PER_CHUNK;
int MPIR_CVAR_CH4_MAX_AM_UNEXPECTED_PACK_BUFFERS_SIZE_BYTE;
int MPIR_CVAR_CH4_OFI_AM_LONG_FORCE_PIPELINE;
int MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG;
int MPIR_CVAR_OFI_SKIP_IPV6;
int MPIR_CVAR_CH4_OFI_ENABLE_AV_TABLE;
int MPIR_CVAR_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS;
int MPIR_CVAR_CH4_OFI_ENABLE_SHARED_CONTEXTS;
int MPIR_CVAR_CH4_OFI_ENABLE_MR_SCALABLE;
int MPIR_CVAR_CH4_OFI_ENABLE_MR_VIRT_ADDRESS;
int MPIR_CVAR_CH4_OFI_ENABLE_MR_ALLOCATED;
int MPIR_CVAR_CH4_OFI_ENABLE_MR_PROV_KEY;
int MPIR_CVAR_CH4_OFI_ENABLE_TAGGED;
int MPIR_CVAR_CH4_OFI_ENABLE_AM;
int MPIR_CVAR_CH4_OFI_ENABLE_RMA;
int MPIR_CVAR_CH4_OFI_ENABLE_ATOMICS;
int MPIR_CVAR_CH4_OFI_FETCH_ATOMIC_IOVECS;
int MPIR_CVAR_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS;
int MPIR_CVAR_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS;
int MPIR_CVAR_CH4_OFI_ENABLE_PT2PT_NOPACK;
int MPIR_CVAR_CH4_OFI_CONTEXT_ID_BITS;
int MPIR_CVAR_CH4_OFI_RANK_BITS;
int MPIR_CVAR_CH4_OFI_TAG_BITS;
int MPIR_CVAR_CH4_OFI_MAJOR_VERSION;
int MPIR_CVAR_CH4_OFI_MINOR_VERSION;
int MPIR_CVAR_CH4_OFI_MAX_VNIS;
int MPIR_CVAR_CH4_OFI_MAX_RMA_SEP_CTX;
int MPIR_CVAR_CH4_OFI_MAX_EAGAIN_RETRY;
int MPIR_CVAR_CH4_OFI_NUM_AM_BUFFERS;
int MPIR_CVAR_CH4_OFI_RMA_PROGRESS_INTERVAL;
int MPIR_CVAR_CH4_OFI_RMA_IOVEC_MAX;
int MPIR_CVAR_CH4_OFI_NUM_PACK_BUFFERS_PER_CHUNK;
int MPIR_CVAR_CH4_OFI_MAX_NUM_PACK_BUFFERS;
int MPIR_CVAR_CH4_OFI_EAGER_MAX_MSG_SIZE;
int MPIR_CVAR_CH4_UCX_MAX_VNIS;
int MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE;
int MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD;
int MPIR_CVAR_CH4_XPMEM_ENABLE;
int MPIR_CVAR_CH4_IPC_XPMEM_P2P_THRESHOLD;
int MPIR_CVAR_BCAST_POSIX_INTRA_ALGORITHM;
int MPIR_CVAR_REDUCE_POSIX_INTRA_ALGORITHM;
int MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM;
int MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM;
int MPIR_CVAR_POSIX_NUM_COLLS_THRESHOLD;
const char * MPIR_CVAR_CH4_SHM_POSIX_EAGER;
const char * MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE;
int MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_NUM_CELLS;
int MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_CELL_SIZE;
int MPIR_CVAR_COLL_SHM_LIMIT_PER_NODE;
int MPIR_CVAR_BCAST_INTRANODE_BUFFER_TOTAL_SIZE;
int MPIR_CVAR_BCAST_INTRANODE_NUM_CELLS;
int MPIR_CVAR_REDUCE_INTRANODE_BUFFER_TOTAL_SIZE;
int MPIR_CVAR_REDUCE_INTRANODE_NUM_CELLS;
int MPIR_CVAR_BCAST_INTRANODE_TREE_KVAL;
const char * MPIR_CVAR_BCAST_INTRANODE_TREE_TYPE;
int MPIR_CVAR_REDUCE_INTRANODE_TREE_KVAL;
const char * MPIR_CVAR_REDUCE_INTRANODE_TREE_TYPE;
int MPIR_CVAR_ENABLE_INTRANODE_TOPOLOGY_AWARE_TREES;
const char * MPIR_CVAR_CH4_NETMOD;
const char * MPIR_CVAR_CH4_SHM;
int MPIR_CVAR_CH4_ROOTS_ONLY_PMI;
int MPIR_CVAR_CH4_RUNTIME_CONF_DEBUG;
const char * MPIR_CVAR_CH4_MT_MODEL;
int MPIR_CVAR_CH4_NUM_VCIS;
const char * MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE;
int MPIR_CVAR_CH4_IOV_DENSITY_MIN;
int MPIR_CVAR_CH4_COMM_CONNECT_TIMEOUT;
int MPIR_CVAR_CH4_RMA_MEM_EFFICIENT;
int MPIR_CVAR_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS;
int MPIR_CVAR_CH4_RMA_AM_PROGRESS_INTERVAL;
int MPIR_CVAR_CH4_RMA_AM_PROGRESS_LOW_FREQ_INTERVAL;
int MPIR_CVAR_ENABLE_HCOLL;
int MPIR_CVAR_COLL_SCHED_DUMP;
int MPIR_CVAR_SHM_RANDOM_ADDR_RETRY;
int MPIR_CVAR_SHM_SYMHEAP_RETRY;

int MPIR_T_cvar_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    int rc;
    const char *tmp_str;
    static int initialized = FALSE;
    MPIR_T_cvar_value_t defaultval;

    /* FIXME any MT issues here? */
    if (initialized)
        return MPI_SUCCESS;
    initialized = TRUE;

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/mpi/coll/alltoall/alltoall.c */
    MPIR_T_cat_add_desc("COLLECTIVE",
        "A category for collective communication variables.");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/mpi/comm/comm_split.c */
    MPIR_T_cat_add_desc("COMMUNICATOR",
        "cvars that control communicator construction and operation");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/mpi/datatype/typerep/dataloop/segment.c */
    MPIR_T_cat_add_desc("DATALOOP",
        "Dataloop-related CVARs");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/mpi/errhan/errutil.c */
    MPIR_T_cat_add_desc("ERROR_HANDLING",
        "cvars that control error handling behavior (stack traces, aborts, etc)");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/mpi/init/init.c */
    MPIR_T_cat_add_desc("THREADS",
        "multi-threading cvars");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/mpi/init/initthread.c */
    MPIR_T_cat_add_desc("DEBUGGER",
        "cvars relevant to the \"MPIR\" debugger interface");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/mpi/init/mpi_init.h */
    MPIR_T_cat_add_desc("DEVELOPER",
        "useful for developers working on MPICH itself");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/mpi/topo/dims_create.c */
    MPIR_T_cat_add_desc("DIMS",
        "Dims_create cvars");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/nameserv/file/file_nameserv.c */
    MPIR_T_cat_add_desc("PROCESS_MANAGER",
        "cvars that control the client-side process manager code");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/util/mpir_handlemem.c */
    MPIR_T_cat_add_desc("MEMORY",
        "affects memory allocation and usage, including MPI object handles");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/util/mpir_nodemap.h */
    MPIR_T_cat_add_desc("NODEMAP",
        "cvars that control behavior of nodemap");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/include/mpir_gpu.h */
    MPIR_T_cat_add_desc("GPU",
        "GPU related cvars");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/include/mpir_request.h */
    MPIR_T_cat_add_desc("REQUEST",
        "A category for requests mangement variables");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/mpid/ch3/channels/nemesis/src/mpid_nem_init.c */
    MPIR_T_cat_add_desc("NEMESIS",
        "cvars that control behavior of the ch3:nemesis channel");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/mpid/ch3/channels/nemesis/src/mpid_nem_lmt.c */
    MPIR_T_cat_add_desc("FT",
        "cvars that control behavior of fault tolerance");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/mpid/ch3/src/mpidi_rma.c */
    MPIR_T_cat_add_desc("CH3",
        "cvars that control behavior of ch3");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/mpid/ch4/netmod/ofi/ofi_init.c */
    MPIR_T_cat_add_desc("CH4_OFI",
        "A category for CH4 OFI netmod variables");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/mpid/ch4/netmod/ucx/ucx_init.c */
    MPIR_T_cat_add_desc("CH4_UCX",
        "A category for CH4 UCX netmod variables");

    /* declared in /tmp/i5118MmFEO/mpich-3.4.2/maint/../src/mpid/ch4/src/ch4_init.c */
    MPIR_T_cat_add_desc("CH4",
        "cvars that control behavior of the CH4 device");

    defaultval.d = 81920;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLGATHER_SHORT_MSG_SIZE, /* name */
        &MPIR_CVAR_ALLGATHER_SHORT_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "For MPI_Allgather and MPI_Allgatherv, the short message algorithm will be used if the send buffer size is < this value (in bytes). (See also: MPIR_CVAR_ALLGATHER_LONG_MSG_SIZE)");
    MPIR_CVAR_ALLGATHER_SHORT_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_ALLGATHER_SHORT_MSG_SIZE", &(MPIR_CVAR_ALLGATHER_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLGATHER_SHORT_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_ALLGATHER_SHORT_MSG_SIZE", &(MPIR_CVAR_ALLGATHER_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLGATHER_SHORT_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_ALLGATHER_SHORT_MSG_SIZE", &(MPIR_CVAR_ALLGATHER_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLGATHER_SHORT_MSG_SIZE");

    defaultval.d = 524288;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLGATHER_LONG_MSG_SIZE, /* name */
        &MPIR_CVAR_ALLGATHER_LONG_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "For MPI_Allgather and MPI_Allgatherv, the long message algorithm will be used if the send buffer size is >= this value (in bytes) (See also: MPIR_CVAR_ALLGATHER_SHORT_MSG_SIZE)");
    MPIR_CVAR_ALLGATHER_LONG_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_ALLGATHER_LONG_MSG_SIZE", &(MPIR_CVAR_ALLGATHER_LONG_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLGATHER_LONG_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_ALLGATHER_LONG_MSG_SIZE", &(MPIR_CVAR_ALLGATHER_LONG_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLGATHER_LONG_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_ALLGATHER_LONG_MSG_SIZE", &(MPIR_CVAR_ALLGATHER_LONG_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLGATHER_LONG_MSG_SIZE");

    defaultval.d = MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select allgather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
brucks             - Force brucks algorithm\
nb                 - Force nonblocking algorithm\
recursive_doubling - Force recursive doubling algorithm\
ring               - Force ring algorithm");
    MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ALLGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLGATHER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ALLGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLGATHER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "brucks"))
            MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM_brucks;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "recursive_doubling"))
            MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM_recursive_doubling;
        else if (0 == strcmp(tmp_str, "ring"))
            MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM_ring;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_ALLGATHER_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLGATHER_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_ALLGATHER_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select allgather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
local_gather_remote_bcast - Force local-gather-remote-bcast algorithm\
nb                        - Force nonblocking algorithm");
    MPIR_CVAR_ALLGATHER_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ALLGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLGATHER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ALLGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLGATHER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ALLGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLGATHER_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ALLGATHER_INTER_ALGORITHM = MPIR_CVAR_ALLGATHER_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "local_gather_remote_bcast"))
            MPIR_CVAR_ALLGATHER_INTER_ALGORITHM = MPIR_CVAR_ALLGATHER_INTER_ALGORITHM_local_gather_remote_bcast;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_ALLGATHER_INTER_ALGORITHM = MPIR_CVAR_ALLGATHER_INTER_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ALLGATHER_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLGATHER_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_ALLGATHER_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Allgather will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_ALLGATHER_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_ALLGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLGATHER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_ALLGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLGATHER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_ALLGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLGATHER_DEVICE_COLLECTIVE");

    defaultval.d = 32768;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE, /* name */
        &MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "The smallest message size that will be used for the pipelined, large-message, ring algorithm in the MPI_Allgatherv implementation.");
    MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_ALLGATHERV_PIPELINE_MSG_SIZE", &(MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLGATHERV_PIPELINE_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_ALLGATHERV_PIPELINE_MSG_SIZE", &(MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLGATHERV_PIPELINE_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE", &(MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE");

    defaultval.d = MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select allgatherv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
brucks             - Force brucks algorithm\
nb                 - Force nonblocking algorithm\
recursive_doubling - Force recursive doubling algorithm\
ring               - Force ring algorithm");
    MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ALLGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLGATHERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ALLGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLGATHERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "brucks"))
            MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM_brucks;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "recursive_doubling"))
            MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM_recursive_doubling;
        else if (0 == strcmp(tmp_str, "ring"))
            MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM_ring;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select allgatherv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb                        - Force nonblocking algorithm\
remote_gather_local_bcast - Force remote-gather-local-bcast algorithm");
    MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ALLGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLGATHERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ALLGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLGATHERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM = MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM = MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "remote_gather_local_bcast"))
            MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM = MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM_remote_gather_local_bcast;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLGATHERV_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_ALLGATHERV_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Allgatherv will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_ALLGATHERV_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_ALLGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLGATHERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_ALLGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLGATHERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_ALLGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLGATHERV_DEVICE_COLLECTIVE");

    defaultval.d = 2048;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLREDUCE_SHORT_MSG_SIZE, /* name */
        &MPIR_CVAR_ALLREDUCE_SHORT_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "the short message algorithm will be used if the send buffer size is <= this value (in bytes)");
    MPIR_CVAR_ALLREDUCE_SHORT_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_ALLREDUCE_SHORT_MSG_SIZE", &(MPIR_CVAR_ALLREDUCE_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLREDUCE_SHORT_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_ALLREDUCE_SHORT_MSG_SIZE", &(MPIR_CVAR_ALLREDUCE_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLREDUCE_SHORT_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_ALLREDUCE_SHORT_MSG_SIZE", &(MPIR_CVAR_ALLREDUCE_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLREDUCE_SHORT_MSG_SIZE");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_MAX_SMP_ALLREDUCE_MSG_SIZE, /* name */
        &MPIR_CVAR_MAX_SMP_ALLREDUCE_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Maximum message size for which SMP-aware allreduce is used.  A value of '0' uses SMP-aware allreduce for all message sizes.");
    MPIR_CVAR_MAX_SMP_ALLREDUCE_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_MAX_SMP_ALLREDUCE_MSG_SIZE", &(MPIR_CVAR_MAX_SMP_ALLREDUCE_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_MAX_SMP_ALLREDUCE_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_MAX_SMP_ALLREDUCE_MSG_SIZE", &(MPIR_CVAR_MAX_SMP_ALLREDUCE_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_MAX_SMP_ALLREDUCE_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_MAX_SMP_ALLREDUCE_MSG_SIZE", &(MPIR_CVAR_MAX_SMP_ALLREDUCE_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_MAX_SMP_ALLREDUCE_MSG_SIZE");

    defaultval.d = MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select allreduce algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb                       - Force nonblocking algorithm\
smp                      - Force smp algorithm\
recursive_doubling       - Force recursive doubling algorithm\
reduce_scatter_allgather - Force reduce scatter allgather algorithm");
    MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ALLREDUCE_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLREDUCE_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ALLREDUCE_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLREDUCE_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "smp"))
            MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM_smp;
        else if (0 == strcmp(tmp_str, "recursive_doubling"))
            MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM_recursive_doubling;
        else if (0 == strcmp(tmp_str, "reduce_scatter_allgather"))
            MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM_reduce_scatter_allgather;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select allreduce algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb                    - Force nonblocking algorithm\
reduce_exchange_bcast - Force reduce-exchange-bcast algorithm");
    MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ALLREDUCE_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLREDUCE_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ALLREDUCE_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLREDUCE_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM = MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM = MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "reduce_exchange_bcast"))
            MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM = MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM_reduce_exchange_bcast;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLREDUCE_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_ALLREDUCE_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Allreduce will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_ALLREDUCE_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_ALLREDUCE_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLREDUCE_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLREDUCE_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_ALLREDUCE_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLREDUCE_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLREDUCE_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_ALLREDUCE_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLREDUCE_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLREDUCE_DEVICE_COLLECTIVE");

    defaultval.d = 256;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLTOALL_SHORT_MSG_SIZE, /* name */
        &MPIR_CVAR_ALLTOALL_SHORT_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "the short message algorithm will be used if the per-destination message size (sendcount*size(sendtype)) is <= this value (See also: MPIR_CVAR_ALLTOALL_MEDIUM_MSG_SIZE)");
    MPIR_CVAR_ALLTOALL_SHORT_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_ALLTOALL_SHORT_MSG_SIZE", &(MPIR_CVAR_ALLTOALL_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLTOALL_SHORT_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_ALLTOALL_SHORT_MSG_SIZE", &(MPIR_CVAR_ALLTOALL_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLTOALL_SHORT_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_ALLTOALL_SHORT_MSG_SIZE", &(MPIR_CVAR_ALLTOALL_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLTOALL_SHORT_MSG_SIZE");

    defaultval.d = 32768;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLTOALL_MEDIUM_MSG_SIZE, /* name */
        &MPIR_CVAR_ALLTOALL_MEDIUM_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "the medium message algorithm will be used if the per-destination message size (sendcount*size(sendtype)) is <= this value and larger than MPIR_CVAR_ALLTOALL_SHORT_MSG_SIZE (See also: MPIR_CVAR_ALLTOALL_SHORT_MSG_SIZE)");
    MPIR_CVAR_ALLTOALL_MEDIUM_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_ALLTOALL_MEDIUM_MSG_SIZE", &(MPIR_CVAR_ALLTOALL_MEDIUM_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLTOALL_MEDIUM_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_ALLTOALL_MEDIUM_MSG_SIZE", &(MPIR_CVAR_ALLTOALL_MEDIUM_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLTOALL_MEDIUM_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_ALLTOALL_MEDIUM_MSG_SIZE", &(MPIR_CVAR_ALLTOALL_MEDIUM_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLTOALL_MEDIUM_MSG_SIZE");

    defaultval.d = 32;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLTOALL_THROTTLE, /* name */
        &MPIR_CVAR_ALLTOALL_THROTTLE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "max no. of irecvs/isends posted at a time in some alltoall algorithms. Setting it to 0 causes all irecvs/isends to be posted at once");
    MPIR_CVAR_ALLTOALL_THROTTLE = defaultval.d;
    rc = MPL_env2int("MPICH_ALLTOALL_THROTTLE", &(MPIR_CVAR_ALLTOALL_THROTTLE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLTOALL_THROTTLE");
    rc = MPL_env2int("MPIR_PARAM_ALLTOALL_THROTTLE", &(MPIR_CVAR_ALLTOALL_THROTTLE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLTOALL_THROTTLE");
    rc = MPL_env2int("MPIR_CVAR_ALLTOALL_THROTTLE", &(MPIR_CVAR_ALLTOALL_THROTTLE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLTOALL_THROTTLE");

    defaultval.d = MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select alltoall algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
brucks                    - Force brucks algorithm\
nb                        - Force nonblocking algorithm\
pairwise                  - Force pairwise algorithm\
pairwise_sendrecv_replace - Force pairwise sendrecv replace algorithm\
scattered                 - Force scattered algorithm");
    MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ALLTOALL_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLTOALL_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ALLTOALL_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLTOALL_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "brucks"))
            MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM_brucks;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "pairwise"))
            MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM_pairwise;
        else if (0 == strcmp(tmp_str, "pairwise_sendrecv_replace"))
            MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM_pairwise_sendrecv_replace;
        else if (0 == strcmp(tmp_str, "scattered"))
            MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM_scattered;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_ALLTOALL_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLTOALL_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_ALLTOALL_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select alltoall algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb                - Force nonblocking algorithm\
pairwise_exchange - Force pairwise exchange algorithm");
    MPIR_CVAR_ALLTOALL_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ALLTOALL_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLTOALL_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ALLTOALL_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLTOALL_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ALLTOALL_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLTOALL_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ALLTOALL_INTER_ALGORITHM = MPIR_CVAR_ALLTOALL_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_ALLTOALL_INTER_ALGORITHM = MPIR_CVAR_ALLTOALL_INTER_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "pairwise_exchange"))
            MPIR_CVAR_ALLTOALL_INTER_ALGORITHM = MPIR_CVAR_ALLTOALL_INTER_ALGORITHM_pairwise_exchange;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ALLTOALL_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLTOALL_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_ALLTOALL_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Alltoall will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_ALLTOALL_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_ALLTOALL_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLTOALL_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLTOALL_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_ALLTOALL_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLTOALL_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLTOALL_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_ALLTOALL_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLTOALL_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLTOALL_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select alltoallv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb                        - Force nonblocking algorithm\
pairwise_sendrecv_replace - Force pairwise_sendrecv_replace algorithm\
scattered                 - Force scattered algorithm");
    MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ALLTOALLV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLTOALLV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ALLTOALLV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLTOALLV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "pairwise_sendrecv_replace"))
            MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM_pairwise_sendrecv_replace;
        else if (0 == strcmp(tmp_str, "scattered"))
            MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM_scattered;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select alltoallv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
pairwise_exchange - Force pairwise exchange algorithm\
nb                - Force nonblocking algorithm");
    MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ALLTOALLV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLTOALLV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ALLTOALLV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLTOALLV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM = MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "pairwise_exchange"))
            MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM = MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM_pairwise_exchange;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM = MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLTOALLV_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_ALLTOALLV_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Alltoallv will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_ALLTOALLV_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_ALLTOALLV_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLTOALLV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLTOALLV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_ALLTOALLV_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLTOALLV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLTOALLV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_ALLTOALLV_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLTOALLV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLTOALLV_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select alltoallw algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb                        - Force nonblocking algorithm\
pairwise_sendrecv_replace - Force pairwise sendrecv replace algorithm\
scattered                 - Force scattered algorithm");
    MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ALLTOALLW_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLTOALLW_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ALLTOALLW_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLTOALLW_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "pairwise_sendrecv_replace"))
            MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM_pairwise_sendrecv_replace;
        else if (0 == strcmp(tmp_str, "scattered"))
            MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM_scattered;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select alltoallw algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb                - Force nonblocking algorithm\
pairwise_exchange - Force pairwise exchange algorithm");
    MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ALLTOALLW_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLTOALLW_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ALLTOALLW_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLTOALLW_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM = MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM = MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "pairwise_exchange"))
            MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM = MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM_pairwise_exchange;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLTOALLW_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_ALLTOALLW_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Alltoallw will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_ALLTOALLW_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_ALLTOALLW_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLTOALLW_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLTOALLW_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_ALLTOALLW_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLTOALLW_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLTOALLW_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_ALLTOALLW_DEVICE_COLLECTIVE", &(MPIR_CVAR_ALLTOALLW_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLTOALLW_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_BARRIER_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_BARRIER_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_BARRIER_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select barrier algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb            - Force nonblocking algorithm\
dissemination - Force dissemination algorithm");
    MPIR_CVAR_BARRIER_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_BARRIER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BARRIER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_BARRIER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BARRIER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_BARRIER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BARRIER_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_BARRIER_INTRA_ALGORITHM = MPIR_CVAR_BARRIER_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_BARRIER_INTRA_ALGORITHM = MPIR_CVAR_BARRIER_INTRA_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "dissemination"))
            MPIR_CVAR_BARRIER_INTRA_ALGORITHM = MPIR_CVAR_BARRIER_INTRA_ALGORITHM_dissemination;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_BARRIER_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_BARRIER_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_BARRIER_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_BARRIER_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select barrier algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
bcast - Force bcast algorithm\
nb    - Force nonblocking algorithm");
    MPIR_CVAR_BARRIER_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_BARRIER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BARRIER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_BARRIER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BARRIER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_BARRIER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BARRIER_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_BARRIER_INTER_ALGORITHM = MPIR_CVAR_BARRIER_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "bcast"))
            MPIR_CVAR_BARRIER_INTER_ALGORITHM = MPIR_CVAR_BARRIER_INTER_ALGORITHM_bcast;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_BARRIER_INTER_ALGORITHM = MPIR_CVAR_BARRIER_INTER_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_BARRIER_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_BARRIER_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_BARRIER_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Barrier will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_BARRIER_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_BARRIER_DEVICE_COLLECTIVE", &(MPIR_CVAR_BARRIER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BARRIER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_BARRIER_DEVICE_COLLECTIVE", &(MPIR_CVAR_BARRIER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BARRIER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_BARRIER_DEVICE_COLLECTIVE", &(MPIR_CVAR_BARRIER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BARRIER_DEVICE_COLLECTIVE");

    defaultval.d = 8;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_BCAST_MIN_PROCS, /* name */
        &MPIR_CVAR_BCAST_MIN_PROCS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Let's define short messages as messages with size < MPIR_CVAR_BCAST_SHORT_MSG_SIZE, and medium messages as messages with size >= MPIR_CVAR_BCAST_SHORT_MSG_SIZE but < MPIR_CVAR_BCAST_LONG_MSG_SIZE, and long messages as messages with size >= MPIR_CVAR_BCAST_LONG_MSG_SIZE. The broadcast algorithms selection procedure is as follows. For short messages or when the number of processes is < MPIR_CVAR_BCAST_MIN_PROCS, we do broadcast using the binomial tree algorithm. Otherwise, for medium messages and with a power-of-two number of processes, we do broadcast based on a scatter followed by a recursive doubling allgather algorithm. Otherwise, for long messages or with non power-of-two number of processes, we do broadcast based on a scatter followed by a ring allgather algorithm. (See also: MPIR_CVAR_BCAST_SHORT_MSG_SIZE, MPIR_CVAR_BCAST_LONG_MSG_SIZE)");
    MPIR_CVAR_BCAST_MIN_PROCS = defaultval.d;
    rc = MPL_env2int("MPICH_BCAST_MIN_PROCS", &(MPIR_CVAR_BCAST_MIN_PROCS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BCAST_MIN_PROCS");
    rc = MPL_env2int("MPIR_PARAM_BCAST_MIN_PROCS", &(MPIR_CVAR_BCAST_MIN_PROCS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BCAST_MIN_PROCS");
    rc = MPL_env2int("MPIR_CVAR_BCAST_MIN_PROCS", &(MPIR_CVAR_BCAST_MIN_PROCS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BCAST_MIN_PROCS");

    defaultval.d = 12288;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_BCAST_SHORT_MSG_SIZE, /* name */
        &MPIR_CVAR_BCAST_SHORT_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Let's define short messages as messages with size < MPIR_CVAR_BCAST_SHORT_MSG_SIZE, and medium messages as messages with size >= MPIR_CVAR_BCAST_SHORT_MSG_SIZE but < MPIR_CVAR_BCAST_LONG_MSG_SIZE, and long messages as messages with size >= MPIR_CVAR_BCAST_LONG_MSG_SIZE. The broadcast algorithms selection procedure is as follows. For short messages or when the number of processes is < MPIR_CVAR_BCAST_MIN_PROCS, we do broadcast using the binomial tree algorithm. Otherwise, for medium messages and with a power-of-two number of processes, we do broadcast based on a scatter followed by a recursive doubling allgather algorithm. Otherwise, for long messages or with non power-of-two number of processes, we do broadcast based on a scatter followed by a ring allgather algorithm. (See also: MPIR_CVAR_BCAST_MIN_PROCS, MPIR_CVAR_BCAST_LONG_MSG_SIZE)");
    MPIR_CVAR_BCAST_SHORT_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_BCAST_SHORT_MSG_SIZE", &(MPIR_CVAR_BCAST_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BCAST_SHORT_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_BCAST_SHORT_MSG_SIZE", &(MPIR_CVAR_BCAST_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BCAST_SHORT_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_BCAST_SHORT_MSG_SIZE", &(MPIR_CVAR_BCAST_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BCAST_SHORT_MSG_SIZE");

    defaultval.d = 524288;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_BCAST_LONG_MSG_SIZE, /* name */
        &MPIR_CVAR_BCAST_LONG_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Let's define short messages as messages with size < MPIR_CVAR_BCAST_SHORT_MSG_SIZE, and medium messages as messages with size >= MPIR_CVAR_BCAST_SHORT_MSG_SIZE but < MPIR_CVAR_BCAST_LONG_MSG_SIZE, and long messages as messages with size >= MPIR_CVAR_BCAST_LONG_MSG_SIZE. The broadcast algorithms selection procedure is as follows. For short messages or when the number of processes is < MPIR_CVAR_BCAST_MIN_PROCS, we do broadcast using the binomial tree algorithm. Otherwise, for medium messages and with a power-of-two number of processes, we do broadcast based on a scatter followed by a recursive doubling allgather algorithm. Otherwise, for long messages or with non power-of-two number of processes, we do broadcast based on a scatter followed by a ring allgather algorithm. (See also: MPIR_CVAR_BCAST_MIN_PROCS, MPIR_CVAR_BCAST_SHORT_MSG_SIZE)");
    MPIR_CVAR_BCAST_LONG_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_BCAST_LONG_MSG_SIZE", &(MPIR_CVAR_BCAST_LONG_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BCAST_LONG_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_BCAST_LONG_MSG_SIZE", &(MPIR_CVAR_BCAST_LONG_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BCAST_LONG_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_BCAST_LONG_MSG_SIZE", &(MPIR_CVAR_BCAST_LONG_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BCAST_LONG_MSG_SIZE");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_MAX_SMP_BCAST_MSG_SIZE, /* name */
        &MPIR_CVAR_MAX_SMP_BCAST_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Maximum message size for which SMP-aware broadcast is used.  A value of '0' uses SMP-aware broadcast for all message sizes.");
    MPIR_CVAR_MAX_SMP_BCAST_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_MAX_SMP_BCAST_MSG_SIZE", &(MPIR_CVAR_MAX_SMP_BCAST_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_MAX_SMP_BCAST_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_MAX_SMP_BCAST_MSG_SIZE", &(MPIR_CVAR_MAX_SMP_BCAST_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_MAX_SMP_BCAST_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_MAX_SMP_BCAST_MSG_SIZE", &(MPIR_CVAR_MAX_SMP_BCAST_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_MAX_SMP_BCAST_MSG_SIZE");

    defaultval.d = MPIR_CVAR_BCAST_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_BCAST_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_BCAST_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select bcast algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
binomial                                - Force Binomial Tree\
nb                                      - Force nonblocking algorithm\
smp                                     - Force smp algorithm\
scatter_recursive_doubling_allgather    - Force Scatter Recursive-Doubling Allgather\
scatter_ring_allgather                  - Force Scatter Ring");
    MPIR_CVAR_BCAST_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_BCAST_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BCAST_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_BCAST_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BCAST_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_BCAST_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BCAST_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_BCAST_INTRA_ALGORITHM = MPIR_CVAR_BCAST_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "binomial"))
            MPIR_CVAR_BCAST_INTRA_ALGORITHM = MPIR_CVAR_BCAST_INTRA_ALGORITHM_binomial;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_BCAST_INTRA_ALGORITHM = MPIR_CVAR_BCAST_INTRA_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "smp"))
            MPIR_CVAR_BCAST_INTRA_ALGORITHM = MPIR_CVAR_BCAST_INTRA_ALGORITHM_smp;
        else if (0 == strcmp(tmp_str, "scatter_recursive_doubling_allgather"))
            MPIR_CVAR_BCAST_INTRA_ALGORITHM = MPIR_CVAR_BCAST_INTRA_ALGORITHM_scatter_recursive_doubling_allgather;
        else if (0 == strcmp(tmp_str, "scatter_ring_allgather"))
            MPIR_CVAR_BCAST_INTRA_ALGORITHM = MPIR_CVAR_BCAST_INTRA_ALGORITHM_scatter_ring_allgather;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_BCAST_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_BCAST_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_BCAST_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_BCAST_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select bcast algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb                      - Force nonblocking algorithm\
remote_send_local_bcast - Force remote-send-local-bcast algorithm");
    MPIR_CVAR_BCAST_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_BCAST_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BCAST_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_BCAST_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BCAST_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_BCAST_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BCAST_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_BCAST_INTER_ALGORITHM = MPIR_CVAR_BCAST_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_BCAST_INTER_ALGORITHM = MPIR_CVAR_BCAST_INTER_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "remote_send_local_bcast"))
            MPIR_CVAR_BCAST_INTER_ALGORITHM = MPIR_CVAR_BCAST_INTER_ALGORITHM_remote_send_local_bcast;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_BCAST_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_BCAST_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_BCAST_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Bcast will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_BCAST_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_BCAST_DEVICE_COLLECTIVE", &(MPIR_CVAR_BCAST_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BCAST_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_BCAST_DEVICE_COLLECTIVE", &(MPIR_CVAR_BCAST_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BCAST_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_BCAST_DEVICE_COLLECTIVE", &(MPIR_CVAR_BCAST_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BCAST_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_EXSCAN_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_EXSCAN_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_EXSCAN_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select allgather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb                 - Force nonblocking algorithm\
recursive_doubling - Force recursive doubling algorithm");
    MPIR_CVAR_EXSCAN_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_EXSCAN_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_EXSCAN_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_EXSCAN_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_EXSCAN_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_EXSCAN_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_EXSCAN_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_EXSCAN_INTRA_ALGORITHM = MPIR_CVAR_EXSCAN_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_EXSCAN_INTRA_ALGORITHM = MPIR_CVAR_EXSCAN_INTRA_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "recursive_doubling"))
            MPIR_CVAR_EXSCAN_INTRA_ALGORITHM = MPIR_CVAR_EXSCAN_INTRA_ALGORITHM_recursive_doubling;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_EXSCAN_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_EXSCAN_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_EXSCAN_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Exscan will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_EXSCAN_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_EXSCAN_DEVICE_COLLECTIVE", &(MPIR_CVAR_EXSCAN_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_EXSCAN_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_EXSCAN_DEVICE_COLLECTIVE", &(MPIR_CVAR_EXSCAN_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_EXSCAN_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_EXSCAN_DEVICE_COLLECTIVE", &(MPIR_CVAR_EXSCAN_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_EXSCAN_DEVICE_COLLECTIVE");

    defaultval.d = 2048;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_GATHER_INTER_SHORT_MSG_SIZE, /* name */
        &MPIR_CVAR_GATHER_INTER_SHORT_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "use the short message algorithm for intercommunicator MPI_Gather if the send buffer size is < this value (in bytes) (See also: MPIR_CVAR_GATHER_VSMALL_MSG_SIZE)");
    MPIR_CVAR_GATHER_INTER_SHORT_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_GATHER_INTER_SHORT_MSG_SIZE", &(MPIR_CVAR_GATHER_INTER_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_GATHER_INTER_SHORT_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_GATHER_INTER_SHORT_MSG_SIZE", &(MPIR_CVAR_GATHER_INTER_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_GATHER_INTER_SHORT_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_GATHER_INTER_SHORT_MSG_SIZE", &(MPIR_CVAR_GATHER_INTER_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_GATHER_INTER_SHORT_MSG_SIZE");

    defaultval.d = MPIR_CVAR_GATHER_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_GATHER_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_GATHER_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select gather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
binomial - Force binomial algorithm\
nb       - Force nonblocking algorithm");
    MPIR_CVAR_GATHER_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_GATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_GATHER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_GATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_GATHER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_GATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_GATHER_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_GATHER_INTRA_ALGORITHM = MPIR_CVAR_GATHER_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "binomial"))
            MPIR_CVAR_GATHER_INTRA_ALGORITHM = MPIR_CVAR_GATHER_INTRA_ALGORITHM_binomial;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_GATHER_INTRA_ALGORITHM = MPIR_CVAR_GATHER_INTRA_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_GATHER_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_GATHER_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_GATHER_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_GATHER_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select gather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
linear                   - Force linear algorithm\
local_gather_remote_send - Force local-gather-remote-send algorithm\
nb                       - Force nonblocking algorithm");
    MPIR_CVAR_GATHER_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_GATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_GATHER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_GATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_GATHER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_GATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_GATHER_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_GATHER_INTER_ALGORITHM = MPIR_CVAR_GATHER_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "linear"))
            MPIR_CVAR_GATHER_INTER_ALGORITHM = MPIR_CVAR_GATHER_INTER_ALGORITHM_linear;
        else if (0 == strcmp(tmp_str, "local_gather_remote_send"))
            MPIR_CVAR_GATHER_INTER_ALGORITHM = MPIR_CVAR_GATHER_INTER_ALGORITHM_local_gather_remote_send;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_GATHER_INTER_ALGORITHM = MPIR_CVAR_GATHER_INTER_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_GATHER_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_GATHER_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_GATHER_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Gather will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_GATHER_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_GATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_GATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_GATHER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_GATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_GATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_GATHER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_GATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_GATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_GATHER_DEVICE_COLLECTIVE");

    defaultval.d = 1024;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_GATHER_VSMALL_MSG_SIZE, /* name */
        &MPIR_CVAR_GATHER_VSMALL_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "use a temporary buffer for intracommunicator MPI_Gather if the send buffer size is < this value (in bytes) (See also: MPIR_CVAR_GATHER_INTER_SHORT_MSG_SIZE)");
    MPIR_CVAR_GATHER_VSMALL_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_GATHER_VSMALL_MSG_SIZE", &(MPIR_CVAR_GATHER_VSMALL_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_GATHER_VSMALL_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_GATHER_VSMALL_MSG_SIZE", &(MPIR_CVAR_GATHER_VSMALL_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_GATHER_VSMALL_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_GATHER_VSMALL_MSG_SIZE", &(MPIR_CVAR_GATHER_VSMALL_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_GATHER_VSMALL_MSG_SIZE");

    defaultval.d = MPIR_CVAR_GATHERV_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_GATHERV_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_GATHERV_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select gatherv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
linear - Force linear algorithm\
nb     - Force nonblocking algorithm");
    MPIR_CVAR_GATHERV_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_GATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_GATHERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_GATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_GATHERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_GATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_GATHERV_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_GATHERV_INTRA_ALGORITHM = MPIR_CVAR_GATHERV_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "linear"))
            MPIR_CVAR_GATHERV_INTRA_ALGORITHM = MPIR_CVAR_GATHERV_INTRA_ALGORITHM_linear;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_GATHERV_INTRA_ALGORITHM = MPIR_CVAR_GATHERV_INTRA_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_GATHERV_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_GATHERV_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_GATHERV_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_GATHERV_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select gatherv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
linear - Force linear algorithm\
nb     - Force nonblocking algorithm");
    MPIR_CVAR_GATHERV_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_GATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_GATHERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_GATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_GATHERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_GATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_GATHERV_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_GATHERV_INTER_ALGORITHM = MPIR_CVAR_GATHERV_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "linear"))
            MPIR_CVAR_GATHERV_INTER_ALGORITHM = MPIR_CVAR_GATHERV_INTER_ALGORITHM_linear;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_GATHERV_INTER_ALGORITHM = MPIR_CVAR_GATHERV_INTER_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_GATHERV_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_GATHERV_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_GATHERV_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Gatherv will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_GATHERV_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_GATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_GATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_GATHERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_GATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_GATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_GATHERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_GATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_GATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_GATHERV_DEVICE_COLLECTIVE");

    defaultval.d = 32;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_GATHERV_INTER_SSEND_MIN_PROCS, /* name */
        &MPIR_CVAR_GATHERV_INTER_SSEND_MIN_PROCS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Use Ssend (synchronous send) for intercommunicator MPI_Gatherv if the \"group B\" size is >= this value.  Specifying \"-1\" always avoids using Ssend.  For backwards compatibility, specifying \"0\" uses the default value.");
    MPIR_CVAR_GATHERV_INTER_SSEND_MIN_PROCS = defaultval.d;
    rc = MPL_env2int("MPICH_GATHERV_INTER_SSEND_MIN_PROCS", &(MPIR_CVAR_GATHERV_INTER_SSEND_MIN_PROCS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_GATHERV_INTER_SSEND_MIN_PROCS");
    rc = MPL_env2int("MPIR_PARAM_GATHERV_INTER_SSEND_MIN_PROCS", &(MPIR_CVAR_GATHERV_INTER_SSEND_MIN_PROCS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_GATHERV_INTER_SSEND_MIN_PROCS");
    rc = MPL_env2int("MPIR_CVAR_GATHERV_INTER_SSEND_MIN_PROCS", &(MPIR_CVAR_GATHERV_INTER_SSEND_MIN_PROCS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_GATHERV_INTER_SSEND_MIN_PROCS");

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLGATHER_RECEXCH_KVAL, /* name */
        &MPIR_CVAR_IALLGATHER_RECEXCH_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for recursive exchange based iallgather");
    MPIR_CVAR_IALLGATHER_RECEXCH_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IALLGATHER_RECEXCH_KVAL", &(MPIR_CVAR_IALLGATHER_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLGATHER_RECEXCH_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IALLGATHER_RECEXCH_KVAL", &(MPIR_CVAR_IALLGATHER_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLGATHER_RECEXCH_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IALLGATHER_RECEXCH_KVAL", &(MPIR_CVAR_IALLGATHER_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLGATHER_RECEXCH_KVAL");

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLGATHER_BRUCKS_KVAL, /* name */
        &MPIR_CVAR_IALLGATHER_BRUCKS_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for radix in brucks based iallgather");
    MPIR_CVAR_IALLGATHER_BRUCKS_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IALLGATHER_BRUCKS_KVAL", &(MPIR_CVAR_IALLGATHER_BRUCKS_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLGATHER_BRUCKS_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IALLGATHER_BRUCKS_KVAL", &(MPIR_CVAR_IALLGATHER_BRUCKS_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLGATHER_BRUCKS_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IALLGATHER_BRUCKS_KVAL", &(MPIR_CVAR_IALLGATHER_BRUCKS_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLGATHER_BRUCKS_KVAL");

    defaultval.d = MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select iallgather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_ring               - Force ring algorithm\
sched_brucks             - Force brucks algorithm\
sched_recursive_doubling - Force recursive doubling algorithm\
gentran_ring       - Force generic transport ring algorithm\
gentran_brucks     - Force generic transport based brucks algorithm\
gentran_recexch_doubling - Force generic transport recursive exchange with neighbours doubling in distance in each phase\
gentran_recexch_halving  - Force generic transport recursive exchange with neighbours halving in distance in each phase");
    MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IALLGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLGATHER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IALLGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLGATHER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_ring"))
            MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM_sched_ring;
        else if (0 == strcmp(tmp_str, "sched_brucks"))
            MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM_sched_brucks;
        else if (0 == strcmp(tmp_str, "sched_recursive_doubling"))
            MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM_sched_recursive_doubling;
        else if (0 == strcmp(tmp_str, "gentran_ring"))
            MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM_gentran_ring;
        else if (0 == strcmp(tmp_str, "gentran_brucks"))
            MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM_gentran_brucks;
        else if (0 == strcmp(tmp_str, "gentran_recexch_doubling"))
            MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM_gentran_recexch_doubling;
        else if (0 == strcmp(tmp_str, "gentran_recexch_halving"))
            MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM_gentran_recexch_halving;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_IALLGATHER_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLGATHER_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_IALLGATHER_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select iallgather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_local_gather_remote_bcast - Force local-gather-remote-bcast algorithm");
    MPIR_CVAR_IALLGATHER_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IALLGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLGATHER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IALLGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLGATHER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IALLGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLGATHER_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IALLGATHER_INTER_ALGORITHM = MPIR_CVAR_IALLGATHER_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IALLGATHER_INTER_ALGORITHM = MPIR_CVAR_IALLGATHER_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_local_gather_remote_bcast"))
            MPIR_CVAR_IALLGATHER_INTER_ALGORITHM = MPIR_CVAR_IALLGATHER_INTER_ALGORITHM_sched_local_gather_remote_bcast;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IALLGATHER_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLGATHER_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_IALLGATHER_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Iallgather will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_IALLGATHER_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_IALLGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLGATHER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_IALLGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLGATHER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_IALLGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLGATHER_DEVICE_COLLECTIVE");

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLGATHERV_RECEXCH_KVAL, /* name */
        &MPIR_CVAR_IALLGATHERV_RECEXCH_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for recursive exchange based iallgatherv");
    MPIR_CVAR_IALLGATHERV_RECEXCH_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IALLGATHERV_RECEXCH_KVAL", &(MPIR_CVAR_IALLGATHERV_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLGATHERV_RECEXCH_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IALLGATHERV_RECEXCH_KVAL", &(MPIR_CVAR_IALLGATHERV_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLGATHERV_RECEXCH_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IALLGATHERV_RECEXCH_KVAL", &(MPIR_CVAR_IALLGATHERV_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLGATHERV_RECEXCH_KVAL");

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLGATHERV_BRUCKS_KVAL, /* name */
        &MPIR_CVAR_IALLGATHERV_BRUCKS_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for radix in brucks based iallgatherv");
    MPIR_CVAR_IALLGATHERV_BRUCKS_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IALLGATHERV_BRUCKS_KVAL", &(MPIR_CVAR_IALLGATHERV_BRUCKS_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLGATHERV_BRUCKS_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IALLGATHERV_BRUCKS_KVAL", &(MPIR_CVAR_IALLGATHERV_BRUCKS_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLGATHERV_BRUCKS_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IALLGATHERV_BRUCKS_KVAL", &(MPIR_CVAR_IALLGATHERV_BRUCKS_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLGATHERV_BRUCKS_KVAL");

    defaultval.d = MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select iallgatherv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_brucks             - Force brucks algorithm\
sched_recursive_doubling - Force recursive doubling algorithm\
sched_ring               - Force ring algorithm\
gentran_recexch_doubling - Force generic transport recursive exchange with neighbours doubling in distance in each phase\
gentran_recexch_halving  - Force generic transport recursive exchange with neighbours halving in distance in each phase\
gentran_ring             - Force generic transport ring algorithm\
gentran_brucks           - Force generic transport based brucks algorithm");
    MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IALLGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLGATHERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IALLGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLGATHERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_brucks"))
            MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM_sched_brucks;
        else if (0 == strcmp(tmp_str, "sched_recursive_doubling"))
            MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM_sched_recursive_doubling;
        else if (0 == strcmp(tmp_str, "sched_ring"))
            MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM_sched_ring;
        else if (0 == strcmp(tmp_str, "gentran_recexch_doubling"))
            MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM_gentran_recexch_doubling;
        else if (0 == strcmp(tmp_str, "gentran_recexch_halving"))
            MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM_gentran_recexch_halving;
        else if (0 == strcmp(tmp_str, "gentran_ring"))
            MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM_gentran_ring;
        else if (0 == strcmp(tmp_str, "gentran_brucks"))
            MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM_gentran_brucks;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select iallgatherv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_remote_gather_local_bcast - Force remote-gather-local-bcast algorithm");
    MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IALLGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLGATHERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IALLGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLGATHERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM = MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM = MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_remote_gather_local_bcast"))
            MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM = MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM_sched_remote_gather_local_bcast;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLGATHERV_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_IALLGATHERV_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Iallgatherv will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_IALLGATHERV_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_IALLGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLGATHERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_IALLGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLGATHERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_IALLGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLGATHERV_DEVICE_COLLECTIVE");

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLREDUCE_TREE_KVAL, /* name */
        &MPIR_CVAR_IALLREDUCE_TREE_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for tree based iallreduce (for tree_kary and tree_knomial)");
    MPIR_CVAR_IALLREDUCE_TREE_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IALLREDUCE_TREE_KVAL", &(MPIR_CVAR_IALLREDUCE_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLREDUCE_TREE_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IALLREDUCE_TREE_KVAL", &(MPIR_CVAR_IALLREDUCE_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLREDUCE_TREE_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IALLREDUCE_TREE_KVAL", &(MPIR_CVAR_IALLREDUCE_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLREDUCE_TREE_KVAL");

    defaultval.str = (const char *) "kary";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_IALLREDUCE_TREE_TYPE, /* name */
        &MPIR_CVAR_IALLREDUCE_TREE_TYPE, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Tree type for tree based ibcast kary      - kary tree type knomial_1 - knomial_1 tree type knomial_2 - knomial_2 tree type");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_IALLREDUCE_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLREDUCE_TREE_TYPE");
    rc = MPL_env2str("MPIR_PARAM_IALLREDUCE_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLREDUCE_TREE_TYPE");
    rc = MPL_env2str("MPIR_CVAR_IALLREDUCE_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLREDUCE_TREE_TYPE");
    if (tmp_str != NULL) {
        MPIR_CVAR_IALLREDUCE_TREE_TYPE = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_IALLREDUCE_TREE_TYPE);
        if (MPIR_CVAR_IALLREDUCE_TREE_TYPE == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_IALLREDUCE_TREE_TYPE");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_IALLREDUCE_TREE_TYPE = NULL;
    }

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLREDUCE_TREE_PIPELINE_CHUNK_SIZE, /* name */
        &MPIR_CVAR_IALLREDUCE_TREE_PIPELINE_CHUNK_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Maximum chunk size (in bytes) for pipelining in tree based iallreduce. Default value is 0, that is, no pipelining by default");
    MPIR_CVAR_IALLREDUCE_TREE_PIPELINE_CHUNK_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_IALLREDUCE_TREE_PIPELINE_CHUNK_SIZE", &(MPIR_CVAR_IALLREDUCE_TREE_PIPELINE_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLREDUCE_TREE_PIPELINE_CHUNK_SIZE");
    rc = MPL_env2int("MPIR_PARAM_IALLREDUCE_TREE_PIPELINE_CHUNK_SIZE", &(MPIR_CVAR_IALLREDUCE_TREE_PIPELINE_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLREDUCE_TREE_PIPELINE_CHUNK_SIZE");
    rc = MPL_env2int("MPIR_CVAR_IALLREDUCE_TREE_PIPELINE_CHUNK_SIZE", &(MPIR_CVAR_IALLREDUCE_TREE_PIPELINE_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLREDUCE_TREE_PIPELINE_CHUNK_SIZE");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLREDUCE_TREE_BUFFER_PER_CHILD, /* name */
        &MPIR_CVAR_IALLREDUCE_TREE_BUFFER_PER_CHILD, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "If set to true, a rank in tree_kary and tree_knomial algorithms will allocate a dedicated buffer for every child it receives data from. This would mean more memory consumption but it would allow preposting of the receives and hence reduce the number of unexpected messages. If set to false, there is only one buffer that is used to receive the data from all the children. The receives are therefore serialized, that is, only one receive can be posted at a time.");
    MPIR_CVAR_IALLREDUCE_TREE_BUFFER_PER_CHILD = defaultval.d;
    rc = MPL_env2bool("MPICH_IALLREDUCE_TREE_BUFFER_PER_CHILD", &(MPIR_CVAR_IALLREDUCE_TREE_BUFFER_PER_CHILD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLREDUCE_TREE_BUFFER_PER_CHILD");
    rc = MPL_env2bool("MPIR_PARAM_IALLREDUCE_TREE_BUFFER_PER_CHILD", &(MPIR_CVAR_IALLREDUCE_TREE_BUFFER_PER_CHILD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLREDUCE_TREE_BUFFER_PER_CHILD");
    rc = MPL_env2bool("MPIR_CVAR_IALLREDUCE_TREE_BUFFER_PER_CHILD", &(MPIR_CVAR_IALLREDUCE_TREE_BUFFER_PER_CHILD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLREDUCE_TREE_BUFFER_PER_CHILD");

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLREDUCE_RECEXCH_KVAL, /* name */
        &MPIR_CVAR_IALLREDUCE_RECEXCH_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for recursive exchange based iallreduce");
    MPIR_CVAR_IALLREDUCE_RECEXCH_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IALLREDUCE_RECEXCH_KVAL", &(MPIR_CVAR_IALLREDUCE_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLREDUCE_RECEXCH_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IALLREDUCE_RECEXCH_KVAL", &(MPIR_CVAR_IALLREDUCE_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLREDUCE_RECEXCH_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IALLREDUCE_RECEXCH_KVAL", &(MPIR_CVAR_IALLREDUCE_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLREDUCE_RECEXCH_KVAL");

    defaultval.d = MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select iallreduce algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_naive                      - Force naive algorithm\
sched_smp                        - Force smp algorithm\
sched_recursive_doubling         - Force recursive doubling algorithm\
sched_reduce_scatter_allgather   - Force reduce scatter allgather algorithm\
gentran_recexch_single_buffer    - Force generic transport recursive exchange with single buffer for receives\
gentran_recexch_multiple_buffer  - Force generic transport recursive exchange with multiple buffers for receives\
gentran_tree                     - Force generic transport tree algorithm\
gentran_ring                     - Force generic transport ring algorithm\
gentran_recexch_reduce_scatter_recexch_allgatherv  - Force generic transport recursive exchange with reduce scatter and allgatherv");
    MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IALLREDUCE_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLREDUCE_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IALLREDUCE_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLREDUCE_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_naive"))
            MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM_sched_naive;
        else if (0 == strcmp(tmp_str, "sched_smp"))
            MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM_sched_smp;
        else if (0 == strcmp(tmp_str, "sched_recursive_doubling"))
            MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM_sched_recursive_doubling;
        else if (0 == strcmp(tmp_str, "sched_reduce_scatter_allgather"))
            MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM_sched_reduce_scatter_allgather;
        else if (0 == strcmp(tmp_str, "gentran_recexch_single_buffer"))
            MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM_gentran_recexch_single_buffer;
        else if (0 == strcmp(tmp_str, "gentran_recexch_multiple_buffer"))
            MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM_gentran_recexch_multiple_buffer;
        else if (0 == strcmp(tmp_str, "gentran_tree"))
            MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM_gentran_tree;
        else if (0 == strcmp(tmp_str, "gentran_ring"))
            MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM_gentran_ring;
        else if (0 == strcmp(tmp_str, "gentran_recexch_reduce_scatter_recexch_allgatherv"))
            MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM_gentran_recexch_reduce_scatter_recexch_allgatherv;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select iallreduce algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_remote_reduce_local_bcast - Force remote-reduce-local-bcast algorithm");
    MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IALLREDUCE_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLREDUCE_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IALLREDUCE_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLREDUCE_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM = MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM = MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_remote_reduce_local_bcast"))
            MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM = MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM_sched_remote_reduce_local_bcast;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLREDUCE_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_IALLREDUCE_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Iallreduce will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_IALLREDUCE_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_IALLREDUCE_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLREDUCE_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLREDUCE_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_IALLREDUCE_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLREDUCE_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLREDUCE_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_IALLREDUCE_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLREDUCE_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLREDUCE_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ialltoall algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_brucks            - Force brucks algorithm\
sched_inplace           - Force inplace algorithm\
sched_pairwise          - Force pairwise algorithm\
sched_permuted_sendrecv - Force permuted sendrecv algorithm\
gentran_ring            - Force generic transport based ring algorithm\
gentran_brucks          - Force generic transport based brucks algorithm\
gentran_scattered       - Force generic transport based scattered algorithm");
    MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IALLTOALL_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALL_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IALLTOALL_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALL_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_brucks"))
            MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM_sched_brucks;
        else if (0 == strcmp(tmp_str, "sched_inplace"))
            MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM_sched_inplace;
        else if (0 == strcmp(tmp_str, "sched_pairwise"))
            MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM_sched_pairwise;
        else if (0 == strcmp(tmp_str, "sched_permuted_sendrecv"))
            MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM_sched_permuted_sendrecv;
        else if (0 == strcmp(tmp_str, "gentran_ring"))
            MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM_gentran_ring;
        else if (0 == strcmp(tmp_str, "gentran_brucks"))
            MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM_gentran_brucks;
        else if (0 == strcmp(tmp_str, "gentran_scattered"))
            MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM_gentran_scattered;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_IALLTOALL_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALL_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_IALLTOALL_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ialltoall algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_pairwise_exchange - Force pairwise exchange algorithm");
    MPIR_CVAR_IALLTOALL_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IALLTOALL_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALL_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IALLTOALL_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALL_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IALLTOALL_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALL_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IALLTOALL_INTER_ALGORITHM = MPIR_CVAR_IALLTOALL_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IALLTOALL_INTER_ALGORITHM = MPIR_CVAR_IALLTOALL_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_pairwise_exchange"))
            MPIR_CVAR_IALLTOALL_INTER_ALGORITHM = MPIR_CVAR_IALLTOALL_INTER_ALGORITHM_sched_pairwise_exchange;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IALLTOALL_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALL_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_IALLTOALL_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Ialltoall will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_IALLTOALL_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_IALLTOALL_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLTOALL_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALL_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_IALLTOALL_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLTOALL_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALL_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_IALLTOALL_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLTOALL_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALL_DEVICE_COLLECTIVE");

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALL_BRUCKS_KVAL, /* name */
        &MPIR_CVAR_IALLTOALL_BRUCKS_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "radix (k) value for generic transport brucks based ialltoall");
    MPIR_CVAR_IALLTOALL_BRUCKS_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IALLTOALL_BRUCKS_KVAL", &(MPIR_CVAR_IALLTOALL_BRUCKS_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALL_BRUCKS_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IALLTOALL_BRUCKS_KVAL", &(MPIR_CVAR_IALLTOALL_BRUCKS_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALL_BRUCKS_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IALLTOALL_BRUCKS_KVAL", &(MPIR_CVAR_IALLTOALL_BRUCKS_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALL_BRUCKS_KVAL");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALL_BRUCKS_BUFFER_PER_NBR, /* name */
        &MPIR_CVAR_IALLTOALL_BRUCKS_BUFFER_PER_NBR, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "If set to true, the gentran based brucks algorithm will allocate dedicated send and receive buffers for every neighbor in the brucks algorithm. Otherwise, it would reuse a single buffer for sending and receiving data to/from neighbors");
    MPIR_CVAR_IALLTOALL_BRUCKS_BUFFER_PER_NBR = defaultval.d;
    rc = MPL_env2bool("MPICH_IALLTOALL_BRUCKS_BUFFER_PER_NBR", &(MPIR_CVAR_IALLTOALL_BRUCKS_BUFFER_PER_NBR));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALL_BRUCKS_BUFFER_PER_NBR");
    rc = MPL_env2bool("MPIR_PARAM_IALLTOALL_BRUCKS_BUFFER_PER_NBR", &(MPIR_CVAR_IALLTOALL_BRUCKS_BUFFER_PER_NBR));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALL_BRUCKS_BUFFER_PER_NBR");
    rc = MPL_env2bool("MPIR_CVAR_IALLTOALL_BRUCKS_BUFFER_PER_NBR", &(MPIR_CVAR_IALLTOALL_BRUCKS_BUFFER_PER_NBR));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALL_BRUCKS_BUFFER_PER_NBR");

    defaultval.d = 64;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALL_SCATTERED_OUTSTANDING_TASKS, /* name */
        &MPIR_CVAR_IALLTOALL_SCATTERED_OUTSTANDING_TASKS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Maximum number of outstanding sends and recvs posted at a time");
    MPIR_CVAR_IALLTOALL_SCATTERED_OUTSTANDING_TASKS = defaultval.d;
    rc = MPL_env2int("MPICH_IALLTOALL_SCATTERED_OUTSTANDING_TASKS", &(MPIR_CVAR_IALLTOALL_SCATTERED_OUTSTANDING_TASKS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALL_SCATTERED_OUTSTANDING_TASKS");
    rc = MPL_env2int("MPIR_PARAM_IALLTOALL_SCATTERED_OUTSTANDING_TASKS", &(MPIR_CVAR_IALLTOALL_SCATTERED_OUTSTANDING_TASKS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALL_SCATTERED_OUTSTANDING_TASKS");
    rc = MPL_env2int("MPIR_CVAR_IALLTOALL_SCATTERED_OUTSTANDING_TASKS", &(MPIR_CVAR_IALLTOALL_SCATTERED_OUTSTANDING_TASKS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALL_SCATTERED_OUTSTANDING_TASKS");

    defaultval.d = 4;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALL_SCATTERED_BATCH_SIZE, /* name */
        &MPIR_CVAR_IALLTOALL_SCATTERED_BATCH_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Number of send/receive tasks that scattered algorithm waits for completion before posting another batch of send/receives of that size");
    MPIR_CVAR_IALLTOALL_SCATTERED_BATCH_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_IALLTOALL_SCATTERED_BATCH_SIZE", &(MPIR_CVAR_IALLTOALL_SCATTERED_BATCH_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALL_SCATTERED_BATCH_SIZE");
    rc = MPL_env2int("MPIR_PARAM_IALLTOALL_SCATTERED_BATCH_SIZE", &(MPIR_CVAR_IALLTOALL_SCATTERED_BATCH_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALL_SCATTERED_BATCH_SIZE");
    rc = MPL_env2int("MPIR_CVAR_IALLTOALL_SCATTERED_BATCH_SIZE", &(MPIR_CVAR_IALLTOALL_SCATTERED_BATCH_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALL_SCATTERED_BATCH_SIZE");

    defaultval.d = MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ialltoallv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_blocked           - Force blocked algorithm\
sched_inplace           - Force inplace algorithm\
sched_pairwise_exchange - Force pairwise exchange algorithm\
gentran_scattered       - Force generic transport based scattered algorithm\
gentran_blocked         - Force generic transport blocked algorithm\
gentran_inplace         - Force generic transport inplace algorithm");
    MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IALLTOALLV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALLV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IALLTOALLV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALLV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_blocked"))
            MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM_sched_blocked;
        else if (0 == strcmp(tmp_str, "sched_inplace"))
            MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM_sched_inplace;
        else if (0 == strcmp(tmp_str, "sched_pairwise_exchange"))
            MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM_sched_pairwise_exchange;
        else if (0 == strcmp(tmp_str, "gentran_scattered"))
            MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM_gentran_scattered;
        else if (0 == strcmp(tmp_str, "gentran_blocked"))
            MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM_gentran_blocked;
        else if (0 == strcmp(tmp_str, "gentran_inplace"))
            MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM_gentran_inplace;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ialltoallv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_pairwise_exchange - Force pairwise exchange algorithm");
    MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IALLTOALLV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALLV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IALLTOALLV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALLV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM = MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM = MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_pairwise_exchange"))
            MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM = MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM_sched_pairwise_exchange;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALLV_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_IALLTOALLV_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Ialltoallv will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_IALLTOALLV_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_IALLTOALLV_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLTOALLV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALLV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_IALLTOALLV_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLTOALLV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALLV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_IALLTOALLV_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLTOALLV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALLV_DEVICE_COLLECTIVE");

    defaultval.d = 64;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALLV_SCATTERED_OUTSTANDING_TASKS, /* name */
        &MPIR_CVAR_IALLTOALLV_SCATTERED_OUTSTANDING_TASKS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Maximum number of outstanding sends and recvs posted at a time");
    MPIR_CVAR_IALLTOALLV_SCATTERED_OUTSTANDING_TASKS = defaultval.d;
    rc = MPL_env2int("MPICH_IALLTOALLV_SCATTERED_OUTSTANDING_TASKS", &(MPIR_CVAR_IALLTOALLV_SCATTERED_OUTSTANDING_TASKS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALLV_SCATTERED_OUTSTANDING_TASKS");
    rc = MPL_env2int("MPIR_PARAM_IALLTOALLV_SCATTERED_OUTSTANDING_TASKS", &(MPIR_CVAR_IALLTOALLV_SCATTERED_OUTSTANDING_TASKS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALLV_SCATTERED_OUTSTANDING_TASKS");
    rc = MPL_env2int("MPIR_CVAR_IALLTOALLV_SCATTERED_OUTSTANDING_TASKS", &(MPIR_CVAR_IALLTOALLV_SCATTERED_OUTSTANDING_TASKS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALLV_SCATTERED_OUTSTANDING_TASKS");

    defaultval.d = 4;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALLV_SCATTERED_BATCH_SIZE, /* name */
        &MPIR_CVAR_IALLTOALLV_SCATTERED_BATCH_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Number of send/receive tasks that scattered algorithm waits for completion before posting another batch of send/receives of that size");
    MPIR_CVAR_IALLTOALLV_SCATTERED_BATCH_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_IALLTOALLV_SCATTERED_BATCH_SIZE", &(MPIR_CVAR_IALLTOALLV_SCATTERED_BATCH_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALLV_SCATTERED_BATCH_SIZE");
    rc = MPL_env2int("MPIR_PARAM_IALLTOALLV_SCATTERED_BATCH_SIZE", &(MPIR_CVAR_IALLTOALLV_SCATTERED_BATCH_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALLV_SCATTERED_BATCH_SIZE");
    rc = MPL_env2int("MPIR_CVAR_IALLTOALLV_SCATTERED_BATCH_SIZE", &(MPIR_CVAR_IALLTOALLV_SCATTERED_BATCH_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALLV_SCATTERED_BATCH_SIZE");

    defaultval.d = MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ialltoallw algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_blocked           - Force blocked algorithm\
sched_inplace           - Force inplace algorithm\
gentran_blocked   - Force genereic transport based blocked algorithm\
gentran_inplace   - Force genereic transport based inplace algorithm");
    MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IALLTOALLW_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALLW_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IALLTOALLW_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALLW_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_blocked"))
            MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM_sched_blocked;
        else if (0 == strcmp(tmp_str, "sched_inplace"))
            MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM_sched_inplace;
        else if (0 == strcmp(tmp_str, "gentran_blocked"))
            MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM_gentran_blocked;
        else if (0 == strcmp(tmp_str, "gentran_inplace"))
            MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM_gentran_inplace;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ialltoallw algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_pairwise_exchange - Force pairwise exchange algorithm");
    MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IALLTOALLW_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALLW_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IALLTOALLW_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALLW_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM = MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM = MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_pairwise_exchange"))
            MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM = MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM_sched_pairwise_exchange;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IALLTOALLW_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_IALLTOALLW_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Ialltoallw will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_IALLTOALLW_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_IALLTOALLW_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLTOALLW_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IALLTOALLW_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_IALLTOALLW_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLTOALLW_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IALLTOALLW_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_IALLTOALLW_DEVICE_COLLECTIVE", &(MPIR_CVAR_IALLTOALLW_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IALLTOALLW_DEVICE_COLLECTIVE");

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IBARRIER_RECEXCH_KVAL, /* name */
        &MPIR_CVAR_IBARRIER_RECEXCH_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for recursive exchange based ibarrier");
    MPIR_CVAR_IBARRIER_RECEXCH_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IBARRIER_RECEXCH_KVAL", &(MPIR_CVAR_IBARRIER_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IBARRIER_RECEXCH_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IBARRIER_RECEXCH_KVAL", &(MPIR_CVAR_IBARRIER_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IBARRIER_RECEXCH_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IBARRIER_RECEXCH_KVAL", &(MPIR_CVAR_IBARRIER_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IBARRIER_RECEXCH_KVAL");

    defaultval.d = MPIR_CVAR_IBARRIER_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IBARRIER_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_IBARRIER_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ibarrier algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_recursive_doubling - Force recursive doubling algorithm\
gentran_recexch          - Force generic transport based recursive exchange algorithm");
    MPIR_CVAR_IBARRIER_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IBARRIER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IBARRIER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IBARRIER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IBARRIER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IBARRIER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IBARRIER_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IBARRIER_INTRA_ALGORITHM = MPIR_CVAR_IBARRIER_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IBARRIER_INTRA_ALGORITHM = MPIR_CVAR_IBARRIER_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_recursive_doubling"))
            MPIR_CVAR_IBARRIER_INTRA_ALGORITHM = MPIR_CVAR_IBARRIER_INTRA_ALGORITHM_sched_recursive_doubling;
        else if (0 == strcmp(tmp_str, "gentran_recexch"))
            MPIR_CVAR_IBARRIER_INTRA_ALGORITHM = MPIR_CVAR_IBARRIER_INTRA_ALGORITHM_gentran_recexch;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IBARRIER_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_IBARRIER_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IBARRIER_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_IBARRIER_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ibarrier algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_bcast - Force bcast algorithm");
    MPIR_CVAR_IBARRIER_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IBARRIER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IBARRIER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IBARRIER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IBARRIER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IBARRIER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IBARRIER_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IBARRIER_INTER_ALGORITHM = MPIR_CVAR_IBARRIER_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IBARRIER_INTER_ALGORITHM = MPIR_CVAR_IBARRIER_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_bcast"))
            MPIR_CVAR_IBARRIER_INTER_ALGORITHM = MPIR_CVAR_IBARRIER_INTER_ALGORITHM_sched_bcast;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IBARRIER_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IBARRIER_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_IBARRIER_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Ibarrier will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_IBARRIER_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_IBARRIER_DEVICE_COLLECTIVE", &(MPIR_CVAR_IBARRIER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IBARRIER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_IBARRIER_DEVICE_COLLECTIVE", &(MPIR_CVAR_IBARRIER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IBARRIER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_IBARRIER_DEVICE_COLLECTIVE", &(MPIR_CVAR_IBARRIER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IBARRIER_DEVICE_COLLECTIVE");

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IBCAST_TREE_KVAL, /* name */
        &MPIR_CVAR_IBCAST_TREE_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for tree (kary, knomial, etc.) based ibcast");
    MPIR_CVAR_IBCAST_TREE_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IBCAST_TREE_KVAL", &(MPIR_CVAR_IBCAST_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IBCAST_TREE_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IBCAST_TREE_KVAL", &(MPIR_CVAR_IBCAST_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IBCAST_TREE_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IBCAST_TREE_KVAL", &(MPIR_CVAR_IBCAST_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IBCAST_TREE_KVAL");

    defaultval.str = (const char *) "kary";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_IBCAST_TREE_TYPE, /* name */
        &MPIR_CVAR_IBCAST_TREE_TYPE, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Tree type for tree based ibcast kary      - kary tree type knomial_1 - knomial_1 tree type knomial_2 - knomial_2 tree type");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_IBCAST_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IBCAST_TREE_TYPE");
    rc = MPL_env2str("MPIR_PARAM_IBCAST_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IBCAST_TREE_TYPE");
    rc = MPL_env2str("MPIR_CVAR_IBCAST_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IBCAST_TREE_TYPE");
    if (tmp_str != NULL) {
        MPIR_CVAR_IBCAST_TREE_TYPE = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_IBCAST_TREE_TYPE);
        if (MPIR_CVAR_IBCAST_TREE_TYPE == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_IBCAST_TREE_TYPE");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_IBCAST_TREE_TYPE = NULL;
    }

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IBCAST_TREE_PIPELINE_CHUNK_SIZE, /* name */
        &MPIR_CVAR_IBCAST_TREE_PIPELINE_CHUNK_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Maximum chunk size (in bytes) for pipelining in tree based ibcast. Default value is 0, that is, no pipelining by default");
    MPIR_CVAR_IBCAST_TREE_PIPELINE_CHUNK_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_IBCAST_TREE_PIPELINE_CHUNK_SIZE", &(MPIR_CVAR_IBCAST_TREE_PIPELINE_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IBCAST_TREE_PIPELINE_CHUNK_SIZE");
    rc = MPL_env2int("MPIR_PARAM_IBCAST_TREE_PIPELINE_CHUNK_SIZE", &(MPIR_CVAR_IBCAST_TREE_PIPELINE_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IBCAST_TREE_PIPELINE_CHUNK_SIZE");
    rc = MPL_env2int("MPIR_CVAR_IBCAST_TREE_PIPELINE_CHUNK_SIZE", &(MPIR_CVAR_IBCAST_TREE_PIPELINE_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IBCAST_TREE_PIPELINE_CHUNK_SIZE");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IBCAST_RING_CHUNK_SIZE, /* name */
        &MPIR_CVAR_IBCAST_RING_CHUNK_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Maximum chunk size (in bytes) for pipelining in ibcast ring algorithm. Default value is 0, that is, no pipelining by default");
    MPIR_CVAR_IBCAST_RING_CHUNK_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_IBCAST_RING_CHUNK_SIZE", &(MPIR_CVAR_IBCAST_RING_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IBCAST_RING_CHUNK_SIZE");
    rc = MPL_env2int("MPIR_PARAM_IBCAST_RING_CHUNK_SIZE", &(MPIR_CVAR_IBCAST_RING_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IBCAST_RING_CHUNK_SIZE");
    rc = MPL_env2int("MPIR_CVAR_IBCAST_RING_CHUNK_SIZE", &(MPIR_CVAR_IBCAST_RING_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IBCAST_RING_CHUNK_SIZE");

    defaultval.d = MPIR_CVAR_IBCAST_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IBCAST_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_IBCAST_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ibcast algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_binomial                             - Force Binomial algorithm\
sched_smp                                  - Force smp algorithm\
sched_scatter_recursive_doubling_allgather - Force Scatter Recursive Doubling Allgather algorithm\
sched_scatter_ring_allgather               - Force Scatter Ring Allgather algorithm\
gentran_tree                               - Force Generic Transport Tree algorithm\
gentran_scatterv_recexch_allgatherv        - Force Generic Transport Scatterv followed by Recursive Exchange Allgatherv algorithm\
gentran_ring                               - Force Generic Transport Ring algorithm");
    MPIR_CVAR_IBCAST_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IBCAST_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IBCAST_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IBCAST_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IBCAST_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IBCAST_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IBCAST_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IBCAST_INTRA_ALGORITHM = MPIR_CVAR_IBCAST_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IBCAST_INTRA_ALGORITHM = MPIR_CVAR_IBCAST_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_binomial"))
            MPIR_CVAR_IBCAST_INTRA_ALGORITHM = MPIR_CVAR_IBCAST_INTRA_ALGORITHM_sched_binomial;
        else if (0 == strcmp(tmp_str, "sched_smp"))
            MPIR_CVAR_IBCAST_INTRA_ALGORITHM = MPIR_CVAR_IBCAST_INTRA_ALGORITHM_sched_smp;
        else if (0 == strcmp(tmp_str, "sched_scatter_recursive_doubling_allgather"))
            MPIR_CVAR_IBCAST_INTRA_ALGORITHM = MPIR_CVAR_IBCAST_INTRA_ALGORITHM_sched_scatter_recursive_doubling_allgather;
        else if (0 == strcmp(tmp_str, "sched_scatter_ring_allgather"))
            MPIR_CVAR_IBCAST_INTRA_ALGORITHM = MPIR_CVAR_IBCAST_INTRA_ALGORITHM_sched_scatter_ring_allgather;
        else if (0 == strcmp(tmp_str, "gentran_tree"))
            MPIR_CVAR_IBCAST_INTRA_ALGORITHM = MPIR_CVAR_IBCAST_INTRA_ALGORITHM_gentran_tree;
        else if (0 == strcmp(tmp_str, "gentran_scatterv_recexch_allgatherv"))
            MPIR_CVAR_IBCAST_INTRA_ALGORITHM = MPIR_CVAR_IBCAST_INTRA_ALGORITHM_gentran_scatterv_recexch_allgatherv;
        else if (0 == strcmp(tmp_str, "gentran_ring"))
            MPIR_CVAR_IBCAST_INTRA_ALGORITHM = MPIR_CVAR_IBCAST_INTRA_ALGORITHM_gentran_ring;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IBCAST_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IBCAST_SCATTERV_KVAL, /* name */
        &MPIR_CVAR_IBCAST_SCATTERV_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for tree based scatter in scatter_recexch_allgather algorithm");
    MPIR_CVAR_IBCAST_SCATTERV_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IBCAST_SCATTERV_KVAL", &(MPIR_CVAR_IBCAST_SCATTERV_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IBCAST_SCATTERV_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IBCAST_SCATTERV_KVAL", &(MPIR_CVAR_IBCAST_SCATTERV_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IBCAST_SCATTERV_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IBCAST_SCATTERV_KVAL", &(MPIR_CVAR_IBCAST_SCATTERV_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IBCAST_SCATTERV_KVAL");

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IBCAST_ALLGATHERV_RECEXCH_KVAL, /* name */
        &MPIR_CVAR_IBCAST_ALLGATHERV_RECEXCH_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for recursive exchange based allgather in scatter_recexch_allgather algorithm");
    MPIR_CVAR_IBCAST_ALLGATHERV_RECEXCH_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IBCAST_ALLGATHERV_RECEXCH_KVAL", &(MPIR_CVAR_IBCAST_ALLGATHERV_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IBCAST_ALLGATHERV_RECEXCH_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IBCAST_ALLGATHERV_RECEXCH_KVAL", &(MPIR_CVAR_IBCAST_ALLGATHERV_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IBCAST_ALLGATHERV_RECEXCH_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IBCAST_ALLGATHERV_RECEXCH_KVAL", &(MPIR_CVAR_IBCAST_ALLGATHERV_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IBCAST_ALLGATHERV_RECEXCH_KVAL");

    defaultval.d = MPIR_CVAR_IBCAST_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IBCAST_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_IBCAST_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ibcast algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_flat - Force flat algorithm");
    MPIR_CVAR_IBCAST_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IBCAST_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IBCAST_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IBCAST_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IBCAST_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IBCAST_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IBCAST_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IBCAST_INTER_ALGORITHM = MPIR_CVAR_IBCAST_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IBCAST_INTER_ALGORITHM = MPIR_CVAR_IBCAST_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_flat"))
            MPIR_CVAR_IBCAST_INTER_ALGORITHM = MPIR_CVAR_IBCAST_INTER_ALGORITHM_sched_flat;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IBCAST_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IBCAST_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_IBCAST_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Ibcast will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_IBCAST_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_IBCAST_DEVICE_COLLECTIVE", &(MPIR_CVAR_IBCAST_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IBCAST_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_IBCAST_DEVICE_COLLECTIVE", &(MPIR_CVAR_IBCAST_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IBCAST_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_IBCAST_DEVICE_COLLECTIVE", &(MPIR_CVAR_IBCAST_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IBCAST_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select iexscan algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_recursive_doubling - Force recursive doubling algorithm");
    MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IEXSCAN_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IEXSCAN_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IEXSCAN_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IEXSCAN_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM = MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM = MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_recursive_doubling"))
            MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM = MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM_sched_recursive_doubling;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IEXSCAN_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_IEXSCAN_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Iexscan will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_IEXSCAN_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_IEXSCAN_DEVICE_COLLECTIVE", &(MPIR_CVAR_IEXSCAN_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IEXSCAN_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_IEXSCAN_DEVICE_COLLECTIVE", &(MPIR_CVAR_IEXSCAN_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IEXSCAN_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_IEXSCAN_DEVICE_COLLECTIVE", &(MPIR_CVAR_IEXSCAN_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IEXSCAN_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_IGATHER_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IGATHER_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_IGATHER_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select igather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_binomial     - Force binomial algorithm\
gentran_tree       - Force genetric transport based tree algorithm");
    MPIR_CVAR_IGATHER_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IGATHER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IGATHER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IGATHER_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IGATHER_INTRA_ALGORITHM = MPIR_CVAR_IGATHER_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IGATHER_INTRA_ALGORITHM = MPIR_CVAR_IGATHER_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_binomial"))
            MPIR_CVAR_IGATHER_INTRA_ALGORITHM = MPIR_CVAR_IGATHER_INTRA_ALGORITHM_sched_binomial;
        else if (0 == strcmp(tmp_str, "gentran_tree"))
            MPIR_CVAR_IGATHER_INTRA_ALGORITHM = MPIR_CVAR_IGATHER_INTRA_ALGORITHM_gentran_tree;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IGATHER_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IGATHER_TREE_KVAL, /* name */
        &MPIR_CVAR_IGATHER_TREE_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for tree based igather");
    MPIR_CVAR_IGATHER_TREE_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IGATHER_TREE_KVAL", &(MPIR_CVAR_IGATHER_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IGATHER_TREE_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IGATHER_TREE_KVAL", &(MPIR_CVAR_IGATHER_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IGATHER_TREE_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IGATHER_TREE_KVAL", &(MPIR_CVAR_IGATHER_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IGATHER_TREE_KVAL");

    defaultval.d = MPIR_CVAR_IGATHER_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IGATHER_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_IGATHER_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select igather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_long  - Force long inter algorithm\
sched_short - Force short inter algorithm");
    MPIR_CVAR_IGATHER_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IGATHER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IGATHER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IGATHER_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IGATHER_INTER_ALGORITHM = MPIR_CVAR_IGATHER_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IGATHER_INTER_ALGORITHM = MPIR_CVAR_IGATHER_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_long"))
            MPIR_CVAR_IGATHER_INTER_ALGORITHM = MPIR_CVAR_IGATHER_INTER_ALGORITHM_sched_long;
        else if (0 == strcmp(tmp_str, "sched_short"))
            MPIR_CVAR_IGATHER_INTER_ALGORITHM = MPIR_CVAR_IGATHER_INTER_ALGORITHM_sched_short;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IGATHER_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IGATHER_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_IGATHER_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Igather will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_IGATHER_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_IGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_IGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IGATHER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_IGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_IGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IGATHER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_IGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_IGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IGATHER_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_IGATHERV_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IGATHERV_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_IGATHERV_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select igatherv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear         - Force linear algorithm\
gentran_linear       - Force generic transport based linear algorithm");
    MPIR_CVAR_IGATHERV_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IGATHERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IGATHERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IGATHERV_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IGATHERV_INTRA_ALGORITHM = MPIR_CVAR_IGATHERV_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IGATHERV_INTRA_ALGORITHM = MPIR_CVAR_IGATHERV_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_IGATHERV_INTRA_ALGORITHM = MPIR_CVAR_IGATHERV_INTRA_ALGORITHM_sched_linear;
        else if (0 == strcmp(tmp_str, "gentran_linear"))
            MPIR_CVAR_IGATHERV_INTRA_ALGORITHM = MPIR_CVAR_IGATHERV_INTRA_ALGORITHM_gentran_linear;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IGATHERV_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_IGATHERV_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IGATHERV_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_IGATHERV_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select igatherv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear - Force linear algorithm");
    MPIR_CVAR_IGATHERV_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IGATHERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IGATHERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IGATHERV_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IGATHERV_INTER_ALGORITHM = MPIR_CVAR_IGATHERV_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IGATHERV_INTER_ALGORITHM = MPIR_CVAR_IGATHERV_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_IGATHERV_INTER_ALGORITHM = MPIR_CVAR_IGATHERV_INTER_ALGORITHM_sched_linear;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IGATHERV_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IGATHERV_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_IGATHERV_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Igatherv will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_IGATHERV_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_IGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_IGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IGATHERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_IGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_IGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IGATHERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_IGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_IGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IGATHERV_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ineighbor_allgather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear    - Force linear algorithm\
gentran_linear  - Force generic transport based linear algorithm");
    MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM_sched_linear;
        else if (0 == strcmp(tmp_str, "gentran_linear"))
            MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM_gentran_linear;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ineighbor_allgather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear    - Force linear algorithm\
gentran_linear  - Force generic transport based linear algorithm");
    MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_INEIGHBOR_ALLGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLGATHER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_INEIGHBOR_ALLGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLGATHER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM_sched_linear;
        else if (0 == strcmp(tmp_str, "gentran_linear"))
            MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM_gentran_linear;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Ineighbor_allgather will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_INEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_INEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_INEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_INEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ineighbor_allgatherv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear          - Force linear algorithm\
gentran_linear        - Force generic transport based linear algorithm");
    MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM_sched_linear;
        else if (0 == strcmp(tmp_str, "gentran_linear"))
            MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM_gentran_linear;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ineighbor_allgatherv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear          - Force linear algorithm\
gentran_linear        - Force generic transport based linear algorithm");
    MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM_sched_linear;
        else if (0 == strcmp(tmp_str, "gentran_linear"))
            MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM_gentran_linear;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Ineighbor_allgatherv will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_INEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_INEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_INEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_INEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ineighbor_alltoall algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear          - Force linear algorithm\
gentran_linear        - Force generic transport based linear algorithm");
    MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM_sched_linear;
        else if (0 == strcmp(tmp_str, "gentran_linear"))
            MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM_gentran_linear;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ineighbor_alltoall algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear          - Force linear algorithm\
gentran_linear        - Force generic transport based linear algorithm");
    MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_INEIGHBOR_ALLTOALL_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLTOALL_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_INEIGHBOR_ALLTOALL_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLTOALL_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM_sched_linear;
        else if (0 == strcmp(tmp_str, "gentran_linear"))
            MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM_gentran_linear;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Ineighbor_alltoall will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_INEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_INEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_INEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_INEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ineighbor_alltoallv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear          - Force linear algorithm\
gentran_linear  - Force generic transport based linear algorithm");
    MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM_sched_linear;
        else if (0 == strcmp(tmp_str, "gentran_linear"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM_gentran_linear;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ineighbor_alltoallv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear          - Force linear algorithm\
gentran_linear  - Force generic transport based linear algorithm");
    MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM_sched_linear;
        else if (0 == strcmp(tmp_str, "gentran_linear"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM_gentran_linear;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Ineighbor_alltoallv will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_INEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_INEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_INEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_INEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ineighbor_alltoallw algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear          - Force linear algorithm\
gentran_linear        - Force generic transport based linear algorithm");
    MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM_sched_linear;
        else if (0 == strcmp(tmp_str, "gentran_linear"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM_gentran_linear;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ineighbor_alltoallw algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear          - Force linear algorithm\
gentran_linear        - Force generic transport based linear algorithm");
    MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM_sched_linear;
        else if (0 == strcmp(tmp_str, "gentran_linear"))
            MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM = MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM_gentran_linear;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_INEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_INEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Ineighbor_alltoallw will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_INEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_INEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_INEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_INEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE", &(MPIR_CVAR_INEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE");

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_TREE_KVAL, /* name */
        &MPIR_CVAR_IREDUCE_TREE_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for tree (kary, knomial, etc.) based ireduce");
    MPIR_CVAR_IREDUCE_TREE_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IREDUCE_TREE_KVAL", &(MPIR_CVAR_IREDUCE_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_TREE_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IREDUCE_TREE_KVAL", &(MPIR_CVAR_IREDUCE_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_TREE_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IREDUCE_TREE_KVAL", &(MPIR_CVAR_IREDUCE_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_TREE_KVAL");

    defaultval.str = (const char *) "kary";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_IREDUCE_TREE_TYPE, /* name */
        &MPIR_CVAR_IREDUCE_TREE_TYPE, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Tree type for tree based ireduce kary      - kary tree knomial_1 - knomial_1 tree knomial_2 - knomial_2 tree");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_IREDUCE_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_TREE_TYPE");
    rc = MPL_env2str("MPIR_PARAM_IREDUCE_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_TREE_TYPE");
    rc = MPL_env2str("MPIR_CVAR_IREDUCE_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_TREE_TYPE");
    if (tmp_str != NULL) {
        MPIR_CVAR_IREDUCE_TREE_TYPE = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_IREDUCE_TREE_TYPE);
        if (MPIR_CVAR_IREDUCE_TREE_TYPE == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_IREDUCE_TREE_TYPE");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_IREDUCE_TREE_TYPE = NULL;
    }

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_TREE_PIPELINE_CHUNK_SIZE, /* name */
        &MPIR_CVAR_IREDUCE_TREE_PIPELINE_CHUNK_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Maximum chunk size (in bytes) for pipelining in tree based ireduce. Default value is 0, that is, no pipelining by default");
    MPIR_CVAR_IREDUCE_TREE_PIPELINE_CHUNK_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_IREDUCE_TREE_PIPELINE_CHUNK_SIZE", &(MPIR_CVAR_IREDUCE_TREE_PIPELINE_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_TREE_PIPELINE_CHUNK_SIZE");
    rc = MPL_env2int("MPIR_PARAM_IREDUCE_TREE_PIPELINE_CHUNK_SIZE", &(MPIR_CVAR_IREDUCE_TREE_PIPELINE_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_TREE_PIPELINE_CHUNK_SIZE");
    rc = MPL_env2int("MPIR_CVAR_IREDUCE_TREE_PIPELINE_CHUNK_SIZE", &(MPIR_CVAR_IREDUCE_TREE_PIPELINE_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_TREE_PIPELINE_CHUNK_SIZE");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_RING_CHUNK_SIZE, /* name */
        &MPIR_CVAR_IREDUCE_RING_CHUNK_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Maximum chunk size (in bytes) for pipelining in ireduce ring algorithm. Default value is 0, that is, no pipelining by default");
    MPIR_CVAR_IREDUCE_RING_CHUNK_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_IREDUCE_RING_CHUNK_SIZE", &(MPIR_CVAR_IREDUCE_RING_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_RING_CHUNK_SIZE");
    rc = MPL_env2int("MPIR_PARAM_IREDUCE_RING_CHUNK_SIZE", &(MPIR_CVAR_IREDUCE_RING_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_RING_CHUNK_SIZE");
    rc = MPL_env2int("MPIR_CVAR_IREDUCE_RING_CHUNK_SIZE", &(MPIR_CVAR_IREDUCE_RING_CHUNK_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_RING_CHUNK_SIZE");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_TREE_BUFFER_PER_CHILD, /* name */
        &MPIR_CVAR_IREDUCE_TREE_BUFFER_PER_CHILD, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "If set to true, a rank in tree algorithms will allocate a dedicated buffer for every child it receives data from. This would mean more memory consumption but it would allow preposting of the receives and hence reduce the number of unexpected messages. If set to false, there is only one buffer that is used to receive the data from all the children. The receives are therefore serialized, that is, only one receive can be posted at a time.");
    MPIR_CVAR_IREDUCE_TREE_BUFFER_PER_CHILD = defaultval.d;
    rc = MPL_env2bool("MPICH_IREDUCE_TREE_BUFFER_PER_CHILD", &(MPIR_CVAR_IREDUCE_TREE_BUFFER_PER_CHILD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_TREE_BUFFER_PER_CHILD");
    rc = MPL_env2bool("MPIR_PARAM_IREDUCE_TREE_BUFFER_PER_CHILD", &(MPIR_CVAR_IREDUCE_TREE_BUFFER_PER_CHILD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_TREE_BUFFER_PER_CHILD");
    rc = MPL_env2bool("MPIR_CVAR_IREDUCE_TREE_BUFFER_PER_CHILD", &(MPIR_CVAR_IREDUCE_TREE_BUFFER_PER_CHILD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_TREE_BUFFER_PER_CHILD");

    defaultval.d = MPIR_CVAR_IREDUCE_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_IREDUCE_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ireduce algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_smp                   - Force smp algorithm\
sched_binomial              - Force binomial algorithm\
sched_reduce_scatter_gather - Force reduce scatter gather algorithm\
gentran_tree                - Force Generic Transport Tree\
gentran_ring                - Force Generic Transport Ring");
    MPIR_CVAR_IREDUCE_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IREDUCE_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IREDUCE_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IREDUCE_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_smp"))
            MPIR_CVAR_IREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_INTRA_ALGORITHM_sched_smp;
        else if (0 == strcmp(tmp_str, "sched_binomial"))
            MPIR_CVAR_IREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_INTRA_ALGORITHM_sched_binomial;
        else if (0 == strcmp(tmp_str, "sched_reduce_scatter_gather"))
            MPIR_CVAR_IREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_INTRA_ALGORITHM_sched_reduce_scatter_gather;
        else if (0 == strcmp(tmp_str, "gentran_tree"))
            MPIR_CVAR_IREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_INTRA_ALGORITHM_gentran_tree;
        else if (0 == strcmp(tmp_str, "gentran_ring"))
            MPIR_CVAR_IREDUCE_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_INTRA_ALGORITHM_gentran_ring;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IREDUCE_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_IREDUCE_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_IREDUCE_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ireduce algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_local_reduce_remote_send - Force local-reduce-remote-send algorithm");
    MPIR_CVAR_IREDUCE_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IREDUCE_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IREDUCE_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IREDUCE_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IREDUCE_INTER_ALGORITHM = MPIR_CVAR_IREDUCE_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IREDUCE_INTER_ALGORITHM = MPIR_CVAR_IREDUCE_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_local_reduce_remote_send"))
            MPIR_CVAR_IREDUCE_INTER_ALGORITHM = MPIR_CVAR_IREDUCE_INTER_ALGORITHM_sched_local_reduce_remote_send;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IREDUCE_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_IREDUCE_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Ireduce will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_IREDUCE_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_IREDUCE_DEVICE_COLLECTIVE", &(MPIR_CVAR_IREDUCE_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_IREDUCE_DEVICE_COLLECTIVE", &(MPIR_CVAR_IREDUCE_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_IREDUCE_DEVICE_COLLECTIVE", &(MPIR_CVAR_IREDUCE_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_DEVICE_COLLECTIVE");

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_SCATTER_RECEXCH_KVAL, /* name */
        &MPIR_CVAR_IREDUCE_SCATTER_RECEXCH_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for recursive exchange based ireduce_scatter");
    MPIR_CVAR_IREDUCE_SCATTER_RECEXCH_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IREDUCE_SCATTER_RECEXCH_KVAL", &(MPIR_CVAR_IREDUCE_SCATTER_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_SCATTER_RECEXCH_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IREDUCE_SCATTER_RECEXCH_KVAL", &(MPIR_CVAR_IREDUCE_SCATTER_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_SCATTER_RECEXCH_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IREDUCE_SCATTER_RECEXCH_KVAL", &(MPIR_CVAR_IREDUCE_SCATTER_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_SCATTER_RECEXCH_KVAL");

    defaultval.d = MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ireduce_scatter algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_noncommutative     - Force noncommutative algorithm\
sched_recursive_doubling - Force recursive doubling algorithm\
sched_pairwise           - Force pairwise algorithm\
sched_recursive_halving  - Force recursive halving algorithm\
gentran_recexch          - Force generic transport recursive exchange algorithm");
    MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IREDUCE_SCATTER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_SCATTER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IREDUCE_SCATTER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_SCATTER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_noncommutative"))
            MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM_sched_noncommutative;
        else if (0 == strcmp(tmp_str, "sched_recursive_doubling"))
            MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM_sched_recursive_doubling;
        else if (0 == strcmp(tmp_str, "sched_pairwise"))
            MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM_sched_pairwise;
        else if (0 == strcmp(tmp_str, "sched_recursive_halving"))
            MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM_sched_recursive_halving;
        else if (0 == strcmp(tmp_str, "gentran_recexch"))
            MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM_gentran_recexch;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ireduce_scatter algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_remote_reduce_local_scatterv - Force remote-reduce-local-scatterv algorithm");
    MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IREDUCE_SCATTER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_SCATTER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IREDUCE_SCATTER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_SCATTER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_remote_reduce_local_scatterv"))
            MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM_sched_remote_reduce_local_scatterv;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_SCATTER_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_IREDUCE_SCATTER_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Ireduce_scatter will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_IREDUCE_SCATTER_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_IREDUCE_SCATTER_DEVICE_COLLECTIVE", &(MPIR_CVAR_IREDUCE_SCATTER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_SCATTER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_IREDUCE_SCATTER_DEVICE_COLLECTIVE", &(MPIR_CVAR_IREDUCE_SCATTER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_SCATTER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_IREDUCE_SCATTER_DEVICE_COLLECTIVE", &(MPIR_CVAR_IREDUCE_SCATTER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_SCATTER_DEVICE_COLLECTIVE");

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_SCATTER_BLOCK_RECEXCH_KVAL, /* name */
        &MPIR_CVAR_IREDUCE_SCATTER_BLOCK_RECEXCH_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for recursive exchange based ireduce_scatter_block");
    MPIR_CVAR_IREDUCE_SCATTER_BLOCK_RECEXCH_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_IREDUCE_SCATTER_BLOCK_RECEXCH_KVAL", &(MPIR_CVAR_IREDUCE_SCATTER_BLOCK_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_SCATTER_BLOCK_RECEXCH_KVAL");
    rc = MPL_env2int("MPIR_PARAM_IREDUCE_SCATTER_BLOCK_RECEXCH_KVAL", &(MPIR_CVAR_IREDUCE_SCATTER_BLOCK_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_SCATTER_BLOCK_RECEXCH_KVAL");
    rc = MPL_env2int("MPIR_CVAR_IREDUCE_SCATTER_BLOCK_RECEXCH_KVAL", &(MPIR_CVAR_IREDUCE_SCATTER_BLOCK_RECEXCH_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_SCATTER_BLOCK_RECEXCH_KVAL");

    defaultval.d = MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ireduce_scatter_block algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_noncommutative     - Force noncommutative algorithm\
sched_recursive_doubling - Force recursive doubling algorithm\
sched_pairwise           - Force pairwise algorithm\
sched_recursive_halving  - Force recursive halving algorithm\
gentran_recexch          - Force generic transport recursive exchange algorithm");
    MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_noncommutative"))
            MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_sched_noncommutative;
        else if (0 == strcmp(tmp_str, "sched_recursive_doubling"))
            MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_sched_recursive_doubling;
        else if (0 == strcmp(tmp_str, "sched_pairwise"))
            MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_sched_pairwise;
        else if (0 == strcmp(tmp_str, "sched_recursive_halving"))
            MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_sched_recursive_halving;
        else if (0 == strcmp(tmp_str, "gentran_recexch"))
            MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_gentran_recexch;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ireduce_scatter_block algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_remote_reduce_local_scatterv - Force remote-reduce-local-scatterv algorithm");
    MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_remote_reduce_local_scatterv"))
            MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM = MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM_sched_remote_reduce_local_scatterv;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_IREDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_IREDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Ireduce_scatter_block will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_IREDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_IREDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE", &(MPIR_CVAR_IREDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_IREDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_IREDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE", &(MPIR_CVAR_IREDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_IREDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_IREDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE", &(MPIR_CVAR_IREDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_IREDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_ISCAN_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ISCAN_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_ISCAN_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select allgather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_smp                  - Force smp algorithm\
sched_recursive_doubling   - Force recursive doubling algorithm\
gentran_recursive_doubling - Force generic transport recursive doubling algorithm");
    MPIR_CVAR_ISCAN_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ISCAN_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ISCAN_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ISCAN_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ISCAN_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ISCAN_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ISCAN_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ISCAN_INTRA_ALGORITHM = MPIR_CVAR_ISCAN_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_ISCAN_INTRA_ALGORITHM = MPIR_CVAR_ISCAN_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_smp"))
            MPIR_CVAR_ISCAN_INTRA_ALGORITHM = MPIR_CVAR_ISCAN_INTRA_ALGORITHM_sched_smp;
        else if (0 == strcmp(tmp_str, "sched_recursive_doubling"))
            MPIR_CVAR_ISCAN_INTRA_ALGORITHM = MPIR_CVAR_ISCAN_INTRA_ALGORITHM_sched_recursive_doubling;
        else if (0 == strcmp(tmp_str, "gentran_recursive_doubling"))
            MPIR_CVAR_ISCAN_INTRA_ALGORITHM = MPIR_CVAR_ISCAN_INTRA_ALGORITHM_gentran_recursive_doubling;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ISCAN_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ISCAN_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_ISCAN_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Iscan will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_ISCAN_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_ISCAN_DEVICE_COLLECTIVE", &(MPIR_CVAR_ISCAN_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ISCAN_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_ISCAN_DEVICE_COLLECTIVE", &(MPIR_CVAR_ISCAN_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ISCAN_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_ISCAN_DEVICE_COLLECTIVE", &(MPIR_CVAR_ISCAN_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ISCAN_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_ISCATTER_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ISCATTER_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_ISCATTER_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select iscatter algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_binomial     - Force binomial algorithm\
gentran_tree       - Force genetric transport based tree algorithm");
    MPIR_CVAR_ISCATTER_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ISCATTER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ISCATTER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ISCATTER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ISCATTER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ISCATTER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ISCATTER_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ISCATTER_INTRA_ALGORITHM = MPIR_CVAR_ISCATTER_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_ISCATTER_INTRA_ALGORITHM = MPIR_CVAR_ISCATTER_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_binomial"))
            MPIR_CVAR_ISCATTER_INTRA_ALGORITHM = MPIR_CVAR_ISCATTER_INTRA_ALGORITHM_sched_binomial;
        else if (0 == strcmp(tmp_str, "gentran_tree"))
            MPIR_CVAR_ISCATTER_INTRA_ALGORITHM = MPIR_CVAR_ISCATTER_INTRA_ALGORITHM_gentran_tree;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ISCATTER_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ISCATTER_TREE_KVAL, /* name */
        &MPIR_CVAR_ISCATTER_TREE_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "k value for tree based iscatter");
    MPIR_CVAR_ISCATTER_TREE_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_ISCATTER_TREE_KVAL", &(MPIR_CVAR_ISCATTER_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ISCATTER_TREE_KVAL");
    rc = MPL_env2int("MPIR_PARAM_ISCATTER_TREE_KVAL", &(MPIR_CVAR_ISCATTER_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ISCATTER_TREE_KVAL");
    rc = MPL_env2int("MPIR_CVAR_ISCATTER_TREE_KVAL", &(MPIR_CVAR_ISCATTER_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ISCATTER_TREE_KVAL");

    defaultval.d = MPIR_CVAR_ISCATTER_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ISCATTER_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_ISCATTER_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select iscatter algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear                    - Force linear algorithm\
sched_remote_send_local_scatter - Force remote-send-local-scatter algorithm");
    MPIR_CVAR_ISCATTER_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ISCATTER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ISCATTER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ISCATTER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ISCATTER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ISCATTER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ISCATTER_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ISCATTER_INTER_ALGORITHM = MPIR_CVAR_ISCATTER_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_ISCATTER_INTER_ALGORITHM = MPIR_CVAR_ISCATTER_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_ISCATTER_INTER_ALGORITHM = MPIR_CVAR_ISCATTER_INTER_ALGORITHM_sched_linear;
        else if (0 == strcmp(tmp_str, "sched_remote_send_local_scatter"))
            MPIR_CVAR_ISCATTER_INTER_ALGORITHM = MPIR_CVAR_ISCATTER_INTER_ALGORITHM_sched_remote_send_local_scatter;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ISCATTER_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ISCATTER_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_ISCATTER_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Iscatter will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_ISCATTER_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_ISCATTER_DEVICE_COLLECTIVE", &(MPIR_CVAR_ISCATTER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ISCATTER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_ISCATTER_DEVICE_COLLECTIVE", &(MPIR_CVAR_ISCATTER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ISCATTER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_ISCATTER_DEVICE_COLLECTIVE", &(MPIR_CVAR_ISCATTER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ISCATTER_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select iscatterv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear    - Force linear algorithm\
gentran_linear  - Force generic transport based linear algorithm");
    MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ISCATTERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ISCATTERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ISCATTERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ISCATTERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM = MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM = MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM = MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM_sched_linear;
        else if (0 == strcmp(tmp_str, "gentran_linear"))
            MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM = MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM_gentran_linear;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_ISCATTERV_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ISCATTERV_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_ISCATTERV_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select iscatterv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
sched_auto - Internal algorithm selection for sched-based algorithms\
sched_linear - Force linear algorithm");
    MPIR_CVAR_ISCATTERV_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ISCATTERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ISCATTERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ISCATTERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ISCATTERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ISCATTERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ISCATTERV_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ISCATTERV_INTER_ALGORITHM = MPIR_CVAR_ISCATTERV_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "sched_auto"))
            MPIR_CVAR_ISCATTERV_INTER_ALGORITHM = MPIR_CVAR_ISCATTERV_INTER_ALGORITHM_sched_auto;
        else if (0 == strcmp(tmp_str, "sched_linear"))
            MPIR_CVAR_ISCATTERV_INTER_ALGORITHM = MPIR_CVAR_ISCATTERV_INTER_ALGORITHM_sched_linear;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ISCATTERV_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ISCATTERV_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_ISCATTERV_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Iscatterv will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_ISCATTERV_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_ISCATTERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_ISCATTERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ISCATTERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_ISCATTERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_ISCATTERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ISCATTERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_ISCATTERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_ISCATTERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ISCATTERV_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ineighbor_allgather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb   - Force nonblocking algorithm");
    MPIR_CVAR_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_NEIGHBOR_ALLGATHER_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLGATHER_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLGATHER_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select ineighbor_allgather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb   - Force nonblocking algorithm");
    MPIR_CVAR_NEIGHBOR_ALLGATHER_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_NEIGHBOR_ALLGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLGATHER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_NEIGHBOR_ALLGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLGATHER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_NEIGHBOR_ALLGATHER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLGATHER_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_NEIGHBOR_ALLGATHER_INTER_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLGATHER_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_NEIGHBOR_ALLGATHER_INTER_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLGATHER_INTER_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_NEIGHBOR_ALLGATHER_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Neighbor_allgather will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_NEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_NEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_NEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_NEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select neighbor_allgatherv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb   - Force nb algorithm");
    MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select neighbor_allgatherv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb   - Force nb algorithm");
    MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Neighbor_allgatherv will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_NEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_NEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_NEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_NEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select neighbor_alltoall algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb   - Force nb algorithm");
    MPIR_CVAR_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_NEIGHBOR_ALLTOALL_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLTOALL_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLTOALL_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select neighbor_alltoall algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb   - Force nb algorithm");
    MPIR_CVAR_NEIGHBOR_ALLTOALL_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_NEIGHBOR_ALLTOALL_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLTOALL_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_NEIGHBOR_ALLTOALL_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLTOALL_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_NEIGHBOR_ALLTOALL_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLTOALL_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_NEIGHBOR_ALLTOALL_INTER_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLTOALL_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_NEIGHBOR_ALLTOALL_INTER_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLTOALL_INTER_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_NEIGHBOR_ALLTOALL_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Neighbor_alltoall will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_NEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_NEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_NEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_NEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select neighbor_alltoallv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb   - Force nb algorithm");
    MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select neighbor_alltoallv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb   - Force nb algorithm");
    MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Neighbor_alltoallv will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_NEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_NEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_NEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_NEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select neighbor_alltoallw algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb   - Force nb algorithm");
    MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select neighbor_alltoallw algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb   - Force nb algorithm");
    MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM = MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_NEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Neighbor_alltoallw will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_NEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_NEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_NEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_NEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE", &(MPIR_CVAR_NEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE");

    defaultval.d = 2048;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_SHORT_MSG_SIZE, /* name */
        &MPIR_CVAR_REDUCE_SHORT_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "the short message algorithm will be used if the send buffer size is <= this value (in bytes)");
    MPIR_CVAR_REDUCE_SHORT_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_REDUCE_SHORT_MSG_SIZE", &(MPIR_CVAR_REDUCE_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_SHORT_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_REDUCE_SHORT_MSG_SIZE", &(MPIR_CVAR_REDUCE_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_SHORT_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_REDUCE_SHORT_MSG_SIZE", &(MPIR_CVAR_REDUCE_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_SHORT_MSG_SIZE");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_MAX_SMP_REDUCE_MSG_SIZE, /* name */
        &MPIR_CVAR_MAX_SMP_REDUCE_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Maximum message size for which SMP-aware reduce is used.  A value of '0' uses SMP-aware reduce for all message sizes.");
    MPIR_CVAR_MAX_SMP_REDUCE_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_MAX_SMP_REDUCE_MSG_SIZE", &(MPIR_CVAR_MAX_SMP_REDUCE_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_MAX_SMP_REDUCE_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_MAX_SMP_REDUCE_MSG_SIZE", &(MPIR_CVAR_MAX_SMP_REDUCE_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_MAX_SMP_REDUCE_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_MAX_SMP_REDUCE_MSG_SIZE", &(MPIR_CVAR_MAX_SMP_REDUCE_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_MAX_SMP_REDUCE_MSG_SIZE");

    defaultval.d = MPIR_CVAR_REDUCE_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_REDUCE_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select reduce algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
binomial              - Force binomial algorithm\
nb                    - Force nonblocking algorithm\
smp                   - Force smp algorithm\
reduce_scatter_gather - Force reduce scatter gather algorithm");
    MPIR_CVAR_REDUCE_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_REDUCE_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_REDUCE_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_REDUCE_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_REDUCE_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "binomial"))
            MPIR_CVAR_REDUCE_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_INTRA_ALGORITHM_binomial;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_REDUCE_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_INTRA_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "smp"))
            MPIR_CVAR_REDUCE_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_INTRA_ALGORITHM_smp;
        else if (0 == strcmp(tmp_str, "reduce_scatter_gather"))
            MPIR_CVAR_REDUCE_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_INTRA_ALGORITHM_reduce_scatter_gather;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_REDUCE_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_REDUCE_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_REDUCE_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select reduce algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
local_reduce_remote_send - Force local-reduce-remote-send algorithm\
nb                       - Force nonblocking algorithm");
    MPIR_CVAR_REDUCE_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_REDUCE_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_REDUCE_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_REDUCE_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_REDUCE_INTER_ALGORITHM = MPIR_CVAR_REDUCE_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "local_reduce_remote_send"))
            MPIR_CVAR_REDUCE_INTER_ALGORITHM = MPIR_CVAR_REDUCE_INTER_ALGORITHM_local_reduce_remote_send;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_REDUCE_INTER_ALGORITHM = MPIR_CVAR_REDUCE_INTER_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_REDUCE_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_REDUCE_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Reduce will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_REDUCE_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_REDUCE_DEVICE_COLLECTIVE", &(MPIR_CVAR_REDUCE_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_REDUCE_DEVICE_COLLECTIVE", &(MPIR_CVAR_REDUCE_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_REDUCE_DEVICE_COLLECTIVE", &(MPIR_CVAR_REDUCE_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_DEVICE_COLLECTIVE");

    defaultval.d = 524288;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE, /* name */
        &MPIR_CVAR_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "the long message algorithm will be used if the operation is commutative and the send buffer size is >= this value (in bytes)");
    MPIR_CVAR_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE", &(MPIR_CVAR_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE", &(MPIR_CVAR_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE", &(MPIR_CVAR_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE");

    defaultval.d = MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select reduce_scatter algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb                 - Force nonblocking algorithm\
noncommutative     - Force noncommutative algorithm\
pairwise           - Force pairwise algorithm\
recursive_doubling - Force recursive doubling algorithm\
recursive_halving  - Force recursive halving algorithm");
    MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_REDUCE_SCATTER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_SCATTER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_REDUCE_SCATTER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_SCATTER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "noncommutative"))
            MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM_noncommutative;
        else if (0 == strcmp(tmp_str, "pairwise"))
            MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM_pairwise;
        else if (0 == strcmp(tmp_str, "recursive_doubling"))
            MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM_recursive_doubling;
        else if (0 == strcmp(tmp_str, "recursive_halving"))
            MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM_recursive_halving;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select reduce_scatter algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb                          - Force nonblocking algorithm\
remote_reduce_local_scatter - Force remote-reduce-local-scatter algorithm");
    MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_REDUCE_SCATTER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_SCATTER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_REDUCE_SCATTER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_SCATTER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "remote_reduce_local_scatter"))
            MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM_remote_reduce_local_scatter;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_SCATTER_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_REDUCE_SCATTER_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Reduce_scatter will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_REDUCE_SCATTER_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_REDUCE_SCATTER_DEVICE_COLLECTIVE", &(MPIR_CVAR_REDUCE_SCATTER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_SCATTER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_REDUCE_SCATTER_DEVICE_COLLECTIVE", &(MPIR_CVAR_REDUCE_SCATTER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_SCATTER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_REDUCE_SCATTER_DEVICE_COLLECTIVE", &(MPIR_CVAR_REDUCE_SCATTER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_SCATTER_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select reduce_scatter_block algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
noncommutative     - Force noncommutative algorithm\
recursive_doubling - Force recursive doubling algorithm\
pairwise           - Force pairwise algorithm\
recursive_halving  - Force recursive halving algorithm\
nb                 - Force nonblocking algorithm");
    MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "noncommutative"))
            MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_noncommutative;
        else if (0 == strcmp(tmp_str, "recursive_doubling"))
            MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_recursive_doubling;
        else if (0 == strcmp(tmp_str, "pairwise"))
            MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_pairwise;
        else if (0 == strcmp(tmp_str, "recursive_halving"))
            MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_recursive_halving;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select reduce_scatter_block algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb                          - Force nonblocking algorithm\
remote_reduce_local_scatter - Force remote-reduce-local-scatter algorithm");
    MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "remote_reduce_local_scatter"))
            MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM = MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM_remote_reduce_local_scatter;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_REDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Reduce_scatter_block will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_REDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_REDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE", &(MPIR_CVAR_REDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_REDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE", &(MPIR_CVAR_REDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_REDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE", &(MPIR_CVAR_REDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_SCAN_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_SCAN_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_SCAN_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select allgather algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
nb                 - Force nonblocking algorithm\
smp                - Force smp algorithm\
recursive_doubling - Force recursive doubling algorithm");
    MPIR_CVAR_SCAN_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_SCAN_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_SCAN_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_SCAN_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_SCAN_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_SCAN_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_SCAN_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_SCAN_INTRA_ALGORITHM = MPIR_CVAR_SCAN_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_SCAN_INTRA_ALGORITHM = MPIR_CVAR_SCAN_INTRA_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "smp"))
            MPIR_CVAR_SCAN_INTRA_ALGORITHM = MPIR_CVAR_SCAN_INTRA_ALGORITHM_smp;
        else if (0 == strcmp(tmp_str, "recursive_doubling"))
            MPIR_CVAR_SCAN_INTRA_ALGORITHM = MPIR_CVAR_SCAN_INTRA_ALGORITHM_recursive_doubling;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_SCAN_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_SCAN_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_SCAN_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Scan will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_SCAN_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_SCAN_DEVICE_COLLECTIVE", &(MPIR_CVAR_SCAN_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_SCAN_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_SCAN_DEVICE_COLLECTIVE", &(MPIR_CVAR_SCAN_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_SCAN_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_SCAN_DEVICE_COLLECTIVE", &(MPIR_CVAR_SCAN_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_SCAN_DEVICE_COLLECTIVE");

    defaultval.d = 2048;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_SCATTER_INTER_SHORT_MSG_SIZE, /* name */
        &MPIR_CVAR_SCATTER_INTER_SHORT_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "use the short message algorithm for intercommunicator MPI_Scatter if the send buffer size is < this value (in bytes)");
    MPIR_CVAR_SCATTER_INTER_SHORT_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_SCATTER_INTER_SHORT_MSG_SIZE", &(MPIR_CVAR_SCATTER_INTER_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_SCATTER_INTER_SHORT_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_SCATTER_INTER_SHORT_MSG_SIZE", &(MPIR_CVAR_SCATTER_INTER_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_SCATTER_INTER_SHORT_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_SCATTER_INTER_SHORT_MSG_SIZE", &(MPIR_CVAR_SCATTER_INTER_SHORT_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_SCATTER_INTER_SHORT_MSG_SIZE");

    defaultval.d = MPIR_CVAR_SCATTER_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_SCATTER_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_SCATTER_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select scatter algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
binomial - Force binomial algorithm\
nb       - Force nonblocking algorithm");
    MPIR_CVAR_SCATTER_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_SCATTER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_SCATTER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_SCATTER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_SCATTER_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_SCATTER_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_SCATTER_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_SCATTER_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "binomial"))
            MPIR_CVAR_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_SCATTER_INTRA_ALGORITHM_binomial;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_SCATTER_INTRA_ALGORITHM = MPIR_CVAR_SCATTER_INTRA_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_SCATTER_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_SCATTER_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_SCATTER_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_SCATTER_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select scatter algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
linear                    - Force linear algorithm\
nb                        - Force nonblocking algorithm\
remote_send_local_scatter - Force remote-send-local-scatter algorithm");
    MPIR_CVAR_SCATTER_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_SCATTER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_SCATTER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_SCATTER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_SCATTER_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_SCATTER_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_SCATTER_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_SCATTER_INTER_ALGORITHM = MPIR_CVAR_SCATTER_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "linear"))
            MPIR_CVAR_SCATTER_INTER_ALGORITHM = MPIR_CVAR_SCATTER_INTER_ALGORITHM_linear;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_SCATTER_INTER_ALGORITHM = MPIR_CVAR_SCATTER_INTER_ALGORITHM_nb;
        else if (0 == strcmp(tmp_str, "remote_send_local_scatter"))
            MPIR_CVAR_SCATTER_INTER_ALGORITHM = MPIR_CVAR_SCATTER_INTER_ALGORITHM_remote_send_local_scatter;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_SCATTER_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_SCATTER_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_SCATTER_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Scatter will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_SCATTER_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_SCATTER_DEVICE_COLLECTIVE", &(MPIR_CVAR_SCATTER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_SCATTER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_SCATTER_DEVICE_COLLECTIVE", &(MPIR_CVAR_SCATTER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_SCATTER_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_SCATTER_DEVICE_COLLECTIVE", &(MPIR_CVAR_SCATTER_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_SCATTER_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_SCATTERV_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_SCATTERV_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_SCATTERV_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select scatterv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
linear - Force linear algorithm\
nb     - Force nonblocking algorithm");
    MPIR_CVAR_SCATTERV_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_SCATTERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_SCATTERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_SCATTERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_SCATTERV_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_SCATTERV_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_SCATTERV_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_SCATTERV_INTRA_ALGORITHM = MPIR_CVAR_SCATTERV_INTRA_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "linear"))
            MPIR_CVAR_SCATTERV_INTRA_ALGORITHM = MPIR_CVAR_SCATTERV_INTRA_ALGORITHM_linear;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_SCATTERV_INTRA_ALGORITHM = MPIR_CVAR_SCATTERV_INTRA_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_SCATTERV_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_SCATTERV_INTER_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_SCATTERV_INTER_ALGORITHM, /* name */
        &MPIR_CVAR_SCATTERV_INTER_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select scatterv algorithm\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE)\
linear - Force linear algorithm\
nb     - Force nonblocking algorithm");
    MPIR_CVAR_SCATTERV_INTER_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_SCATTERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_SCATTERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_SCATTERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_SCATTERV_INTER_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_SCATTERV_INTER_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_SCATTERV_INTER_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_SCATTERV_INTER_ALGORITHM = MPIR_CVAR_SCATTERV_INTER_ALGORITHM_auto;
        else if (0 == strcmp(tmp_str, "linear"))
            MPIR_CVAR_SCATTERV_INTER_ALGORITHM = MPIR_CVAR_SCATTERV_INTER_ALGORITHM_linear;
        else if (0 == strcmp(tmp_str, "nb"))
            MPIR_CVAR_SCATTERV_INTER_ALGORITHM = MPIR_CVAR_SCATTERV_INTER_ALGORITHM_nb;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_SCATTERV_INTER_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_SCATTERV_DEVICE_COLLECTIVE, /* name */
        &MPIR_CVAR_SCATTERV_DEVICE_COLLECTIVE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "This CVAR is only used when MPIR_CVAR_DEVICE_COLLECTIVES is set to \"percoll\".  If set to true, MPI_Scatterv will allow the device to override the MPIR-level collective algorithms.  The device might still call the MPIR-level algorithms manually.  If set to false, the device-override will be disabled.");
    MPIR_CVAR_SCATTERV_DEVICE_COLLECTIVE = defaultval.d;
    rc = MPL_env2bool("MPICH_SCATTERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_SCATTERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_SCATTERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_PARAM_SCATTERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_SCATTERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_SCATTERV_DEVICE_COLLECTIVE");
    rc = MPL_env2bool("MPIR_CVAR_SCATTERV_DEVICE_COLLECTIVE", &(MPIR_CVAR_SCATTERV_DEVICE_COLLECTIVE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_SCATTERV_DEVICE_COLLECTIVE");

    defaultval.d = MPIR_CVAR_DEVICE_COLLECTIVES_percoll;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_DEVICE_COLLECTIVES, /* name */
        &MPIR_CVAR_DEVICE_COLLECTIVES, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select whether the device can override the\
MPIR-level collective algorithms.\
all     - Always prefer the device collectives\
none    - Never pick the device collectives\
percoll - Use the per-collective CVARs to decide");
    MPIR_CVAR_DEVICE_COLLECTIVES = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_DEVICE_COLLECTIVES", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_DEVICE_COLLECTIVES");
    rc = MPL_env2str("MPIR_PARAM_DEVICE_COLLECTIVES", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_DEVICE_COLLECTIVES");
    rc = MPL_env2str("MPIR_CVAR_DEVICE_COLLECTIVES", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_DEVICE_COLLECTIVES");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "all"))
            MPIR_CVAR_DEVICE_COLLECTIVES = MPIR_CVAR_DEVICE_COLLECTIVES_all;
        else if (0 == strcmp(tmp_str, "none"))
            MPIR_CVAR_DEVICE_COLLECTIVES = MPIR_CVAR_DEVICE_COLLECTIVES_none;
        else if (0 == strcmp(tmp_str, "percoll"))
            MPIR_CVAR_DEVICE_COLLECTIVES = MPIR_CVAR_DEVICE_COLLECTIVES_percoll;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_DEVICE_COLLECTIVES", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_COLLECTIVE_FALLBACK_silent;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_COLLECTIVE_FALLBACK, /* name */
        &MPIR_CVAR_COLLECTIVE_FALLBACK, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to control what the MPI library should do if the\
user-specified collective algorithm does not work for the\
arguments passed in by the user.\
error   - throw an error\
print   - print an error message and fallback to the internally selected algorithm\
silent  - silently fallback to the internally selected algorithm");
    MPIR_CVAR_COLLECTIVE_FALLBACK = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_COLLECTIVE_FALLBACK", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_COLLECTIVE_FALLBACK");
    rc = MPL_env2str("MPIR_PARAM_COLLECTIVE_FALLBACK", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_COLLECTIVE_FALLBACK");
    rc = MPL_env2str("MPIR_CVAR_COLLECTIVE_FALLBACK", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_COLLECTIVE_FALLBACK");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "error"))
            MPIR_CVAR_COLLECTIVE_FALLBACK = MPIR_CVAR_COLLECTIVE_FALLBACK_error;
        else if (0 == strcmp(tmp_str, "print"))
            MPIR_CVAR_COLLECTIVE_FALLBACK = MPIR_CVAR_COLLECTIVE_FALLBACK_print;
        else if (0 == strcmp(tmp_str, "silent"))
            MPIR_CVAR_COLLECTIVE_FALLBACK = MPIR_CVAR_COLLECTIVE_FALLBACK_silent;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_COLLECTIVE_FALLBACK", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.str = (const char *) "";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE, /* name */
        &MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Defines the location of tuning file.");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_COLL_SELECTION_TUNING_JSON_FILE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_COLL_SELECTION_TUNING_JSON_FILE");
    rc = MPL_env2str("MPIR_PARAM_COLL_SELECTION_TUNING_JSON_FILE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_COLL_SELECTION_TUNING_JSON_FILE");
    rc = MPL_env2str("MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE");
    if (tmp_str != NULL) {
        MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE);
        if (MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE = NULL;
    }

    defaultval.d = 8;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_PROGRESS_MAX_COLLS, /* name */
        &MPIR_CVAR_PROGRESS_MAX_COLLS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Maximum number of collective operations at a time that the progress engine should make progress on");
    MPIR_CVAR_PROGRESS_MAX_COLLS = defaultval.d;
    rc = MPL_env2int("MPICH_PROGRESS_MAX_COLLS", &(MPIR_CVAR_PROGRESS_MAX_COLLS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_PROGRESS_MAX_COLLS");
    rc = MPL_env2int("MPIR_PARAM_PROGRESS_MAX_COLLS", &(MPIR_CVAR_PROGRESS_MAX_COLLS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_PROGRESS_MAX_COLLS");
    rc = MPL_env2int("MPIR_CVAR_PROGRESS_MAX_COLLS", &(MPIR_CVAR_PROGRESS_MAX_COLLS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_PROGRESS_MAX_COLLS");

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_COMM_SPLIT_USE_QSORT, /* name */
        &MPIR_CVAR_COMM_SPLIT_USE_QSORT, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COMMUNICATOR", /* category */
        "Use qsort(3) in the implementation of MPI_Comm_split instead of bubble sort.");
    MPIR_CVAR_COMM_SPLIT_USE_QSORT = defaultval.d;
    rc = MPL_env2bool("MPICH_COMM_SPLIT_USE_QSORT", &(MPIR_CVAR_COMM_SPLIT_USE_QSORT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_COMM_SPLIT_USE_QSORT");
    rc = MPL_env2bool("MPIR_PARAM_COMM_SPLIT_USE_QSORT", &(MPIR_CVAR_COMM_SPLIT_USE_QSORT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_COMM_SPLIT_USE_QSORT");
    rc = MPL_env2bool("MPIR_CVAR_COMM_SPLIT_USE_QSORT", &(MPIR_CVAR_COMM_SPLIT_USE_QSORT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_COMM_SPLIT_USE_QSORT");

    defaultval.d = 2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CTXID_EAGER_SIZE, /* name */
        &MPIR_CVAR_CTXID_EAGER_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "THREADS", /* category */
        "The MPIR_CVAR_CTXID_EAGER_SIZE environment variable allows you to specify how many words in the context ID mask will be set aside for the eager allocation protocol.  If the application is running out of context IDs, reducing this value may help.");
    MPIR_CVAR_CTXID_EAGER_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_CTXID_EAGER_SIZE", &(MPIR_CVAR_CTXID_EAGER_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CTXID_EAGER_SIZE");
    rc = MPL_env2int("MPIR_PARAM_CTXID_EAGER_SIZE", &(MPIR_CVAR_CTXID_EAGER_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CTXID_EAGER_SIZE");
    rc = MPL_env2int("MPIR_CVAR_CTXID_EAGER_SIZE", &(MPIR_CVAR_CTXID_EAGER_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CTXID_EAGER_SIZE");

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_DATALOOP_FAST_SEEK, /* name */
        &MPIR_CVAR_DATALOOP_FAST_SEEK, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "DATALOOP", /* category */
        "use a datatype-specialized algorithm to shortcut seeking to the correct location in a noncontiguous buffer");
    MPIR_CVAR_DATALOOP_FAST_SEEK = defaultval.d;
    rc = MPL_env2int("MPICH_DATALOOP_FAST_SEEK", &(MPIR_CVAR_DATALOOP_FAST_SEEK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_DATALOOP_FAST_SEEK");
    rc = MPL_env2int("MPIR_PARAM_DATALOOP_FAST_SEEK", &(MPIR_CVAR_DATALOOP_FAST_SEEK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_DATALOOP_FAST_SEEK");
    rc = MPL_env2int("MPIR_CVAR_DATALOOP_FAST_SEEK", &(MPIR_CVAR_DATALOOP_FAST_SEEK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_DATALOOP_FAST_SEEK");

    defaultval.d = 64;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_PROCTABLE_SIZE, /* name */
        &MPIR_CVAR_PROCTABLE_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "DEBUGGER", /* category */
        "Size of the \"MPIR\" debugger interface proctable (process table).");
    MPIR_CVAR_PROCTABLE_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_PROCTABLE_SIZE", &(MPIR_CVAR_PROCTABLE_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_PROCTABLE_SIZE");
    rc = MPL_env2int("MPIR_PARAM_PROCTABLE_SIZE", &(MPIR_CVAR_PROCTABLE_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_PROCTABLE_SIZE");
    rc = MPL_env2int("MPIR_CVAR_PROCTABLE_SIZE", &(MPIR_CVAR_PROCTABLE_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_PROCTABLE_SIZE");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_PROCTABLE_PRINT, /* name */
        &MPIR_CVAR_PROCTABLE_PRINT, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "DEBUGGER", /* category */
        "If true, dump the proctable entries at MPII_Wait_for_debugger-time.");
    MPIR_CVAR_PROCTABLE_PRINT = defaultval.d;
    rc = MPL_env2bool("MPICH_PROCTABLE_PRINT", &(MPIR_CVAR_PROCTABLE_PRINT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_PROCTABLE_PRINT");
    rc = MPL_env2bool("MPIR_PARAM_PROCTABLE_PRINT", &(MPIR_CVAR_PROCTABLE_PRINT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_PROCTABLE_PRINT");
    rc = MPL_env2bool("MPIR_CVAR_PROCTABLE_PRINT", &(MPIR_CVAR_PROCTABLE_PRINT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_PROCTABLE_PRINT");

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_PRINT_ERROR_STACK, /* name */
        &MPIR_CVAR_PRINT_ERROR_STACK, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "ERROR_HANDLING", /* category */
        "If true, print an error stack trace at error handling time.");
    MPIR_CVAR_PRINT_ERROR_STACK = defaultval.d;
    rc = MPL_env2bool("MPICH_PRINT_ERROR_STACK", &(MPIR_CVAR_PRINT_ERROR_STACK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_PRINT_ERROR_STACK");
    rc = MPL_env2bool("MPIR_PARAM_PRINT_ERROR_STACK", &(MPIR_CVAR_PRINT_ERROR_STACK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_PRINT_ERROR_STACK");
    rc = MPL_env2bool("MPIR_CVAR_PRINT_ERROR_STACK", &(MPIR_CVAR_PRINT_ERROR_STACK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_PRINT_ERROR_STACK");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CHOP_ERROR_STACK, /* name */
        &MPIR_CVAR_CHOP_ERROR_STACK, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "ERROR_HANDLING", /* category */
        "If >0, truncate error stack output lines this many characters wide.  If 0, do not truncate, and if <0 use a sensible default.");
    MPIR_CVAR_CHOP_ERROR_STACK = defaultval.d;
    rc = MPL_env2int("MPICH_CHOP_ERROR_STACK", &(MPIR_CVAR_CHOP_ERROR_STACK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CHOP_ERROR_STACK");
    rc = MPL_env2int("MPIR_PARAM_CHOP_ERROR_STACK", &(MPIR_CVAR_CHOP_ERROR_STACK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CHOP_ERROR_STACK");
    rc = MPL_env2int("MPIR_CVAR_CHOP_ERROR_STACK", &(MPIR_CVAR_CHOP_ERROR_STACK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CHOP_ERROR_STACK");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_SUPPRESS_ABORT_MESSAGE, /* name */
        &MPIR_CVAR_SUPPRESS_ABORT_MESSAGE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "ERROR_HANDLING", /* category */
        "Disable printing of abort error message.");
    MPIR_CVAR_SUPPRESS_ABORT_MESSAGE = defaultval.d;
    rc = MPL_env2bool("MPICH_SUPPRESS_ABORT_MESSAGE", &(MPIR_CVAR_SUPPRESS_ABORT_MESSAGE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_SUPPRESS_ABORT_MESSAGE");
    rc = MPL_env2bool("MPIR_PARAM_SUPPRESS_ABORT_MESSAGE", &(MPIR_CVAR_SUPPRESS_ABORT_MESSAGE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_SUPPRESS_ABORT_MESSAGE");
    rc = MPL_env2bool("MPIR_CVAR_SUPPRESS_ABORT_MESSAGE", &(MPIR_CVAR_SUPPRESS_ABORT_MESSAGE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_SUPPRESS_ABORT_MESSAGE");

    defaultval.str = (const char *) "MPI_THREAD_SINGLE";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_DEFAULT_THREAD_LEVEL, /* name */
        &MPIR_CVAR_DEFAULT_THREAD_LEVEL, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "THREADS", /* category */
        "Sets the default thread level to use when using MPI_INIT. This variable is case-insensitive.");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_DEFAULT_THREAD_LEVEL", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_DEFAULT_THREAD_LEVEL");
    rc = MPL_env2str("MPIR_PARAM_DEFAULT_THREAD_LEVEL", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_DEFAULT_THREAD_LEVEL");
    rc = MPL_env2str("MPIR_CVAR_DEFAULT_THREAD_LEVEL", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_DEFAULT_THREAD_LEVEL");
    if (tmp_str != NULL) {
        MPIR_CVAR_DEFAULT_THREAD_LEVEL = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_DEFAULT_THREAD_LEVEL);
        if (MPIR_CVAR_DEFAULT_THREAD_LEVEL == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_DEFAULT_THREAD_LEVEL");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_DEFAULT_THREAD_LEVEL = NULL;
    }

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ASYNC_PROGRESS, /* name */
        &MPIR_CVAR_ASYNC_PROGRESS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "THREADS", /* category */
        "If set to true, MPICH will initiate an additional thread to make asynchronous progress on all communication operations including point-to-point, collective, one-sided operations and I/O.  Setting this variable will automatically increase the thread-safety level to MPI_THREAD_MULTIPLE.  While this improves the progress semantics, it might cause a small amount of performance overhead for regular MPI operations.  The user is encouraged to leave one or more hardware threads vacant in order to prevent contention between the application threads and the progress thread(s).  The impact of oversubscription is highly system dependent but may be substantial in some cases, hence this recommendation.");
    MPIR_CVAR_ASYNC_PROGRESS = defaultval.d;
    rc = MPL_env2bool("MPICH_ASYNC_PROGRESS", &(MPIR_CVAR_ASYNC_PROGRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ASYNC_PROGRESS");
    rc = MPL_env2bool("MPIR_PARAM_ASYNC_PROGRESS", &(MPIR_CVAR_ASYNC_PROGRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ASYNC_PROGRESS");
    rc = MPL_env2bool("MPIR_CVAR_ASYNC_PROGRESS", &(MPIR_CVAR_ASYNC_PROGRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ASYNC_PROGRESS");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_DEBUG_HOLD, /* name */
        &MPIR_CVAR_DEBUG_HOLD, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "DEBUGGER", /* category */
        "If true, causes processes to wait in MPI_Init and MPI_Initthread for a debugger to be attached.  Once the debugger has attached, the variable 'hold' should be set to 0 in order to allow the process to continue (e.g., in gdb, \"set hold=0\").");
    MPIR_CVAR_DEBUG_HOLD = defaultval.d;
    rc = MPL_env2bool("MPICH_DEBUG_HOLD", &(MPIR_CVAR_DEBUG_HOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_DEBUG_HOLD");
    rc = MPL_env2bool("MPIR_PARAM_DEBUG_HOLD", &(MPIR_CVAR_DEBUG_HOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_DEBUG_HOLD");
    rc = MPL_env2bool("MPIR_CVAR_DEBUG_HOLD", &(MPIR_CVAR_DEBUG_HOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_DEBUG_HOLD");

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ERROR_CHECKING, /* name */
        &MPIR_CVAR_ERROR_CHECKING, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "ERROR_HANDLING", /* category */
        "If true, perform checks for errors, typically to verify valid inputs to MPI routines.  Only effective when MPICH is configured with --enable-error-checking=runtime .");
    MPIR_CVAR_ERROR_CHECKING = defaultval.d;
    rc = MPL_env2bool("MPICH_ERROR_CHECKING", &(MPIR_CVAR_ERROR_CHECKING));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ERROR_CHECKING");
    rc = MPL_env2bool("MPIR_PARAM_ERROR_CHECKING", &(MPIR_CVAR_ERROR_CHECKING));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ERROR_CHECKING");
    rc = MPL_env2bool("MPIR_CVAR_ERROR_CHECKING", &(MPIR_CVAR_ERROR_CHECKING));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ERROR_CHECKING");

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_MEMDUMP, /* name */
        &MPIR_CVAR_MEMDUMP, /* address */
        1, /* count */
        MPI_T_VERBOSITY_MPIDEV_DETAIL,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "DEVELOPER", /* category */
        "If true, list any memory that was allocated by MPICH and that remains allocated when MPI_Finalize completes.");
    MPIR_CVAR_MEMDUMP = defaultval.d;
    rc = MPL_env2bool("MPICH_MEMDUMP", &(MPIR_CVAR_MEMDUMP));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_MEMDUMP");
    rc = MPL_env2bool("MPIR_PARAM_MEMDUMP", &(MPIR_CVAR_MEMDUMP));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_MEMDUMP");
    rc = MPL_env2bool("MPIR_CVAR_MEMDUMP", &(MPIR_CVAR_MEMDUMP));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_MEMDUMP");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_MEM_CATEGORY_INFORMATION, /* name */
        &MPIR_CVAR_MEM_CATEGORY_INFORMATION, /* address */
        1, /* count */
        MPI_T_VERBOSITY_MPIDEV_DETAIL,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "DEVELOPER", /* category */
        "If true, print a summary of memory allocation by category. The category definitions are found in mpl_trmem.h.");
    MPIR_CVAR_MEM_CATEGORY_INFORMATION = defaultval.d;
    rc = MPL_env2bool("MPICH_MEM_CATEGORY_INFORMATION", &(MPIR_CVAR_MEM_CATEGORY_INFORMATION));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_MEM_CATEGORY_INFORMATION");
    rc = MPL_env2bool("MPIR_PARAM_MEM_CATEGORY_INFORMATION", &(MPIR_CVAR_MEM_CATEGORY_INFORMATION));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_MEM_CATEGORY_INFORMATION");
    rc = MPL_env2bool("MPIR_CVAR_MEM_CATEGORY_INFORMATION", &(MPIR_CVAR_MEM_CATEGORY_INFORMATION));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_MEM_CATEGORY_INFORMATION");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_DIMS_VERBOSE, /* name */
        &MPIR_CVAR_DIMS_VERBOSE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_MPIDEV_DETAIL,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "DIMS", /* category */
        "If true, enable verbose output about the actions of the implementation of MPI_Dims_create.");
    MPIR_CVAR_DIMS_VERBOSE = defaultval.d;
    rc = MPL_env2bool("MPICH_DIMS_VERBOSE", &(MPIR_CVAR_DIMS_VERBOSE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_DIMS_VERBOSE");
    rc = MPL_env2bool("MPIR_PARAM_DIMS_VERBOSE", &(MPIR_CVAR_DIMS_VERBOSE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_DIMS_VERBOSE");
    rc = MPL_env2bool("MPIR_CVAR_DIMS_VERBOSE", &(MPIR_CVAR_DIMS_VERBOSE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_DIMS_VERBOSE");

    defaultval.str = (const char *) NULL;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_NAMESERV_FILE_PUBDIR, /* name */
        &MPIR_CVAR_NAMESERV_FILE_PUBDIR, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "PROCESS_MANAGER", /* category */
        "Sets the directory to use for MPI service publishing in the file nameserv implementation.  Allows the user to override where the publish and lookup information is placed for connect/accept based applications.");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_NAMEPUB_DIR", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NAMEPUB_DIR");
    rc = MPL_env2str("MPIR_PARAM_NAMEPUB_DIR", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NAMEPUB_DIR");
    rc = MPL_env2str("MPIR_CVAR_NAMEPUB_DIR", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NAMEPUB_DIR");
    rc = MPL_env2str("MPICH_NAMESERV_FILE_PUBDIR", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NAMESERV_FILE_PUBDIR");
    rc = MPL_env2str("MPIR_PARAM_NAMESERV_FILE_PUBDIR", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NAMESERV_FILE_PUBDIR");
    rc = MPL_env2str("MPIR_CVAR_NAMESERV_FILE_PUBDIR", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NAMESERV_FILE_PUBDIR");
    if (tmp_str != NULL) {
        MPIR_CVAR_NAMESERV_FILE_PUBDIR = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_NAMESERV_FILE_PUBDIR);
        if (MPIR_CVAR_NAMESERV_FILE_PUBDIR == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_NAMESERV_FILE_PUBDIR");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_NAMESERV_FILE_PUBDIR = NULL;
    }

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ABORT_ON_LEAKED_HANDLES, /* name */
        &MPIR_CVAR_ABORT_ON_LEAKED_HANDLES, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "MEMORY", /* category */
        "If true, MPI will call MPI_Abort at MPI_Finalize if any MPI object handles have been leaked.  For example, if MPI_Comm_dup is called without calling a corresponding MPI_Comm_free.  For uninteresting reasons, enabling this option may prevent all known object leaks from being reported.  MPICH must have been configure with \"--enable-g=handlealloc\" or better in order for this functionality to work.");
    MPIR_CVAR_ABORT_ON_LEAKED_HANDLES = defaultval.d;
    rc = MPL_env2bool("MPICH_ABORT_ON_LEAKED_HANDLES", &(MPIR_CVAR_ABORT_ON_LEAKED_HANDLES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ABORT_ON_LEAKED_HANDLES");
    rc = MPL_env2bool("MPIR_PARAM_ABORT_ON_LEAKED_HANDLES", &(MPIR_CVAR_ABORT_ON_LEAKED_HANDLES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ABORT_ON_LEAKED_HANDLES");
    rc = MPL_env2bool("MPIR_CVAR_ABORT_ON_LEAKED_HANDLES", &(MPIR_CVAR_ABORT_ON_LEAKED_HANDLES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ABORT_ON_LEAKED_HANDLES");

    defaultval.str = (const char *) "auto";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_NETLOC_NODE_FILE, /* name */
        &MPIR_CVAR_NETLOC_NODE_FILE, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "DEBUGGER", /* category */
        "Subnet json file");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_NETLOC_NODE_FILE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NETLOC_NODE_FILE");
    rc = MPL_env2str("MPIR_PARAM_NETLOC_NODE_FILE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NETLOC_NODE_FILE");
    rc = MPL_env2str("MPIR_CVAR_NETLOC_NODE_FILE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NETLOC_NODE_FILE");
    if (tmp_str != NULL) {
        MPIR_CVAR_NETLOC_NODE_FILE = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_NETLOC_NODE_FILE);
        if (MPIR_CVAR_NETLOC_NODE_FILE == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_NETLOC_NODE_FILE");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_NETLOC_NODE_FILE = NULL;
    }

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NOLOCAL, /* name */
        &MPIR_CVAR_NOLOCAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "NODEMAP", /* category */
        "If true, force all processes to operate as though all processes are located on another node.  For example, this disables shared memory communication hierarchical collectives.");
    MPIR_CVAR_NOLOCAL = defaultval.d;
    rc = MPL_env2bool("MPICH_NO_LOCAL", &(MPIR_CVAR_NOLOCAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NO_LOCAL");
    rc = MPL_env2bool("MPIR_PARAM_NO_LOCAL", &(MPIR_CVAR_NOLOCAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NO_LOCAL");
    rc = MPL_env2bool("MPIR_CVAR_NO_LOCAL", &(MPIR_CVAR_NOLOCAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NO_LOCAL");
    rc = MPL_env2bool("MPICH_NOLOCAL", &(MPIR_CVAR_NOLOCAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NOLOCAL");
    rc = MPL_env2bool("MPIR_PARAM_NOLOCAL", &(MPIR_CVAR_NOLOCAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NOLOCAL");
    rc = MPL_env2bool("MPIR_CVAR_NOLOCAL", &(MPIR_CVAR_NOLOCAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NOLOCAL");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ODD_EVEN_CLIQUES, /* name */
        &MPIR_CVAR_ODD_EVEN_CLIQUES, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "NODEMAP", /* category */
        "If true, odd procs on a node are seen as local to each other, and even procs on a node are seen as local to each other.  Used for debugging on a single machine. Deprecated in favor of MPIR_CVAR_NUM_CLIQUES.");
    MPIR_CVAR_ODD_EVEN_CLIQUES = defaultval.d;
    rc = MPL_env2bool("MPICH_EVEN_ODD_CLIQUES", &(MPIR_CVAR_ODD_EVEN_CLIQUES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_EVEN_ODD_CLIQUES");
    rc = MPL_env2bool("MPIR_PARAM_EVEN_ODD_CLIQUES", &(MPIR_CVAR_ODD_EVEN_CLIQUES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_EVEN_ODD_CLIQUES");
    rc = MPL_env2bool("MPIR_CVAR_EVEN_ODD_CLIQUES", &(MPIR_CVAR_ODD_EVEN_CLIQUES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_EVEN_ODD_CLIQUES");
    rc = MPL_env2bool("MPICH_ODD_EVEN_CLIQUES", &(MPIR_CVAR_ODD_EVEN_CLIQUES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ODD_EVEN_CLIQUES");
    rc = MPL_env2bool("MPIR_PARAM_ODD_EVEN_CLIQUES", &(MPIR_CVAR_ODD_EVEN_CLIQUES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ODD_EVEN_CLIQUES");
    rc = MPL_env2bool("MPIR_CVAR_ODD_EVEN_CLIQUES", &(MPIR_CVAR_ODD_EVEN_CLIQUES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ODD_EVEN_CLIQUES");

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NUM_CLIQUES, /* name */
        &MPIR_CVAR_NUM_CLIQUES, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "NODEMAP", /* category */
        "Specify the number of cliques that should be used to partition procs on a local node. Procs with the same clique number are seen as local to each other. Used for debugging on a single machine.");
    MPIR_CVAR_NUM_CLIQUES = defaultval.d;
    rc = MPL_env2int("MPICH_NUM_CLIQUES", &(MPIR_CVAR_NUM_CLIQUES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NUM_CLIQUES");
    rc = MPL_env2int("MPIR_PARAM_NUM_CLIQUES", &(MPIR_CVAR_NUM_CLIQUES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NUM_CLIQUES");
    rc = MPL_env2int("MPIR_CVAR_NUM_CLIQUES", &(MPIR_CVAR_NUM_CLIQUES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NUM_CLIQUES");
    rc = MPL_env2int("MPICH_NUM_CLIQUES", &(MPIR_CVAR_NUM_CLIQUES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NUM_CLIQUES");
    rc = MPL_env2int("MPIR_PARAM_NUM_CLIQUES", &(MPIR_CVAR_NUM_CLIQUES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NUM_CLIQUES");
    rc = MPL_env2int("MPIR_CVAR_NUM_CLIQUES", &(MPIR_CVAR_NUM_CLIQUES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NUM_CLIQUES");

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_COLL_ALIAS_CHECK, /* name */
        &MPIR_CVAR_COLL_ALIAS_CHECK, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Enable checking of aliasing in collective operations");
    MPIR_CVAR_COLL_ALIAS_CHECK = defaultval.d;
    rc = MPL_env2int("MPICH_COLL_ALIAS_CHECK", &(MPIR_CVAR_COLL_ALIAS_CHECK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_COLL_ALIAS_CHECK");
    rc = MPL_env2int("MPIR_PARAM_COLL_ALIAS_CHECK", &(MPIR_CVAR_COLL_ALIAS_CHECK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_COLL_ALIAS_CHECK");
    rc = MPL_env2int("MPIR_CVAR_COLL_ALIAS_CHECK", &(MPIR_CVAR_COLL_ALIAS_CHECK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_COLL_ALIAS_CHECK");

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ENABLE_GPU, /* name */
        &MPIR_CVAR_ENABLE_GPU, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "GPU", /* category */
        "Control MPICH GPU support. If set to 0, all GPU support is disabled and we do not query the buffer type internally because we assume no GPU buffer is use.");
    MPIR_CVAR_ENABLE_GPU = defaultval.d;
    rc = MPL_env2int("MPICH_ENABLE_GPU", &(MPIR_CVAR_ENABLE_GPU));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ENABLE_GPU");
    rc = MPL_env2int("MPIR_PARAM_ENABLE_GPU", &(MPIR_CVAR_ENABLE_GPU));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ENABLE_GPU");
    rc = MPL_env2int("MPIR_CVAR_ENABLE_GPU", &(MPIR_CVAR_ENABLE_GPU));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ENABLE_GPU");

    defaultval.d = 8;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REQUEST_POLL_FREQ, /* name */
        &MPIR_CVAR_REQUEST_POLL_FREQ, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "REQUEST", /* category */
        "How frequent to poll during completion calls (wait/test) in terms of number of processed requests before polling.");
    MPIR_CVAR_REQUEST_POLL_FREQ = defaultval.d;
    rc = MPL_env2int("MPICH_REQUEST_POLL_FREQ", &(MPIR_CVAR_REQUEST_POLL_FREQ));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REQUEST_POLL_FREQ");
    rc = MPL_env2int("MPIR_PARAM_REQUEST_POLL_FREQ", &(MPIR_CVAR_REQUEST_POLL_FREQ));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REQUEST_POLL_FREQ");
    rc = MPL_env2int("MPIR_CVAR_REQUEST_POLL_FREQ", &(MPIR_CVAR_REQUEST_POLL_FREQ));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REQUEST_POLL_FREQ");

    defaultval.d = 64;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REQUEST_BATCH_SIZE, /* name */
        &MPIR_CVAR_REQUEST_BATCH_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "REQUEST", /* category */
        "The number of requests to make completion as a batch in MPI_Waitall and MPI_Testall implementation. A large number is likely to cause more cache misses.");
    MPIR_CVAR_REQUEST_BATCH_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_REQUEST_BATCH_SIZE", &(MPIR_CVAR_REQUEST_BATCH_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REQUEST_BATCH_SIZE");
    rc = MPL_env2int("MPIR_PARAM_REQUEST_BATCH_SIZE", &(MPIR_CVAR_REQUEST_BATCH_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REQUEST_BATCH_SIZE");
    rc = MPL_env2int("MPIR_CVAR_REQUEST_BATCH_SIZE", &(MPIR_CVAR_REQUEST_BATCH_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REQUEST_BATCH_SIZE");

    defaultval.d = 1000;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_POLLS_BEFORE_YIELD, /* name */
        &MPIR_CVAR_POLLS_BEFORE_YIELD, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "NEMESIS", /* category */
        "When MPICH is in a busy waiting loop, it will periodically call a function to yield the processor.  This cvar sets the number of loops before the yield function is called.  A value of 0 disables yielding.");
    MPIR_CVAR_POLLS_BEFORE_YIELD = defaultval.d;
    rc = MPL_env2int("MPICH_POLLS_BEFORE_YIELD", &(MPIR_CVAR_POLLS_BEFORE_YIELD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_POLLS_BEFORE_YIELD");
    rc = MPL_env2int("MPIR_PARAM_POLLS_BEFORE_YIELD", &(MPIR_CVAR_POLLS_BEFORE_YIELD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_POLLS_BEFORE_YIELD");
    rc = MPL_env2int("MPIR_CVAR_POLLS_BEFORE_YIELD", &(MPIR_CVAR_POLLS_BEFORE_YIELD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_POLLS_BEFORE_YIELD");

    defaultval.str = (const char *) NULL;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_OFI_USE_PROVIDER, /* name */
        &MPIR_CVAR_OFI_USE_PROVIDER, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_MPIDEV_DETAIL,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "DEVELOPER", /* category */
        "If non-null, choose an OFI provider by name. If using with the CH4 device and using an older libfabric installation than the recommended version to accompany this MPICH version, unexpected results may occur.");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_OFI_USE_PROVIDER", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_OFI_USE_PROVIDER");
    rc = MPL_env2str("MPIR_PARAM_OFI_USE_PROVIDER", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_OFI_USE_PROVIDER");
    rc = MPL_env2str("MPIR_CVAR_OFI_USE_PROVIDER", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_OFI_USE_PROVIDER");
    if (tmp_str != NULL) {
        MPIR_CVAR_OFI_USE_PROVIDER = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_OFI_USE_PROVIDER);
        if (MPIR_CVAR_OFI_USE_PROVIDER == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_OFI_USE_PROVIDER");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_OFI_USE_PROVIDER = NULL;
    }

    defaultval.str = (const char *) NULL;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_CH3_INTERFACE_HOSTNAME, /* name */
        &MPIR_CVAR_CH3_INTERFACE_HOSTNAME, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH3", /* category */
        "If non-NULL, this cvar specifies the IP address that other processes should use when connecting to this process. This cvar is mutually exclusive with the MPIR_CVAR_CH3_NETWORK_IFACE cvar and it is an error to set them both.");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_INTERFACE_HOSTNAME", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_INTERFACE_HOSTNAME");
    rc = MPL_env2str("MPIR_PARAM_INTERFACE_HOSTNAME", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_INTERFACE_HOSTNAME");
    rc = MPL_env2str("MPIR_CVAR_INTERFACE_HOSTNAME", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_INTERFACE_HOSTNAME");
    rc = MPL_env2str("MPICH_CH3_INTERFACE_HOSTNAME", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_INTERFACE_HOSTNAME");
    rc = MPL_env2str("MPIR_PARAM_CH3_INTERFACE_HOSTNAME", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_INTERFACE_HOSTNAME");
    rc = MPL_env2str("MPIR_CVAR_CH3_INTERFACE_HOSTNAME", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_INTERFACE_HOSTNAME");
    if (tmp_str != NULL) {
        MPIR_CVAR_CH3_INTERFACE_HOSTNAME = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_CH3_INTERFACE_HOSTNAME);
        if (MPIR_CVAR_CH3_INTERFACE_HOSTNAME == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_CH3_INTERFACE_HOSTNAME");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_CH3_INTERFACE_HOSTNAME = NULL;
    }

    defaultval.range = (MPIR_T_cvar_range_value_t) {0,0};
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_PORT_RANGE, /* name */
        &MPIR_CVAR_CH3_PORT_RANGE, /* address */
        2, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "The MPIR_CVAR_CH3_PORT_RANGE environment variable allows you to specify the range of TCP ports to be used by the process manager and the MPICH library. The format of this variable is <low>:<high>.  To specify any available port, use 0:0.");
    MPIR_CVAR_CH3_PORT_RANGE = defaultval.range;
    rc = MPL_env2range("MPICH_PORTRANGE", &(MPIR_CVAR_CH3_PORT_RANGE.low), &(MPIR_CVAR_CH3_PORT_RANGE.high));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_PORTRANGE");
    rc = MPL_env2range("MPICH_PORT_RANGE", &(MPIR_CVAR_CH3_PORT_RANGE.low), &(MPIR_CVAR_CH3_PORT_RANGE.high));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_PORT_RANGE");
    rc = MPL_env2range("MPIR_PARAM_PORTRANGE", &(MPIR_CVAR_CH3_PORT_RANGE.low), &(MPIR_CVAR_CH3_PORT_RANGE.high));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_PORTRANGE");
    rc = MPL_env2range("MPIR_PARAM_PORT_RANGE", &(MPIR_CVAR_CH3_PORT_RANGE.low), &(MPIR_CVAR_CH3_PORT_RANGE.high));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_PORT_RANGE");
    rc = MPL_env2range("MPIR_CVAR_PORTRANGE", &(MPIR_CVAR_CH3_PORT_RANGE.low), &(MPIR_CVAR_CH3_PORT_RANGE.high));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_PORTRANGE");
    rc = MPL_env2range("MPIR_CVAR_PORT_RANGE", &(MPIR_CVAR_CH3_PORT_RANGE.low), &(MPIR_CVAR_CH3_PORT_RANGE.high));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_PORT_RANGE");
    rc = MPL_env2range("MPICH_CH3_PORT_RANGE", &(MPIR_CVAR_CH3_PORT_RANGE.low), &(MPIR_CVAR_CH3_PORT_RANGE.high));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_PORT_RANGE");
    rc = MPL_env2range("MPIR_PARAM_CH3_PORT_RANGE", &(MPIR_CVAR_CH3_PORT_RANGE.low), &(MPIR_CVAR_CH3_PORT_RANGE.high));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_PORT_RANGE");
    rc = MPL_env2range("MPIR_CVAR_CH3_PORT_RANGE", &(MPIR_CVAR_CH3_PORT_RANGE.low), &(MPIR_CVAR_CH3_PORT_RANGE.high));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_PORT_RANGE");

    defaultval.str = (const char *) NULL;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE, /* name */
        &MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "NEMESIS", /* category */
        "If non-NULL, this cvar specifies which pseudo-ethernet interface the tcp netmod should use (e.g., \"eth1\", \"ib0\"). Note, this is a Linux-specific cvar. This cvar is mutually exclusive with the MPIR_CVAR_CH3_INTERFACE_HOSTNAME cvar and it is an error to set them both.");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_NETWORK_IFACE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NETWORK_IFACE");
    rc = MPL_env2str("MPIR_PARAM_NETWORK_IFACE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NETWORK_IFACE");
    rc = MPL_env2str("MPIR_CVAR_NETWORK_IFACE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NETWORK_IFACE");
    rc = MPL_env2str("MPICH_NEMESIS_TCP_NETWORK_IFACE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEMESIS_TCP_NETWORK_IFACE");
    rc = MPL_env2str("MPIR_PARAM_NEMESIS_TCP_NETWORK_IFACE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEMESIS_TCP_NETWORK_IFACE");
    rc = MPL_env2str("MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE");
    if (tmp_str != NULL) {
        MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE);
        if (MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE = NULL;
    }

    defaultval.d = 10;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEMESIS_TCP_HOST_LOOKUP_RETRIES, /* name */
        &MPIR_CVAR_NEMESIS_TCP_HOST_LOOKUP_RETRIES, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "NEMESIS", /* category */
        "This cvar controls the number of times to retry the gethostbyname() function before giving up.");
    MPIR_CVAR_NEMESIS_TCP_HOST_LOOKUP_RETRIES = defaultval.d;
    rc = MPL_env2int("MPICH_NEMESIS_TCP_HOST_LOOKUP_RETRIES", &(MPIR_CVAR_NEMESIS_TCP_HOST_LOOKUP_RETRIES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEMESIS_TCP_HOST_LOOKUP_RETRIES");
    rc = MPL_env2int("MPIR_PARAM_NEMESIS_TCP_HOST_LOOKUP_RETRIES", &(MPIR_CVAR_NEMESIS_TCP_HOST_LOOKUP_RETRIES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEMESIS_TCP_HOST_LOOKUP_RETRIES");
    rc = MPL_env2int("MPIR_CVAR_NEMESIS_TCP_HOST_LOOKUP_RETRIES", &(MPIR_CVAR_NEMESIS_TCP_HOST_LOOKUP_RETRIES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEMESIS_TCP_HOST_LOOKUP_RETRIES");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEMESIS_ENABLE_CKPOINT, /* name */
        &MPIR_CVAR_NEMESIS_ENABLE_CKPOINT, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "NEMESIS", /* category */
        "If true, enables checkpointing support and returns an error if checkpointing library cannot be initialized.");
    MPIR_CVAR_NEMESIS_ENABLE_CKPOINT = defaultval.d;
    rc = MPL_env2bool("MPICH_NEMESIS_ENABLE_CKPOINT", &(MPIR_CVAR_NEMESIS_ENABLE_CKPOINT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEMESIS_ENABLE_CKPOINT");
    rc = MPL_env2bool("MPIR_PARAM_NEMESIS_ENABLE_CKPOINT", &(MPIR_CVAR_NEMESIS_ENABLE_CKPOINT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEMESIS_ENABLE_CKPOINT");
    rc = MPL_env2bool("MPIR_CVAR_NEMESIS_ENABLE_CKPOINT", &(MPIR_CVAR_NEMESIS_ENABLE_CKPOINT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEMESIS_ENABLE_CKPOINT");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEMESIS_SHM_EAGER_MAX_SZ, /* name */
        &MPIR_CVAR_NEMESIS_SHM_EAGER_MAX_SZ, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "NEMESIS", /* category */
        "This cvar controls the message size at which Nemesis switches from eager to rendezvous mode for shared memory. If this cvar is set to -1, then Nemesis will choose an appropriate value.");
    MPIR_CVAR_NEMESIS_SHM_EAGER_MAX_SZ = defaultval.d;
    rc = MPL_env2int("MPICH_NEMESIS_SHM_EAGER_MAX_SZ", &(MPIR_CVAR_NEMESIS_SHM_EAGER_MAX_SZ));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEMESIS_SHM_EAGER_MAX_SZ");
    rc = MPL_env2int("MPIR_PARAM_NEMESIS_SHM_EAGER_MAX_SZ", &(MPIR_CVAR_NEMESIS_SHM_EAGER_MAX_SZ));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEMESIS_SHM_EAGER_MAX_SZ");
    rc = MPL_env2int("MPIR_CVAR_NEMESIS_SHM_EAGER_MAX_SZ", &(MPIR_CVAR_NEMESIS_SHM_EAGER_MAX_SZ));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEMESIS_SHM_EAGER_MAX_SZ");

    defaultval.d = -2;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_NEMESIS_SHM_READY_EAGER_MAX_SZ, /* name */
        &MPIR_CVAR_NEMESIS_SHM_READY_EAGER_MAX_SZ, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "NEMESIS", /* category */
        "This cvar controls the message size at which Nemesis switches from eager to rendezvous mode for ready-send messages.  If this cvar is set to -1, then ready messages will always be sent eagerly.  If this cvar is set to -2, then Nemesis will choose an appropriate value.");
    MPIR_CVAR_NEMESIS_SHM_READY_EAGER_MAX_SZ = defaultval.d;
    rc = MPL_env2int("MPICH_NEMESIS_SHM_READY_EAGER_MAX_SZ", &(MPIR_CVAR_NEMESIS_SHM_READY_EAGER_MAX_SZ));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEMESIS_SHM_READY_EAGER_MAX_SZ");
    rc = MPL_env2int("MPIR_PARAM_NEMESIS_SHM_READY_EAGER_MAX_SZ", &(MPIR_CVAR_NEMESIS_SHM_READY_EAGER_MAX_SZ));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEMESIS_SHM_READY_EAGER_MAX_SZ");
    rc = MPL_env2int("MPIR_CVAR_NEMESIS_SHM_READY_EAGER_MAX_SZ", &(MPIR_CVAR_NEMESIS_SHM_READY_EAGER_MAX_SZ));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEMESIS_SHM_READY_EAGER_MAX_SZ");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ENABLE_FT, /* name */
        &MPIR_CVAR_ENABLE_FT, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "FT", /* category */
        "Enable fault tolerance functions");
    MPIR_CVAR_ENABLE_FT = defaultval.d;
    rc = MPL_env2bool("MPICH_ENABLE_FT", &(MPIR_CVAR_ENABLE_FT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ENABLE_FT");
    rc = MPL_env2bool("MPIR_PARAM_ENABLE_FT", &(MPIR_CVAR_ENABLE_FT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ENABLE_FT");
    rc = MPL_env2bool("MPIR_CVAR_ENABLE_FT", &(MPIR_CVAR_ENABLE_FT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ENABLE_FT");

    defaultval.str = (const char *) "";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_NEMESIS_NETMOD, /* name */
        &MPIR_CVAR_NEMESIS_NETMOD, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "NEMESIS", /* category */
        "If non-empty, this cvar specifies which network module should be used for communication. This variable is case-insensitive.");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_NEMESIS_NETMOD", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_NEMESIS_NETMOD");
    rc = MPL_env2str("MPIR_PARAM_NEMESIS_NETMOD", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_NEMESIS_NETMOD");
    rc = MPL_env2str("MPIR_CVAR_NEMESIS_NETMOD", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_NEMESIS_NETMOD");
    if (tmp_str != NULL) {
        MPIR_CVAR_NEMESIS_NETMOD = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_NEMESIS_NETMOD);
        if (MPIR_CVAR_NEMESIS_NETMOD == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_NEMESIS_NETMOD");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_NEMESIS_NETMOD = NULL;
    }

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_ENABLE_HCOLL, /* name */
        &MPIR_CVAR_CH3_ENABLE_HCOLL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "If true, enable HCOLL collectives.");
    MPIR_CVAR_CH3_ENABLE_HCOLL = defaultval.d;
    rc = MPL_env2bool("MPICH_CH3_ENABLE_HCOLL", &(MPIR_CVAR_CH3_ENABLE_HCOLL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_ENABLE_HCOLL");
    rc = MPL_env2bool("MPIR_PARAM_CH3_ENABLE_HCOLL", &(MPIR_CVAR_CH3_ENABLE_HCOLL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_ENABLE_HCOLL");
    rc = MPL_env2bool("MPIR_CVAR_CH3_ENABLE_HCOLL", &(MPIR_CVAR_CH3_ENABLE_HCOLL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_ENABLE_HCOLL");

    defaultval.d = 180;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_COMM_CONNECT_TIMEOUT, /* name */
        &MPIR_CVAR_CH3_COMM_CONNECT_TIMEOUT, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_GROUP_EQ,
        defaultval,
        "CH3", /* category */
        "The default time out period in seconds for a connection attempt to the server communicator where the named port exists but no pending accept. User can change the value for a specified connection through its info argument.");
    MPIR_CVAR_CH3_COMM_CONNECT_TIMEOUT = defaultval.d;
    rc = MPL_env2int("MPICH_CH3_COMM_CONNECT_TIMEOUT", &(MPIR_CVAR_CH3_COMM_CONNECT_TIMEOUT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_COMM_CONNECT_TIMEOUT");
    rc = MPL_env2int("MPIR_PARAM_CH3_COMM_CONNECT_TIMEOUT", &(MPIR_CVAR_CH3_COMM_CONNECT_TIMEOUT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_COMM_CONNECT_TIMEOUT");
    rc = MPL_env2int("MPIR_CVAR_CH3_COMM_CONNECT_TIMEOUT", &(MPIR_CVAR_CH3_COMM_CONNECT_TIMEOUT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_COMM_CONNECT_TIMEOUT");

    defaultval.d = 65536;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_RMA_OP_PIGGYBACK_LOCK_DATA_SIZE, /* name */
        &MPIR_CVAR_CH3_RMA_OP_PIGGYBACK_LOCK_DATA_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "Specify the threshold of data size of a RMA operation which can be piggybacked with a LOCK message. It is always a positive value and should not be smaller than MPIDI_RMA_IMMED_BYTES. If user sets it as a small value, for middle and large data size, we will lose performance because of always waiting for round-trip of LOCK synchronization; if user sets it as a large value, we need to consume more memory on target side to buffer this lock request when lock is not satisfied.");
    MPIR_CVAR_CH3_RMA_OP_PIGGYBACK_LOCK_DATA_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_CH3_RMA_OP_PIGGYBACK_LOCK_DATA_SIZE", &(MPIR_CVAR_CH3_RMA_OP_PIGGYBACK_LOCK_DATA_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_RMA_OP_PIGGYBACK_LOCK_DATA_SIZE");
    rc = MPL_env2int("MPIR_PARAM_CH3_RMA_OP_PIGGYBACK_LOCK_DATA_SIZE", &(MPIR_CVAR_CH3_RMA_OP_PIGGYBACK_LOCK_DATA_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_RMA_OP_PIGGYBACK_LOCK_DATA_SIZE");
    rc = MPL_env2int("MPIR_CVAR_CH3_RMA_OP_PIGGYBACK_LOCK_DATA_SIZE", &(MPIR_CVAR_CH3_RMA_OP_PIGGYBACK_LOCK_DATA_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_RMA_OP_PIGGYBACK_LOCK_DATA_SIZE");

    defaultval.d = 65536;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_RMA_ACTIVE_REQ_THRESHOLD, /* name */
        &MPIR_CVAR_CH3_RMA_ACTIVE_REQ_THRESHOLD, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "Threshold of number of active requests to trigger blocking waiting in operation routines. When the value is negative, we never blockingly wait in operation routines. When the value is zero, we always trigger blocking waiting in operation routines to wait until no. of active requests becomes zero. When the value is positive, we do blocking waiting in operation routines to wait until no. of active requests being reduced to this value.");
    MPIR_CVAR_CH3_RMA_ACTIVE_REQ_THRESHOLD = defaultval.d;
    rc = MPL_env2int("MPICH_CH3_RMA_ACTIVE_REQ_THRESHOLD", &(MPIR_CVAR_CH3_RMA_ACTIVE_REQ_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_RMA_ACTIVE_REQ_THRESHOLD");
    rc = MPL_env2int("MPIR_PARAM_CH3_RMA_ACTIVE_REQ_THRESHOLD", &(MPIR_CVAR_CH3_RMA_ACTIVE_REQ_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_RMA_ACTIVE_REQ_THRESHOLD");
    rc = MPL_env2int("MPIR_CVAR_CH3_RMA_ACTIVE_REQ_THRESHOLD", &(MPIR_CVAR_CH3_RMA_ACTIVE_REQ_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_RMA_ACTIVE_REQ_THRESHOLD");

    defaultval.d = 128;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_RMA_POKE_PROGRESS_REQ_THRESHOLD, /* name */
        &MPIR_CVAR_CH3_RMA_POKE_PROGRESS_REQ_THRESHOLD, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "Threshold at which the RMA implementation attempts to complete requests while completing RMA operations and while using the lazy synchonization approach.  Change this value if programs fail because they run out of requests or other internal resources");
    MPIR_CVAR_CH3_RMA_POKE_PROGRESS_REQ_THRESHOLD = defaultval.d;
    rc = MPL_env2int("MPICH_CH3_RMA_POKE_PROGRESS_REQ_THRESHOLD", &(MPIR_CVAR_CH3_RMA_POKE_PROGRESS_REQ_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_RMA_POKE_PROGRESS_REQ_THRESHOLD");
    rc = MPL_env2int("MPIR_PARAM_CH3_RMA_POKE_PROGRESS_REQ_THRESHOLD", &(MPIR_CVAR_CH3_RMA_POKE_PROGRESS_REQ_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_RMA_POKE_PROGRESS_REQ_THRESHOLD");
    rc = MPL_env2int("MPIR_CVAR_CH3_RMA_POKE_PROGRESS_REQ_THRESHOLD", &(MPIR_CVAR_CH3_RMA_POKE_PROGRESS_REQ_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_RMA_POKE_PROGRESS_REQ_THRESHOLD");

    defaultval.d = 1024;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_RMA_SCALABLE_FENCE_PROCESS_NUM, /* name */
        &MPIR_CVAR_CH3_RMA_SCALABLE_FENCE_PROCESS_NUM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "Specify the threshold of switching the algorithm used in FENCE from the basic algorithm to the scalable algorithm. The value can be nagative, zero or positive. When the number of processes is larger than or equal to this value, FENCE will use a scalable algorithm which do not use O(P) data structure; when the number of processes is smaller than the value, FENCE will use a basic but fast algorithm which requires an O(P) data structure.");
    MPIR_CVAR_CH3_RMA_SCALABLE_FENCE_PROCESS_NUM = defaultval.d;
    rc = MPL_env2int("MPICH_CH3_RMA_SCALABLE_FENCE_PROCESS_NUM", &(MPIR_CVAR_CH3_RMA_SCALABLE_FENCE_PROCESS_NUM));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_RMA_SCALABLE_FENCE_PROCESS_NUM");
    rc = MPL_env2int("MPIR_PARAM_CH3_RMA_SCALABLE_FENCE_PROCESS_NUM", &(MPIR_CVAR_CH3_RMA_SCALABLE_FENCE_PROCESS_NUM));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_RMA_SCALABLE_FENCE_PROCESS_NUM");
    rc = MPL_env2int("MPIR_CVAR_CH3_RMA_SCALABLE_FENCE_PROCESS_NUM", &(MPIR_CVAR_CH3_RMA_SCALABLE_FENCE_PROCESS_NUM));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_RMA_SCALABLE_FENCE_PROCESS_NUM");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_RMA_DELAY_ISSUING_FOR_PIGGYBACKING, /* name */
        &MPIR_CVAR_CH3_RMA_DELAY_ISSUING_FOR_PIGGYBACKING, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "Specify if delay issuing of RMA operations for piggybacking LOCK/UNLOCK/FLUSH is enabled. It can be either 0 or 1. When it is set to 1, the issuing of LOCK message is delayed until origin process see the first RMA operation and piggyback LOCK with that operation, and the origin process always keeps the current last operation until the ending synchronization call in order to piggyback UNLOCK/FLUSH with that operation. When it is set to 0, in WIN_LOCK/UNLOCK case, the LOCK message is sent out as early as possible, in WIN_LOCK_ALL/UNLOCK_ALL case, the origin process still tries to piggyback LOCK message with the first operation; for UNLOCK/FLUSH message, the origin process no longer keeps the current last operation but only piggyback UNLOCK/FLUSH if there is an operation avaliable in the ending synchronization call.");
    MPIR_CVAR_CH3_RMA_DELAY_ISSUING_FOR_PIGGYBACKING = defaultval.d;
    rc = MPL_env2int("MPICH_CH3_RMA_DELAY_ISSUING_FOR_PIGGYBACKING", &(MPIR_CVAR_CH3_RMA_DELAY_ISSUING_FOR_PIGGYBACKING));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_RMA_DELAY_ISSUING_FOR_PIGGYBACKING");
    rc = MPL_env2int("MPIR_PARAM_CH3_RMA_DELAY_ISSUING_FOR_PIGGYBACKING", &(MPIR_CVAR_CH3_RMA_DELAY_ISSUING_FOR_PIGGYBACKING));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_RMA_DELAY_ISSUING_FOR_PIGGYBACKING");
    rc = MPL_env2int("MPIR_CVAR_CH3_RMA_DELAY_ISSUING_FOR_PIGGYBACKING", &(MPIR_CVAR_CH3_RMA_DELAY_ISSUING_FOR_PIGGYBACKING));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_RMA_DELAY_ISSUING_FOR_PIGGYBACKING");

    defaultval.d = 262144;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_RMA_SLOTS_SIZE, /* name */
        &MPIR_CVAR_CH3_RMA_SLOTS_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "Number of RMA slots during window creation. Each slot contains a linked list of target elements. The distribution of ranks among slots follows a round-robin pattern. Requires a positive value.");
    MPIR_CVAR_CH3_RMA_SLOTS_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_CH3_RMA_SLOTS_SIZE", &(MPIR_CVAR_CH3_RMA_SLOTS_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_RMA_SLOTS_SIZE");
    rc = MPL_env2int("MPIR_PARAM_CH3_RMA_SLOTS_SIZE", &(MPIR_CVAR_CH3_RMA_SLOTS_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_RMA_SLOTS_SIZE");
    rc = MPL_env2int("MPIR_CVAR_CH3_RMA_SLOTS_SIZE", &(MPIR_CVAR_CH3_RMA_SLOTS_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_RMA_SLOTS_SIZE");

    defaultval.d = 655360;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_RMA_TARGET_LOCK_DATA_BYTES, /* name */
        &MPIR_CVAR_CH3_RMA_TARGET_LOCK_DATA_BYTES, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "Size (in bytes) of available lock data this window can provided. If current buffered lock data is more than this value, the process will drop the upcoming operation data. Requires a positive calue.");
    MPIR_CVAR_CH3_RMA_TARGET_LOCK_DATA_BYTES = defaultval.d;
    rc = MPL_env2int("MPICH_CH3_RMA_TARGET_LOCK_DATA_BYTES", &(MPIR_CVAR_CH3_RMA_TARGET_LOCK_DATA_BYTES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_RMA_TARGET_LOCK_DATA_BYTES");
    rc = MPL_env2int("MPIR_PARAM_CH3_RMA_TARGET_LOCK_DATA_BYTES", &(MPIR_CVAR_CH3_RMA_TARGET_LOCK_DATA_BYTES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_RMA_TARGET_LOCK_DATA_BYTES");
    rc = MPL_env2int("MPIR_CVAR_CH3_RMA_TARGET_LOCK_DATA_BYTES", &(MPIR_CVAR_CH3_RMA_TARGET_LOCK_DATA_BYTES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_RMA_TARGET_LOCK_DATA_BYTES");

    defaultval.d = 131072;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE, /* name */
        &MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "This cvar controls the message size at which CH3 switches from eager to rendezvous mode.");
    MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_CH3_EAGER_MAX_MSG_SIZE", &(MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_EAGER_MAX_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_CH3_EAGER_MAX_MSG_SIZE", &(MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_EAGER_MAX_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE", &(MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_PG_VERBOSE, /* name */
        &MPIR_CVAR_CH3_PG_VERBOSE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_GROUP_EQ,
        defaultval,
        "CH3", /* category */
        "If set, print the PG state on finalize.");
    MPIR_CVAR_CH3_PG_VERBOSE = defaultval.d;
    rc = MPL_env2bool("MPICH_CH3_PG_VERBOSE", &(MPIR_CVAR_CH3_PG_VERBOSE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_PG_VERBOSE");
    rc = MPL_env2bool("MPIR_PARAM_CH3_PG_VERBOSE", &(MPIR_CVAR_CH3_PG_VERBOSE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_PG_VERBOSE");
    rc = MPL_env2bool("MPIR_CVAR_CH3_PG_VERBOSE", &(MPIR_CVAR_CH3_PG_VERBOSE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_PG_VERBOSE");

    defaultval.d = 256;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_RMA_OP_WIN_POOL_SIZE, /* name */
        &MPIR_CVAR_CH3_RMA_OP_WIN_POOL_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "Size of the window-private RMA operations pool (in number of operations) that stores information about RMA operations that could not be issued immediately.  Requires a positive value.");
    MPIR_CVAR_CH3_RMA_OP_WIN_POOL_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_CH3_RMA_OP_WIN_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_OP_WIN_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_RMA_OP_WIN_POOL_SIZE");
    rc = MPL_env2int("MPIR_PARAM_CH3_RMA_OP_WIN_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_OP_WIN_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_RMA_OP_WIN_POOL_SIZE");
    rc = MPL_env2int("MPIR_CVAR_CH3_RMA_OP_WIN_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_OP_WIN_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_RMA_OP_WIN_POOL_SIZE");

    defaultval.d = 16384;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_RMA_OP_GLOBAL_POOL_SIZE, /* name */
        &MPIR_CVAR_CH3_RMA_OP_GLOBAL_POOL_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "Size of the Global RMA operations pool (in number of operations) that stores information about RMA operations that could not be issued immediatly.  Requires a positive value.");
    MPIR_CVAR_CH3_RMA_OP_GLOBAL_POOL_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_CH3_RMA_OP_GLOBAL_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_OP_GLOBAL_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_RMA_OP_GLOBAL_POOL_SIZE");
    rc = MPL_env2int("MPIR_PARAM_CH3_RMA_OP_GLOBAL_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_OP_GLOBAL_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_RMA_OP_GLOBAL_POOL_SIZE");
    rc = MPL_env2int("MPIR_CVAR_CH3_RMA_OP_GLOBAL_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_OP_GLOBAL_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_RMA_OP_GLOBAL_POOL_SIZE");

    defaultval.d = 256;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_RMA_TARGET_WIN_POOL_SIZE, /* name */
        &MPIR_CVAR_CH3_RMA_TARGET_WIN_POOL_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "Size of the window-private RMA target pool (in number of targets) that stores information about RMA targets that could not be issued immediately.  Requires a positive value.");
    MPIR_CVAR_CH3_RMA_TARGET_WIN_POOL_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_CH3_RMA_TARGET_WIN_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_TARGET_WIN_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_RMA_TARGET_WIN_POOL_SIZE");
    rc = MPL_env2int("MPIR_PARAM_CH3_RMA_TARGET_WIN_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_TARGET_WIN_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_RMA_TARGET_WIN_POOL_SIZE");
    rc = MPL_env2int("MPIR_CVAR_CH3_RMA_TARGET_WIN_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_TARGET_WIN_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_RMA_TARGET_WIN_POOL_SIZE");

    defaultval.d = 16384;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_RMA_TARGET_GLOBAL_POOL_SIZE, /* name */
        &MPIR_CVAR_CH3_RMA_TARGET_GLOBAL_POOL_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "Size of the Global RMA targets pool (in number of targets) that stores information about RMA targets that could not be issued immediatly.  Requires a positive value.");
    MPIR_CVAR_CH3_RMA_TARGET_GLOBAL_POOL_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_CH3_RMA_TARGET_GLOBAL_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_TARGET_GLOBAL_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_RMA_TARGET_GLOBAL_POOL_SIZE");
    rc = MPL_env2int("MPIR_PARAM_CH3_RMA_TARGET_GLOBAL_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_TARGET_GLOBAL_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_RMA_TARGET_GLOBAL_POOL_SIZE");
    rc = MPL_env2int("MPIR_CVAR_CH3_RMA_TARGET_GLOBAL_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_TARGET_GLOBAL_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_RMA_TARGET_GLOBAL_POOL_SIZE");

    defaultval.d = 256;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH3_RMA_TARGET_LOCK_ENTRY_WIN_POOL_SIZE, /* name */
        &MPIR_CVAR_CH3_RMA_TARGET_LOCK_ENTRY_WIN_POOL_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH3", /* category */
        "Size of the window-private RMA lock entries pool (in number of lock entries) that stores information about RMA lock requests that could not be satisfied immediatly.  Requires a positive value.");
    MPIR_CVAR_CH3_RMA_TARGET_LOCK_ENTRY_WIN_POOL_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_CH3_RMA_TARGET_LOCK_ENTRY_WIN_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_TARGET_LOCK_ENTRY_WIN_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH3_RMA_TARGET_LOCK_ENTRY_WIN_POOL_SIZE");
    rc = MPL_env2int("MPIR_PARAM_CH3_RMA_TARGET_LOCK_ENTRY_WIN_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_TARGET_LOCK_ENTRY_WIN_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH3_RMA_TARGET_LOCK_ENTRY_WIN_POOL_SIZE");
    rc = MPL_env2int("MPIR_CVAR_CH3_RMA_TARGET_LOCK_ENTRY_WIN_POOL_SIZE", &(MPIR_CVAR_CH3_RMA_TARGET_LOCK_ENTRY_WIN_POOL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH3_RMA_TARGET_LOCK_ENTRY_WIN_POOL_SIZE");

    defaultval.d = 69632;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_AM_PACK_BUFFER_SIZE, /* name */
        &MPIR_CVAR_CH4_AM_PACK_BUFFER_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4", /* category */
        "Specifies the number of buffers for packing/unpacking active messages in each block of the pool. The size here should be greater or equal to the max of the eager buffer limit of SHM and NETMOD.");
    MPIR_CVAR_CH4_AM_PACK_BUFFER_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_AM_PACK_BUFFER_SIZE", &(MPIR_CVAR_CH4_AM_PACK_BUFFER_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_AM_PACK_BUFFER_SIZE");
    rc = MPL_env2int("MPIR_PARAM_CH4_AM_PACK_BUFFER_SIZE", &(MPIR_CVAR_CH4_AM_PACK_BUFFER_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_AM_PACK_BUFFER_SIZE");
    rc = MPL_env2int("MPIR_CVAR_CH4_AM_PACK_BUFFER_SIZE", &(MPIR_CVAR_CH4_AM_PACK_BUFFER_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_AM_PACK_BUFFER_SIZE");

    defaultval.d = 16;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_NUM_AM_PACK_BUFFERS_PER_CHUNK, /* name */
        &MPIR_CVAR_CH4_NUM_AM_PACK_BUFFERS_PER_CHUNK, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4", /* category */
        "Specifies the number of buffers for packing/unpacking active messages in each block of the pool.");
    MPIR_CVAR_CH4_NUM_AM_PACK_BUFFERS_PER_CHUNK = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_NUM_AM_PACK_BUFFERS_PER_CHUNK", &(MPIR_CVAR_CH4_NUM_AM_PACK_BUFFERS_PER_CHUNK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_NUM_AM_PACK_BUFFERS_PER_CHUNK");
    rc = MPL_env2int("MPIR_PARAM_CH4_NUM_AM_PACK_BUFFERS_PER_CHUNK", &(MPIR_CVAR_CH4_NUM_AM_PACK_BUFFERS_PER_CHUNK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_NUM_AM_PACK_BUFFERS_PER_CHUNK");
    rc = MPL_env2int("MPIR_CVAR_CH4_NUM_AM_PACK_BUFFERS_PER_CHUNK", &(MPIR_CVAR_CH4_NUM_AM_PACK_BUFFERS_PER_CHUNK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_NUM_AM_PACK_BUFFERS_PER_CHUNK");

    defaultval.d = 8388608;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_MAX_AM_UNEXPECTED_PACK_BUFFERS_SIZE_BYTE, /* name */
        &MPIR_CVAR_CH4_MAX_AM_UNEXPECTED_PACK_BUFFERS_SIZE_BYTE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4", /* category */
        "Specifies the max number of buffers for packing/unpacking active messages in the pool.");
    MPIR_CVAR_CH4_MAX_AM_UNEXPECTED_PACK_BUFFERS_SIZE_BYTE = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_MAX_AM_UNEXPECTED_PACK_BUFFERS_SIZE_BYTE", &(MPIR_CVAR_CH4_MAX_AM_UNEXPECTED_PACK_BUFFERS_SIZE_BYTE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_MAX_AM_UNEXPECTED_PACK_BUFFERS_SIZE_BYTE");
    rc = MPL_env2int("MPIR_PARAM_CH4_MAX_AM_UNEXPECTED_PACK_BUFFERS_SIZE_BYTE", &(MPIR_CVAR_CH4_MAX_AM_UNEXPECTED_PACK_BUFFERS_SIZE_BYTE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_MAX_AM_UNEXPECTED_PACK_BUFFERS_SIZE_BYTE");
    rc = MPL_env2int("MPIR_CVAR_CH4_MAX_AM_UNEXPECTED_PACK_BUFFERS_SIZE_BYTE", &(MPIR_CVAR_CH4_MAX_AM_UNEXPECTED_PACK_BUFFERS_SIZE_BYTE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_MAX_AM_UNEXPECTED_PACK_BUFFERS_SIZE_BYTE");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_AM_LONG_FORCE_PIPELINE, /* name */
        &MPIR_CVAR_CH4_OFI_AM_LONG_FORCE_PIPELINE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "DEVELOPER", /* category */
        "For long message to be sent using pipeline rather than default RDMA read.");
    MPIR_CVAR_CH4_OFI_AM_LONG_FORCE_PIPELINE = defaultval.d;
    rc = MPL_env2bool("MPICH_CH4_OFI_AM_LONG_FORCE_PIPELINE", &(MPIR_CVAR_CH4_OFI_AM_LONG_FORCE_PIPELINE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_AM_LONG_FORCE_PIPELINE");
    rc = MPL_env2bool("MPIR_PARAM_CH4_OFI_AM_LONG_FORCE_PIPELINE", &(MPIR_CVAR_CH4_OFI_AM_LONG_FORCE_PIPELINE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_AM_LONG_FORCE_PIPELINE");
    rc = MPL_env2bool("MPIR_CVAR_CH4_OFI_AM_LONG_FORCE_PIPELINE", &(MPIR_CVAR_CH4_OFI_AM_LONG_FORCE_PIPELINE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_AM_LONG_FORCE_PIPELINE");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG, /* name */
        &MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "Prints out the configuration of each capability selected via the capability sets interface.");
    MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_CAPABILITY_SETS_DEBUG", &(MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_CAPABILITY_SETS_DEBUG");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_CAPABILITY_SETS_DEBUG", &(MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_CAPABILITY_SETS_DEBUG");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG", &(MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG");

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_OFI_SKIP_IPV6, /* name */
        &MPIR_CVAR_OFI_SKIP_IPV6, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "DEVELOPER", /* category */
        "Skip IPv6 providers.");
    MPIR_CVAR_OFI_SKIP_IPV6 = defaultval.d;
    rc = MPL_env2bool("MPICH_OFI_SKIP_IPV6", &(MPIR_CVAR_OFI_SKIP_IPV6));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_OFI_SKIP_IPV6");
    rc = MPL_env2bool("MPIR_PARAM_OFI_SKIP_IPV6", &(MPIR_CVAR_OFI_SKIP_IPV6));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_OFI_SKIP_IPV6");
    rc = MPL_env2bool("MPIR_CVAR_OFI_SKIP_IPV6", &(MPIR_CVAR_OFI_SKIP_IPV6));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_OFI_SKIP_IPV6");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_ENABLE_AV_TABLE, /* name */
        &MPIR_CVAR_CH4_OFI_ENABLE_AV_TABLE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If true, the OFI addressing information will be stored with an FI_AV_TABLE. If false, an FI_AV_MAP will be used.");
    MPIR_CVAR_CH4_OFI_ENABLE_AV_TABLE = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_ENABLE_AV_TABLE", &(MPIR_CVAR_CH4_OFI_ENABLE_AV_TABLE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_ENABLE_AV_TABLE");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_ENABLE_AV_TABLE", &(MPIR_CVAR_CH4_OFI_ENABLE_AV_TABLE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_ENABLE_AV_TABLE");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_ENABLE_AV_TABLE", &(MPIR_CVAR_CH4_OFI_ENABLE_AV_TABLE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_ENABLE_AV_TABLE");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS, /* name */
        &MPIR_CVAR_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If true, use OFI scalable endpoints.");
    MPIR_CVAR_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS", &(MPIR_CVAR_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS", &(MPIR_CVAR_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS", &(MPIR_CVAR_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_ENABLE_SHARED_CONTEXTS, /* name */
        &MPIR_CVAR_CH4_OFI_ENABLE_SHARED_CONTEXTS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If set to false (zero), MPICH does not use OFI shared contexts. If set to -1, it is determined by the OFI capability sets based on the provider. Otherwise, MPICH tries to use OFI shared contexts. If they are unavailable, it'll fall back to the mode without shared contexts.");
    MPIR_CVAR_CH4_OFI_ENABLE_SHARED_CONTEXTS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_ENABLE_SHARED_CONTEXTS", &(MPIR_CVAR_CH4_OFI_ENABLE_SHARED_CONTEXTS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_ENABLE_SHARED_CONTEXTS");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_ENABLE_SHARED_CONTEXTS", &(MPIR_CVAR_CH4_OFI_ENABLE_SHARED_CONTEXTS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_ENABLE_SHARED_CONTEXTS");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_ENABLE_SHARED_CONTEXTS", &(MPIR_CVAR_CH4_OFI_ENABLE_SHARED_CONTEXTS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_ENABLE_SHARED_CONTEXTS");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_ENABLE_MR_SCALABLE, /* name */
        &MPIR_CVAR_CH4_OFI_ENABLE_MR_SCALABLE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "This variable is only provided for backward compatibility. When using OFI versions 1.5+, use the other memory region variables. If true, MR_SCALABLE for OFI memory regions. If false, MR_BASIC for OFI memory regions.");
    MPIR_CVAR_CH4_OFI_ENABLE_MR_SCALABLE = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_ENABLE_MR_SCALABLE", &(MPIR_CVAR_CH4_OFI_ENABLE_MR_SCALABLE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_ENABLE_MR_SCALABLE");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_ENABLE_MR_SCALABLE", &(MPIR_CVAR_CH4_OFI_ENABLE_MR_SCALABLE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_ENABLE_MR_SCALABLE");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_ENABLE_MR_SCALABLE", &(MPIR_CVAR_CH4_OFI_ENABLE_MR_SCALABLE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_ENABLE_MR_SCALABLE");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_ENABLE_MR_VIRT_ADDRESS, /* name */
        &MPIR_CVAR_CH4_OFI_ENABLE_MR_VIRT_ADDRESS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If true, enable virtual addressing for OFI memory regions. This variable is only meaningful for OFI versions 1.5+. It is equivelent to using FI_MR_BASIC in versions of OFI older than 1.5.");
    MPIR_CVAR_CH4_OFI_ENABLE_MR_VIRT_ADDRESS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_ENABLE_MR_VIRT_ADDRESS", &(MPIR_CVAR_CH4_OFI_ENABLE_MR_VIRT_ADDRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_ENABLE_MR_VIRT_ADDRESS");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_ENABLE_MR_VIRT_ADDRESS", &(MPIR_CVAR_CH4_OFI_ENABLE_MR_VIRT_ADDRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_ENABLE_MR_VIRT_ADDRESS");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_ENABLE_MR_VIRT_ADDRESS", &(MPIR_CVAR_CH4_OFI_ENABLE_MR_VIRT_ADDRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_ENABLE_MR_VIRT_ADDRESS");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_ENABLE_MR_ALLOCATED, /* name */
        &MPIR_CVAR_CH4_OFI_ENABLE_MR_ALLOCATED, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If true, require all OFI memory regions must be backed by physical memory pages at the time the registration call is made. This variable is only meaningful for OFI versions 1.5+. It is equivelent to using FI_MR_BASIC in versions of OFI older than 1.5.");
    MPIR_CVAR_CH4_OFI_ENABLE_MR_ALLOCATED = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_ENABLE_MR_ALLOCATED", &(MPIR_CVAR_CH4_OFI_ENABLE_MR_ALLOCATED));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_ENABLE_MR_ALLOCATED");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_ENABLE_MR_ALLOCATED", &(MPIR_CVAR_CH4_OFI_ENABLE_MR_ALLOCATED));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_ENABLE_MR_ALLOCATED");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_ENABLE_MR_ALLOCATED", &(MPIR_CVAR_CH4_OFI_ENABLE_MR_ALLOCATED));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_ENABLE_MR_ALLOCATED");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_ENABLE_MR_PROV_KEY, /* name */
        &MPIR_CVAR_CH4_OFI_ENABLE_MR_PROV_KEY, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If true, enable provider supplied key for OFI memory regions. This variable is only meaningful for OFI versions 1.5+. It is equivelent to using FI_MR_BASIC in versions of OFI older than 1.5.");
    MPIR_CVAR_CH4_OFI_ENABLE_MR_PROV_KEY = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_ENABLE_MR_PROV_KEY", &(MPIR_CVAR_CH4_OFI_ENABLE_MR_PROV_KEY));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_ENABLE_MR_PROV_KEY");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_ENABLE_MR_PROV_KEY", &(MPIR_CVAR_CH4_OFI_ENABLE_MR_PROV_KEY));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_ENABLE_MR_PROV_KEY");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_ENABLE_MR_PROV_KEY", &(MPIR_CVAR_CH4_OFI_ENABLE_MR_PROV_KEY));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_ENABLE_MR_PROV_KEY");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_ENABLE_TAGGED, /* name */
        &MPIR_CVAR_CH4_OFI_ENABLE_TAGGED, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If true, use tagged message transmission functions in OFI.");
    MPIR_CVAR_CH4_OFI_ENABLE_TAGGED = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_ENABLE_TAGGED", &(MPIR_CVAR_CH4_OFI_ENABLE_TAGGED));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_ENABLE_TAGGED");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_ENABLE_TAGGED", &(MPIR_CVAR_CH4_OFI_ENABLE_TAGGED));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_ENABLE_TAGGED");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_ENABLE_TAGGED", &(MPIR_CVAR_CH4_OFI_ENABLE_TAGGED));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_ENABLE_TAGGED");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_ENABLE_AM, /* name */
        &MPIR_CVAR_CH4_OFI_ENABLE_AM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If true, enable OFI active message support.");
    MPIR_CVAR_CH4_OFI_ENABLE_AM = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_ENABLE_AM", &(MPIR_CVAR_CH4_OFI_ENABLE_AM));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_ENABLE_AM");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_ENABLE_AM", &(MPIR_CVAR_CH4_OFI_ENABLE_AM));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_ENABLE_AM");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_ENABLE_AM", &(MPIR_CVAR_CH4_OFI_ENABLE_AM));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_ENABLE_AM");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_ENABLE_RMA, /* name */
        &MPIR_CVAR_CH4_OFI_ENABLE_RMA, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If true, enable OFI RMA support for MPI RMA operations. OFI support for basic RMA is always required to implement large messgage transfers in the active message code path.");
    MPIR_CVAR_CH4_OFI_ENABLE_RMA = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_ENABLE_RMA", &(MPIR_CVAR_CH4_OFI_ENABLE_RMA));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_ENABLE_RMA");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_ENABLE_RMA", &(MPIR_CVAR_CH4_OFI_ENABLE_RMA));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_ENABLE_RMA");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_ENABLE_RMA", &(MPIR_CVAR_CH4_OFI_ENABLE_RMA));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_ENABLE_RMA");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_ENABLE_ATOMICS, /* name */
        &MPIR_CVAR_CH4_OFI_ENABLE_ATOMICS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If true, enable OFI Atomics support.");
    MPIR_CVAR_CH4_OFI_ENABLE_ATOMICS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_ENABLE_ATOMICS", &(MPIR_CVAR_CH4_OFI_ENABLE_ATOMICS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_ENABLE_ATOMICS");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_ENABLE_ATOMICS", &(MPIR_CVAR_CH4_OFI_ENABLE_ATOMICS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_ENABLE_ATOMICS");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_ENABLE_ATOMICS", &(MPIR_CVAR_CH4_OFI_ENABLE_ATOMICS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_ENABLE_ATOMICS");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_FETCH_ATOMIC_IOVECS, /* name */
        &MPIR_CVAR_CH4_OFI_FETCH_ATOMIC_IOVECS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "Specifies the maximum number of iovecs that can be used by the OFI provider for fetch_atomic operations. The default value is -1, indicating that no value is set.");
    MPIR_CVAR_CH4_OFI_FETCH_ATOMIC_IOVECS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_FETCH_ATOMIC_IOVECS", &(MPIR_CVAR_CH4_OFI_FETCH_ATOMIC_IOVECS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_FETCH_ATOMIC_IOVECS");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_FETCH_ATOMIC_IOVECS", &(MPIR_CVAR_CH4_OFI_FETCH_ATOMIC_IOVECS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_FETCH_ATOMIC_IOVECS");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_FETCH_ATOMIC_IOVECS", &(MPIR_CVAR_CH4_OFI_FETCH_ATOMIC_IOVECS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_FETCH_ATOMIC_IOVECS");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS, /* name */
        &MPIR_CVAR_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If true, enable MPI data auto progress.");
    MPIR_CVAR_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS", &(MPIR_CVAR_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS", &(MPIR_CVAR_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS", &(MPIR_CVAR_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS, /* name */
        &MPIR_CVAR_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If true, enable MPI control auto progress.");
    MPIR_CVAR_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS", &(MPIR_CVAR_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS", &(MPIR_CVAR_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS", &(MPIR_CVAR_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_ENABLE_PT2PT_NOPACK, /* name */
        &MPIR_CVAR_CH4_OFI_ENABLE_PT2PT_NOPACK, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If true, enable iovec for pt2pt.");
    MPIR_CVAR_CH4_OFI_ENABLE_PT2PT_NOPACK = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_ENABLE_PT2PT_NOPACK", &(MPIR_CVAR_CH4_OFI_ENABLE_PT2PT_NOPACK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_ENABLE_PT2PT_NOPACK");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_ENABLE_PT2PT_NOPACK", &(MPIR_CVAR_CH4_OFI_ENABLE_PT2PT_NOPACK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_ENABLE_PT2PT_NOPACK");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_ENABLE_PT2PT_NOPACK", &(MPIR_CVAR_CH4_OFI_ENABLE_PT2PT_NOPACK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_ENABLE_PT2PT_NOPACK");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_CONTEXT_ID_BITS, /* name */
        &MPIR_CVAR_CH4_OFI_CONTEXT_ID_BITS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "Specifies the number of bits that will be used for matching the context ID. The default value is -1, indicating that no value is set and that the default will be defined in the ofi_types.h file.");
    MPIR_CVAR_CH4_OFI_CONTEXT_ID_BITS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_CONTEXT_ID_BITS", &(MPIR_CVAR_CH4_OFI_CONTEXT_ID_BITS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_CONTEXT_ID_BITS");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_CONTEXT_ID_BITS", &(MPIR_CVAR_CH4_OFI_CONTEXT_ID_BITS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_CONTEXT_ID_BITS");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_CONTEXT_ID_BITS", &(MPIR_CVAR_CH4_OFI_CONTEXT_ID_BITS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_CONTEXT_ID_BITS");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_RANK_BITS, /* name */
        &MPIR_CVAR_CH4_OFI_RANK_BITS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "Specifies the number of bits that will be used for matching the MPI rank. The default value is -1, indicating that no value is set and that the default will be defined in the ofi_types.h file.");
    MPIR_CVAR_CH4_OFI_RANK_BITS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_RANK_BITS", &(MPIR_CVAR_CH4_OFI_RANK_BITS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_RANK_BITS");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_RANK_BITS", &(MPIR_CVAR_CH4_OFI_RANK_BITS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_RANK_BITS");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_RANK_BITS", &(MPIR_CVAR_CH4_OFI_RANK_BITS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_RANK_BITS");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_TAG_BITS, /* name */
        &MPIR_CVAR_CH4_OFI_TAG_BITS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "Specifies the number of bits that will be used for matching the user tag. The default value is -1, indicating that no value is set and that the default will be defined in the ofi_types.h file.");
    MPIR_CVAR_CH4_OFI_TAG_BITS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_TAG_BITS", &(MPIR_CVAR_CH4_OFI_TAG_BITS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_TAG_BITS");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_TAG_BITS", &(MPIR_CVAR_CH4_OFI_TAG_BITS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_TAG_BITS");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_TAG_BITS", &(MPIR_CVAR_CH4_OFI_TAG_BITS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_TAG_BITS");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_MAJOR_VERSION, /* name */
        &MPIR_CVAR_CH4_OFI_MAJOR_VERSION, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "Specifies the major version of the OFI library. The default is the major version of the OFI library used with MPICH. If using this CVAR, it is recommended that the user also specifies a specific OFI provider.");
    MPIR_CVAR_CH4_OFI_MAJOR_VERSION = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_MAJOR_VERSION", &(MPIR_CVAR_CH4_OFI_MAJOR_VERSION));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_MAJOR_VERSION");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_MAJOR_VERSION", &(MPIR_CVAR_CH4_OFI_MAJOR_VERSION));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_MAJOR_VERSION");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_MAJOR_VERSION", &(MPIR_CVAR_CH4_OFI_MAJOR_VERSION));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_MAJOR_VERSION");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_MINOR_VERSION, /* name */
        &MPIR_CVAR_CH4_OFI_MINOR_VERSION, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "Specifies the major version of the OFI library. The default is the minor version of the OFI library used with MPICH. If using this CVAR, it is recommended that the user also specifies a specific OFI provider.");
    MPIR_CVAR_CH4_OFI_MINOR_VERSION = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_MINOR_VERSION", &(MPIR_CVAR_CH4_OFI_MINOR_VERSION));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_MINOR_VERSION");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_MINOR_VERSION", &(MPIR_CVAR_CH4_OFI_MINOR_VERSION));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_MINOR_VERSION");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_MINOR_VERSION", &(MPIR_CVAR_CH4_OFI_MINOR_VERSION));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_MINOR_VERSION");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_MAX_VNIS, /* name */
        &MPIR_CVAR_CH4_OFI_MAX_VNIS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If set to positive, this CVAR specifies the maximum number of CH4 VNIs that OFI netmod exposes. If set to 0 (the default) or bigger than MPIR_CVAR_CH4_NUM_VCIS, the number of exposed VNIs is set to MPIR_CVAR_CH4_NUM_VCIS.");
    MPIR_CVAR_CH4_OFI_MAX_VNIS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_MAX_VNIS", &(MPIR_CVAR_CH4_OFI_MAX_VNIS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_MAX_VNIS");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_MAX_VNIS", &(MPIR_CVAR_CH4_OFI_MAX_VNIS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_MAX_VNIS");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_MAX_VNIS", &(MPIR_CVAR_CH4_OFI_MAX_VNIS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_MAX_VNIS");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_MAX_RMA_SEP_CTX, /* name */
        &MPIR_CVAR_CH4_OFI_MAX_RMA_SEP_CTX, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If set to positive, this CVAR specifies the maximum number of transmit contexts RMA can utilize in a scalable endpoint. This value is effective only when scalable endpoint is available, otherwise it will be ignored.");
    MPIR_CVAR_CH4_OFI_MAX_RMA_SEP_CTX = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_MAX_RMA_SEP_CTX", &(MPIR_CVAR_CH4_OFI_MAX_RMA_SEP_CTX));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_MAX_RMA_SEP_CTX");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_MAX_RMA_SEP_CTX", &(MPIR_CVAR_CH4_OFI_MAX_RMA_SEP_CTX));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_MAX_RMA_SEP_CTX");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_MAX_RMA_SEP_CTX", &(MPIR_CVAR_CH4_OFI_MAX_RMA_SEP_CTX));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_MAX_RMA_SEP_CTX");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_MAX_EAGAIN_RETRY, /* name */
        &MPIR_CVAR_CH4_OFI_MAX_EAGAIN_RETRY, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "If set to positive, this CVAR specifies the maximum number of retries of an ofi operations before returning MPIX_ERR_EAGAIN. This value is effective only when the communicator has the MPI_OFI_set_eagain info hint set to true.");
    MPIR_CVAR_CH4_OFI_MAX_EAGAIN_RETRY = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_MAX_EAGAIN_RETRY", &(MPIR_CVAR_CH4_OFI_MAX_EAGAIN_RETRY));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_MAX_EAGAIN_RETRY");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_MAX_EAGAIN_RETRY", &(MPIR_CVAR_CH4_OFI_MAX_EAGAIN_RETRY));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_MAX_EAGAIN_RETRY");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_MAX_EAGAIN_RETRY", &(MPIR_CVAR_CH4_OFI_MAX_EAGAIN_RETRY));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_MAX_EAGAIN_RETRY");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_NUM_AM_BUFFERS, /* name */
        &MPIR_CVAR_CH4_OFI_NUM_AM_BUFFERS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "Specifies the number of buffers for receiving active messages.");
    MPIR_CVAR_CH4_OFI_NUM_AM_BUFFERS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_NUM_AM_BUFFERS", &(MPIR_CVAR_CH4_OFI_NUM_AM_BUFFERS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_NUM_AM_BUFFERS");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_NUM_AM_BUFFERS", &(MPIR_CVAR_CH4_OFI_NUM_AM_BUFFERS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_NUM_AM_BUFFERS");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_NUM_AM_BUFFERS", &(MPIR_CVAR_CH4_OFI_NUM_AM_BUFFERS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_NUM_AM_BUFFERS");

    defaultval.d = 100;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_RMA_PROGRESS_INTERVAL, /* name */
        &MPIR_CVAR_CH4_OFI_RMA_PROGRESS_INTERVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "Specifies the interval for manually flushing RMA operations when automatic progress is not enabled. It the underlying OFI provider supports auto data progress, this value is ignored. If the value is -1, this optimization will be turned off.");
    MPIR_CVAR_CH4_OFI_RMA_PROGRESS_INTERVAL = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_RMA_PROGRESS_INTERVAL", &(MPIR_CVAR_CH4_OFI_RMA_PROGRESS_INTERVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_RMA_PROGRESS_INTERVAL");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_RMA_PROGRESS_INTERVAL", &(MPIR_CVAR_CH4_OFI_RMA_PROGRESS_INTERVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_RMA_PROGRESS_INTERVAL");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_RMA_PROGRESS_INTERVAL", &(MPIR_CVAR_CH4_OFI_RMA_PROGRESS_INTERVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_RMA_PROGRESS_INTERVAL");

    defaultval.d = 16384;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_RMA_IOVEC_MAX, /* name */
        &MPIR_CVAR_CH4_OFI_RMA_IOVEC_MAX, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "Specifies the maximum number of iovecs to allocate for RMA operations to/from noncontiguous buffers.");
    MPIR_CVAR_CH4_OFI_RMA_IOVEC_MAX = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_RMA_IOVEC_MAX", &(MPIR_CVAR_CH4_OFI_RMA_IOVEC_MAX));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_RMA_IOVEC_MAX");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_RMA_IOVEC_MAX", &(MPIR_CVAR_CH4_OFI_RMA_IOVEC_MAX));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_RMA_IOVEC_MAX");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_RMA_IOVEC_MAX", &(MPIR_CVAR_CH4_OFI_RMA_IOVEC_MAX));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_RMA_IOVEC_MAX");

    defaultval.d = 16;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_NUM_PACK_BUFFERS_PER_CHUNK, /* name */
        &MPIR_CVAR_CH4_OFI_NUM_PACK_BUFFERS_PER_CHUNK, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "Specifies the number of buffers for packing/unpacking messages in each block of the pool.");
    MPIR_CVAR_CH4_OFI_NUM_PACK_BUFFERS_PER_CHUNK = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_NUM_PACK_BUFFERS_PER_CHUNK", &(MPIR_CVAR_CH4_OFI_NUM_PACK_BUFFERS_PER_CHUNK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_NUM_PACK_BUFFERS_PER_CHUNK");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_NUM_PACK_BUFFERS_PER_CHUNK", &(MPIR_CVAR_CH4_OFI_NUM_PACK_BUFFERS_PER_CHUNK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_NUM_PACK_BUFFERS_PER_CHUNK");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_NUM_PACK_BUFFERS_PER_CHUNK", &(MPIR_CVAR_CH4_OFI_NUM_PACK_BUFFERS_PER_CHUNK));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_NUM_PACK_BUFFERS_PER_CHUNK");

    defaultval.d = 256;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_MAX_NUM_PACK_BUFFERS, /* name */
        &MPIR_CVAR_CH4_OFI_MAX_NUM_PACK_BUFFERS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "Specifies the max number of buffers for packing/unpacking messages in the pool.");
    MPIR_CVAR_CH4_OFI_MAX_NUM_PACK_BUFFERS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_MAX_NUM_PACK_BUFFERS", &(MPIR_CVAR_CH4_OFI_MAX_NUM_PACK_BUFFERS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_MAX_NUM_PACK_BUFFERS");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_MAX_NUM_PACK_BUFFERS", &(MPIR_CVAR_CH4_OFI_MAX_NUM_PACK_BUFFERS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_MAX_NUM_PACK_BUFFERS");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_MAX_NUM_PACK_BUFFERS", &(MPIR_CVAR_CH4_OFI_MAX_NUM_PACK_BUFFERS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_MAX_NUM_PACK_BUFFERS");

    defaultval.d = -1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_OFI_EAGER_MAX_MSG_SIZE, /* name */
        &MPIR_CVAR_CH4_OFI_EAGER_MAX_MSG_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_OFI", /* category */
        "This cvar controls the message size at which OFI native path switches from eager to rendezvous mode. It does not affect the AM path eager limit. Having this gives a way to reliably test native non-path. If the number is positive, OFI will init the MPIDI_OFI_global.max_msg_size to the value of cvar. If the number is negative, OFI will init the MPIDI_OFI_globa.max_msg_size using whatever provider gives (which might be unlimited for socket provider).");
    MPIR_CVAR_CH4_OFI_EAGER_MAX_MSG_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_OFI_EAGER_MAX_MSG_SIZE", &(MPIR_CVAR_CH4_OFI_EAGER_MAX_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_OFI_EAGER_MAX_MSG_SIZE");
    rc = MPL_env2int("MPIR_PARAM_CH4_OFI_EAGER_MAX_MSG_SIZE", &(MPIR_CVAR_CH4_OFI_EAGER_MAX_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_OFI_EAGER_MAX_MSG_SIZE");
    rc = MPL_env2int("MPIR_CVAR_CH4_OFI_EAGER_MAX_MSG_SIZE", &(MPIR_CVAR_CH4_OFI_EAGER_MAX_MSG_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_OFI_EAGER_MAX_MSG_SIZE");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_UCX_MAX_VNIS, /* name */
        &MPIR_CVAR_CH4_UCX_MAX_VNIS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4_UCX", /* category */
        "If set to positive, this CVAR specifies the maximum number of CH4 VNIs that UCX netmod exposes. If set to 0 (the default) or bigger than MPIR_CVAR_CH4_NUM_VCIS, the number of exposed VNIs is set to MPIR_CVAR_CH4_NUM_VCIS.");
    MPIR_CVAR_CH4_UCX_MAX_VNIS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_UCX_MAX_VNIS", &(MPIR_CVAR_CH4_UCX_MAX_VNIS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_UCX_MAX_VNIS");
    rc = MPL_env2int("MPIR_PARAM_CH4_UCX_MAX_VNIS", &(MPIR_CVAR_CH4_UCX_MAX_VNIS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_UCX_MAX_VNIS");
    rc = MPL_env2int("MPIR_CVAR_CH4_UCX_MAX_VNIS", &(MPIR_CVAR_CH4_UCX_MAX_VNIS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_UCX_MAX_VNIS");

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE, /* name */
        &MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH4", /* category */
        "By default, we will cache ipc handle. To manually disable ipc handle cache, user can set this variable to 0.");
    MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_IPC_GPU_HANDLE_CACHE", &(MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_IPC_GPU_HANDLE_CACHE");
    rc = MPL_env2int("MPIR_PARAM_CH4_IPC_GPU_HANDLE_CACHE", &(MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_IPC_GPU_HANDLE_CACHE");
    rc = MPL_env2int("MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE", &(MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE");

    defaultval.d = 32768;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD, /* name */
        &MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH4", /* category */
        "If a send message size is greater than or equal to MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD (in bytes), then enable GPU-based single copy protocol for intranode communication. The environment variable is valid only when then GPU IPC shmmod is enabled.");
    MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_IPC_GPU_P2P_THRESHOLD", &(MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_IPC_GPU_P2P_THRESHOLD");
    rc = MPL_env2int("MPIR_PARAM_CH4_IPC_GPU_P2P_THRESHOLD", &(MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_IPC_GPU_P2P_THRESHOLD");
    rc = MPL_env2int("MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD", &(MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD");

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_XPMEM_ENABLE, /* name */
        &MPIR_CVAR_CH4_XPMEM_ENABLE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH4", /* category */
        "To manually disable XPMEM set to 0. The environment variable is valid only when the XPMEM submodule is enabled.");
    MPIR_CVAR_CH4_XPMEM_ENABLE = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_XPMEM_ENABLE", &(MPIR_CVAR_CH4_XPMEM_ENABLE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_XPMEM_ENABLE");
    rc = MPL_env2int("MPIR_PARAM_CH4_XPMEM_ENABLE", &(MPIR_CVAR_CH4_XPMEM_ENABLE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_XPMEM_ENABLE");
    rc = MPL_env2int("MPIR_CVAR_CH4_XPMEM_ENABLE", &(MPIR_CVAR_CH4_XPMEM_ENABLE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_XPMEM_ENABLE");

    defaultval.d = 4096;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_IPC_XPMEM_P2P_THRESHOLD, /* name */
        &MPIR_CVAR_CH4_IPC_XPMEM_P2P_THRESHOLD, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH4", /* category */
        "If a send message size is greater than or equal to MPIR_CVAR_CH4_IPC_XPMEM_P2P_THRESHOLD (in bytes), then enable XPMEM-based single copy protocol for intranode communication. The environment variable is valid only when the XPMEM submodule is enabled.");
    MPIR_CVAR_CH4_IPC_XPMEM_P2P_THRESHOLD = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_IPC_XPMEM_P2P_THRESHOLD", &(MPIR_CVAR_CH4_IPC_XPMEM_P2P_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_IPC_XPMEM_P2P_THRESHOLD");
    rc = MPL_env2int("MPIR_PARAM_CH4_IPC_XPMEM_P2P_THRESHOLD", &(MPIR_CVAR_CH4_IPC_XPMEM_P2P_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_IPC_XPMEM_P2P_THRESHOLD");
    rc = MPL_env2int("MPIR_CVAR_CH4_IPC_XPMEM_P2P_THRESHOLD", &(MPIR_CVAR_CH4_IPC_XPMEM_P2P_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_IPC_XPMEM_P2P_THRESHOLD");

    defaultval.d = MPIR_CVAR_BCAST_POSIX_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_BCAST_POSIX_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_BCAST_POSIX_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select algorithm for intra-node bcast\
mpir           - Fallback to MPIR collectives\
release_gather - Force shm optimized algo using release, gather primitives\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE)");
    MPIR_CVAR_BCAST_POSIX_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_BCAST_POSIX_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BCAST_POSIX_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_BCAST_POSIX_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BCAST_POSIX_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_BCAST_POSIX_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BCAST_POSIX_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "mpir"))
            MPIR_CVAR_BCAST_POSIX_INTRA_ALGORITHM = MPIR_CVAR_BCAST_POSIX_INTRA_ALGORITHM_mpir;
        else if (0 == strcmp(tmp_str, "release_gather"))
            MPIR_CVAR_BCAST_POSIX_INTRA_ALGORITHM = MPIR_CVAR_BCAST_POSIX_INTRA_ALGORITHM_release_gather;
        else if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_BCAST_POSIX_INTRA_ALGORITHM = MPIR_CVAR_BCAST_POSIX_INTRA_ALGORITHM_auto;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_BCAST_POSIX_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_REDUCE_POSIX_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_POSIX_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_REDUCE_POSIX_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select algorithm for intra-node reduce\
mpir           - Fallback to MPIR collectives\
release_gather - Force shm optimized algo using release, gather primitives\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE)");
    MPIR_CVAR_REDUCE_POSIX_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_REDUCE_POSIX_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_POSIX_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_REDUCE_POSIX_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_POSIX_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_REDUCE_POSIX_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_POSIX_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "mpir"))
            MPIR_CVAR_REDUCE_POSIX_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_POSIX_INTRA_ALGORITHM_mpir;
        else if (0 == strcmp(tmp_str, "release_gather"))
            MPIR_CVAR_REDUCE_POSIX_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_POSIX_INTRA_ALGORITHM_release_gather;
        else if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_REDUCE_POSIX_INTRA_ALGORITHM = MPIR_CVAR_REDUCE_POSIX_INTRA_ALGORITHM_auto;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_REDUCE_POSIX_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select algorithm for intra-node allreduce\
mpir           - Fallback to MPIR collectives\
release_gather - Force shm optimized algo using release, gather primitives\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE)");
    MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_ALLREDUCE_POSIX_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ALLREDUCE_POSIX_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_ALLREDUCE_POSIX_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ALLREDUCE_POSIX_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "mpir"))
            MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM = MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM_mpir;
        else if (0 == strcmp(tmp_str, "release_gather"))
            MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM = MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM_release_gather;
        else if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM = MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM_auto;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_ALLREDUCE_POSIX_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM_auto;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM, /* name */
        &MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Variable to select algorithm for intra-node barrier\
mpir           - Fallback to MPIR collectives\
release_gather - Force shm optimized algo using release, gather primitives\
auto - Internal algorithm selection (can be overridden with MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE)");
    MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM = defaultval.d;
    tmp_str=NULL;
    rc = MPL_env2str("MPICH_BARRIER_POSIX_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BARRIER_POSIX_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_PARAM_BARRIER_POSIX_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BARRIER_POSIX_INTRA_ALGORITHM");
    rc = MPL_env2str("MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM");
    if (tmp_str != NULL) {
        if (0 == strcmp(tmp_str, "mpir"))
            MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM = MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM_mpir;
        else if (0 == strcmp(tmp_str, "release_gather"))
            MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM = MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM_release_gather;
        else if (0 == strcmp(tmp_str, "auto"))
            MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM = MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM_auto;
        else {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,MPI_ERR_OTHER, "**cvar_val", "**cvar_val %s %s", "MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM", tmp_str);
            goto fn_fail;
        }
    }

    defaultval.d = 5;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_POSIX_NUM_COLLS_THRESHOLD, /* name */
        &MPIR_CVAR_POSIX_NUM_COLLS_THRESHOLD, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Use posix optimized collectives (release_gather) only when the total number of Bcast, Reduce, Barrier, and Allreduce calls on the node level communicator is more than this threshold.");
    MPIR_CVAR_POSIX_NUM_COLLS_THRESHOLD = defaultval.d;
    rc = MPL_env2int("MPICH_POSIX_NUM_COLLS_THRESHOLD", &(MPIR_CVAR_POSIX_NUM_COLLS_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_POSIX_NUM_COLLS_THRESHOLD");
    rc = MPL_env2int("MPIR_PARAM_POSIX_NUM_COLLS_THRESHOLD", &(MPIR_CVAR_POSIX_NUM_COLLS_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_POSIX_NUM_COLLS_THRESHOLD");
    rc = MPL_env2int("MPIR_CVAR_POSIX_NUM_COLLS_THRESHOLD", &(MPIR_CVAR_POSIX_NUM_COLLS_THRESHOLD));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_POSIX_NUM_COLLS_THRESHOLD");

    defaultval.str = (const char *) "";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_CH4_SHM_POSIX_EAGER, /* name */
        &MPIR_CVAR_CH4_SHM_POSIX_EAGER, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH4", /* category */
        "If non-empty, this cvar specifies which shm posix eager module to use");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_CH4_SHM_POSIX_EAGER", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_SHM_POSIX_EAGER");
    rc = MPL_env2str("MPIR_PARAM_CH4_SHM_POSIX_EAGER", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_SHM_POSIX_EAGER");
    rc = MPL_env2str("MPIR_CVAR_CH4_SHM_POSIX_EAGER", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_SHM_POSIX_EAGER");
    if (tmp_str != NULL) {
        MPIR_CVAR_CH4_SHM_POSIX_EAGER = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_CH4_SHM_POSIX_EAGER);
        if (MPIR_CVAR_CH4_SHM_POSIX_EAGER == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_CH4_SHM_POSIX_EAGER");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_CH4_SHM_POSIX_EAGER = NULL;
    }

    defaultval.str = (const char *) "";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE, /* name */
        &MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Defines the location of tuning file.");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE");
    rc = MPL_env2str("MPIR_PARAM_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE");
    rc = MPL_env2str("MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE");
    if (tmp_str != NULL) {
        MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE);
        if (MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE = NULL;
    }

    defaultval.d = 64;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_NUM_CELLS, /* name */
        &MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_NUM_CELLS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH4", /* category */
        "The number of cells used for the depth of the iqueue.");
    MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_NUM_CELLS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_SHM_POSIX_IQUEUE_NUM_CELLS", &(MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_NUM_CELLS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_SHM_POSIX_IQUEUE_NUM_CELLS");
    rc = MPL_env2int("MPIR_PARAM_CH4_SHM_POSIX_IQUEUE_NUM_CELLS", &(MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_NUM_CELLS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_SHM_POSIX_IQUEUE_NUM_CELLS");
    rc = MPL_env2int("MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_NUM_CELLS", &(MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_NUM_CELLS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_NUM_CELLS");

    defaultval.d = 69632;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_CELL_SIZE, /* name */
        &MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_CELL_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH4", /* category */
        "Size of each cell. 4KB * 17 is default to avoid a cache aliasing issue.");
    MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_CELL_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_SHM_POSIX_IQUEUE_CELL_SIZE", &(MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_CELL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_SHM_POSIX_IQUEUE_CELL_SIZE");
    rc = MPL_env2int("MPIR_PARAM_CH4_SHM_POSIX_IQUEUE_CELL_SIZE", &(MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_CELL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_SHM_POSIX_IQUEUE_CELL_SIZE");
    rc = MPL_env2int("MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_CELL_SIZE", &(MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_CELL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_SHM_POSIX_IQUEUE_CELL_SIZE");

    defaultval.d = 65536;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_COLL_SHM_LIMIT_PER_NODE, /* name */
        &MPIR_CVAR_COLL_SHM_LIMIT_PER_NODE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Maximum shared memory created per node for optimized intra-node collectives (in KB)");
    MPIR_CVAR_COLL_SHM_LIMIT_PER_NODE = defaultval.d;
    rc = MPL_env2int("MPICH_COLL_SHM_LIMIT_PER_NODE", &(MPIR_CVAR_COLL_SHM_LIMIT_PER_NODE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_COLL_SHM_LIMIT_PER_NODE");
    rc = MPL_env2int("MPIR_PARAM_COLL_SHM_LIMIT_PER_NODE", &(MPIR_CVAR_COLL_SHM_LIMIT_PER_NODE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_COLL_SHM_LIMIT_PER_NODE");
    rc = MPL_env2int("MPIR_CVAR_COLL_SHM_LIMIT_PER_NODE", &(MPIR_CVAR_COLL_SHM_LIMIT_PER_NODE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_COLL_SHM_LIMIT_PER_NODE");

    defaultval.d = 32768;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_BCAST_INTRANODE_BUFFER_TOTAL_SIZE, /* name */
        &MPIR_CVAR_BCAST_INTRANODE_BUFFER_TOTAL_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Total size of the bcast buffer (in bytes)");
    MPIR_CVAR_BCAST_INTRANODE_BUFFER_TOTAL_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_BCAST_INTRANODE_BUFFER_TOTAL_SIZE", &(MPIR_CVAR_BCAST_INTRANODE_BUFFER_TOTAL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BCAST_INTRANODE_BUFFER_TOTAL_SIZE");
    rc = MPL_env2int("MPIR_PARAM_BCAST_INTRANODE_BUFFER_TOTAL_SIZE", &(MPIR_CVAR_BCAST_INTRANODE_BUFFER_TOTAL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BCAST_INTRANODE_BUFFER_TOTAL_SIZE");
    rc = MPL_env2int("MPIR_CVAR_BCAST_INTRANODE_BUFFER_TOTAL_SIZE", &(MPIR_CVAR_BCAST_INTRANODE_BUFFER_TOTAL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BCAST_INTRANODE_BUFFER_TOTAL_SIZE");

    defaultval.d = 4;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_BCAST_INTRANODE_NUM_CELLS, /* name */
        &MPIR_CVAR_BCAST_INTRANODE_NUM_CELLS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Number of cells the bcast buffer is divided into");
    MPIR_CVAR_BCAST_INTRANODE_NUM_CELLS = defaultval.d;
    rc = MPL_env2int("MPICH_BCAST_INTRANODE_NUM_CELLS", &(MPIR_CVAR_BCAST_INTRANODE_NUM_CELLS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BCAST_INTRANODE_NUM_CELLS");
    rc = MPL_env2int("MPIR_PARAM_BCAST_INTRANODE_NUM_CELLS", &(MPIR_CVAR_BCAST_INTRANODE_NUM_CELLS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BCAST_INTRANODE_NUM_CELLS");
    rc = MPL_env2int("MPIR_CVAR_BCAST_INTRANODE_NUM_CELLS", &(MPIR_CVAR_BCAST_INTRANODE_NUM_CELLS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BCAST_INTRANODE_NUM_CELLS");

    defaultval.d = 32768;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_INTRANODE_BUFFER_TOTAL_SIZE, /* name */
        &MPIR_CVAR_REDUCE_INTRANODE_BUFFER_TOTAL_SIZE, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Total size of the reduce buffer per rank (in bytes)");
    MPIR_CVAR_REDUCE_INTRANODE_BUFFER_TOTAL_SIZE = defaultval.d;
    rc = MPL_env2int("MPICH_REDUCE_INTRANODE_BUFFER_TOTAL_SIZE", &(MPIR_CVAR_REDUCE_INTRANODE_BUFFER_TOTAL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_INTRANODE_BUFFER_TOTAL_SIZE");
    rc = MPL_env2int("MPIR_PARAM_REDUCE_INTRANODE_BUFFER_TOTAL_SIZE", &(MPIR_CVAR_REDUCE_INTRANODE_BUFFER_TOTAL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_INTRANODE_BUFFER_TOTAL_SIZE");
    rc = MPL_env2int("MPIR_CVAR_REDUCE_INTRANODE_BUFFER_TOTAL_SIZE", &(MPIR_CVAR_REDUCE_INTRANODE_BUFFER_TOTAL_SIZE));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_INTRANODE_BUFFER_TOTAL_SIZE");

    defaultval.d = 4;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_INTRANODE_NUM_CELLS, /* name */
        &MPIR_CVAR_REDUCE_INTRANODE_NUM_CELLS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Number of cells the reduce buffer is divided into, for each rank");
    MPIR_CVAR_REDUCE_INTRANODE_NUM_CELLS = defaultval.d;
    rc = MPL_env2int("MPICH_REDUCE_INTRANODE_NUM_CELLS", &(MPIR_CVAR_REDUCE_INTRANODE_NUM_CELLS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_INTRANODE_NUM_CELLS");
    rc = MPL_env2int("MPIR_PARAM_REDUCE_INTRANODE_NUM_CELLS", &(MPIR_CVAR_REDUCE_INTRANODE_NUM_CELLS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_INTRANODE_NUM_CELLS");
    rc = MPL_env2int("MPIR_CVAR_REDUCE_INTRANODE_NUM_CELLS", &(MPIR_CVAR_REDUCE_INTRANODE_NUM_CELLS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_INTRANODE_NUM_CELLS");

    defaultval.d = 64;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_BCAST_INTRANODE_TREE_KVAL, /* name */
        &MPIR_CVAR_BCAST_INTRANODE_TREE_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "K value for the kary/knomial tree for intra-node bcast");
    MPIR_CVAR_BCAST_INTRANODE_TREE_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_BCAST_INTRANODE_TREE_KVAL", &(MPIR_CVAR_BCAST_INTRANODE_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BCAST_INTRANODE_TREE_KVAL");
    rc = MPL_env2int("MPIR_PARAM_BCAST_INTRANODE_TREE_KVAL", &(MPIR_CVAR_BCAST_INTRANODE_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BCAST_INTRANODE_TREE_KVAL");
    rc = MPL_env2int("MPIR_CVAR_BCAST_INTRANODE_TREE_KVAL", &(MPIR_CVAR_BCAST_INTRANODE_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BCAST_INTRANODE_TREE_KVAL");

    defaultval.str = (const char *) "kary";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_BCAST_INTRANODE_TREE_TYPE, /* name */
        &MPIR_CVAR_BCAST_INTRANODE_TREE_TYPE, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Tree type for intra-node bcast tree kary      - kary tree type knomial_1 - knomial_1 tree type (ranks are added in order from the left side) knomial_2 - knomial_2 tree type (ranks are added in order from the right side)");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_BCAST_INTRANODE_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_BCAST_INTRANODE_TREE_TYPE");
    rc = MPL_env2str("MPIR_PARAM_BCAST_INTRANODE_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_BCAST_INTRANODE_TREE_TYPE");
    rc = MPL_env2str("MPIR_CVAR_BCAST_INTRANODE_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_BCAST_INTRANODE_TREE_TYPE");
    if (tmp_str != NULL) {
        MPIR_CVAR_BCAST_INTRANODE_TREE_TYPE = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_BCAST_INTRANODE_TREE_TYPE);
        if (MPIR_CVAR_BCAST_INTRANODE_TREE_TYPE == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_BCAST_INTRANODE_TREE_TYPE");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_BCAST_INTRANODE_TREE_TYPE = NULL;
    }

    defaultval.d = 4;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_REDUCE_INTRANODE_TREE_KVAL, /* name */
        &MPIR_CVAR_REDUCE_INTRANODE_TREE_KVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "K value for the kary/knomial tree for intra-node reduce");
    MPIR_CVAR_REDUCE_INTRANODE_TREE_KVAL = defaultval.d;
    rc = MPL_env2int("MPICH_REDUCE_INTRANODE_TREE_KVAL", &(MPIR_CVAR_REDUCE_INTRANODE_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_INTRANODE_TREE_KVAL");
    rc = MPL_env2int("MPIR_PARAM_REDUCE_INTRANODE_TREE_KVAL", &(MPIR_CVAR_REDUCE_INTRANODE_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_INTRANODE_TREE_KVAL");
    rc = MPL_env2int("MPIR_CVAR_REDUCE_INTRANODE_TREE_KVAL", &(MPIR_CVAR_REDUCE_INTRANODE_TREE_KVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_INTRANODE_TREE_KVAL");

    defaultval.str = (const char *) "kary";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_REDUCE_INTRANODE_TREE_TYPE, /* name */
        &MPIR_CVAR_REDUCE_INTRANODE_TREE_TYPE, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Tree type for intra-node reduce tree kary      - kary tree type knomial_1 - knomial_1 tree type (ranks are added in order from the left side) knomial_2 - knomial_2 tree type (ranks are added in order from the right side)");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_REDUCE_INTRANODE_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_REDUCE_INTRANODE_TREE_TYPE");
    rc = MPL_env2str("MPIR_PARAM_REDUCE_INTRANODE_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_REDUCE_INTRANODE_TREE_TYPE");
    rc = MPL_env2str("MPIR_CVAR_REDUCE_INTRANODE_TREE_TYPE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_REDUCE_INTRANODE_TREE_TYPE");
    if (tmp_str != NULL) {
        MPIR_CVAR_REDUCE_INTRANODE_TREE_TYPE = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_REDUCE_INTRANODE_TREE_TYPE);
        if (MPIR_CVAR_REDUCE_INTRANODE_TREE_TYPE == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_REDUCE_INTRANODE_TREE_TYPE");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_REDUCE_INTRANODE_TREE_TYPE = NULL;
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ENABLE_INTRANODE_TOPOLOGY_AWARE_TREES, /* name */
        &MPIR_CVAR_ENABLE_INTRANODE_TOPOLOGY_AWARE_TREES, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Enable collective specific intra-node trees which leverage the memory hierarchy of a machine. Depends on hwloc to extract the binding information of each rank. Pick a leader rank per package (socket), then create a per_package tree for ranks on a same package, package leaders tree for package leaders. For Bcast - Assemble the per_package and package_leaders tree in such a way that leaders interact among themselves first before interacting with package local ranks. Both the package_leaders and per_package trees are left skewed (children are added from left to right, first child to be added is the first one to be processed in traversal) For Reduce - Assemble the per_package and package_leaders tree in such a way that a leader rank interacts with its package local ranks first, then with the other package leaders. Both the per_package and package_leaders tree is right skewed (children are added in reverse order, first child to be added is the last one to be processed in traversal) The tree radix and tree type of package_leaders and per_package tree is MPIR_CVAR_BCAST{REDUCE}_INTRANODE_TREE_KVAL and MPIR_CVAR_BCAST{REDUCE}_INTRANODE_TREE_TYPE respectively for bast and reduce. But of as now topology aware trees are only kary. knomial is to be implemented.");
    MPIR_CVAR_ENABLE_INTRANODE_TOPOLOGY_AWARE_TREES = defaultval.d;
    rc = MPL_env2int("MPICH_ENABLE_INTRANODE_TOPOLOGY_AWARE_TREES", &(MPIR_CVAR_ENABLE_INTRANODE_TOPOLOGY_AWARE_TREES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ENABLE_INTRANODE_TOPOLOGY_AWARE_TREES");
    rc = MPL_env2int("MPIR_PARAM_ENABLE_INTRANODE_TOPOLOGY_AWARE_TREES", &(MPIR_CVAR_ENABLE_INTRANODE_TOPOLOGY_AWARE_TREES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ENABLE_INTRANODE_TOPOLOGY_AWARE_TREES");
    rc = MPL_env2int("MPIR_CVAR_ENABLE_INTRANODE_TOPOLOGY_AWARE_TREES", &(MPIR_CVAR_ENABLE_INTRANODE_TOPOLOGY_AWARE_TREES));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ENABLE_INTRANODE_TOPOLOGY_AWARE_TREES");

    defaultval.str = (const char *) "";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_CH4_NETMOD, /* name */
        &MPIR_CVAR_CH4_NETMOD, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH4", /* category */
        "If non-empty, this cvar specifies which network module to use");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_CH4_NETMOD", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_NETMOD");
    rc = MPL_env2str("MPIR_PARAM_CH4_NETMOD", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_NETMOD");
    rc = MPL_env2str("MPIR_CVAR_CH4_NETMOD", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_NETMOD");
    if (tmp_str != NULL) {
        MPIR_CVAR_CH4_NETMOD = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_CH4_NETMOD);
        if (MPIR_CVAR_CH4_NETMOD == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_CH4_NETMOD");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_CH4_NETMOD = NULL;
    }

    defaultval.str = (const char *) "";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_CH4_SHM, /* name */
        &MPIR_CVAR_CH4_SHM, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH4", /* category */
        "If non-empty, this cvar specifies which shm module to use");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_CH4_SHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_SHM");
    rc = MPL_env2str("MPIR_PARAM_CH4_SHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_SHM");
    rc = MPL_env2str("MPIR_CVAR_CH4_SHM", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_SHM");
    if (tmp_str != NULL) {
        MPIR_CVAR_CH4_SHM = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_CH4_SHM);
        if (MPIR_CVAR_CH4_SHM == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_CH4_SHM");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_CH4_SHM = NULL;
    }

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_ROOTS_ONLY_PMI, /* name */
        &MPIR_CVAR_CH4_ROOTS_ONLY_PMI, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4", /* category */
        "Enables an optimized business card exchange over PMI for node root processes only.");
    MPIR_CVAR_CH4_ROOTS_ONLY_PMI = defaultval.d;
    rc = MPL_env2bool("MPICH_CH4_ROOTS_ONLY_PMI", &(MPIR_CVAR_CH4_ROOTS_ONLY_PMI));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_ROOTS_ONLY_PMI");
    rc = MPL_env2bool("MPIR_PARAM_CH4_ROOTS_ONLY_PMI", &(MPIR_CVAR_CH4_ROOTS_ONLY_PMI));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_ROOTS_ONLY_PMI");
    rc = MPL_env2bool("MPIR_CVAR_CH4_ROOTS_ONLY_PMI", &(MPIR_CVAR_CH4_ROOTS_ONLY_PMI));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_ROOTS_ONLY_PMI");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_RUNTIME_CONF_DEBUG, /* name */
        &MPIR_CVAR_CH4_RUNTIME_CONF_DEBUG, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH4", /* category */
        "If enabled, CH4-level runtime configurations are printed out");
    MPIR_CVAR_CH4_RUNTIME_CONF_DEBUG = defaultval.d;
    rc = MPL_env2bool("MPICH_CH4_RUNTIME_CONF_DEBUG", &(MPIR_CVAR_CH4_RUNTIME_CONF_DEBUG));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_RUNTIME_CONF_DEBUG");
    rc = MPL_env2bool("MPIR_PARAM_CH4_RUNTIME_CONF_DEBUG", &(MPIR_CVAR_CH4_RUNTIME_CONF_DEBUG));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_RUNTIME_CONF_DEBUG");
    rc = MPL_env2bool("MPIR_CVAR_CH4_RUNTIME_CONF_DEBUG", &(MPIR_CVAR_CH4_RUNTIME_CONF_DEBUG));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_RUNTIME_CONF_DEBUG");

    defaultval.str = (const char *) "";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_CH4_MT_MODEL, /* name */
        &MPIR_CVAR_CH4_MT_MODEL, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "CH4", /* category */
        "Specifies the CH4 multi-threading model. Possible values are: direct (default) handoff");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_CH4_MT_MODEL", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_MT_MODEL");
    rc = MPL_env2str("MPIR_PARAM_CH4_MT_MODEL", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_MT_MODEL");
    rc = MPL_env2str("MPIR_CVAR_CH4_MT_MODEL", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_MT_MODEL");
    if (tmp_str != NULL) {
        MPIR_CVAR_CH4_MT_MODEL = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_CH4_MT_MODEL);
        if (MPIR_CVAR_CH4_MT_MODEL == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_CH4_MT_MODEL");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_CH4_MT_MODEL = NULL;
    }

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_NUM_VCIS, /* name */
        &MPIR_CVAR_CH4_NUM_VCIS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4", /* category */
        "Sets the number of VCIs that user needs (should be a subset of MPIDI_CH4_MAX_VCIS).");
    MPIR_CVAR_CH4_NUM_VCIS = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_NUM_VCIS", &(MPIR_CVAR_CH4_NUM_VCIS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_NUM_VCIS");
    rc = MPL_env2int("MPIR_PARAM_CH4_NUM_VCIS", &(MPIR_CVAR_CH4_NUM_VCIS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_NUM_VCIS");
    rc = MPL_env2int("MPIR_CVAR_CH4_NUM_VCIS", &(MPIR_CVAR_CH4_NUM_VCIS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_NUM_VCIS");

    defaultval.str = (const char *) "";
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_CHAR,
        MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE, /* name */
        &MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE, /* address */
        MPIR_CVAR_MAX_STRLEN, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Defines the location of tuning file.");
    tmp_str = defaultval.str;
    rc = MPL_env2str("MPICH_CH4_COLL_SELECTION_TUNING_JSON_FILE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_COLL_SELECTION_TUNING_JSON_FILE");
    rc = MPL_env2str("MPIR_PARAM_CH4_COLL_SELECTION_TUNING_JSON_FILE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_COLL_SELECTION_TUNING_JSON_FILE");
    rc = MPL_env2str("MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE", &tmp_str);
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE");
    if (tmp_str != NULL) {
        MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE = MPL_strdup(tmp_str);
        MPIR_CVAR_assert(MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE);
        if (MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE == NULL) {
            MPIR_CHKMEM_SETERR(mpi_errno, strlen(tmp_str), "dup of string for MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE");
            goto fn_fail;
        }
    }
    else {
        MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE = NULL;
    }

    defaultval.d = 16384;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_IOV_DENSITY_MIN, /* name */
        &MPIR_CVAR_CH4_IOV_DENSITY_MIN, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4", /* category */
        "Defines the threshold of high-density datatype. The density is calculated by (datatype_size / datatype_num_contig_blocks).");
    MPIR_CVAR_CH4_IOV_DENSITY_MIN = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_IOV_DENSITY_MIN", &(MPIR_CVAR_CH4_IOV_DENSITY_MIN));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_IOV_DENSITY_MIN");
    rc = MPL_env2int("MPIR_PARAM_CH4_IOV_DENSITY_MIN", &(MPIR_CVAR_CH4_IOV_DENSITY_MIN));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_IOV_DENSITY_MIN");
    rc = MPL_env2int("MPIR_CVAR_CH4_IOV_DENSITY_MIN", &(MPIR_CVAR_CH4_IOV_DENSITY_MIN));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_IOV_DENSITY_MIN");

    defaultval.d = 180;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_COMM_CONNECT_TIMEOUT, /* name */
        &MPIR_CVAR_CH4_COMM_CONNECT_TIMEOUT, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_GROUP_EQ,
        defaultval,
        "CH4", /* category */
        "The default time out period in seconds for a connection attempt to the server communicator where the named port exists but no pending accept. User can change the value for a specified connection through its info argument.");
    MPIR_CVAR_CH4_COMM_CONNECT_TIMEOUT = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_COMM_CONNECT_TIMEOUT", &(MPIR_CVAR_CH4_COMM_CONNECT_TIMEOUT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_COMM_CONNECT_TIMEOUT");
    rc = MPL_env2int("MPIR_PARAM_CH4_COMM_CONNECT_TIMEOUT", &(MPIR_CVAR_CH4_COMM_CONNECT_TIMEOUT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_COMM_CONNECT_TIMEOUT");
    rc = MPL_env2int("MPIR_CVAR_CH4_COMM_CONNECT_TIMEOUT", &(MPIR_CVAR_CH4_COMM_CONNECT_TIMEOUT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_COMM_CONNECT_TIMEOUT");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_RMA_MEM_EFFICIENT, /* name */
        &MPIR_CVAR_CH4_RMA_MEM_EFFICIENT, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_GROUP_EQ,
        defaultval,
        "CH4", /* category */
        "If true, memory-saving mode is on, per-target object is released at the epoch end call. If false, performance-efficient mode is on, all allocated target objects are cached and freed at win_finalize.");
    MPIR_CVAR_CH4_RMA_MEM_EFFICIENT = defaultval.d;
    rc = MPL_env2bool("MPICH_CH4_RMA_MEM_EFFICIENT", &(MPIR_CVAR_CH4_RMA_MEM_EFFICIENT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_RMA_MEM_EFFICIENT");
    rc = MPL_env2bool("MPIR_PARAM_CH4_RMA_MEM_EFFICIENT", &(MPIR_CVAR_CH4_RMA_MEM_EFFICIENT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_RMA_MEM_EFFICIENT");
    rc = MPL_env2bool("MPIR_CVAR_CH4_RMA_MEM_EFFICIENT", &(MPIR_CVAR_CH4_RMA_MEM_EFFICIENT));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_RMA_MEM_EFFICIENT");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS, /* name */
        &MPIR_CVAR_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4", /* category */
        "If true, allows RMA synchronization calls to dynamically reduce the frequency of internal progress polling for incoming RMA active messages received on the target process. The RMA synchronization call initially polls progress with a low frequency (defined by MPIR_CVAR_CH4_RMA_AM_PROGRESS_LOW_FREQ_INTERVAL) to reduce synchronization overhead. Once any RMA active message has been received, it will always poll progress once at every synchronization call to ensure prompt target-side progress. Effective only for passive target synchronization MPI_Win_flush{_all} and MPI_Win_flush_local{_all}.");
    MPIR_CVAR_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS = defaultval.d;
    rc = MPL_env2bool("MPICH_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS", &(MPIR_CVAR_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS");
    rc = MPL_env2bool("MPIR_PARAM_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS", &(MPIR_CVAR_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS");
    rc = MPL_env2bool("MPIR_CVAR_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS", &(MPIR_CVAR_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS");

    defaultval.d = 1;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_RMA_AM_PROGRESS_INTERVAL, /* name */
        &MPIR_CVAR_CH4_RMA_AM_PROGRESS_INTERVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4", /* category */
        "Specifies a static interval of progress polling for incoming RMA active messages received on the target process. Effective only for passive-target synchronization MPI_Win_flush{_all} and MPI_Win_flush_local{_all}. Interval indicates the number of performed flush calls before polling. It is counted globally across all windows. Invalid when MPIR_CVAR_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS is true.");
    MPIR_CVAR_CH4_RMA_AM_PROGRESS_INTERVAL = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_RMA_AM_PROGRESS_INTERVAL", &(MPIR_CVAR_CH4_RMA_AM_PROGRESS_INTERVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_RMA_AM_PROGRESS_INTERVAL");
    rc = MPL_env2int("MPIR_PARAM_CH4_RMA_AM_PROGRESS_INTERVAL", &(MPIR_CVAR_CH4_RMA_AM_PROGRESS_INTERVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_RMA_AM_PROGRESS_INTERVAL");
    rc = MPL_env2int("MPIR_CVAR_CH4_RMA_AM_PROGRESS_INTERVAL", &(MPIR_CVAR_CH4_RMA_AM_PROGRESS_INTERVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_RMA_AM_PROGRESS_INTERVAL");

    defaultval.d = 100;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_CH4_RMA_AM_PROGRESS_LOW_FREQ_INTERVAL, /* name */
        &MPIR_CVAR_CH4_RMA_AM_PROGRESS_LOW_FREQ_INTERVAL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "CH4", /* category */
        "Specifies the interval of progress polling with low frequency for incoming RMA active message received on the target process. Effective only for passive-target synchronization MPI_Win_flush{_all} and MPI_Win_flush_local{_all}. Interval indicates the number of performed flush calls before polling. It is counted globally across all windows. Used when MPIR_CVAR_CH4_RMA_ENABLE_DYNAMIC_AM_PROGRESS is true.");
    MPIR_CVAR_CH4_RMA_AM_PROGRESS_LOW_FREQ_INTERVAL = defaultval.d;
    rc = MPL_env2int("MPICH_CH4_RMA_AM_PROGRESS_LOW_FREQ_INTERVAL", &(MPIR_CVAR_CH4_RMA_AM_PROGRESS_LOW_FREQ_INTERVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_CH4_RMA_AM_PROGRESS_LOW_FREQ_INTERVAL");
    rc = MPL_env2int("MPIR_PARAM_CH4_RMA_AM_PROGRESS_LOW_FREQ_INTERVAL", &(MPIR_CVAR_CH4_RMA_AM_PROGRESS_LOW_FREQ_INTERVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_CH4_RMA_AM_PROGRESS_LOW_FREQ_INTERVAL");
    rc = MPL_env2int("MPIR_CVAR_CH4_RMA_AM_PROGRESS_LOW_FREQ_INTERVAL", &(MPIR_CVAR_CH4_RMA_AM_PROGRESS_LOW_FREQ_INTERVAL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_CH4_RMA_AM_PROGRESS_LOW_FREQ_INTERVAL");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_ENABLE_HCOLL, /* name */
        &MPIR_CVAR_ENABLE_HCOLL, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_LOCAL,
        defaultval,
        "COLLECTIVE", /* category */
        "Enable hcoll collective support.");
    MPIR_CVAR_ENABLE_HCOLL = defaultval.d;
    rc = MPL_env2bool("MPICH_ENABLE_HCOLL", &(MPIR_CVAR_ENABLE_HCOLL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_ENABLE_HCOLL");
    rc = MPL_env2bool("MPIR_PARAM_ENABLE_HCOLL", &(MPIR_CVAR_ENABLE_HCOLL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_ENABLE_HCOLL");
    rc = MPL_env2bool("MPIR_CVAR_ENABLE_HCOLL", &(MPIR_CVAR_ENABLE_HCOLL));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_ENABLE_HCOLL");

    defaultval.d = 0;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_COLL_SCHED_DUMP, /* name */
        &MPIR_CVAR_COLL_SCHED_DUMP, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_ALL_EQ,
        defaultval,
        "COLLECTIVE", /* category */
        "Print schedule data for nonblocking collective operations.");
    MPIR_CVAR_COLL_SCHED_DUMP = defaultval.d;
    rc = MPL_env2bool("MPICH_COLL_SCHED_DUMP", &(MPIR_CVAR_COLL_SCHED_DUMP));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_COLL_SCHED_DUMP");
    rc = MPL_env2bool("MPIR_PARAM_COLL_SCHED_DUMP", &(MPIR_CVAR_COLL_SCHED_DUMP));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_COLL_SCHED_DUMP");
    rc = MPL_env2bool("MPIR_CVAR_COLL_SCHED_DUMP", &(MPIR_CVAR_COLL_SCHED_DUMP));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_COLL_SCHED_DUMP");

    defaultval.d = 100;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_SHM_RANDOM_ADDR_RETRY, /* name */
        &MPIR_CVAR_SHM_RANDOM_ADDR_RETRY, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_GROUP,
        defaultval,
        "MEMORY", /* category */
        "The default number of retries for generating a random address. A retrying involves only local operations.");
    MPIR_CVAR_SHM_RANDOM_ADDR_RETRY = defaultval.d;
    rc = MPL_env2int("MPICH_SHM_RANDOM_ADDR_RETRY", &(MPIR_CVAR_SHM_RANDOM_ADDR_RETRY));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_SHM_RANDOM_ADDR_RETRY");
    rc = MPL_env2int("MPIR_PARAM_SHM_RANDOM_ADDR_RETRY", &(MPIR_CVAR_SHM_RANDOM_ADDR_RETRY));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_SHM_RANDOM_ADDR_RETRY");
    rc = MPL_env2int("MPIR_CVAR_SHM_RANDOM_ADDR_RETRY", &(MPIR_CVAR_SHM_RANDOM_ADDR_RETRY));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_SHM_RANDOM_ADDR_RETRY");

    defaultval.d = 100;
    MPIR_T_CVAR_REGISTER_STATIC(
        MPI_INT,
        MPIR_CVAR_SHM_SYMHEAP_RETRY, /* name */
        &MPIR_CVAR_SHM_SYMHEAP_RETRY, /* address */
        1, /* count */
        MPI_T_VERBOSITY_USER_BASIC,
        MPI_T_SCOPE_GROUP,
        defaultval,
        "MEMORY", /* category */
        "The default number of retries for allocating a symmetric heap in shared memory. A retrying involves collective communication over the group in the shared memory.");
    MPIR_CVAR_SHM_SYMHEAP_RETRY = defaultval.d;
    rc = MPL_env2int("MPICH_SHM_SYMHEAP_RETRY", &(MPIR_CVAR_SHM_SYMHEAP_RETRY));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPICH_SHM_SYMHEAP_RETRY");
    rc = MPL_env2int("MPIR_PARAM_SHM_SYMHEAP_RETRY", &(MPIR_CVAR_SHM_SYMHEAP_RETRY));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_PARAM_SHM_SYMHEAP_RETRY");
    rc = MPL_env2int("MPIR_CVAR_SHM_SYMHEAP_RETRY", &(MPIR_CVAR_SHM_SYMHEAP_RETRY));
    MPIR_ERR_CHKANDJUMP1((-1 == rc),mpi_errno,MPI_ERR_OTHER,"**envvarparse","**envvarparse %s","MPIR_CVAR_SHM_SYMHEAP_RETRY");

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

int MPIR_T_cvar_finalize(void)
{
    int mpi_errno = MPI_SUCCESS;

    MPL_free((char *)MPIR_CVAR_IALLREDUCE_TREE_TYPE);
    MPIR_CVAR_IALLREDUCE_TREE_TYPE = NULL;

    MPL_free((char *)MPIR_CVAR_IBCAST_TREE_TYPE);
    MPIR_CVAR_IBCAST_TREE_TYPE = NULL;

    MPL_free((char *)MPIR_CVAR_IREDUCE_TREE_TYPE);
    MPIR_CVAR_IREDUCE_TREE_TYPE = NULL;

    MPL_free((char *)MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE);
    MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE = NULL;

    MPL_free((char *)MPIR_CVAR_DEFAULT_THREAD_LEVEL);
    MPIR_CVAR_DEFAULT_THREAD_LEVEL = NULL;

    MPL_free((char *)MPIR_CVAR_NAMESERV_FILE_PUBDIR);
    MPIR_CVAR_NAMESERV_FILE_PUBDIR = NULL;

    MPL_free((char *)MPIR_CVAR_NETLOC_NODE_FILE);
    MPIR_CVAR_NETLOC_NODE_FILE = NULL;

    MPL_free((char *)MPIR_CVAR_OFI_USE_PROVIDER);
    MPIR_CVAR_OFI_USE_PROVIDER = NULL;

    MPL_free((char *)MPIR_CVAR_CH3_INTERFACE_HOSTNAME);
    MPIR_CVAR_CH3_INTERFACE_HOSTNAME = NULL;

    MPL_free((char *)MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE);
    MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE = NULL;

    MPL_free((char *)MPIR_CVAR_NEMESIS_NETMOD);
    MPIR_CVAR_NEMESIS_NETMOD = NULL;

    MPL_free((char *)MPIR_CVAR_CH4_SHM_POSIX_EAGER);
    MPIR_CVAR_CH4_SHM_POSIX_EAGER = NULL;

    MPL_free((char *)MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE);
    MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE = NULL;

    MPL_free((char *)MPIR_CVAR_BCAST_INTRANODE_TREE_TYPE);
    MPIR_CVAR_BCAST_INTRANODE_TREE_TYPE = NULL;

    MPL_free((char *)MPIR_CVAR_REDUCE_INTRANODE_TREE_TYPE);
    MPIR_CVAR_REDUCE_INTRANODE_TREE_TYPE = NULL;

    MPL_free((char *)MPIR_CVAR_CH4_NETMOD);
    MPIR_CVAR_CH4_NETMOD = NULL;

    MPL_free((char *)MPIR_CVAR_CH4_SHM);
    MPIR_CVAR_CH4_SHM = NULL;

    MPL_free((char *)MPIR_CVAR_CH4_MT_MODEL);
    MPIR_CVAR_CH4_MT_MODEL = NULL;

    MPL_free((char *)MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE);
    MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE = NULL;

    return mpi_errno;
}

int MPIR_MPIR_CVAR_GROUP_COLL_ALGO_from_str(const char *s) {
    if (strcmp(s, "    MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM_mpir")==0) return     MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM_mpir;
    else if (strcmp(s, "    MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM_release_gather")==0) return     MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM_release_gather;
    else if (strcmp(s, "    MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM_auto")==0) return     MPIR_CVAR_BARRIER_POSIX_INTRA_ALGORITHM_auto;
    else return -1;
}
