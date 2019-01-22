/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2010 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

/* Automatically generated
 *   by:   ./maint/extractcvars
 *   on:   Wed Nov 21 11:33:20 2018
 *
 * DO NOT EDIT!!!
 */

#if !defined(MPIR_CVARS_H_INCLUDED)
#define MPIR_CVARS_H_INCLUDED

#include "mpitimpl.h" /* for MPIR_T_cvar_range_value_t */

/* Initializes cvar values from the environment */
int MPIR_T_cvar_init(void);
int MPIR_T_cvar_finalize(void);

/* Extern declarations for each cvar
 * (definitions in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/util/cvar/mpir_cvars.c) */

/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/errhan/errutil.c */
extern int MPIR_CVAR_PRINT_ERROR_STACK;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/errhan/errutil.c */
extern int MPIR_CVAR_CHOP_ERROR_STACK;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/topo/dims_create.c */
extern int MPIR_CVAR_DIMS_VERBOSE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/exscan/exscan.c */
extern char * MPIR_CVAR_EXSCAN_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/exscan/exscan.c */
extern int MPIR_CVAR_EXSCAN_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce_scatter/ireduce_scatter.c */
extern int MPIR_CVAR_IREDUCE_SCATTER_RECEXCH_KVAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce_scatter/ireduce_scatter.c */
extern char * MPIR_CVAR_IREDUCE_SCATTER_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce_scatter/ireduce_scatter.c */
extern char * MPIR_CVAR_IREDUCE_SCATTER_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce_scatter/ireduce_scatter.c */
extern int MPIR_CVAR_IREDUCE_SCATTER_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ialltoallw/ialltoallw.c */
extern char * MPIR_CVAR_IALLTOALLW_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ialltoallw/ialltoallw.c */
extern char * MPIR_CVAR_IALLTOALLW_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ialltoallw/ialltoallw.c */
extern int MPIR_CVAR_IALLTOALLW_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_alltoallw/neighbor_alltoallw.c */
extern char * MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_alltoallw/neighbor_alltoallw.c */
extern char * MPIR_CVAR_NEIGHBOR_ALLTOALLW_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_alltoallw/neighbor_alltoallw.c */
extern int MPIR_CVAR_NEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ialltoallv/ialltoallv.c */
extern char * MPIR_CVAR_IALLTOALLV_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ialltoallv/ialltoallv.c */
extern char * MPIR_CVAR_IALLTOALLV_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ialltoallv/ialltoallv.c */
extern int MPIR_CVAR_IALLTOALLV_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allgatherv/allgatherv.c */
extern int MPIR_CVAR_ALLGATHERV_PIPELINE_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allgatherv/allgatherv.c */
extern char * MPIR_CVAR_ALLGATHERV_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allgatherv/allgatherv.c */
extern char * MPIR_CVAR_ALLGATHERV_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allgatherv/allgatherv.c */
extern int MPIR_CVAR_ALLGATHERV_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allreduce/allreduce.c */
extern int MPIR_CVAR_ALLREDUCE_SHORT_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allreduce/allreduce.c */
extern int MPIR_CVAR_ENABLE_SMP_COLLECTIVES;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allreduce/allreduce.c */
extern int MPIR_CVAR_ENABLE_SMP_ALLREDUCE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allreduce/allreduce.c */
extern int MPIR_CVAR_MAX_SMP_ALLREDUCE_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allreduce/allreduce.c */
extern char * MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allreduce/allreduce.c */
extern char * MPIR_CVAR_ALLREDUCE_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allreduce/allreduce.c */
extern int MPIR_CVAR_ALLREDUCE_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/igather/igather.c */
extern char * MPIR_CVAR_IGATHER_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/igather/igather.c */
extern int MPIR_CVAR_IGATHER_TREE_KVAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/igather/igather.c */
extern char * MPIR_CVAR_IGATHER_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/igather/igather.c */
extern int MPIR_CVAR_IGATHER_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_alltoall/neighbor_alltoall.c */
extern char * MPIR_CVAR_NEIGHBOR_ALLTOALL_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_alltoall/neighbor_alltoall.c */
extern char * MPIR_CVAR_NEIGHBOR_ALLTOALL_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_alltoall/neighbor_alltoall.c */
extern int MPIR_CVAR_NEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ialltoall/ialltoall.c */
extern char * MPIR_CVAR_IALLTOALL_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ialltoall/ialltoall.c */
extern char * MPIR_CVAR_IALLTOALL_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ialltoall/ialltoall.c */
extern int MPIR_CVAR_IALLTOALL_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/scatter/scatter.c */
extern int MPIR_CVAR_SCATTER_INTER_SHORT_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/scatter/scatter.c */
extern char * MPIR_CVAR_SCATTER_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/scatter/scatter.c */
extern char * MPIR_CVAR_SCATTER_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/scatter/scatter.c */
extern int MPIR_CVAR_SCATTER_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_allgatherv/neighbor_allgatherv.c */
extern char * MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_allgatherv/neighbor_allgatherv.c */
extern char * MPIR_CVAR_NEIGHBOR_ALLGATHERV_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_allgatherv/neighbor_allgatherv.c */
extern int MPIR_CVAR_NEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce/ireduce.c */
extern int MPIR_CVAR_IREDUCE_TREE_KVAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce/ireduce.c */
extern char * MPIR_CVAR_IREDUCE_TREE_TYPE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce/ireduce.c */
extern int MPIR_CVAR_IREDUCE_TREE_PIPELINE_CHUNK_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce/ireduce.c */
extern int MPIR_CVAR_IREDUCE_RING_CHUNK_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce/ireduce.c */
extern int MPIR_CVAR_IREDUCE_TREE_BUFFER_PER_CHILD;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce/ireduce.c */
extern char * MPIR_CVAR_IREDUCE_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce/ireduce.c */
extern char * MPIR_CVAR_IREDUCE_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce/ireduce.c */
extern int MPIR_CVAR_IREDUCE_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/reduce_scatter_block/reduce_scatter_block.c */
extern char * MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/reduce_scatter_block/reduce_scatter_block.c */
extern char * MPIR_CVAR_REDUCE_SCATTER_BLOCK_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/reduce_scatter_block/reduce_scatter_block.c */
extern int MPIR_CVAR_REDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/gatherv/gatherv_allcomm_linear.c */
extern int MPIR_CVAR_GATHERV_INTER_SSEND_MIN_PROCS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/gatherv/gatherv.c */
extern char * MPIR_CVAR_GATHERV_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/gatherv/gatherv.c */
extern char * MPIR_CVAR_GATHERV_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/gatherv/gatherv.c */
extern int MPIR_CVAR_GATHERV_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iscatter/iscatter.c */
extern char * MPIR_CVAR_ISCATTER_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iscatter/iscatter.c */
extern int MPIR_CVAR_ISCATTER_TREE_KVAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iscatter/iscatter.c */
extern char * MPIR_CVAR_ISCATTER_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iscatter/iscatter.c */
extern int MPIR_CVAR_ISCATTER_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/transports/gentran/gentran_impl.c */
extern int MPIR_CVAR_PROGRESS_MAX_COLLS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_allgather/neighbor_allgather.c */
extern char * MPIR_CVAR_NEIGHBOR_ALLGATHER_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_allgather/neighbor_allgather.c */
extern char * MPIR_CVAR_NEIGHBOR_ALLGATHER_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_allgather/neighbor_allgather.c */
extern int MPIR_CVAR_NEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allgather/allgather.c */
extern int MPIR_CVAR_ALLGATHER_SHORT_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allgather/allgather.c */
extern int MPIR_CVAR_ALLGATHER_LONG_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allgather/allgather.c */
extern char * MPIR_CVAR_ALLGATHER_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allgather/allgather.c */
extern char * MPIR_CVAR_ALLGATHER_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/allgather/allgather.c */
extern int MPIR_CVAR_ALLGATHER_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/alltoallv/alltoallv.c */
extern char * MPIR_CVAR_ALLTOALLV_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/alltoallv/alltoallv.c */
extern char * MPIR_CVAR_ALLTOALLV_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/alltoallv/alltoallv.c */
extern int MPIR_CVAR_ALLTOALLV_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/alltoall/alltoall.c */
extern int MPIR_CVAR_ALLTOALL_SHORT_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/alltoall/alltoall.c */
extern int MPIR_CVAR_ALLTOALL_MEDIUM_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/alltoall/alltoall.c */
extern int MPIR_CVAR_ALLTOALL_THROTTLE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/alltoall/alltoall.c */
extern char * MPIR_CVAR_ALLTOALL_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/alltoall/alltoall.c */
extern char * MPIR_CVAR_ALLTOALL_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/alltoall/alltoall.c */
extern int MPIR_CVAR_ALLTOALL_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallreduce/iallreduce.c */
extern int MPIR_CVAR_IALLREDUCE_TREE_KVAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallreduce/iallreduce.c */
extern int MPIR_CVAR_IALLREDUCE_TREE_PIPELINE_CHUNK_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallreduce/iallreduce.c */
extern int MPIR_CVAR_IALLREDUCE_TREE_BUFFER_PER_CHILD;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallreduce/iallreduce.c */
extern int MPIR_CVAR_IALLREDUCE_RECEXCH_KVAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallreduce/iallreduce.c */
extern char * MPIR_CVAR_IALLREDUCE_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallreduce/iallreduce.c */
extern char * MPIR_CVAR_IALLREDUCE_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallreduce/iallreduce.c */
extern int MPIR_CVAR_IALLREDUCE_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iexscan/iexscan.c */
extern char * MPIR_CVAR_IEXSCAN_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iexscan/iexscan.c */
extern int MPIR_CVAR_IEXSCAN_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_alltoallw/ineighbor_alltoallw.c */
extern char * MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_alltoallw/ineighbor_alltoallw.c */
extern char * MPIR_CVAR_INEIGHBOR_ALLTOALLW_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_alltoallw/ineighbor_alltoallw.c */
extern int MPIR_CVAR_INEIGHBOR_ALLTOALLW_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/reduce_scatter/reduce_scatter.c */
extern int MPIR_CVAR_REDUCE_SCATTER_COMMUTATIVE_LONG_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/reduce_scatter/reduce_scatter.c */
extern char * MPIR_CVAR_REDUCE_SCATTER_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/reduce_scatter/reduce_scatter.c */
extern char * MPIR_CVAR_REDUCE_SCATTER_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/reduce_scatter/reduce_scatter.c */
extern int MPIR_CVAR_REDUCE_SCATTER_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_alltoallv/ineighbor_alltoallv.c */
extern char * MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_alltoallv/ineighbor_alltoallv.c */
extern char * MPIR_CVAR_INEIGHBOR_ALLTOALLV_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_alltoallv/ineighbor_alltoallv.c */
extern int MPIR_CVAR_INEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_alltoallv/neighbor_alltoallv.c */
extern char * MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_alltoallv/neighbor_alltoallv.c */
extern char * MPIR_CVAR_NEIGHBOR_ALLTOALLV_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/neighbor_alltoallv/neighbor_alltoallv.c */
extern int MPIR_CVAR_NEIGHBOR_ALLTOALLV_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ibarrier/ibarrier.c */
extern int MPIR_CVAR_IBARRIER_RECEXCH_KVAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ibarrier/ibarrier.c */
extern char * MPIR_CVAR_IBARRIER_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ibarrier/ibarrier.c */
extern char * MPIR_CVAR_IBARRIER_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ibarrier/ibarrier.c */
extern int MPIR_CVAR_IBARRIER_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/bcast/bcast.c */
extern int MPIR_CVAR_BCAST_MIN_PROCS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/bcast/bcast.c */
extern int MPIR_CVAR_BCAST_SHORT_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/bcast/bcast.c */
extern int MPIR_CVAR_BCAST_LONG_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/bcast/bcast.c */
extern int MPIR_CVAR_ENABLE_SMP_BCAST;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/bcast/bcast.c */
extern int MPIR_CVAR_MAX_SMP_BCAST_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/bcast/bcast.c */
extern char * MPIR_CVAR_BCAST_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/bcast/bcast.c */
extern char * MPIR_CVAR_BCAST_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/bcast/bcast.c */
extern int MPIR_CVAR_BCAST_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/igatherv/igatherv.c */
extern char * MPIR_CVAR_IGATHERV_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/igatherv/igatherv.c */
extern char * MPIR_CVAR_IGATHERV_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/igatherv/igatherv.c */
extern int MPIR_CVAR_IGATHERV_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallgatherv/iallgatherv.c */
extern int MPIR_CVAR_IALLGATHERV_RECEXCH_KVAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallgatherv/iallgatherv.c */
extern char * MPIR_CVAR_IALLGATHERV_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallgatherv/iallgatherv.c */
extern char * MPIR_CVAR_IALLGATHERV_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallgatherv/iallgatherv.c */
extern int MPIR_CVAR_IALLGATHERV_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iscan/iscan.c */
extern char * MPIR_CVAR_ISCAN_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iscan/iscan.c */
extern int MPIR_CVAR_ISCAN_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/scan/scan.c */
extern char * MPIR_CVAR_SCAN_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/scan/scan.c */
extern int MPIR_CVAR_SCAN_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_alltoall/ineighbor_alltoall.c */
extern char * MPIR_CVAR_INEIGHBOR_ALLTOALL_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_alltoall/ineighbor_alltoall.c */
extern char * MPIR_CVAR_INEIGHBOR_ALLTOALL_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_alltoall/ineighbor_alltoall.c */
extern int MPIR_CVAR_INEIGHBOR_ALLTOALL_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/reduce/reduce.c */
extern int MPIR_CVAR_REDUCE_SHORT_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/reduce/reduce.c */
extern int MPIR_CVAR_ENABLE_SMP_REDUCE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/reduce/reduce.c */
extern int MPIR_CVAR_MAX_SMP_REDUCE_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/reduce/reduce.c */
extern char * MPIR_CVAR_REDUCE_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/reduce/reduce.c */
extern char * MPIR_CVAR_REDUCE_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/reduce/reduce.c */
extern int MPIR_CVAR_REDUCE_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/alltoallw/alltoallw.c */
extern char * MPIR_CVAR_ALLTOALLW_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/alltoallw/alltoallw.c */
extern char * MPIR_CVAR_ALLTOALLW_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/alltoallw/alltoallw.c */
extern int MPIR_CVAR_ALLTOALLW_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/barrier/barrier.c */
extern int MPIR_CVAR_ENABLE_SMP_BARRIER;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/barrier/barrier.c */
extern char * MPIR_CVAR_BARRIER_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/barrier/barrier.c */
extern char * MPIR_CVAR_BARRIER_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/barrier/barrier.c */
extern int MPIR_CVAR_BARRIER_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iscatterv/iscatterv.c */
extern char * MPIR_CVAR_ISCATTERV_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iscatterv/iscatterv.c */
extern char * MPIR_CVAR_ISCATTERV_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iscatterv/iscatterv.c */
extern int MPIR_CVAR_ISCATTERV_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/src/coll_impl.c */
extern int MPIR_CVAR_DEVICE_COLLECTIVES;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_allgatherv/ineighbor_allgatherv.c */
extern char * MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_allgatherv/ineighbor_allgatherv.c */
extern char * MPIR_CVAR_INEIGHBOR_ALLGATHERV_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_allgatherv/ineighbor_allgatherv.c */
extern int MPIR_CVAR_INEIGHBOR_ALLGATHERV_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/scatterv/scatterv.c */
extern char * MPIR_CVAR_SCATTERV_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/scatterv/scatterv.c */
extern char * MPIR_CVAR_SCATTERV_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/scatterv/scatterv.c */
extern int MPIR_CVAR_SCATTERV_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/gather/gather_intra_binomial.c */
extern int MPIR_CVAR_GATHER_VSMALL_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/gather/gather.c */
extern int MPIR_CVAR_GATHER_INTER_SHORT_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/gather/gather.c */
extern char * MPIR_CVAR_GATHER_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/gather/gather.c */
extern char * MPIR_CVAR_GATHER_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/gather/gather.c */
extern int MPIR_CVAR_GATHER_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce_scatter_block/ireduce_scatter_block.c */
extern int MPIR_CVAR_IREDUCE_SCATTER_BLOCK_RECEXCH_KVAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce_scatter_block/ireduce_scatter_block.c */
extern char * MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce_scatter_block/ireduce_scatter_block.c */
extern char * MPIR_CVAR_IREDUCE_SCATTER_BLOCK_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ireduce_scatter_block/ireduce_scatter_block.c */
extern int MPIR_CVAR_IREDUCE_SCATTER_BLOCK_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallgather/iallgather.c */
extern int MPIR_CVAR_IALLGATHER_RECEXCH_KVAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallgather/iallgather.c */
extern int MPIR_CVAR_IALLGATHER_BRUCKS_KVAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallgather/iallgather.c */
extern char * MPIR_CVAR_IALLGATHER_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallgather/iallgather.c */
extern char * MPIR_CVAR_IALLGATHER_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/iallgather/iallgather.c */
extern int MPIR_CVAR_IALLGATHER_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ibcast/ibcast.c */
extern int MPIR_CVAR_IBCAST_TREE_KVAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ibcast/ibcast.c */
extern char * MPIR_CVAR_IBCAST_TREE_TYPE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ibcast/ibcast.c */
extern int MPIR_CVAR_IBCAST_TREE_PIPELINE_CHUNK_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ibcast/ibcast.c */
extern int MPIR_CVAR_IBCAST_RING_CHUNK_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ibcast/ibcast.c */
extern char * MPIR_CVAR_IBCAST_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ibcast/ibcast.c */
extern int MPIR_CVAR_IBCAST_SCATTER_KVAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ibcast/ibcast.c */
extern int MPIR_CVAR_IBCAST_ALLGATHER_RECEXCH_KVAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ibcast/ibcast.c */
extern char * MPIR_CVAR_IBCAST_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ibcast/ibcast.c */
extern int MPIR_CVAR_IBCAST_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_allgather/ineighbor_allgather.c */
extern char * MPIR_CVAR_INEIGHBOR_ALLGATHER_INTRA_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_allgather/ineighbor_allgather.c */
extern char * MPIR_CVAR_INEIGHBOR_ALLGATHER_INTER_ALGORITHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/coll/ineighbor_allgather/ineighbor_allgather.c */
extern int MPIR_CVAR_INEIGHBOR_ALLGATHER_DEVICE_COLLECTIVE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/init/finalize.c */
extern int MPIR_CVAR_MEMDUMP;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/init/finalize.c */
extern int MPIR_CVAR_MEM_CATEGORY_INFORMATION;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/init/initthread.c */
extern int MPIR_CVAR_DEBUG_HOLD;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/init/initthread.c */
extern int MPIR_CVAR_ERROR_CHECKING;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/init/initthread.c */
extern char * MPIR_CVAR_NETLOC_NODE_FILE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/init/abort.c */
extern int MPIR_CVAR_SUPPRESS_ABORT_MESSAGE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/init/init.c */
extern int MPIR_CVAR_ASYNC_PROGRESS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/init/init.c */
extern char * MPIR_CVAR_DEFAULT_THREAD_LEVEL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/comm/comm_split.c */
extern int MPIR_CVAR_COMM_SPLIT_USE_QSORT;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/comm/contextid.c */
extern int MPIR_CVAR_CTXID_EAGER_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/debugger/dbginit.c */
extern int MPIR_CVAR_PROCTABLE_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpi/debugger/dbginit.c */
extern int MPIR_CVAR_PROCTABLE_PRINT;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/nameserv/file/file_nameserv.c */
extern char * MPIR_CVAR_NAMESERV_FILE_PUBDIR;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/util/mem/handlemem.c */
extern int MPIR_CVAR_ABORT_ON_LEAKED_HANDLES;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/util/nodemap/build_nodemap.h */
extern int MPIR_CVAR_NOLOCAL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/util/nodemap/build_nodemap.h */
extern int MPIR_CVAR_ODD_EVEN_CLIQUES;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/util/nodemap/build_nodemap.h */
extern int MPIR_CVAR_NUM_CLIQUES;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/include/mpir_request.h */
extern int MPIR_CVAR_REQUEST_POLL_FREQ;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/include/mpir_request.h */
extern int MPIR_CVAR_REQUEST_BATCH_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/include/mpir_err.h */
extern int MPIR_CVAR_COLL_ALIAS_CHECK;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_ENABLE_DATA;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_ENABLE_AV_TABLE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_ENABLE_SHARED_CONTEXTS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_ENABLE_MR_SCALABLE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_ENABLE_TAGGED;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_ENABLE_AM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_ENABLE_RMA;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_ENABLE_ATOMICS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_FETCH_ATOMIC_IOVECS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_ENABLE_CONTROL_AUTO_PROGRESS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_ENABLE_PT2PT_NOPACK;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_CONTEXT_ID_BITS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_RANK_BITS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_TAG_BITS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_MAJOR_VERSION;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_MINOR_VERSION;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_MAX_VNIS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_MAX_RMA_SEP_CTX;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_MAX_EAGAIN_RETRY;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/netmod/ofi/ofi_init.h */
extern int MPIR_CVAR_CH4_OFI_NUM_AM_BUFFERS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/src/ch4_spawn.h */
extern int MPIR_CVAR_CH4_COMM_CONNECT_TIMEOUT;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/src/ch4r_win.h */
extern int MPIR_CVAR_CH4_RMA_MEM_EFFICIENT;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/src/ch4_init.h */
extern char * MPIR_CVAR_CH4_NETMOD;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/src/ch4_init.h */
extern char * MPIR_CVAR_CH4_SHM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/src/ch4_init.h */
extern int MPIR_CVAR_CH4_ROOTS_ONLY_PMI;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/src/ch4_init.h */
extern int MPIR_CVAR_CH4_RUNTIME_CONF_DEBUG;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/src/ch4_init.h */
extern char * MPIR_CVAR_CH4_MT_MODEL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/src/ch4r_symheap.h */
extern int MPIR_CVAR_CH4_RANDOM_ADDR_RETRY;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch4/src/ch4r_symheap.h */
extern int MPIR_CVAR_CH4_SHM_SYMHEAP_RETRY;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/common/hcoll/hcoll_init.c */
extern int MPIR_CVAR_ENABLE_HCOLL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/common/sched/mpidu_sched.c */
extern int MPIR_CVAR_COLL_SCHED_DUMP;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/netmod/ofi/ofi_init.c */
extern char * MPIR_CVAR_OFI_USE_PROVIDER;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/netmod/ofi/ofi_init.c */
extern int MPIR_CVAR_OFI_DUMP_PROVIDERS;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/netmod/mxm/mxm_init.c */
extern int MPIR_CVAR_NEMESIS_MXM_BULK_CONNECT;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/netmod/mxm/mxm_init.c */
extern int MPIR_CVAR_NEMESIS_MXM_BULK_DISCONNECT;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/netmod/mxm/mxm_init.c */
extern int MPIR_CVAR_NEMESIS_MXM_HUGEPAGE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/netmod/tcp/tcp_init.c */
extern char * MPIR_CVAR_CH3_INTERFACE_HOSTNAME;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/netmod/tcp/tcp_init.c */
extern MPIR_T_cvar_range_value_t MPIR_CVAR_CH3_PORT_RANGE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/netmod/tcp/tcp_init.c */
extern char * MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/netmod/tcp/tcp_init.c */
extern int MPIR_CVAR_NEMESIS_TCP_HOST_LOOKUP_RETRIES;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/include/mpid_nem_inline.h */
extern int MPIR_CVAR_POLLS_BEFORE_YIELD;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/src/mpid_nem_network.c */
extern char * MPIR_CVAR_NEMESIS_NETMOD;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/src/mpid_nem_init.c */
extern int MPIR_CVAR_NEMESIS_SHM_EAGER_MAX_SZ;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/src/mpid_nem_init.c */
extern int MPIR_CVAR_NEMESIS_SHM_READY_EAGER_MAX_SZ;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/src/mpid_nem_ckpt.c */
extern int MPIR_CVAR_NEMESIS_ENABLE_CKPOINT;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/src/mpid_nem_lmt_dma.c */
extern int MPIR_CVAR_NEMESIS_LMT_DMA_THRESHOLD;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/channels/nemesis/src/mpid_nem_lmt.c */
extern int MPIR_CVAR_ENABLE_FT;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/ch3u_rma_sync.c */
extern int MPIR_CVAR_CH3_RMA_SCALABLE_FENCE_PROCESS_NUM;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/ch3u_rma_sync.c */
extern int MPIR_CVAR_CH3_RMA_DELAY_ISSUING_FOR_PIGGYBACKING;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/mpid_vc.c */
extern int MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/mpid_rma.c */
extern int MPIR_CVAR_CH3_RMA_SLOTS_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/mpid_rma.c */
extern int MPIR_CVAR_CH3_RMA_TARGET_LOCK_DATA_BYTES;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/mpidi_rma.c */
extern int MPIR_CVAR_CH3_RMA_OP_WIN_POOL_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/mpidi_rma.c */
extern int MPIR_CVAR_CH3_RMA_OP_GLOBAL_POOL_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/mpidi_rma.c */
extern int MPIR_CVAR_CH3_RMA_TARGET_WIN_POOL_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/mpidi_rma.c */
extern int MPIR_CVAR_CH3_RMA_TARGET_GLOBAL_POOL_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/mpidi_rma.c */
extern int MPIR_CVAR_CH3_RMA_TARGET_LOCK_ENTRY_WIN_POOL_SIZE;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/ch3u_rma_progress.c */
extern int MPIR_CVAR_CH3_RMA_ACTIVE_REQ_THRESHOLD;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/ch3u_rma_progress.c */
extern int MPIR_CVAR_CH3_RMA_POKE_PROGRESS_REQ_THRESHOLD;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/ch3u_comm.c */
extern int MPIR_CVAR_CH3_ENABLE_HCOLL;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/ch3u_port.c */
extern int MPIR_CVAR_CH3_COMM_CONNECT_TIMEOUT;
/* declared in /tmp/F8kHb7tPTK/mpich-3.3/maint/../src/mpid/ch3/src/ch3u_rma_ops.c */
extern int MPIR_CVAR_CH3_RMA_OP_PIGGYBACK_LOCK_DATA_SIZE;

/* TODO: this should be defined elsewhere */
#define MPIR_CVAR_assert MPIR_Assert

/* Arbitrary, simplifies interaction with external interfaces like MPI_T_ */
#define MPIR_CVAR_MAX_STRLEN (384)

#endif /* MPIR_CVARS_H_INCLUDED */
