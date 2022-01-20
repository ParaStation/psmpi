/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef MPID_COLL_H_INCLUDED
#define MPID_COLL_H_INCLUDED

#include "mpiimpl.h"
#ifdef HAVE_LIBHCOLL
#include "../../common/hcoll/hcoll.h"
#endif

#ifdef MPID_PSP_HCOLL_STATS
void MPIDI_PSP_stats_hcoll_counter_inc(MPIDI_PSP_stats_collops_enum_t);
#endif

#undef MPIDI_PSP_WITH_PSCOM_COLLECTIVES
#ifdef MPIDI_PSP_WITH_PSCOM_COLLECTIVES
void MPID_PSP_group_init(MPIR_Comm *comm_ptr);
void MPID_PSP_group_cleanup(MPIR_Comm *comm_ptr);
#endif

static inline int MPID_Barrier(MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

#ifdef HAVE_LIBHCOLL
    mpi_errno = hcoll_Barrier(comm, errflag);
    if (mpi_errno == MPI_SUCCESS) {
#ifdef MPID_PSP_HCOLL_STATS
        MPIDI_PSP_stats_hcoll_counter_inc(mpidi_psp_stats_collops_enum__barrier);
#endif
        goto fn_exit;
    }
#endif

#ifdef MPID_PSP_MSA_AWARE_COLLOPS
    if ((comm->hierarchy_kind == MPIR_COMM_HIERARCHY_KIND__NODE) && (comm->local_comm != NULL))
        mpi_errno = MPIR_Barrier_impl(comm->local_comm, errflag);
    else
        mpi_errno = MPIR_Barrier_impl(comm, errflag);
#else
    mpi_errno = MPIR_Barrier_impl(comm, errflag);
#endif

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Bcast(void *buffer, MPI_Aint count, MPI_Datatype datatype, int root,
                             MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

#ifdef HAVE_LIBHCOLL
    int typesize;
    MPIR_Datatype_get_size_macro(datatype, typesize);
    if (unlikely(count * typesize == 0) && comm->hcoll_priv.is_hcoll_init) {
        goto fn_exit; /* do shortcut here (as it seems that HCOLL has problems with zero-byte messages) */
    }
    mpi_errno = hcoll_Bcast(buffer, count, datatype, root, comm, errflag);
    if (mpi_errno == MPI_SUCCESS) {
#ifdef MPID_PSP_HCOLL_STATS
        MPIDI_PSP_stats_hcoll_counter_inc(mpidi_psp_stats_collops_enum__bcast);
#endif
        goto fn_exit;
    }
#endif

#ifdef MPID_PSP_MSA_AWARE_COLLOPS
    if ((comm->hierarchy_kind == MPIR_COMM_HIERARCHY_KIND__NODE) && (comm->local_comm != NULL))
        mpi_errno = MPIR_Bcast_impl(buffer, count, datatype, root, comm->local_comm, errflag);
    else
        mpi_errno = MPIR_Bcast_impl(buffer, count, datatype, root, comm, errflag);
#else
    mpi_errno = MPIR_Bcast_impl(buffer, count, datatype, root, comm, errflag);
#endif

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Allreduce(const void *sendbuf, void *recvbuf, MPI_Aint count,
                                 MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm,
                                 MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

#ifdef HAVE_LIBHCOLL
    int typesize;
    MPIR_Datatype_get_size_macro(datatype, typesize);
    if (unlikely(count * typesize == 0) && comm->hcoll_priv.is_hcoll_init) {
        goto fn_exit; /* do shortcut here (as it seems that HCOLL has problems with zero-byte messages) */
    }
    mpi_errno = hcoll_Allreduce(sendbuf, recvbuf, count, datatype, op, comm, errflag);
    if (mpi_errno == MPI_SUCCESS) {
#ifdef MPID_PSP_HCOLL_STATS
        MPIDI_PSP_stats_hcoll_counter_inc(mpidi_psp_stats_collops_enum__allreduce);
#endif
        goto fn_exit;
    }
#endif

#ifdef MPID_PSP_MSA_AWARE_COLLOPS
    if ((comm->hierarchy_kind == MPIR_COMM_HIERARCHY_KIND__NODE) && (comm->local_comm != NULL))
        mpi_errno = MPIR_Allreduce_impl(sendbuf, recvbuf, count, datatype, op, comm->local_comm, errflag);
    else
        mpi_errno = MPIR_Allreduce_impl(sendbuf, recvbuf, count, datatype, op, comm, errflag);
#else
    mpi_errno = MPIR_Allreduce_impl(sendbuf, recvbuf, count, datatype, op, comm, errflag);
#endif

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Allgather(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                                 void *recvbuf, MPI_Aint recvcount, MPI_Datatype recvtype,
                                 MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

#ifdef HAVE_LIBHCOLL
    int stypesize, rtypesize;
    MPIR_Datatype_get_size_macro(sendtype, stypesize);
    MPIR_Datatype_get_size_macro(recvtype, rtypesize);
    if (unlikely(((sendcount * stypesize == 0) && (sendbuf != MPI_IN_PLACE)) || (recvcount * rtypesize == 0)) && comm->hcoll_priv.is_hcoll_init) {
        goto fn_exit; /* do shortcut here (as it seems that HCOLL has problems with zero-byte messages) */
    }
    mpi_errno = hcoll_Allgather(sendbuf, sendcount, sendtype, recvbuf,
                                recvcount, recvtype, comm, errflag);
    if (mpi_errno == MPI_SUCCESS) {
#ifdef MPID_PSP_HCOLL_STATS
        MPIDI_PSP_stats_hcoll_counter_inc(mpidi_psp_stats_collops_enum__allgather);
#endif
        goto fn_exit;
    }
#endif
    mpi_errno = MPIR_Allgather_impl(sendbuf, sendcount, sendtype, recvbuf,
                                    recvcount, recvtype, comm, errflag);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Allgatherv(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                                  void *recvbuf, const MPI_Aint recvcounts[], const MPI_Aint displs[],
                                  MPI_Datatype recvtype, MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Allgatherv_impl(sendbuf, sendcount, sendtype, recvbuf,
                                     recvcounts, displs, recvtype, comm,
                                     errflag);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Scatter(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                               void *recvbuf, MPI_Aint recvcount, MPI_Datatype recvtype,
                               int root, MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Scatter_impl(sendbuf, sendcount, sendtype, recvbuf,
                                  recvcount, recvtype, root, comm, errflag);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Scatterv(const void *sendbuf, const MPI_Aint sendcounts[], const MPI_Aint displs[],
                                MPI_Datatype sendtype, void *recvbuf, MPI_Aint recvcount,
                                MPI_Datatype recvtype, int root, MPIR_Comm * comm,
                                MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Scatterv_impl(sendbuf, sendcounts, displs, sendtype,
                                   recvbuf, recvcount, recvtype, root, comm,
                                   errflag);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Gather(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                              void *recvbuf, MPI_Aint recvcount, MPI_Datatype recvtype,
                              int root, MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Gather_impl(sendbuf, sendcount, sendtype, recvbuf,
                                 recvcount, recvtype, root, comm, errflag);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Gatherv(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                               void *recvbuf, const MPI_Aint recvcounts[], const MPI_Aint displs[],
                               MPI_Datatype recvtype, int root, MPIR_Comm * comm,
                               MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Gatherv_impl(sendbuf, sendcount, sendtype, recvbuf,
                                  recvcounts, displs, recvtype, root, comm,
                                  errflag);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Alltoall(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                                void *recvbuf, MPI_Aint recvcount, MPI_Datatype recvtype,
                                MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

#ifdef HAVE_LIBHCOLL
    int stypesize, rtypesize;
    MPIR_Datatype_get_size_macro(sendtype, stypesize);
    MPIR_Datatype_get_size_macro(recvtype, rtypesize);
    if (unlikely(((sendcount * stypesize == 0) && (sendbuf != MPI_IN_PLACE)) || (recvcount * rtypesize == 0)) && comm->hcoll_priv.is_hcoll_init) {
        goto fn_exit; /* do shortcut here (as it seems that HCOLL has problems with zero-byte messages) */
    }
    mpi_errno = hcoll_Alltoall(sendbuf, sendcount, sendtype, recvbuf,
                               recvcount, recvtype, comm, errflag);
    if (mpi_errno == MPI_SUCCESS) {
#ifdef MPID_PSP_HCOLL_STATS
        MPIDI_PSP_stats_hcoll_counter_inc(mpidi_psp_stats_collops_enum__alltoall);
#endif
        goto fn_exit;
    }
#endif

    mpi_errno = MPIR_Alltoall_impl(sendbuf, sendcount, sendtype, recvbuf,
                                   recvcount, recvtype, comm, errflag);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Alltoallv(const void *sendbuf, const MPI_Aint sendcounts[], const MPI_Aint sdispls[],
                                 MPI_Datatype sendtype, void *recvbuf, const MPI_Aint recvcounts[],
                                 const MPI_Aint rdispls[], MPI_Datatype recvtype, MPIR_Comm * comm,
                                 MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

#ifdef HAVE_LIBHCOLL
    mpi_errno = hcoll_Alltoallv(sendbuf, sendcounts, sdispls, sendtype,
                                recvbuf, recvcounts, rdispls, recvtype,
                                comm, errflag);
    if (mpi_errno == MPI_SUCCESS) {
#ifdef MPID_PSP_HCOLL_STATS
        MPIDI_PSP_stats_hcoll_counter_inc(mpidi_psp_stats_collops_enum__alltoallv);
#endif
        goto fn_exit;
    }
#endif

    mpi_errno = MPIR_Alltoallv_impl(sendbuf, sendcounts, sdispls, sendtype,
                                    recvbuf, recvcounts, rdispls, recvtype,
                                    comm, errflag);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Alltoallw(const void *sendbuf, const MPI_Aint sendcounts[], const MPI_Aint sdispls[],
                                 const MPI_Datatype sendtypes[], void *recvbuf,
                                 const MPI_Aint recvcounts[], const MPI_Aint rdispls[],
                                 const MPI_Datatype recvtypes[], MPIR_Comm * comm_ptr,
                                 MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Alltoallw_impl(sendbuf, sendcounts, sdispls, sendtypes,
                                    recvbuf, recvcounts, rdispls, recvtypes,
                                    comm_ptr, errflag);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Reduce(const void *sendbuf, void *recvbuf, MPI_Aint count,
                              MPI_Datatype datatype, MPI_Op op, int root,
                              MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

#ifdef HAVE_LIBHCOLL
    int typesize;
    MPIR_Datatype_get_size_macro(datatype, typesize);
    if (unlikely(count * typesize == 0) && comm->hcoll_priv.is_hcoll_init) {
        goto fn_exit; /* do shortcut here (as it seems that HCOLL has problems with zero-byte messages) */
    }
    mpi_errno = hcoll_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm, errflag);
    if (mpi_errno == MPI_SUCCESS) {
#ifdef MPID_PSP_HCOLL_STATS
        MPIDI_PSP_stats_hcoll_counter_inc(mpidi_psp_stats_collops_enum__reduce);
#endif
        goto fn_exit;
    }
#endif

#ifdef MPID_PSP_MSA_AWARE_COLLOPS
    if ((comm->hierarchy_kind == MPIR_COMM_HIERARCHY_KIND__NODE) && (comm->local_comm != NULL))
        mpi_errno = MPIR_Reduce_impl(sendbuf, recvbuf, count, datatype, op, root, comm->local_comm, errflag);
    else
        mpi_errno = MPIR_Reduce_impl(sendbuf, recvbuf, count, datatype, op, root, comm, errflag);
#else
    mpi_errno = MPIR_Reduce_impl(sendbuf, recvbuf, count, datatype, op, root, comm, errflag);
#endif

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Reduce_scatter(const void *sendbuf, void *recvbuf, const MPI_Aint recvcounts[],
                                      MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm_ptr,
                                      MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Reduce_scatter_impl(sendbuf, recvbuf, recvcounts,
                                         datatype, op, comm_ptr, errflag);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Reduce_scatter_block(const void *sendbuf, void *recvbuf,
                                            MPI_Aint recvcount, MPI_Datatype datatype,
                                            MPI_Op op, MPIR_Comm * comm_ptr,
                                            MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Reduce_scatter_block_impl(sendbuf, recvbuf, recvcount,
                                               datatype, op, comm_ptr,
                                               errflag);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Scan(const void *sendbuf, void *recvbuf, MPI_Aint count,
                            MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm,
                            MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

#ifdef MPID_PSP_MSA_AWARE_COLLOPS
    if ((comm->hierarchy_kind == MPIR_COMM_HIERARCHY_KIND__NODE) && (comm->local_comm != NULL))
        mpi_errno = MPIR_Scan_impl(sendbuf, recvbuf, count, datatype, op, comm->local_comm, errflag);
    else
        mpi_errno = MPIR_Scan_impl(sendbuf, recvbuf, count, datatype, op, comm, errflag);
#else
    mpi_errno = MPIR_Scan_impl(sendbuf, recvbuf, count, datatype, op, comm, errflag);
#endif

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Exscan(const void *sendbuf, void *recvbuf, MPI_Aint count,
                              MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm,
                              MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Exscan_impl(sendbuf, recvbuf, count, datatype, op, comm,
                                 errflag);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Neighbor_allgather(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                                          void *recvbuf, MPI_Aint recvcount, MPI_Datatype recvtype,
                                          MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Neighbor_allgather_impl(sendbuf, sendcount, sendtype,
                                             recvbuf, recvcount, recvtype,
                                             comm);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Neighbor_allgatherv(const void *sendbuf, MPI_Aint sendcount,
                                           MPI_Datatype sendtype, void *recvbuf,
                                           const MPI_Aint recvcounts[], const MPI_Aint displs[],
                                           MPI_Datatype recvtype, MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Neighbor_allgatherv_impl(sendbuf, sendcount, sendtype,
                                              recvbuf, recvcounts, displs,
                                              recvtype, comm);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Neighbor_alltoallv(const void *sendbuf, const MPI_Aint sendcounts[],
                                          const MPI_Aint sdispls[], MPI_Datatype sendtype,
                                          void *recvbuf, const MPI_Aint recvcounts[],
                                          const MPI_Aint rdispls[], MPI_Datatype recvtype,
                                          MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Neighbor_alltoallv_impl(sendbuf, sendcounts, sdispls,
                                             sendtype, recvbuf, recvcounts,
                                             rdispls, recvtype, comm);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Neighbor_alltoallw(const void *sendbuf, const MPI_Aint sendcounts[],
                                          const MPI_Aint sdispls[], const MPI_Datatype sendtypes[],
                                          void *recvbuf, const MPI_Aint recvcounts[],
                                          const MPI_Aint rdispls[], const MPI_Datatype recvtypes[],
                                          MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Neighbor_alltoallw_impl(sendbuf, sendcounts, sdispls,
                                             sendtypes, recvbuf, recvcounts,
                                             rdispls, recvtypes, comm);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Neighbor_alltoall(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                                         void *recvbuf, MPI_Aint recvcount, MPI_Datatype recvtype,
                                         MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Neighbor_alltoall_impl(sendbuf, sendcount, sendtype,
                                            recvbuf, recvcount, recvtype,
                                            comm);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Ineighbor_allgather(const void *sendbuf, MPI_Aint sendcount,
                                           MPI_Datatype sendtype, void *recvbuf, MPI_Aint recvcount,
                                           MPI_Datatype recvtype, MPIR_Comm * comm,
                                           MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Ineighbor_allgather_impl(sendbuf, sendcount, sendtype,
                                              recvbuf, recvcount, recvtype,
                                              comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Ineighbor_allgatherv(const void *sendbuf, MPI_Aint sendcount,
                                            MPI_Datatype sendtype, void *recvbuf,
                                            const MPI_Aint recvcounts[], const MPI_Aint displs[],
                                            MPI_Datatype recvtype, MPIR_Comm * comm,
                                            MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Ineighbor_allgatherv_impl(sendbuf, sendcount, sendtype,
                                               recvbuf, recvcounts, displs,
                                               recvtype, comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Ineighbor_alltoall(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                                          void *recvbuf, MPI_Aint recvcount, MPI_Datatype recvtype,
                                          MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Ineighbor_alltoall_impl(sendbuf, sendcount, sendtype,
                                             recvbuf, recvcount, recvtype,
                                             comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Ineighbor_alltoallv(const void *sendbuf, const MPI_Aint sendcounts[],
                                           const MPI_Aint sdispls[], MPI_Datatype sendtype,
                                           void *recvbuf, const MPI_Aint recvcounts[],
                                           const MPI_Aint rdispls[], MPI_Datatype recvtype,
                                           MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Ineighbor_alltoallv_impl(sendbuf, sendcounts, sdispls,
                                              sendtype, recvbuf, recvcounts,
                                              rdispls, recvtype, comm,
                                              request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Ineighbor_alltoallw(const void *sendbuf, const MPI_Aint sendcounts[],
                                           const MPI_Aint sdispls[], const MPI_Datatype sendtypes[],
                                           void *recvbuf, const MPI_Aint recvcounts[],
                                           const MPI_Aint rdispls[], const MPI_Datatype recvtypes[],
                                           MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Ineighbor_alltoallw_impl(sendbuf, sendcounts, sdispls,
                                              sendtypes, recvbuf, recvcounts,
                                              rdispls, recvtypes, comm,
                                              request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Ibarrier(MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Ibarrier_impl(comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Ibcast(void *buffer, MPI_Aint count, MPI_Datatype datatype, int root,
                              MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Ibcast_impl(buffer, count, datatype, root, comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Iallgather(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                                  void *recvbuf, MPI_Aint recvcount, MPI_Datatype recvtype,
                                  MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Iallgather_impl(sendbuf, sendcount, sendtype, recvbuf,
                                     recvcount, recvtype, comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Iallgatherv(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                                   void *recvbuf, const MPI_Aint recvcounts[], const MPI_Aint displs[],
                                   MPI_Datatype recvtype, MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Iallgatherv_impl(sendbuf, sendcount, sendtype, recvbuf,
                                      recvcounts, displs, recvtype, comm,
                                      request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Iallreduce(const void *sendbuf, void *recvbuf, MPI_Aint count,
                                  MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm,
                                  MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Iallreduce_impl(sendbuf, recvbuf, count, datatype, op,
                                     comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Ialltoall(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                                 void *recvbuf, MPI_Aint recvcount, MPI_Datatype recvtype,
                                 MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Ialltoall_impl(sendbuf, sendcount, sendtype, recvbuf,
                                    recvcount, recvtype, comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Ialltoallv(const void *sendbuf, const MPI_Aint sendcounts[],
                                  const MPI_Aint sdispls[], MPI_Datatype sendtype,
                                  void *recvbuf, const MPI_Aint recvcounts[],
                                  const MPI_Aint rdispls[], MPI_Datatype recvtype,
                                  MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Ialltoallv_impl(sendbuf, sendcounts, sdispls, sendtype,
                                     recvbuf, recvcounts, rdispls, recvtype,
                                     comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Ialltoallw(const void *sendbuf, const MPI_Aint sendcounts[],
                                  const MPI_Aint sdispls[], const MPI_Datatype sendtypes[],
                                  void *recvbuf, const MPI_Aint recvcounts[],
                                  const MPI_Aint rdispls[], const MPI_Datatype recvtypes[],
                                  MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Ialltoallw_impl(sendbuf, sendcounts, sdispls, sendtypes,
                                     recvbuf, recvcounts, rdispls, recvtypes,
                                     comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Iexscan(const void *sendbuf, void *recvbuf, MPI_Aint count,
                               MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm,
                               MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Iexscan_impl(sendbuf, recvbuf, count, datatype, op, comm,
                                  request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Igather(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                               void *recvbuf, MPI_Aint recvcount, MPI_Datatype recvtype,
                               int root, MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Igather_impl(sendbuf, sendcount, sendtype, recvbuf,
                                  recvcount, recvtype, root, comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Igatherv(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                                void *recvbuf, const MPI_Aint recvcounts[], const MPI_Aint displs[],
                                MPI_Datatype recvtype, int root, MPIR_Comm * comm,
                                MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Igatherv_impl(sendbuf, sendcount, sendtype, recvbuf,
                                   recvcounts, displs, recvtype, root, comm,
                                   request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Ireduce_scatter_block(const void *sendbuf, void *recvbuf, MPI_Aint recvcount,
                                             MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm,
                                             MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Ireduce_scatter_block_impl(sendbuf, recvbuf, recvcount,
                                                datatype, op, comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Ireduce_scatter(const void *sendbuf, void *recvbuf, const MPI_Aint recvcounts[],
                                       MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm,
                                       MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Ireduce_scatter_impl(sendbuf, recvbuf, recvcounts,
                                          datatype, op, comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Ireduce(const void *sendbuf, void *recvbuf, MPI_Aint count, MPI_Datatype datatype,
                               MPI_Op op, int root, MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Ireduce_impl(sendbuf, recvbuf, count, datatype, op, root,
                                  comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Iscan(const void *sendbuf, void *recvbuf, MPI_Aint count, MPI_Datatype datatype,
                             MPI_Op op, MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Iscan_impl(sendbuf, recvbuf, count, datatype, op, comm,
                                request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Iscatter(const void *sendbuf, MPI_Aint sendcount, MPI_Datatype sendtype,
                                void *recvbuf, MPI_Aint recvcount, MPI_Datatype recvtype,
                                int root, MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Iscatter_impl(sendbuf, sendcount, sendtype, recvbuf,
                                   recvcount, recvtype, root, comm, request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Iscatterv(const void *sendbuf, const MPI_Aint sendcounts[],
                                 const MPI_Aint displs[], MPI_Datatype sendtype,
                                 void *recvbuf, MPI_Aint recvcount, MPI_Datatype recvtype,
                                 int root, MPIR_Comm * comm, MPIR_Request **request)
{
    int mpi_errno = MPI_SUCCESS;

    mpi_errno = MPIR_Iscatterv_impl(sendbuf, sendcounts, displs, sendtype,
                                    recvbuf, recvcount, recvtype, root, comm,
                                    request);

    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static inline int MPID_Barrier_init(MPIR_Comm * comm, MPIR_Info * info,
                                    MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Barrier_init_impl(comm, info, request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Bcast_init(void *buffer, MPI_Aint count,
                                  MPI_Datatype datatype, int root,
                                  MPIR_Comm * comm_ptr, MPIR_Info * info_ptr,
                                  MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Bcast_init_impl(buffer, count, datatype, root, comm_ptr,
                                     info_ptr, request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Gather_init(const void *sendbuf, MPI_Aint sendcount,
                                   MPI_Datatype sendtype, void *recvbuf,
                                   MPI_Aint recvcount, MPI_Datatype recvtype,
                                   int root, MPIR_Comm * comm, MPIR_Info * info,
                                   MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Gather_init_impl(sendbuf, sendcount, sendtype, recvbuf,
                                      recvcount, recvtype, root, comm, info,
                                      request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Gatherv_init(const void *sendbuf, MPI_Aint sendcount,
                                    MPI_Datatype sendtype, void *recvbuf,
                                    const MPI_Aint recvcounts[],
                                    const MPI_Aint displs[],
                                    MPI_Datatype recvtype,
                                    int root, MPIR_Comm * comm,
                                    MPIR_Info * info, MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Gatherv_init_impl(sendbuf, sendcount, sendtype, recvbuf,
                                       recvcounts, displs, recvtype, root, comm,
                                       info, request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Scatter_init(const void *sendbuf, MPI_Aint sendcount,
                                    MPI_Datatype sendtype, void *recvbuf,
                                    MPI_Aint recvcount, MPI_Datatype recvtype,
                                    int root, MPIR_Comm * comm,
                                    MPIR_Info * info, MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Scatter_init_impl(sendbuf, sendcount, sendtype, recvbuf,
                                       recvcount, recvtype, root, comm, info,
                                       request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Scatterv_init(const void *sendbuf,
                                     const MPI_Aint sendcounts[],
                                     const MPI_Aint displs[],
                                     MPI_Datatype sendtype, void *recvbuf,
                                     MPI_Aint recvcount, MPI_Datatype recvtype,
                                     int root, MPIR_Comm * comm,
                                     MPIR_Info * info, MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Scatterv_init_impl(sendbuf, sendcounts, displs, sendtype,
                                        recvbuf, recvcount, recvtype, root,
                                        comm, info, request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Allgather_init(const void *sendbuf, MPI_Aint sendcount,
                                      MPI_Datatype sendtype, void *recvbuf,
                                      MPI_Aint recvcount, MPI_Datatype recvtype,
                                      MPIR_Comm * comm_ptr,
                                      MPIR_Info * info_ptr,
                                      MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Allgather_init_impl(sendbuf, sendcount, sendtype, recvbuf,
                                         recvcount, recvtype, comm_ptr,
                                         info_ptr, request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Allgatherv_init(const void *sendbuf, MPI_Aint sendcount,
                                       MPI_Datatype sendtype, void *recvbuf,
                                       const MPI_Aint * recvcounts,
                                       const MPI_Aint * displs,
                                       MPI_Datatype recvtype,
                                       MPIR_Comm * comm_ptr,
                                       MPIR_Info * info_ptr,
                                       MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Allgatherv_init_impl(sendbuf, sendcount, sendtype, recvbuf,
                                          recvcounts, displs, recvtype,
                                          comm_ptr, info_ptr, request);

    return mpi_errno;
}

static inline int MPID_Alltoall_init(const void *sendbuf, MPI_Aint sendcount,
                                     MPI_Datatype sendtype, void *recvbuf,
                                     MPI_Aint recvcount, MPI_Datatype recvtype,
                                     MPIR_Comm * comm_ptr, MPIR_Info * info_ptr,
                                     MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Alltoall_init_impl(sendbuf, sendcount, sendtype, recvbuf,
                                        recvcount, recvtype, comm_ptr, info_ptr,
                                        request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Alltoallv_init(const void *sendbuf,
                                      const MPI_Aint sendcounts[],
                                      const MPI_Aint sdispls[],
                                      MPI_Datatype sendtype,
                                      void *recvbuf,const MPI_Aint recvcounts[],
                                      const MPI_Aint rdispls[],
                                      MPI_Datatype recvtype,
                                      MPIR_Comm * comm_ptr,
                                      MPIR_Info * info_ptr,
                                      MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Alltoallv_init_impl(sendbuf, sendcounts, sdispls, sendtype,
                                         recvbuf, recvcounts, rdispls, recvtype,
                                         comm_ptr, info_ptr, request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Alltoallw_init(const void *sendbuf,
                                      const MPI_Aint sendcounts[],
                                      const MPI_Aint sdispls[],
                                      const MPI_Datatype sendtypes[],
                                      void *recvbuf,
                                      const MPI_Aint recvcounts[],
                                      const MPI_Aint rdispls[],
                                      const MPI_Datatype recvtypes[],
                                      MPIR_Comm * comm_ptr,
                                      MPIR_Info * info_ptr,
                                      MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Alltoallw_init_impl(sendbuf, sendcounts, sdispls,
                                         sendtypes, recvbuf, recvcounts,
                                         rdispls, recvtypes, comm_ptr, info_ptr,
                                         request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Reduce_init(const void *sendbuf, void *recvbuf,
                                   MPI_Aint count, MPI_Datatype datatype,
                                   MPI_Op op, int root, MPIR_Comm * comm_ptr,
                                   MPIR_Info * info_ptr,
                                   MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Reduce_init_impl(sendbuf, recvbuf, count, datatype, op,
                                      root, comm_ptr, info_ptr, request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Allreduce_init(const void *sendbuf, void *recvbuf,
                                      MPI_Aint count, MPI_Datatype datatype,
                                      MPI_Op op, MPIR_Comm * comm_ptr,
                                      MPIR_Info * info_ptr,
                                      MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Allreduce_init_impl(sendbuf, recvbuf, count, datatype, op,
                                         comm_ptr, info_ptr, request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Reduce_scatter_init(const void *sendbuf, void *recvbuf,
                                           const MPI_Aint recvcounts[],
                                           MPI_Datatype datatype, MPI_Op op,
                                           MPIR_Comm * comm, MPIR_Info * info,
                                           MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Reduce_scatter_init_impl(sendbuf, recvbuf, recvcounts,
                                              datatype, op, comm, info,
                                              request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Reduce_scatter_block_init(const void *sendbuf,
                                                 void *recvbuf,
                                                 MPI_Aint recvcount,
                                                 MPI_Datatype datatype,
                                                 MPI_Op op,
                                                 MPIR_Comm * comm,
                                                 MPIR_Info * info,
                                                 MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Reduce_scatter_block_init_impl(sendbuf, recvbuf, recvcount,
                                                    datatype, op, comm, info,
                                                    request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Scan_init(const void *sendbuf, void *recvbuf,
                                 MPI_Aint count, MPI_Datatype datatype,
                                 MPI_Op op, MPIR_Comm * comm, MPIR_Info * info,
                                 MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Scan_init_impl(sendbuf, recvbuf, count, datatype, op, comm,
                                    info, request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Exscan_init(const void *sendbuf, void *recvbuf,
                                   MPI_Aint count, MPI_Datatype datatype,
                                   MPI_Op op, MPIR_Comm * comm,
                                   MPIR_Info * info, MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Exscan_init_impl(sendbuf, recvbuf, count, datatype, op,
                                      comm, info, request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Neighbor_allgather_init(const void *sendbuf,
                                               MPI_Aint sendcount,
                                               MPI_Datatype sendtype,
                                               void *recvbuf,
                                               MPI_Aint recvcount,
                                               MPI_Datatype recvtype,
                                               MPIR_Comm * comm,
                                               MPIR_Info * info,
                                               MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Neighbor_allgather_init_impl(sendbuf, sendcount, sendtype,
                                                  recvbuf, recvcount, recvtype,
                                                  comm, info, request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Neighbor_allgatherv_init(const void *sendbuf,
                                                MPI_Aint sendcount,
                                                MPI_Datatype sendtype,
                                                void *recvbuf,
                                                const MPI_Aint recvcounts[],
                                                const MPI_Aint displs[],
                                                MPI_Datatype recvtype,
                                                MPIR_Comm * comm,
                                                MPIR_Info * info,
                                                MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Neighbor_allgatherv_init_impl(sendbuf, sendcount, sendtype,
                                                   recvbuf, recvcounts, displs,
                                                   recvtype, comm, info,
                                                   request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Neighbor_alltoall_init(const void *sendbuf,
                                              MPI_Aint sendcount,
                                              MPI_Datatype sendtype,
                                              void *recvbuf,
                                              MPI_Aint recvcount,
                                              MPI_Datatype recvtype,
                                              MPIR_Comm * comm,
                                              MPIR_Info * info,
                                              MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Neighbor_alltoall_init_impl(sendbuf, sendcount, sendtype,
                                                 recvbuf, recvcount, recvtype,
                                                 comm, info, request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Neighbor_alltoallv_init(const void *sendbuf,
                                               const MPI_Aint sendcounts[],
                                               const MPI_Aint sdispls[],
                                               MPI_Datatype sendtype,
                                               void *recvbuf,
                                               const MPI_Aint recvcounts[],
                                               const MPI_Aint rdispls[],
                                               MPI_Datatype recvtype,
                                               MPIR_Comm * comm,
                                               MPIR_Info * info,
                                               MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Neighbor_alltoallv_init_impl(sendbuf, sendcounts, sdispls,
                                                  sendtype, recvbuf, recvcounts,
                                                  rdispls, recvtype, comm, info,
                                                  request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

static inline int MPID_Neighbor_alltoallw_init(const void *sendbuf,
                                               const MPI_Aint sendcounts[],
                                               const MPI_Aint sdispls[],
                                               const MPI_Datatype sendtypes[],
                                               void *recvbuf,
                                               const MPI_Aint recvcounts[],
                                               const MPI_Aint rdispls[],
                                               const MPI_Datatype recvtypes[],
                                               MPIR_Comm * comm,
                                               MPIR_Info * info,
                                               MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Neighbor_alltoallw_init_impl(sendbuf, sendcounts, sdispls,
                                                  sendtypes, recvbuf,
                                                  recvcounts, rdispls,
                                                  recvtypes, comm, info,
                                                  request);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}


#ifdef MPIDI_PSP_WITH_CUDA_AWARENESS
int MPIR_Reduce_local_impl(const void *inbuf, void *inoutbuf, MPI_Aint count, MPI_Datatype datatype, MPI_Op op);
int MPID_PSP_Reduce_local_for_cuda(const void *inbuf, void *inoutbuf, MPI_Aint count, MPI_Datatype datatype, MPI_Op op);
#define MPID_REDUCE_LOCAL_HOOK(inbuf, inoutbuf, count, datatype, op) \
	MPID_PSP_Reduce_local_for_cuda(inbuf, inoutbuf, count, datatype, op)

static inline
int MPIDI_PSP_needs_staging(const void *ptr)
{
	return (ptr == MPI_IN_PLACE)? 0 : pscom_is_gpu_mem(ptr);
}
#endif /* MPIDI_PSP_WITH_CUDA_AWARENESS */

#endif /* MPID_COLL_H_INCLUDED */
