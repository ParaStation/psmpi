/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

int MPIR_Alltoallw_allcomm_nb(const void *sendbuf, const int sendcounts[], const int sdispls[],
                              const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
                              const int rdispls[], const MPI_Datatype recvtypes[],
                              MPIR_Comm * comm_ptr, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Request *req_ptr = NULL;

    /* just call the nonblocking version and wait on it */
    mpi_errno =
        MPIR_Ialltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
                        recvtypes, comm_ptr, &req_ptr);
    MPIR_ERR_CHECK(mpi_errno);

    mpi_errno = MPIC_Wait(req_ptr, errflag);
    MPIR_ERR_CHECK(mpi_errno);
    MPIR_Request_free(req_ptr);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
