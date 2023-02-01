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

#include "mpidimpl.h"
#include "mpid_psp_request.h"

/*
 * TODO: this function has to differentiate between
 *       persistent requests and partitioned requests.
 */
static
int MPID_Start(MPIR_Request * req)
{
    int mpi_errno = MPI_SUCCESS;

    switch (req->kind) {
        case MPIR_REQUEST_KIND__PREQUEST_SEND:
        case MPIR_REQUEST_KIND__PREQUEST_RECV:
            mpi_errno = MPID_PSP_persistent_start(req);
            break;
        case MPIR_REQUEST_KIND__PART_SEND:
            MPID_PSP_LOCKFREE_CALL(mpi_errno = MPID_PSP_psend_start(req));
            break;
        case MPIR_REQUEST_KIND__PART_RECV:
            MPID_PSP_LOCKFREE_CALL(mpi_errno = MPID_PSP_precv_start(req));
            break;
        case MPIR_REQUEST_KIND__PREQUEST_COLL:
            mpi_errno = MPIR_Persist_coll_start(req);
            break;
        default:
            mpi_errno = MPI_ERR_INTERN;
    }

    return mpi_errno;
}


int MPID_Startall(int count, MPIR_Request * requests[])
{
    int mpi_errno = MPI_SUCCESS;

    while (count) {
        mpi_errno = MPID_Start(*requests);
        if (mpi_errno != MPI_SUCCESS)
            break;

        requests++;
        count--;
    }

    return mpi_errno;
}
