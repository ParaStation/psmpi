/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2024 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"
#include "mpid_psp_datatype.h"
#include "mpid_psp_packed_msg.h"
#include "mpid_psp_request.h"

int MPIDI_PSP_compute_acc_op(void *origin_addr, int origin_count,
                             MPI_Datatype origin_datatype, void *target_addr,
                             int target_count, MPI_Datatype target_datatype,
                             MPI_Op op, int packed_source_buf)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_ENTER;

    mpi_errno = MPIR_Typerep_op(origin_addr, origin_count, origin_datatype,
                                target_addr, target_count, target_datatype,
                                op, packed_source_buf, -1);

    MPIR_FUNC_EXIT;
    return mpi_errno;
}
