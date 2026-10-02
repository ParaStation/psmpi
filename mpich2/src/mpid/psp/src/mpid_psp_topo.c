/*
 * ParaStation
 *
 * Copyright (C) 2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"

int MPID_Get_node_id(MPIR_Comm * comm, int rank, int *id_p)
{
    MPIR_Lpid lpid = MPIR_comm_rank_to_lpid(comm, rank);
    int world_idx = MPIR_LPID_WORLD_INDEX(lpid);

    if (world_idx == 0) {
        // rank is within the own MPI_COMM_WORLD -> use map
        *id_p = MPIR_Process.node_map[lpid];
    } else {
        // node ids of remote process groups are unknown...
        *id_p = -1;
    }

    return MPI_SUCCESS;
}
