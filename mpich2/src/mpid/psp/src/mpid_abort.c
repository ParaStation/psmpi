/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"

int MPID_Abort(MPIR_Comm * comm_ptr, int mpi_errno, int exit_code, const char *error_msg)
{
    MPIR_Comm *comm_self_ptr;

    MPIR_Comm_get_ptr(MPI_COMM_SELF, comm_self_ptr);

    if ((comm_ptr == comm_self_ptr) && (!MPIDI_Process.env.finalize.barrier)) {

        /* Experimental extension:
         * Properly deregister from PMI in the COMM_SELF case
         * so that other processes can still carry on. */

        MPIR_pmi_finalize();
    }

    MPL_error_printf("%s\n", error_msg);

    switch (MPIDI_Process.env.hard_abort) {
        case 0:
            exit(exit_code);
        case 1:
            MPIR_pmi_abort(exit_code, error_msg);
            // fall through
        case 2:
            _exit(exit_code);
        default:
            abort();
    }
    return MPI_ERR_INTERN;
}
