/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2022 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"

static
int _getenv_i(const char *env_name, int _default)
{
	char *val = getenv(env_name);
	return val ? atoi(val) : _default;
}

int MPID_Abort(MPIR_Comm * comm_ptr, int mpi_errno, int exit_code,
	       const char *error_msg)
{
	int termination_call;
	MPIR_Comm *comm_self_ptr;

	MPIR_Comm_get_ptr (MPI_COMM_SELF, comm_self_ptr);

	if( (comm_ptr == comm_self_ptr) && (!_getenv_i("PSP_FINALIZE_BARRIER", 1)) ) {

		/* Experimental extension:
		   Properly deregister from PMI in the COMM_SELF case
		   so that other processes can still carry on.*/

		MPIR_pmi_finalize();
	}

	MPL_error_printf("%s\n", error_msg);

	termination_call = _getenv_i("PSP_HARD_ABORT", 0);

	switch (termination_call) {
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
