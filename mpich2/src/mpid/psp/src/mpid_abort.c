/*
 * ParaStation
 *
 * Copyright (C) 2006-2010 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include "mpidimpl.h"
#include "pmi.h"

static
int _getenv_i(const char *env_name, int _default)
{
	char *val = getenv(env_name);
	return val ? atoi(val) : _default;
}

#define FCNAME "MPID_Abort"
#define FUNCNAME MPID_Abort
int MPID_Abort(MPID_Comm * comm_ptr, int mpi_errno, int exit_code,
	       const char *error_msg)
{
	MPID_Comm *comm_self_ptr;

	MPID_Comm_get_ptr(MPI_COMM_SELF, comm_self_ptr);

	if( (comm_ptr == comm_self_ptr) && (!_getenv_i("PSP_FINALIZE_BARRIER", 1)) ) {

		/* Experimental extension:
		   Properly deregister from PMI in the COMM_SELF case
		   so that other processes can still carry on.*/

		PMI_Finalize();
	}

	MPL_error_printf("%s", error_msg);

	exit(exit_code);
	return MPI_ERR_INTERN;
}
#undef FUNCNAME
#undef FCNAME
