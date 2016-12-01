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

#define FCNAME "MPID_Abort"
#define FUNCNAME MPID_Abort
int MPID_Abort(MPID_Comm * comm, int mpi_errno, int exit_code,
	       const char *error_msg)
{
	/* printf("ps--- %s() called\n", __func__); */

	MPL_error_printf("%s", error_msg);

	exit(exit_code);
	return MPI_ERR_INTERN;
}
#undef FUNCNAME
#undef FCNAME
