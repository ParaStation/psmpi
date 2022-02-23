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
#include <unistd.h>


static char processorName[MPI_MAX_PROCESSOR_NAME];
static int  setProcessorName = 0;
static size_t processorNameLen = 0;


int MPID_Get_processor_name(char * name, int namelen, int * resultlen)
{
	int mpi_errno = MPI_SUCCESS;

	if (!name) { mpi_errno = MPI_ERR_ARG; goto out; }

	if (!setProcessorName) {
		size_t len = sizeof(processorName);

		setProcessorName = 1;
		if (gethostname(processorName, len) != 0) {
			strncpy(processorName, "???", len);
		}

		processorNameLen = strlen(processorName);
	}

	if (processorNameLen > 0) {
		/* MPL_strncpy only copies until (and including) the null,
		   unlink strncpy, it does not blank pad.  This is a good thing
		   here, because users don't always allocated MPI_MAX_PROCESSOR_NAME
		   characters */
		MPL_strncpy(name, processorName, namelen);
		if (resultlen) *resultlen = processorNameLen;
	} else {
		mpi_errno = MPIR_Err_create_code(MPI_SUCCESS,
						 MPIR_ERR_RECOVERABLE,
						 "MPID_Get_processor_name",
						 __LINE__,
						 MPI_ERR_OTHER,
						 "**mpi_get_processor_name",0);
	}

 out:
	return mpi_errno;
}
