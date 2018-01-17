/*
 * ParaStation
 *
 * Copyright (C) 2006,2007 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include "mpidimpl.h"
#include <unistd.h>

// This must be the last include before sysmbols are defined:
#include "mpid_visibility.h"


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
		/* MPIU_Strncpy only copies until (and including) the null,
		   unlink strncpy, it does not blank pad.  This is a good thing
		   here, because users don't always allocated MPI_MAX_PROCESSOR_NAME
		   characters */
		MPIU_Strncpy(name, processorName, namelen);
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
