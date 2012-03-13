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

#define WARN_NOT_IMPLEMENTED						\
do {									\
	static int warned = 0;						\
	if (!warned) {							\
		warned = 1;						\
		fprintf(stderr, "Warning: %s() not implemented\n", __func__); \
	}								\
} while (0)
/*@
   MPID_Open_port - Open an MPI Port

   Input Arguments:
.  MPI_Info info - info

   Output Arguments:
.  char *port_name - port name

   Notes:

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_OTHER
@*/
int MPID_Open_port(MPID_Info *info_ptr, char *port_name)
{
	WARN_NOT_IMPLEMENTED;

	return MPI_ERR_UNSUPPORTED_OPERATION;
}


/*@
   MPID_Close_port - Close port

   Input Parameter:
.  port_name - Name of MPI port to close

   Notes:

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_OTHER

@*/
int MPID_Close_port(const char *port_name)
{
	WARN_NOT_IMPLEMENTED;
	return MPI_ERR_UNSUPPORTED_OPERATION;
}

/*@
   MPID_Comm_accept - MPID entry point for MPI_Comm_accept

   Input Parameters:
+  port_name - port name
.  info - info
.  root - root
-  comm - communicator

   Output Parameters:
.  MPI_Comm *newcomm - new communicator

  Return Value:
  'MPI_SUCCESS' or a valid MPI error code.
@*/
int MPID_Comm_accept(char * port_name, MPID_Info * info, int root,
		     MPID_Comm * comm, MPID_Comm ** newcomm_ptr)
{
	WARN_NOT_IMPLEMENTED;
	return MPI_ERR_UNSUPPORTED_OPERATION;
}

/*@
   MPID_Comm_connect - MPID entry point for MPI_Comm_connect

   Input Parameters:
+  port_name - port name
.  info - info
.  root - root
-  comm - communicator

   Output Parameters:
.  newcomm_ptr - new intercommunicator

  Return Value:
  'MPI_SUCCESS' or a valid MPI error code.
@*/
int MPID_Comm_connect(const char * port_name, MPID_Info * info, int root,
		      MPID_Comm * comm, MPID_Comm ** newcomm_ptr)
{
	WARN_NOT_IMPLEMENTED;
	return MPI_ERR_UNSUPPORTED_OPERATION;
}

/* int MPID_Comm_disconnect(MPID_Comm *); */

/* FIXME: Correct description of function */
/*@
   MPID_Comm_spawn_multiple -

   Input Arguments:
+  int count - count
.  char *array_of_commands[] - commands
.  char* *array_of_argv[] - arguments
.  int array_of_maxprocs[] - maxprocs
.  MPI_Info array_of_info[] - infos
.  int root - root
-  MPI_Comm comm - communicator

   Output Arguments:
+  MPI_Comm *intercomm - intercommunicator
-  int array_of_errcodes[] - error codes

   Notes:

.N Errors
.N MPI_SUCCESS
@*/
#define FCNAME "MPID_Comm_spawn_multiple"
#define FUNCNAME MPID_Comm_spawn_multiple
int MPID_Comm_spawn_multiple(int count, char *array_of_commands[],
			     char ** array_of_argv[], int array_of_maxprocs[],
			     MPID_Info * array_of_info_ptrs[], int root,
			     MPID_Comm * comm_ptr, MPID_Comm ** intercomm,
			     int array_of_errcodes[])
{
	WARN_NOT_IMPLEMENTED;
	return MPI_ERR_UNSUPPORTED_OPERATION;
}
#undef FUNCNAME
#undef FCNAME
