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

#ifndef _MPIDPOST_H_
#define _MPIDPOST_H_

#include "mpid_coll.h"

/* FIXME: mpidpost.h is included by mpiimpl.h .  However, mpiimpl.h should
   refer only to the ADI3 prototypes and should never include prototypes
   specific to any particular device.  Factor the include files to maintain
   better modularity by providing mpiimpl.h with only the definitions that it
   needs

   See mpipost.h of the CH3.
   */
/*
 * Channel API prototypes
 */

/*@
  MPIDI_PSP_Progress_start - Mark the beginning of a progress epoch.

  Input Parameters:
. state - pointer to a MPID_Progress_state object

  Return value:
  An MPI error code.

  NOTE:
  This routine need only be called if the code might call
  MPIDI_PSP_Progress_wait().  It is normally used as follows example:
.vb
      if (*req->cc_ptr != 0)
	  {
          MPID_Progress_state state;

          MPIDI_PSP_Progress_start(&state);
          {
              while(*req->cc_ptr != 0)
              {
                  MPIDI_PSP_Progress_wait(&state);
              }
          }
          MPIDI_PSP_Progress_end(&state);
      }
.ve

  IMPLEMENTORS:
  A multi-threaded implementation might save the current value of a request
  completion counter in the state.
@*/
void MPIDI_PSP_Progress_start(MPID_Progress_state * state);


/*@
  MPIDI_PSP_Progress_wait - Give the channel implementation an opportunity to
  make progress on outstanding communication requests.

  Input Parameters:
. state - pointer to the same MPID_Progress_state object passed to
  MPIDI_PSP_Progress_start

  Return value:
  An MPI error code.

  NOTE:
  MPIDI_PSP_Progress_start/end() need to be called.

  IMPLEMENTORS:
  A multi-threaded implementation would return immediately if the a request
  had been completed between the call to
  MPIDI_PSP_Progress_start() and MPIDI_PSP_Progress_wait().  This could be
  implemented by checking a request completion counter
  in the progress state against a global counter, and returning if they did
  not match.
@*/
int MPIDI_PSP_Progress_wait(MPID_Progress_state * state);


/*@
  MPIDI_PSP_Progress_end - Mark the end of a progress epoch.

  Input Parameters:
. state - pointer to the same MPID_Progress_state object passed to
  MPIDI_PSP_Progress_start

  Return value:
  An MPI error code.
@*/
void MPIDI_PSP_Progress_end(MPID_Progress_state * state);


/*@
  MPIDI_PSP_Progress_test - Give the channel implementation an opportunity to
  make progress on outstanding communication requests.

  Return value:
  An MPI error code.

  NOTE:
  This function implicitly marks the beginning and end of a progress epoch.
@*/
int MPIDI_PSP_Progress_test(void);


/*@
  MPIDI_PSP_Progress_poke - Give the channel implementation a moment of
  opportunity to make progress on outstanding communication.

  Return value:
  An mpi error code.

  IMPLEMENTORS:
  This routine is similar to MPIDI_PSP_Progress_test but may not be as
  thorough in its attempt to satisfy all outstanding
  communication.
@*/
int MPIDI_PSP_Progress_poke(void);

/*
 * Device level request management macros
 */

#define MPID_Prequest_free_hook(req_) do {} while(0)

/*
 * Device level progress engine macros
 */
#define MPID_Progress_start(progress_state_) MPIDI_PSP_Progress_start(progress_state_)
#define MPID_Progress_wait(progress_state_)  MPIDI_PSP_Progress_wait(progress_state_)
#define MPID_Progress_end(progress_state_)   MPIDI_PSP_Progress_end(progress_state_)
/* This is static inline instead of macro because otherwise MPID_Progress_test will
 * be a chain of macros and therefore can not be used as a callback function */
static inline int MPID_Progress_test(MPID_Progress_state * state) /* state is unused */
{
    return MPIDI_PSP_Progress_test();
}
#define MPID_Progress_poke()		     MPIDI_PSP_Progress_poke()

struct MPIR_Comm;
int MPIDI_GPID_GetAllInComm(MPIR_Comm *comm_ptr, int local_size,
			   MPIDI_Gpid local_gpids[], int *singlePG);
int MPIDI_GPID_ToLpidArray(int size, MPIDI_Gpid gpid[], uint64_t lpid[]);
int MPID_Create_intercomm_from_lpids(MPIR_Comm *newcomm_ptr,
			   int size, const uint64_t lpids[]);
int MPIDI_PG_ForwardPGInfo( MPIR_Comm *peer_ptr, MPIR_Comm *comm_ptr,
			   int nPGids, const MPIDI_Gpid gpids[],
			   int root, int remote_leader, int cts_tag,
			   pscom_connection_t *con, char *all_ports, pscom_socket_t *pscom_socket );
int MPID_Intercomm_exchange_map( MPIR_Comm *local_comm_ptr, int local_leader,
                                 MPIR_Comm *peer_comm_ptr, int remote_leader,
                                 int *remote_size, uint64_t **remote_lpids,
                                 int *is_low_group);

int MPIDI_GPID_Get(MPIR_Comm *comm_ptr, int rank, MPIDI_Gpid gpid[]);

#define MPID_ICCREATE_REMOTECOMM_HOOK(peer_comm_ptr, local_comm_ptr, remote_size, remote_gpids, local_leader) \
  MPIDI_PG_ForwardPGInfo(peer_comm_ptr, local_comm_ptr, remote_size, remote_gpids, local_leader, remote_leader, cts_tag, NULL, NULL, NULL)


/* ULFM support */
MPL_STATIC_INLINE_PREFIX int MPID_Comm_AS_enabled(MPIR_Comm * comm_ptr)
{
	/* This function must return 1 in the default case and should not be ignored
	 * by the implementation. */
	return 1;
}


MPL_STATIC_INLINE_PREFIX int MPID_Request_is_anysource(MPIR_Request * request_ptr)
{
	int ret = 0;

	if (request_ptr->kind == MPIR_REQUEST_KIND__RECV &&
	    request_ptr->dev.kind.recv.common.pscom_req) {
		ret = request_ptr->dev.kind.recv.common.pscom_req->connection == NULL;
	}

	return ret;
}

int MPIDI_PSP_Isend(const void *buf, MPI_Aint count, MPI_Datatype datatype,
		    int dest, int tag, MPIR_Comm *comm, int context_offset,
		    MPIR_Request **request);
MPL_STATIC_INLINE_PREFIX int MPID_Isend(const void *buf, MPI_Aint count, MPI_Datatype datatype,
					int dest, int tag, MPIR_Comm *comm, int context_offset,
					MPIR_Request **request)
{
	*request = NULL;
	return MPIDI_PSP_Isend(buf, count, datatype, dest, tag, comm, context_offset, request);
}

int MPIDI_PSP_Issend(const void * buf, MPI_Aint count, MPI_Datatype datatype,
		     int rank, int tag, MPIR_Comm * comm, int context_offset,
		     MPIR_Request ** request);
MPL_STATIC_INLINE_PREFIX int MPID_Issend(const void *buf, MPI_Aint count, MPI_Datatype datatype,
					 int dest, int tag, MPIR_Comm *comm, int context_offset,
					 MPIR_Request **request)
{
	*request = NULL;
	return MPIDI_PSP_Issend(buf, count, datatype, dest, tag, comm, context_offset, request);
}

int MPIDI_PSP_Irecv(void * buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
		    MPIR_Comm * comm, int context_offset, MPIR_Request ** request);
MPL_STATIC_INLINE_PREFIX int MPID_Irecv(void * buf, MPI_Aint count, MPI_Datatype datatype, int rank, int tag,
					MPIR_Comm * comm, int context_offset, MPIR_Request ** request)
{
	*request = NULL;
	return MPIDI_PSP_Irecv(buf, count, datatype, rank, tag, comm, context_offset, request);
}

int MPIDI_PSP_Imrecv(void *buf, int count, MPI_Datatype datatype, MPIR_Request *message,
		     MPIR_Request **request);
MPL_STATIC_INLINE_PREFIX int MPID_Imrecv(void *buf, int count, MPI_Datatype datatype, MPIR_Request *message,
					 MPIR_Request **request)
{
	*request = NULL;
	return MPIDI_PSP_Imrecv(buf, count, datatype, message, request);
}

/*
  Device override hooks for asynchronous progress threads
*/
MPL_STATIC_INLINE_PREFIX int MPID_Init_async_thread(void)
{
    return MPIR_Init_async_thread();
}

MPL_STATIC_INLINE_PREFIX int MPID_Finalize_async_thread(void)
{
    return MPIR_Finalize_async_thread();
}

MPL_STATIC_INLINE_PREFIX int MPID_Test(MPIR_Request * request_ptr, int *flag, MPI_Status * status)
{
    return MPIR_Test_impl(request_ptr, flag, status);
}

MPL_STATIC_INLINE_PREFIX int MPID_Testall(int count, MPIR_Request * request_ptrs[],
                                          int *flag, MPI_Status array_of_statuses[],
                                          int requests_property)
{
    return MPIR_Testall_impl(count, request_ptrs, flag, array_of_statuses, requests_property);
}

MPL_STATIC_INLINE_PREFIX int MPID_Testany(int count, MPIR_Request * request_ptrs[],
                                          int *indx, int *flag, MPI_Status * status)
{
    return MPIR_Testany_impl(count, request_ptrs, indx, flag, status);
}

MPL_STATIC_INLINE_PREFIX int MPID_Testsome(int incount, MPIR_Request * request_ptrs[],
                                           int *outcount, int array_of_indices[],
                                           MPI_Status array_of_statuses[])
{
    return MPIR_Testsome_impl(incount, request_ptrs, outcount, array_of_indices, array_of_statuses);
}

MPL_STATIC_INLINE_PREFIX int MPID_Waitall(int count, MPIR_Request * request_ptrs[],
                                          MPI_Status array_of_statuses[], int request_properties)
{
    return MPIR_Waitall_impl(count, request_ptrs, array_of_statuses, request_properties);
}

MPL_STATIC_INLINE_PREFIX int MPID_Wait(MPIR_Request * request_ptr, MPI_Status * status)
{
    return MPIR_Wait_impl(request_ptr, status);
}

MPL_STATIC_INLINE_PREFIX int MPID_Waitany(int count, MPIR_Request * request_ptrs[],
                                          int *indx, MPI_Status * status)
{
    return MPIR_Waitany_impl(count, request_ptrs, indx, status);
}

MPL_STATIC_INLINE_PREFIX int MPID_Waitsome(int incount, MPIR_Request * request_ptrs[],
                                           int *outcount, int array_of_indices[],
                                           MPI_Status array_of_statuses[])
{
    return MPIR_Waitsome_impl(incount, request_ptrs, outcount, array_of_indices, array_of_statuses);
}
#endif /* _MPIDPOST_H_ */
