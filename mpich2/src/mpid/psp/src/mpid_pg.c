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

int MPIDI_GPID_Get( MPIR_Comm *comm_ptr, int rank, MPIDI_Gpid *gpid )
{
	MPIDI_VC_t *vc;

	vc = comm_ptr->vcrt->vcr[rank];

	/* Get the process group id as an int */
	gpid->gpid[0] = vc->pg->id_num;
	gpid->gpid[1] = vc->pg_rank;

	return 0;
}

/* see intercomm_create.c: */
int MPIDI_GPID_GetAllInComm(MPIR_Comm *comm_ptr, int local_size,
			   MPIDI_Gpid local_gpids[], int *singlePG)
{
	int i;
	MPIDI_Gpid *gpid = local_gpids;
	int lastPGID = -1;

	if(singlePG) *singlePG = 1;
	for (i=0; i<comm_ptr->local_size; i++) {
		MPIDI_GPID_Get(comm_ptr, i, gpid);

		if (lastPGID != gpid->gpid[0]) {
			if (i == 0) {
				lastPGID = gpid->gpid[0];
			} else {
				if(singlePG) *singlePG = 0;
			}
		}
		gpid++;
	}
	return 0;
}

int MPIDI_GPID_ToLpidArray(int size, MPIDI_Gpid gpid[], uint64_t lpid[])
{
	int i, mpi_errno = MPI_SUCCESS;
	int pgid;
	MPIDI_PG_t *pg;

	for (i=0; i<size; i++) {

		pg = MPIDI_Process.my_pg;

		do {

			if(!pg) {
				/* Unknown process group! This can happen if two (or more) PG have been spawned in the meanwhile...
				   The best we can do here is to create a new one:
				   (size unknown, but at least as big as the current PG rank plus one)
				*/
				MPIDI_PG_t *new_pg;

				MPIDI_PG_Create(gpid->gpid[1]+1, gpid->gpid[0], NULL, &new_pg);
				assert(new_pg->lpids[gpid->gpid[1]] == MPIDI_PSP_INVALID_LPID);
				if (!MPIDI_Process.next_lpid) MPIDI_Process.next_lpid = MPIR_Process.comm_world->local_size;
				lpid[i] = MPIDI_Process.next_lpid++;
				new_pg->lpids[gpid->gpid[1]] = lpid[i];

				break;

			} else {

				pgid = pg->id_num;

				if (pgid == gpid->gpid[0]) {
					/* found the process group.  gpid->gpid[1] is the rank in this process group */

					/* Sanity check on size */
					if(gpid->gpid[1] >= pg->size) {
						/* This can happen if a new PG was created (see above) but the initially chosen size was too small
						   (which is quite likely to happen). Now, we have to re-size the PG: (ugly but effective...)
						*/
						int k = pg->size;

						pg->size = gpid->gpid[1]+1;
						pg->vcr = MPL_realloc(pg->vcr, sizeof(MPIDI_VC_t*)*pg->size, MPL_MEM_OBJECT);
						pg->lpids = MPL_realloc(pg->lpids, sizeof(uint64_t)*pg->size, MPL_MEM_OBJECT);
						pg->cons = MPL_realloc(pg->cons, sizeof(pscom_connection_t*)*pg->size, MPL_MEM_OBJECT);

						for(; k<pg->size; k++) {
							pg->vcr[k] = NULL;
							pg->lpids[k] = MPIDI_PSP_INVALID_LPID;
							pg->cons[k] = NULL;
						}
					}

					if(!pg->vcr[gpid->gpid[1]]) {
						assert(pg->lpids[gpid->gpid[1]] == MPIDI_PSP_INVALID_LPID);
						/* VCR not yet initialized (connection establishment still needed)
						   Assign next free LPID (MPIDI_Process.next_lpid):
						*/
						if (!MPIDI_Process.next_lpid) MPIDI_Process.next_lpid = MPIR_Process.comm_world->local_size;
						lpid[i] = MPIDI_Process.next_lpid++;
						/*printf("(%d) LPID NOT found! Assigned next lipd: %" PRIu64 "\n", getpid(), lpid[i]);*/
						pg->lpids[gpid->gpid[1]] = lpid[i];
						break;
					}

					lpid[i] = pg->vcr[gpid->gpid[1]]->lpid;

					break;
				}
				pg = pg->next;
			}
		} while (1);

		gpid++;
	}

	return mpi_errno;
}


#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
static int MPIDI_PSP_get_num_topology_levels(MPIDI_PG_t *pg);
static void MPIDI_PSP_pack_topology_badges(int** pack_msg, int* msg_size, MPIDI_PG_t *pg);
static void MPIDI_PSP_unpack_topology_badges(int* pack_msg, int pg_size, int num_levels, MPIDI_PSP_topo_level_t **levels);
static int MPIDI_PSP_add_topo_levels_to_pg(MPIDI_PG_t *pg, MPIDI_PSP_topo_level_t *level);
#endif


/* The following is a temporary hook to ensure that all processes in
   a communicator have a set of process groups.

   All arguments are input (all processes in comm must have gpids)

   First: all processes check to see if they have information on all
   of the process groups mentioned by id in the array of gpids.

   The local result is LANDed with Allreduce.
   If any process is missing process group information, then the
   root process broadcasts the process group information as a string;
   each process then uses this information to update to local process group
   information and then connects to the still missing remote partners.
*/
int MPIDI_PG_ForwardPGInfo( MPIR_Comm *peer_comm_ptr, MPIR_Comm *comm_ptr,
			   int nPGids, const MPIDI_Gpid gpids[], int root, int remote_leader, int cts_tag,
			   pscom_connection_t *peer_con, char *all_ports, pscom_socket_t *pscom_socket )
{
	MPIR_Errflag_t errflag = FALSE;
	int mpi_errno = MPI_SUCCESS;
	pscom_err_t rc;

	int i, j;
	MPIDI_PG_t *pg;
	int id_num;

	const MPIDI_Gpid *gpid_ptr;
	int all_found_local = 1;
	int all_found_remote;
	int pg_count_root = 0;
	int pg_count_local = 0;
	int pg_count_remote = 0;

	pg = MPIDI_Process.my_pg;
	do {
		assert(pg);
		pg_count_local++;
		pg = pg->next;

	} while (pg);

	gpid_ptr = gpids;
	for (i=0; i<nPGids; i++) {
		pg = MPIDI_Process.my_pg;
		do {
			if(!pg){
				/* We don't know this pgid... */
				all_found_local = 0;
				break;
			}
			id_num = pg->id_num;
			if(id_num == gpid_ptr->gpid[0]) {
				/* Found PG, but is the respective pg_rank also there? */
				for(j=0; j<pg->size; j++) {
					if( pg->vcr[j] && (pg->vcr[j]->pg_rank == gpid_ptr->gpid[1]) ) {
						/* Found pg_rank! */
						break;
					}
				}
				if(j==pg->size) {
					/* We don't know this pg_rank of the known pgid... */
					all_found_local = 0;
					break;
				}
			}
			pg = pg->next;
		} while (id_num != gpid_ptr->gpid[0]);
		gpid_ptr++;
	}

	/* See if everyone in local comm is happy: */
	mpi_errno = MPIR_Allreduce_impl( MPI_IN_PLACE, &all_found_local, 1, MPI_INT, MPI_LAND, comm_ptr, &errflag );
	assert(mpi_errno == MPI_SUCCESS);

	/* See if remote procs are happy, too: */
	if(comm_ptr->rank == root) {
		if(peer_comm_ptr) {
			mpi_errno = MPIC_Sendrecv(&all_found_local,  1, MPI_INT,
						  remote_leader, cts_tag,
						  &all_found_remote, 1, MPI_INT,
						  remote_leader, cts_tag,
						  peer_comm_ptr, MPI_STATUS_IGNORE, &errflag);
			assert(mpi_errno == MPI_SUCCESS);
		} else {
			assert(peer_con);
			pscom_send(peer_con, NULL, 0, &all_found_local, sizeof(int));
			rc = pscom_recv_from(peer_con, NULL, 0, &all_found_remote, sizeof(int));
			assert(rc == PSCOM_SUCCESS);
		}
	}

	/* Check if we can stop this here because all procs involved are happy: */
	mpi_errno = MPIR_Bcast_impl(&all_found_remote, 1, MPI_INT, root, comm_ptr, &errflag);
	assert(mpi_errno == MPI_SUCCESS);

	if(all_found_local && all_found_remote) {
		/* Oh Happy Day! :-) We can leave this here without further ado!
		   (Quite likely we are dealing here with a non-spawn case...)
		*/
		return MPI_SUCCESS;
	}

	if(comm_ptr->rank == root) {

		/* Initially, make sure that all remote PGs are known at root! */

		int* local_pg_ids;
		int* local_pg_sizes;
		int* remote_pg_ids;
		int* remote_pg_sizes;
		int  new_pg_count = 0;
#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
		int  max_pg_count;
		int*  local_pg_topo_levels;
		int** local_pg_topo_badges;
		int*  local_pg_topo_msglen;
		int*  remote_pg_topo_levels;
		int** remote_pg_topo_badges;
		int*  remote_pg_topo_msglen;
#endif

		if(peer_comm_ptr) {
			mpi_errno = MPIC_Sendrecv(&pg_count_local, 1, MPI_INT,
						  remote_leader, cts_tag,
						  &pg_count_remote, 1, MPI_INT,
						  remote_leader, cts_tag,
						  peer_comm_ptr, MPI_STATUS_IGNORE, &errflag);
			assert(mpi_errno == MPI_SUCCESS);

		} else {
			assert(peer_con);
			pscom_send(peer_con, NULL, 0, &pg_count_local, sizeof(int));
			rc = pscom_recv_from(peer_con, NULL, 0, &pg_count_remote, sizeof(int));
			assert(rc == PSCOM_SUCCESS);
		}

		local_pg_ids   = MPL_malloc(pg_count_local * sizeof(int), MPL_MEM_OBJECT);
		local_pg_sizes = MPL_malloc(pg_count_local * sizeof(int), MPL_MEM_OBJECT);
		remote_pg_ids   = MPL_malloc(pg_count_remote * sizeof(int), MPL_MEM_OBJECT);
		remote_pg_sizes = MPL_malloc(pg_count_remote * sizeof(int), MPL_MEM_OBJECT);

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
		max_pg_count = pg_count_local > pg_count_remote ? pg_count_local : pg_count_remote;

		local_pg_topo_levels = MPL_malloc(max_pg_count * sizeof(int), MPL_MEM_OBJECT);
		local_pg_topo_badges = MPL_malloc(max_pg_count * sizeof(int*), MPL_MEM_OBJECT);
		local_pg_topo_msglen = MPL_malloc(max_pg_count * sizeof(int), MPL_MEM_OBJECT);
		remote_pg_topo_levels = MPL_malloc(max_pg_count * sizeof(int), MPL_MEM_OBJECT);
		remote_pg_topo_badges = MPL_malloc(max_pg_count * sizeof(int*), MPL_MEM_OBJECT);
		remote_pg_topo_msglen = MPL_malloc(max_pg_count * sizeof(int), MPL_MEM_OBJECT);
		for(i=0; i<max_pg_count; i++) {
			local_pg_topo_badges[i] = NULL;
			local_pg_topo_msglen[i] = 0;
			remote_pg_topo_badges[i] = NULL;
			remote_pg_topo_msglen[i] = 0;
		}
#endif

		pg = MPIDI_Process.my_pg;
		for(i=0; i<pg_count_local; i++) {
			local_pg_ids[i] = pg->id_num;
			local_pg_sizes[i] = pg->size;
#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
			local_pg_topo_levels[i] = MPIDI_PSP_get_num_topology_levels(pg);
			MPIDI_PSP_pack_topology_badges(&local_pg_topo_badges[i], &local_pg_topo_msglen[i], pg);
#endif
			pg = pg->next;
		}

		if(peer_comm_ptr) {
			mpi_errno = MPIC_Sendrecv(local_pg_ids, pg_count_local, MPI_INT,
						  remote_leader, cts_tag,
						  remote_pg_ids, pg_count_remote, MPI_INT,
						  remote_leader, cts_tag,
						  peer_comm_ptr, MPI_STATUS_IGNORE, &errflag);
			assert(mpi_errno == MPI_SUCCESS);

			mpi_errno = MPIC_Sendrecv(local_pg_sizes, pg_count_local, MPI_INT,
						  remote_leader, cts_tag,
						  remote_pg_sizes, pg_count_remote, MPI_INT,
						  remote_leader, cts_tag,
						  peer_comm_ptr, MPI_STATUS_IGNORE, &errflag);
			assert(mpi_errno == MPI_SUCCESS);

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
			mpi_errno = MPIC_Sendrecv(local_pg_topo_levels, pg_count_local, MPI_INT,
						  remote_leader, cts_tag,
						  remote_pg_topo_levels, pg_count_remote, MPI_INT,
						  remote_leader, cts_tag,
						  peer_comm_ptr, MPI_STATUS_IGNORE, &errflag);
			assert(mpi_errno == MPI_SUCCESS);

			mpi_errno = MPIC_Sendrecv(local_pg_topo_msglen, pg_count_local, MPI_INT,
						  remote_leader, cts_tag,
						  remote_pg_topo_msglen, pg_count_remote, MPI_INT,
						  remote_leader, cts_tag,
						  peer_comm_ptr, MPI_STATUS_IGNORE, &errflag);
			assert(mpi_errno == MPI_SUCCESS);

			for(i=0; i<max_pg_count; i++) {
				remote_pg_topo_badges[i] = MPL_malloc(remote_pg_topo_msglen[i] * sizeof(int), MPL_MEM_OBJECT);
				mpi_errno = MPIC_Sendrecv(local_pg_topo_badges[i], local_pg_topo_msglen[i], MPI_BYTE,
							  remote_leader, cts_tag,
							  remote_pg_topo_badges[i], remote_pg_topo_msglen[i], MPI_BYTE,
							  remote_leader, cts_tag,
							  peer_comm_ptr, MPI_STATUS_IGNORE, &errflag);
				assert(mpi_errno == MPI_SUCCESS);
				MPL_free(local_pg_topo_badges[i]);
			}
#endif
		} else {
			assert(peer_con);
			pscom_send(peer_con, NULL, 0, local_pg_ids, pg_count_local * sizeof(int));
			rc = pscom_recv_from(peer_con, NULL, 0, remote_pg_ids, pg_count_remote * sizeof(int));
			assert(rc == PSCOM_SUCCESS);

			pscom_send(peer_con, NULL, 0, local_pg_sizes, pg_count_local * sizeof(int));
			rc = pscom_recv_from(peer_con, NULL, 0, remote_pg_sizes, pg_count_remote * sizeof(int));
			assert(rc == PSCOM_SUCCESS);
#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
			pscom_send(peer_con, NULL, 0, local_pg_topo_levels, pg_count_local * sizeof(int));
			rc = pscom_recv_from(peer_con, NULL, 0, remote_pg_topo_levels, pg_count_remote * sizeof(int));
			assert(rc == PSCOM_SUCCESS);

			pscom_send(peer_con, NULL, 0, local_pg_topo_msglen, pg_count_local * sizeof(int));
			rc = pscom_recv_from(peer_con, NULL, 0, remote_pg_topo_msglen, pg_count_remote * sizeof(int));
			assert(rc == PSCOM_SUCCESS);

			for(i=0; i<max_pg_count; i++) {
				remote_pg_topo_badges[i] = MPL_malloc(remote_pg_topo_msglen[i] * sizeof(int), MPL_MEM_OBJECT);
				pscom_send(peer_con, NULL, 0, local_pg_topo_badges[i], local_pg_topo_msglen[i]);
				rc = pscom_recv_from(peer_con, NULL, 0, remote_pg_topo_badges[i], remote_pg_topo_msglen[i]);
				assert(rc == PSCOM_SUCCESS);
				MPL_free(local_pg_topo_badges[i]);
			}
#endif
		}

		for(i=0; i<pg_count_remote; i++) {

			int found = 0;
			int needed = 0;

			pg = MPIDI_Process.my_pg;
			for(j=0; j<pg_count_local; j++) {
				assert(pg);
				if(remote_pg_ids[i] == pg->id_num) {
					found = 1;
					break;
				}
				pg = pg->next;
			}

			if(!found) {
				/* Unknown Process Group at root: Check if it is actually needed! */
				gpid_ptr = gpids;
				for (j=0; j<nPGids; j++) {
					if(gpid_ptr->gpid[0] == remote_pg_ids[i]) {
						needed = 1;
						break;
					}
					gpid_ptr++;
				}

				if(needed) {
					MPIDI_PSP_topo_level_t* levels = NULL;
#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
					if(remote_pg_topo_msglen[i]) {
						MPIDI_PSP_unpack_topology_badges(remote_pg_topo_badges[i], remote_pg_sizes[i], remote_pg_topo_levels[i], &levels);
					} else {
						MPL_free(remote_pg_topo_badges[i]);
					}
					remote_pg_topo_badges[i] = NULL;
#endif
					MPIDI_PG_Create(remote_pg_sizes[i], remote_pg_ids[i], levels, NULL);
					new_pg_count++;
				}
			}
#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
			else {
				if(!pg->topo_levels && remote_pg_topo_levels[i] && remote_pg_topo_msglen[i]) { // PG already added (in MPIDI_GPID_ToLpidArray) but still without topo information
					MPIDI_PSP_topo_level_t* levels = NULL;
					MPIDI_PSP_unpack_topology_badges(remote_pg_topo_badges[i], remote_pg_sizes[i], remote_pg_topo_levels[i], &levels);
					remote_pg_topo_badges[i] = NULL;
					MPIDI_PSP_add_topo_levels_to_pg(pg, levels);
				}
			}
#endif
		}

		MPL_free(local_pg_ids);
		MPL_free(local_pg_sizes);
		MPL_free(remote_pg_ids);
		MPL_free(remote_pg_sizes);

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
		for(i=0; i<max_pg_count; i++) {
			if(remote_pg_topo_badges[i]) {
				MPL_free(remote_pg_topo_badges[i]);
			}
		}
		MPL_free(local_pg_topo_levels);
		MPL_free(local_pg_topo_badges);
		MPL_free(local_pg_topo_msglen);
		MPL_free(remote_pg_topo_levels);
		MPL_free(remote_pg_topo_badges);
		MPL_free(remote_pg_topo_msglen);
#endif
		pg_count_root = pg_count_local + new_pg_count;
		pg = MPIDI_Process.my_pg;
	}

	mpi_errno = MPIR_Bcast_impl(&pg_count_root, 1, MPI_INT, root, comm_ptr, &errflag);
	assert(mpi_errno == MPI_SUCCESS);

	for(i=0; i<pg_count_root; i++) {

		int found = 0;
		int needed = 0;

		int pg_size;
		int pg_id_num;
#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
		int *pg_topo_badges;
		int pg_topo_msglen;
		int pg_topo_num_levels;
#endif
		if(comm_ptr->rank == root) {
			assert(pg);
			pg_id_num = pg->id_num;
			pg_size   = pg->size;
#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
			MPIDI_PSP_pack_topology_badges(&pg_topo_badges, &pg_topo_msglen, pg);
			pg_topo_num_levels = MPIDI_PSP_get_num_topology_levels(pg);
#endif
		}

		mpi_errno = MPIR_Bcast_impl(&pg_size, 1, MPI_INT, root, comm_ptr, &errflag);
		assert(mpi_errno == MPI_SUCCESS);

		mpi_errno = MPIR_Bcast_impl(&pg_id_num, 1, MPI_INT, root, comm_ptr, &errflag);
		assert(mpi_errno == MPI_SUCCESS);

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
		mpi_errno = MPIR_Bcast_impl(&pg_topo_num_levels, 1, MPI_INT, root, comm_ptr, &errflag);
		assert(mpi_errno == MPI_SUCCESS);

		mpi_errno = MPIR_Bcast_impl(&pg_topo_msglen, 1, MPI_INT, root, comm_ptr, &errflag);
		assert(mpi_errno == MPI_SUCCESS);

		if(comm_ptr->rank != root) {
			pg_topo_badges = MPL_malloc(pg_topo_msglen * sizeof(int), MPL_MEM_OBJECT);
		}
		mpi_errno = MPIR_Bcast_impl(pg_topo_badges, pg_topo_msglen, MPI_BYTE, root, comm_ptr, &errflag);
		assert(mpi_errno == MPI_SUCCESS);
#endif
		if(comm_ptr->rank != root) {

			pg = MPIDI_Process.my_pg;
			for(j=0; j<pg_count_local; j++) {
				assert(pg);

				if(pg_id_num == pg->id_num) {
					found = 1;
					break;
				}
				pg = pg->next;
			}

			if(!found) {
				/* Unknown Process Group: Check if it is actually needed! */
				gpid_ptr = gpids;
				for (j=0; j<nPGids; j++) {
					if(gpid_ptr->gpid[0] == pg_id_num) {
						needed = 1;
						break;
					}
					gpid_ptr++;
				}

				if(needed) {
					/* New Process Group: */
					MPIDI_PSP_topo_level_t* levels = NULL;
#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
					if(pg_topo_msglen) {
						MPIDI_PSP_unpack_topology_badges(pg_topo_badges, pg_size, pg_topo_num_levels, &levels);
					} else {
						MPL_free(pg_topo_badges);
					}
					pg_topo_badges = NULL;
#endif
					MPIDI_PG_Create(pg_size, pg_id_num, levels, NULL);
				}
			}
#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
			else {
				if(!pg->topo_levels && pg_topo_num_levels && pg_topo_msglen) { // PG already added (in MPIDI_GPID_ToLpidArray) but still without topo information
					MPIDI_PSP_topo_level_t* levels = NULL;
					MPIDI_PSP_unpack_topology_badges(pg_topo_badges, pg_size, pg_topo_num_levels, &levels);
					pg_topo_badges = NULL;
					MPIDI_PSP_add_topo_levels_to_pg(pg, levels);
				}
			}
#endif
		}

		if(comm_ptr->rank == root) {
			pg = pg->next;
		}

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
		if(pg_topo_badges) {
			MPL_free(pg_topo_badges);
		}
#endif
	}

	/* Now it's time to establish the still needed connection: */
	{
		int local_size = 0;
		int remote_size = 0;

		pscom_port_str_t *all_ports_local = NULL;
		pscom_port_str_t *all_ports_remote = NULL;
		pscom_socket_t *comm_socket;

		MPIDI_Gpid my_gpid;
		MPIDI_Gpid *local_gpids_by_comm;
		MPIDI_Gpid *remote_gpids_by_comm;

		/* If we get here via MPIR_Intercomm_create_impl(), we have to open a new socket.
		   If not, a socket should already be opened in MPID_Comm_connect()/accept()... */
		if(pscom_socket) {
			comm_socket = pscom_socket;
			all_ports_local = (pscom_port_str_t*)all_ports;
		} else {
			MPIR_Comm intercomm_dummy;
			assert(!all_ports);
			/* We just want to get the socket, but open_all_ports() expects an (inter)comm.
			   So we fetch it via an intercomm_dummy: */
			all_ports_local = MPID_PSP_open_all_ports(root, comm_ptr, &intercomm_dummy);
			comm_socket = intercomm_dummy.pscom_socket;
		}

		local_size = comm_ptr->local_size;

		if(comm_ptr->rank == root) {
			if(peer_comm_ptr) {
				mpi_errno = MPIC_Sendrecv(&local_size,  1, MPI_INT,
							  remote_leader, cts_tag,
							  &remote_size, 1, MPI_INT,
							  remote_leader, cts_tag,
							  peer_comm_ptr, MPI_STATUS_IGNORE, &errflag);
				assert(mpi_errno == MPI_SUCCESS);

				all_ports_remote = MPL_malloc(remote_size * sizeof(pscom_port_str_t), MPL_MEM_STRINGS);
				memset(all_ports_remote, 0, remote_size * sizeof(pscom_port_str_t));

				mpi_errno = MPIC_Sendrecv(all_ports_local, local_size * sizeof(pscom_port_str_t), MPI_CHAR,
							  remote_leader, cts_tag,
							  all_ports_remote, remote_size * sizeof(pscom_port_str_t), MPI_CHAR,
							  remote_leader, cts_tag,
							  peer_comm_ptr, MPI_STATUS_IGNORE, &errflag);
				assert(mpi_errno == MPI_SUCCESS);
			} else {
				assert(peer_con);

				pscom_send(peer_con, NULL, 0, &local_size, sizeof(int));
				rc = pscom_recv_from(peer_con, NULL, 0, &remote_size, sizeof(int));
				assert(rc == PSCOM_SUCCESS);

				all_ports_remote = MPL_malloc(remote_size * sizeof(pscom_port_str_t), MPL_MEM_STRINGS);
				memset(all_ports_remote, 0, remote_size * sizeof(pscom_port_str_t));

				pscom_send(peer_con, NULL, 0, all_ports_local, local_size * sizeof(pscom_port_str_t));
				rc = pscom_recv_from(peer_con, NULL, 0, all_ports_remote, remote_size * sizeof(pscom_port_str_t));
				assert(rc == PSCOM_SUCCESS);
			}
		}

		mpi_errno = MPIR_Bcast_impl(&remote_size, 1, MPI_INT, root, comm_ptr, &errflag);
		assert(mpi_errno == MPI_SUCCESS);

		if(comm_ptr->rank != root) {
			all_ports_remote = MPL_malloc(remote_size * sizeof(pscom_port_str_t), MPL_MEM_STRINGS);
			memset(all_ports_remote, 0, remote_size * sizeof(pscom_port_str_t));
		}

		mpi_errno = MPIR_Bcast_impl(all_ports_remote, remote_size * sizeof(pscom_port_str_t), MPI_CHAR, root, comm_ptr, &errflag);
		assert(mpi_errno == MPI_SUCCESS);

		MPIDI_GPID_Get(comm_ptr, comm_ptr->rank, &my_gpid);

		local_gpids_by_comm = (MPIDI_Gpid*)MPL_malloc(local_size * sizeof(MPIDI_Gpid), MPL_MEM_OBJECT);
		remote_gpids_by_comm = (MPIDI_Gpid*)MPL_malloc(remote_size * sizeof(MPIDI_Gpid), MPL_MEM_OBJECT);

		mpi_errno = MPIR_Gather_allcomm_auto(&my_gpid, sizeof(MPIDI_Gpid), MPI_CHAR,
					      local_gpids_by_comm, sizeof(MPIDI_Gpid), MPI_CHAR,
					      root, comm_ptr, &errflag);
		assert(mpi_errno == MPI_SUCCESS);

		if(comm_ptr->rank == root) {
			if(peer_comm_ptr) {
				mpi_errno = MPIC_Sendrecv(local_gpids_by_comm, sizeof(MPIDI_Gpid) * local_size, MPI_CHAR,
							  remote_leader, cts_tag,
							  remote_gpids_by_comm, sizeof(MPIDI_Gpid) * remote_size, MPI_CHAR,
							  remote_leader, cts_tag,
							  peer_comm_ptr, MPI_STATUS_IGNORE, &errflag);
				assert(mpi_errno == MPI_SUCCESS);
			} else {
				assert(peer_con);
				pscom_send(peer_con, NULL, 0, local_gpids_by_comm, sizeof(MPIDI_Gpid) * local_size);
				rc = pscom_recv_from(peer_con, NULL, 0, remote_gpids_by_comm, sizeof(MPIDI_Gpid) * remote_size);
				assert(rc == PSCOM_SUCCESS);
			}
		}

		mpi_errno = MPIR_Bcast_impl(remote_gpids_by_comm, sizeof(MPIDI_Gpid) * remote_size, MPI_CHAR, root, comm_ptr, &errflag);
		assert(mpi_errno == MPI_SUCCESS);

		/* FIRST RUN: call pscom_connect_socket_str() to establish all needed connections */
		gpid_ptr = gpids;
		for (i=0; i<nPGids; i++) {
			pg = MPIDI_Process.my_pg;
			do {
				id_num = pg->id_num;

				for (j=0; j<pg->size; j++) {

					if((gpid_ptr->gpid[0] == id_num) && (gpid_ptr->gpid[1] == j)) {

						if(pg->vcr[j]) {
							assert(j==pg->vcr[j]->pg_rank);
						} else {
							if(!pg->cons[j]) {
								int pos;
								MPIDI_Gpid *remote_gpid_ptr;
								pscom_connection_t *con = pscom_open_connection(comm_socket);

								remote_gpid_ptr = remote_gpids_by_comm;
								for(pos=0; pos<remote_size; pos++) {

									if((remote_gpid_ptr->gpid[0] == gpid_ptr->gpid[0]) &&
									   (remote_gpid_ptr->gpid[1] == gpid_ptr->gpid[1])) {
										break;
									}

									remote_gpid_ptr++;
								}
								assert(pos<remote_size);

								rc = pscom_connect_socket_str(con, all_ports_remote[pos]);
								assert(rc == PSCOM_SUCCESS);

								pg->cons[j] = con;
							} else {
								assert(pg->lpids[j] != MPIDI_PSP_INVALID_LPID);
							}
						}
					}
				}

				pg = pg->next;

			} while (pg);

			gpid_ptr++;
		}


		/* Workaround for timing of pscom ondemand connections. Be sure both sides have called
		   pscom_connect_socket_str() before using the connections: */
		MPIR_Barrier_impl(comm_ptr, &errflag);
		if(comm_ptr->rank == root) {
			if(peer_comm_ptr) {
				mpi_errno = MPIC_Sendrecv(NULL, 0, MPI_CHAR, remote_leader, cts_tag, NULL, 0, MPI_CHAR,
							  remote_leader, cts_tag,  peer_comm_ptr, MPI_STATUS_IGNORE, &errflag);
				assert(mpi_errno == MPI_SUCCESS);
			} else {
				int dummy = -1;
				assert(peer_con);
				pscom_send(peer_con, NULL, 0, &dummy, sizeof(int));
				rc = pscom_recv_from(peer_con, NULL, 0, &dummy, sizeof(int));
				assert(rc == PSCOM_SUCCESS);
			}
		}
		MPIR_Barrier_impl(comm_ptr, &errflag);

		/* SECOND RUN: store, check and warm-up all new connections: */
		gpid_ptr = gpids;
		for (i=0; i<nPGids; i++) {
			pg = MPIDI_Process.my_pg;
			do {
				id_num = pg->id_num;

				for (j=0; j<pg->size; j++) {

					if((gpid_ptr->gpid[0] == id_num) && (gpid_ptr->gpid[1] == j)) {

						if(pg->vcr[j]) {
							assert(j==pg->vcr[j]->pg_rank);
						} else {
							int pos;
							MPIDI_Gpid *remote_gpid_ptr;
							pscom_connection_t *con = pg->cons[j];

							remote_gpid_ptr = remote_gpids_by_comm;
							for(pos=0; pos<remote_size; pos++) {

								if((remote_gpid_ptr->gpid[0] == gpid_ptr->gpid[0]) &&
								   (remote_gpid_ptr->gpid[1] == gpid_ptr->gpid[1])) {
									break;
								}

								remote_gpid_ptr++;
							}
							assert(pos<remote_size);

							assert(con);

							if (pg->lpids[j] != MPIDI_PSP_INVALID_LPID) {
								pg->vcr[j] = MPIDI_VC_Create(pg, j, con, pg->lpids[j]);
							} else {
								if (!MPIDI_Process.next_lpid) MPIDI_Process.next_lpid = MPIR_Process.comm_world->local_size;

								/* Using the next so far unused lpid > np.*/
								pg->vcr[j] = MPIDI_VC_Create(pg, j, con, MPIDI_Process.next_lpid++);
							}

							/* Sanity check and connection warm-up: */
							if(!MPIDI_Process.env.enable_ondemand_spawn)
							{
								int remote_pg_id;
								int remote_pg_rank;

								/*
								printf("(%d) [%d|%d] --> [%d|%d] %s\n", getpid(), MPIDI_Process.my_pg->id_num, MPIR_Process.comm_world->rank,
									  gpid_ptr[0], gpid_ptr[1], all_ports_remote[pos]);
								*/

								if(MPIDI_Process.my_pg->id_num < gpid_ptr->gpid[0]) {
									pscom_send(con, NULL, 0, &MPIDI_Process.my_pg->id_num, sizeof(int));
									rc = pscom_recv_from(con, NULL, 0, &remote_pg_id, sizeof(int));
									assert(rc == PSCOM_SUCCESS);
									assert(remote_pg_id == gpid_ptr->gpid[0]);

									pscom_send(con, NULL, 0, &MPIR_Process.comm_world->rank, sizeof(int));
									rc = pscom_recv_from(con, NULL, 0, &remote_pg_rank, sizeof(int));
									assert(rc == PSCOM_SUCCESS);
									assert(remote_pg_rank == gpid_ptr->gpid[1]);
								} else {
									rc = pscom_recv_from(con, NULL, 0, &remote_pg_id, sizeof(int));
									assert(rc == PSCOM_SUCCESS);
									assert(remote_pg_id == gpid_ptr->gpid[0]);
									pscom_send(con, NULL, 0, &MPIDI_Process.my_pg->id_num, sizeof(int));

									rc = pscom_recv_from(con, NULL, 0, &remote_pg_rank, sizeof(int));
									assert(rc == PSCOM_SUCCESS);
									assert(remote_pg_rank == gpid_ptr->gpid[1]);
									pscom_send(con, NULL, 0, &MPIR_Process.comm_world->rank, sizeof(int));
								}
							}
						}
					}
				}

				pg = pg->next;

			} while (pg);

			gpid_ptr++;
		}

		MPL_free(local_gpids_by_comm);
		MPL_free(remote_gpids_by_comm);
		MPL_free(all_ports_remote);
		if(!pscom_socket) {
			MPL_free(all_ports_local);
		}

		pscom_stop_listen(comm_socket);
	}

	return MPI_SUCCESS;
}


#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS

static
int MPIDI_PSP_get_num_topology_levels(MPIDI_PG_t *pg)
{
	int level_count = 0;
	MPIDI_PSP_topo_level_t *tl = pg->topo_levels;

	while(tl) {
		level_count++;
		tl=tl->next;
	}
	return level_count;
}

static
void MPIDI_PSP_pack_topology_badges(int** pack_msg, int* pack_size, MPIDI_PG_t *pg)
{
	int i;
	int* msg;
	MPIDI_PSP_topo_level_t *tl = pg->topo_levels;

	*pack_size = MPIDI_PSP_get_num_topology_levels(pg) * (pg->size + 3) * sizeof(int);
	*pack_msg = MPL_malloc(*pack_size * sizeof(int), MPL_MEM_OBJECT);

	msg = *pack_msg;
	while(tl) { // FIX ME: non-global badges and "dummy" tables need not to be exchanged!
		for(i=0; i<pg->size; i++, msg++) {
			if(tl->badge_table) {
				*msg = tl->badge_table[i];
			} else {
				*msg = MPIDI_PSP_TOPO_BADGE__NULL;
			}
		}
		*msg = tl->degree;  msg++;
		*msg = tl->max_badge;  msg++;
		*msg = tl->badges_are_global;  msg++;
		tl=tl->next;
	}
}

static
void MPIDI_PSP_unpack_topology_badges(int* pack_msg, int pg_size, int num_levels, MPIDI_PSP_topo_level_t **levels)
{
	int i, j;
	int* msg;
	MPIDI_PSP_topo_level_t *level;

	*levels = NULL;

	msg = pack_msg;
	for(i=0; i<num_levels; i++) {

		level = MPL_malloc(sizeof(MPIDI_PSP_topo_level_t), MPL_MEM_OBJECT);

		if(*msg != MPIDI_PSP_TOPO_BADGE__NULL) {
			level->badge_table = MPL_malloc(pg_size * sizeof(int), MPL_MEM_OBJECT);
			for(j=0; j<pg_size; j++) {
				level->badge_table[j] = msg[j];
			}
		} else { // just a "dummy" table
			assert(msg[pg_size + 1] == MPIDI_PSP_TOPO_BADGE__NULL);
			level->badge_table = NULL;
		}
		level->degree = msg[pg_size];
		level->max_badge = msg[pg_size + 1];
		level->badges_are_global = msg[pg_size + 2];
		msg += (pg_size + 3);

		level->next = *levels;
		*levels = level;
	}
	MPL_free(pack_msg);
}

static
int MPIDI_PSP_add_topo_level_to_pg(MPIDI_PG_t *pg, MPIDI_PSP_topo_level_t *level)
{
	MPIDI_PSP_topo_level_t * tlnext = pg->topo_levels;

	if(!tlnext || tlnext->degree < level->degree) {
		level->next = tlnext;
		pg->topo_levels = level;
	} else {
		assert(tlnext->degree != level->degree);
		while (tlnext->next && tlnext->degree > level->degree)
		{
			tlnext = tlnext->next;
		}
		level->next = tlnext->next;
		tlnext->next = level;
	}
	level->pg = pg;

	return MPI_SUCCESS;
}

static
int MPIDI_PSP_add_topo_levels_to_pg(MPIDI_PG_t *pg, MPIDI_PSP_topo_level_t *levels)
{
	while(levels) {
		MPIDI_PSP_topo_level_t *level_next = levels->next;
		MPIDI_PSP_add_topo_level_to_pg(pg, levels);
		levels = level_next;
	}

	return MPI_SUCCESS;
}

static
int MPIDI_PSP_add_flat_level_to_pg(MPIDI_PG_t *pg, int degree)
{
	MPIDI_PSP_topo_level_t * level = MPL_malloc(sizeof(MPIDI_PSP_topo_level_t), MPL_MEM_OBJECT);

	level->degree = degree;
	level->badges_are_global = 1;
	level->max_badge = MPIDI_PSP_TOPO_BADGE__NULL;
	level->badge_table = NULL;

	return MPIDI_PSP_add_topo_level_to_pg(pg, level);
}

#endif /* MPID_PSP_TOPOLOGY_AWARE_COLLOPS */


int MPIDI_PG_Create(int pg_size, int pg_id_num, MPIDI_PSP_topo_level_t *levels, MPIDI_PG_t ** pg_ptr)
{
	MPIDI_PG_t * pg = NULL, *pgnext;
	int i;
	int mpi_errno = MPI_SUCCESS;
	MPIR_CHKPMEM_DECL(4);

	MPIR_FUNC_ENTER;

	MPIR_CHKPMEM_MALLOC(pg, MPIDI_PG_t* ,sizeof(MPIDI_PG_t), mpi_errno, "pg", MPL_MEM_OBJECT);
	MPIR_CHKPMEM_MALLOC(pg->vcr, MPIDI_VC_t**, sizeof(MPIDI_VC_t)*pg_size, mpi_errno,"pg->vcr", MPL_MEM_OBJECT);
	MPIR_CHKPMEM_MALLOC(pg->lpids, uint64_t*, sizeof(uint64_t)*pg_size, mpi_errno,"pg->lpids", MPL_MEM_OBJECT);
	MPIR_CHKPMEM_MALLOC(pg->cons, pscom_connection_t **, sizeof(pscom_connection_t*)*pg_size, mpi_errno,"pg->cons", MPL_MEM_OBJECT);

	pg->size = pg_size;
	pg->id_num = pg_id_num;
	pg->refcnt = 0;
#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
	pg->topo_levels = NULL;
#endif
	for(i=0; i<pg_size; i++) {
		pg->vcr[i] = NULL;
		pg->lpids[i] = MPIDI_PSP_INVALID_LPID;
		pg->cons[i] = NULL;
	}

	/* Add pg's at the tail so that comm world is always the first pg */
	pg->next = NULL;

	if (!MPIDI_Process.my_pg)
	{
		/* The first process group is always the world group */
		MPIDI_Process.my_pg = pg;
	}
	else
	{
		pgnext = MPIDI_Process.my_pg;
		while (pgnext->next)
		{
			pgnext = pgnext->next;
		}
		pgnext->next = pg;
	}

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
	MPIDI_PSP_add_topo_levels_to_pg(pg, levels);

	if(pg != MPIDI_Process.my_pg) { // This is for the rare case that joined PGs do not feature the same set of level degrees!

		MPIDI_PSP_topo_level_t *level = pg->topo_levels;
		while(level) { // If not known, add a flat badge table (as a "dummy") with same the degree to the home PG:
			MPIDI_PSP_topo_level_t *level_next = level->next;
			if(level->badges_are_global && !MPIDI_PSP_check_pg_for_level(level->degree, MPIDI_Process.my_pg, NULL)) {
				MPIDI_PSP_add_flat_level_to_pg(MPIDI_Process.my_pg, level->degree);
			}
			level = level_next;
		}
	}
#endif

	if(pg_ptr) *pg_ptr = pg;

fn_exit:
	MPIR_FUNC_EXIT;
	return mpi_errno;

fn_fail:
	MPIR_CHKPMEM_REAP();
	goto fn_exit;
}

MPIDI_PG_t* MPIDI_PG_Destroy(MPIDI_PG_t * pg_ptr)
{
	int j;
	MPIDI_PG_t * pg_next = pg_ptr->next;

	/* Check if this is the PG of the local COMM_WORLD.
	   If not, ensure that the list of PGs does not get broken: */
	if(pg_ptr != MPIDI_Process.my_pg) {
		MPIDI_PG_t * pg_run = MPIDI_Process.my_pg;
		assert(pg_run);
		while(pg_run->next != pg_ptr) {
			pg_run = pg_run->next;
			assert(pg_run);
		}
		pg_run->next = pg_next;
	}


	for (j=0; j<pg_ptr->size; j++) {

		assert( (pg_ptr->refcnt>0) || ((pg_ptr->refcnt==0)&&(!pg_ptr->vcr[j])) );

		if (pg_ptr->vcr[j]) {
			/* If MPIDI_PG_Destroy() is called with still existing connections,
			   then this PG has not been disconnected before. Hence, this is most
			   likely the common case, where an MPI_Finalize() is tearing down the
			   current session. Therefore, we just close the still open connections
			   and free the related VCR without any decreasing of reference counters:
			*/
			pscom_close_connection(pg_ptr->vcr[j]->con);
			MPL_free(pg_ptr->vcr[j]);

		} else  {

			if(pg_ptr->cons[j]) {
				/* If we come here, this rank has already been disconected in an
				   MPI sense but due to the 'lazy disconnect' feature, the old
				   pscom connection is still open. Hence, close it right here:
				*/
				pscom_close_connection(pg_ptr->cons[j]);
			}
		}
	}

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
	while(pg_ptr->topo_levels) {
		MPIDI_PSP_topo_level_t *level = pg_ptr->topo_levels;
		pg_ptr->topo_levels = level->next;
		MPL_free(level->badge_table);
		MPL_free(level);
	}
#endif
	MPL_free(pg_ptr->cons);
	MPL_free(pg_ptr->lpids);
	MPL_free(pg_ptr->vcr);
	MPL_free(pg_ptr);

	return pg_next;
}

/* Taken from MPIDI_PG_IdToNum() of CH3: */
void MPIDI_PG_Convert_id(char *pg_id_name, int *pg_id_num)
{
	const char *p = (const char *)pg_id_name;
	int pgid = 0;

	while (*p) {
		pgid += *p++;
		pgid += (pgid << 10);
		pgid ^= (pgid >> 6);
	}
	pgid += (pgid << 3);
	pgid ^= (pgid >> 11);
	pgid += (pgid << 15);

	/* restrict to 31 bits */
	*pg_id_num = (pgid & 0x7fffffff);
}
