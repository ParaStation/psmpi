/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Authors:	Jens Hauke <hauke@par-tec.com>
 *         	Carsten Clauss <clauss@par-tec.com>
 */

#include <assert.h>
#include <unistd.h>
#include "mpidimpl.h"
#include "mpi-ext.h"
#include "mpl.h"
#include "errno.h"

struct MPIR_Commops MPIR_PSP_Comm_fns;
extern struct MPIR_Commops  *MPIR_Comm_fns;

int MPID_PSP_split_type(MPIR_Comm * comm_ptr, int split_type, int key,
			MPIR_Info * info_ptr, MPIR_Comm ** newcomm_ptr)
{
	int mpi_errno = MPI_SUCCESS;

	if(split_type == MPI_COMM_TYPE_SHARED) {
		int color;

		if(!MPIDI_Process.env.enable_smp_awareness) {
			// pretend that all ranks live on their own nodes:
			color = comm_ptr->rank;
		} else {
			color = MPIDI_Process.smp_node_id;
		}

		mpi_errno = MPIR_Comm_split_impl(comm_ptr, color, key, newcomm_ptr);

	} else if(split_type == MPIX_COMM_TYPE_MODULE) {
		int color;

		if(!MPIDI_Process.env.enable_msa_awareness) {
			// assume that all ranks live in the same module:
			color = 0;
		} else {
			color = MPIDI_Process.msa_module_id;
		}

		mpi_errno = MPIR_Comm_split_impl(comm_ptr, color, key, newcomm_ptr);

	} else if (split_type == MPIX_COMM_TYPE_NEIGHBORHOOD) {
		// we don't know how to handle this split types -> so hand it back to the upper MPICH layer:
		mpi_errno = MPIR_Comm_split_type(comm_ptr, split_type, key, info_ptr, newcomm_ptr);
	} else {
		mpi_errno = MPIR_Comm_split_impl(comm_ptr,  MPI_UNDEFINED, key, newcomm_ptr);
	}

	return mpi_errno;
}


#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS

int MPIDI_PSP_check_pg_for_level(int degree, MPIDI_PG_t *pg, MPIDI_PSP_topo_level_t **level)
{
	MPIDI_PSP_topo_level_t *tl = pg->topo_levels;

	while(tl) {
		if(tl->degree == degree) {
			if(level) *level = tl;
			return 1;
		}
		tl=tl->next;
	}
	return 0;
}

static
int MPIDI_PSP_create_badge_table(int my_badge, int pg_rank, int pg_size, int *max_badge, int **badge_table, int normalize)
{
	int rc;
	int grank;
	int remote_badge;

	if(*badge_table != NULL) {

		for (grank = 0; grank < pg_size; grank++) {
			if(my_badge == (*badge_table)[grank]) {
				my_badge = grank;
				break;
			}
		}

		MPL_free(*badge_table);
	}

	*badge_table = MPL_malloc(pg_size * sizeof(int), MPL_MEM_OBJECT);

	if(pg_rank != 0) {

		/* gather: */
		pscom_connection_t *con = MPIDI_Process.grank2con[0];
		assert(con);
		pscom_send(con, NULL, 0, &my_badge, sizeof(int));

		/* bcast: */
		rc = pscom_recv_from(con, NULL, 0, *badge_table, pg_size*sizeof(int));
		assert(rc == PSCOM_SUCCESS);

	} else {

		/* gather: */
		(*badge_table)[0] = my_badge;
		for(grank=1; grank < pg_size; grank++) {
			pscom_connection_t *con = MPIDI_Process.grank2con[grank];
			assert(con);
			rc = pscom_recv_from(con, NULL, 0, &remote_badge, sizeof(int));
			assert(rc == PSCOM_SUCCESS);
			(*badge_table)[grank] = remote_badge;
		}

		/* bcast: */
		for(grank=1; grank < pg_size; grank++) {
			pscom_connection_t *con = MPIDI_Process.grank2con[grank];
			pscom_send(con, NULL, 0, *badge_table, pg_size*sizeof(int));
		}
	}

	*max_badge = (*badge_table)[0];
	for(grank=1; grank < pg_size; grank++) {
		if((*badge_table)[grank] > *max_badge) *max_badge = (*badge_table)[grank];
	}

	if(*max_badge >= pg_size && normalize) {
		MPIDI_PSP_create_badge_table(my_badge, pg_rank, pg_size, max_badge, badge_table, normalize);
	}

	return MPI_SUCCESS;
}

static
int MPIDI_PSP_comm_is_global(MPIR_Comm *comm)
{
	int i;
	for(i=0; i<comm->local_size; i++) {
		if(comm->vcr[i]->pg != MPIDI_Process.my_pg) {
			return 1;
		}
	}
	return 0;
}

static
int MPIDI_PSP_get_max_badge_by_level(MPIDI_PSP_topo_level_t *level)
{
	MPIDI_PG_t *pg = MPIDI_Process.my_pg;
	int max_badge = level->max_badge;

	// check also the remote process groups:
	while(pg->next) {
		MPIDI_PSP_topo_level_t *ext_level = NULL;
		if(MPIDI_PSP_check_pg_for_level(level->degree, pg->next, &ext_level)) {
			assert(ext_level);
			if(ext_level->max_badge > max_badge) {
				max_badge = ext_level->max_badge;
			}
		}
		pg = pg->next;
	}

	return max_badge;
}

static
int MPIDI_PSP_get_badge_by_level_and_comm_rank(MPIR_Comm *comm, MPIDI_PSP_topo_level_t *level, int rank)
{
	MPIDI_PSP_topo_level_t *ext_level = NULL;
	assert(level->pg == MPIDI_Process.my_pg); // level must be local!

	if(likely(comm->vcr[rank]->pg == MPIDI_Process.my_pg)) { // rank is in local process group

		if(unlikely(!level->badge_table)) { // "dummy" level
			assert(level->max_badge == -1);
			goto badge_unknown;
		}

		if(!level->badges_are_global) {

			if(unlikely(MPIDI_PSP_comm_is_global(comm))) {
				// if own badges are not global, these are treated as "unknown" by other PGs
				goto badge_unknown;
			}
		}

		return level->badge_table[comm->vcr[rank]->pg_rank];

	} else { // rank is in a remote process group

		if(MPIDI_PSP_check_pg_for_level(level->degree, comm->vcr[rank]->pg, &ext_level)) {
			// found remote level with identical degree
			assert(ext_level);

			if(ext_level->badges_are_global) {
				assert(ext_level->badge_table); // <- "dummy" levels only valid on home PG!

				return ext_level->badge_table[comm->vcr[rank]->pg_rank];
			}
		}
	}

badge_unknown:
	return MPIDI_PSP_get_max_badge_by_level(level) + 1; // plus 1 as wildcard for an unknown badge
}

static
int MPIDI_PSP_comm_is_flat_on_level(MPIR_Comm *comm, MPIDI_PSP_topo_level_t *level)
{
	int i;
	int my_badge;

	assert(level->pg == MPIDI_Process.my_pg); // level must be local!
	my_badge = MPIDI_PSP_get_badge_by_level_and_comm_rank(comm, level, comm->rank);

	for(i=0; i<comm->local_size; i++) {
		if(MPIDI_PSP_get_badge_by_level_and_comm_rank(comm, level, i) != my_badge) {
			return 0;
		}
	}
	return 1;
}

int MPID_Get_badge(MPIR_Comm *comm, int rank, int *badge_p)
{
	MPIDI_PSP_topo_level_t *tl = MPIDI_Process.my_pg->topo_levels;

	if(tl == NULL) {
		*badge_p = -1;
		return MPI_ERR_OTHER;
	}

	while(tl->next && MPIDI_PSP_comm_is_flat_on_level(comm, tl)) {
		assert(tl->badge_table);
		tl = tl->next;
	}

	*badge_p = MPIDI_PSP_get_badge_by_level_and_comm_rank(comm, tl, rank);
	return MPI_SUCCESS;
}

int MPID_Get_max_badge(MPIR_Comm *comm, int *max_badge_p)
{
	MPIDI_PSP_topo_level_t *tl = MPIDI_Process.my_pg->topo_levels;

	if(tl == NULL) {
		*max_badge_p = 0;
		return MPI_ERR_OTHER;
	}

	while(tl->next && MPIDI_PSP_comm_is_flat_on_level(comm, tl)) {
		assert(tl->badge_table);
		tl = tl->next;
	}

	*max_badge_p =  MPIDI_PSP_get_max_badge_by_level(tl) + 1; // plus 1 for the "unknown badge" wildcard
	return MPI_SUCCESS;
}

#endif /* MPID_PSP_TOPOLOGY_AWARE_COLLOPS */


int MPID_Get_node_id(MPIR_Comm *comm, int rank, int *id_p)
{
	/* The node IDs are unique, but do not have to be ordered and contiguous,
	   nor do they have to be limited in value by the number of nodes!
	*/
	*id_p = MPIDI_Process.smp_node_id;
	return MPI_SUCCESS;
}

int MPID_Get_max_node_id(MPIR_Comm *comm, int *max_id_p)
{
	/* Since the node IDs are not necessarily ordered and contiguous,
	   we cannot determine a meaningful maximum here and therefore
	   exit with a non-fatal error. This shall then only disable
	   the creation of SMP-aware  communicators in the higher
	   MPICH layer (see MPIR_Find_local_and_external()).
	*/
	*max_id_p = 0;
	return MPI_ERR_OTHER;
}


int MPID_PSP_comm_init(int has_parent)
{
	MPIR_Comm * comm;
	int grank;
	int pg_id_num;
	char *parent_port;
	MPIDI_PG_t * pg_ptr;
	MPIDI_VCRT_t * vcrt;
	int mpi_errno = MPI_SUCCESS;

	int pg_rank = MPIDI_Process.my_pg_rank;
	int pg_size = MPIDI_Process.my_pg_size;
	char* pg_id_name = MPIDI_Process.pg_id_name;

	MPIDI_PSP_topo_level_t *topo_levels = NULL;


	/* Initialize and overload Comm_ops (currently merely used for comm_split_type) */
	memset(&MPIR_PSP_Comm_fns, 0, sizeof(MPIR_PSP_Comm_fns));
	MPIR_Comm_fns = &MPIR_PSP_Comm_fns;
	MPIR_Comm_fns->split_type = MPID_PSP_split_type;


	/*
	 * Initialize the MPI_COMM_WORLD object
	 */
	comm = MPIR_Process.comm_world;
	comm->rank        = pg_rank;
	comm->remote_size = pg_size;
	comm->local_size  = pg_size;

	vcrt = MPIDI_VCRT_Create(comm->remote_size);
	assert(vcrt);
	MPID_PSP_comm_set_vcrt(comm, vcrt);


	if(MPIDI_Process.env.enable_msa_awareness) {

		if(MPIDI_Process.msa_module_id < 0) {
			/* If no msa_module_id is set explicitly, use the appnum for this: */
			MPIDI_Process.msa_module_id = MPIR_Process.attrs.appnum;
		}

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
		if(MPIDI_Process.env.enable_msa_aware_collops) {

			int* module_badge_table = NULL;
			int module_max_badge = 0;
			MPIDI_PSP_topo_level_t *level = NULL;

			MPIDI_PSP_create_badge_table(MPIDI_Process.msa_module_id, pg_rank, pg_size, &module_max_badge, &module_badge_table, 0 /* normalize*/);
			assert(module_badge_table);

			level = MPL_malloc(sizeof(MPIDI_PSP_topo_level_t), MPL_MEM_OBJECT);
			level->badge_table = module_badge_table;
			level->max_badge = module_max_badge;
			level->degree = MPIDI_PSP_TOPO_LEVEL__MODULES;
			level->badges_are_global = 1;

			level->next = topo_levels;
			topo_levels = level;
		}
#endif
	}

	if(MPIDI_Process.env.enable_smp_awareness) {

		if(MPIDI_Process.smp_node_id < 0) {
			/* If no smp_node_id is set explicitly, use the pscom's node_id for this:
			   (...which is an int and might be negative. However, since we know that it actually
			   corresponds to the IPv4 address of the node, it is safe to force the most significant
			   bit to be unset so that it is positive and can thus also be used as a split color.)
			*/
			MPIDI_Process.smp_node_id = ((MPIR_Process.comm_world->pscom_socket->local_con_info.node_id)<<1)>>1;
		}

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
		if(MPIDI_Process.env.enable_smp_aware_collops) {

			int* node_badge_table = NULL;
			int node_max_badge = 0;
			MPIDI_PSP_topo_level_t *level = NULL;

			MPIDI_PSP_create_badge_table(MPIDI_Process.smp_node_id, pg_rank, pg_size, &node_max_badge, &node_badge_table, 1 /* normalize*/);
			assert(node_badge_table);

			level = MPL_malloc(sizeof(MPIDI_PSP_topo_level_t), MPL_MEM_OBJECT);
			level->badge_table = node_badge_table;
			level->max_badge = node_max_badge;
			level->degree = MPIDI_PSP_TOPO_LEVEL__NODES;
			level->badges_are_global = 0;

			level->next = topo_levels;
			topo_levels = level;
		}
#endif
	}


	/* Create my home PG for MPI_COMM_WORLD: */
	MPIDI_PG_Convert_id(pg_id_name, &pg_id_num);
	MPIDI_PG_Create(pg_size, pg_id_num, topo_levels, &pg_ptr);
	assert(pg_ptr == MPIDI_Process.my_pg);

	for (grank = 0; grank < pg_size; grank++) {
		/* MPIR_CheckDisjointLpids() in mpi/comm/intercomm_create.c expect
		   lpid to be smaller than 4096!!!
		   Else you will see an "Fatal error in MPI_Intercomm_create"
		*/

		pscom_connection_t *con = MPIDI_Process.grank2con[grank];

		pg_ptr->vcr[grank] = MPIDI_VC_Create(pg_ptr, grank, con, grank);
		comm->vcr[grank] = MPIDI_VC_Dup(pg_ptr->vcr[grank]);
	}

	mpi_errno = MPIR_Comm_commit(comm);
	assert(mpi_errno == MPI_SUCCESS);


	/*
	 * Initialize the MPI_COMM_SELF object
	 */
	comm = MPIR_Process.comm_self;
	comm->rank        = 0;
	comm->remote_size = 1;
	comm->local_size  = 1;

	vcrt = MPIDI_VCRT_Create(comm->remote_size);
	assert(vcrt);
	MPID_PSP_comm_set_vcrt(comm, vcrt);

	comm->vcr[0] = MPIDI_VC_Dup(MPIR_Process.comm_world->vcr[pg_rank]);

	mpi_errno = MPIR_Comm_commit(comm);
	assert(mpi_errno == MPI_SUCCESS);

	if (has_parent) {
		MPIR_Comm * comm_parent;

		mpi_errno = MPID_PSP_GetParentPort(&parent_port);
		assert(mpi_errno == MPI_SUCCESS);

		mpi_errno = MPID_Comm_connect(parent_port, NULL, 0,
					      MPIR_Process.comm_world, &comm_parent);
		if (mpi_errno != MPI_SUCCESS) {
			fprintf(stderr, "MPI_Comm_connect(parent) failed!\n");
			goto fn_fail;
		}

		assert(comm_parent != NULL);
		MPL_strncpy(comm_parent->name, "MPI_COMM_PARENT", MPI_MAX_OBJECT_NAME);
		MPIR_Process.comm_parent = comm_parent;
	}

fn_exit:
	return mpi_errno;
fn_fail:
	goto fn_exit;
}

int MPID_Comm_get_lpid(MPIR_Comm *comm_ptr, int idx, int * lpid_ptr, bool is_remote)
{
	if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM || is_remote) {
		*lpid_ptr = comm_ptr->vcr[idx]->lpid;
	} else {
		*lpid_ptr = comm_ptr->local_vcr[idx]->lpid;
	}

	return MPI_SUCCESS;
}

int MPID_Create_intercomm_from_lpids(MPIR_Comm *newcomm_ptr, int size, const int lpids[])
{
	int mpi_errno = MPI_SUCCESS;
	MPIR_Comm *commworld_ptr;
	MPIDI_VCRT_t *vcrt;
	int i;

	commworld_ptr = MPIR_Process.comm_world;
	/* Setup the communicator's vc table: remote group */
	vcrt = MPIDI_VCRT_Create(size);
	assert(vcrt);
	MPID_PSP_comm_set_vcrt(newcomm_ptr, vcrt);

	for (i=0; i<size; i++) {
		MPIDI_VC_t *vcr = NULL;

		/* For rank i in the new communicator, find the corresponding
		   virtual connection.  For lpids less than the size of comm_world,
		   we can just take the corresponding entry from comm_world.
		   Otherwise, we need to search through the process groups.
		*/
		/* printf( "[%d] Remote rank %d has lpid %d\n",
		   MPIR_Process.comm_world->rank, i, lpids[i] ); */
		if ((lpids[i] >=0) && (lpids[i] < commworld_ptr->remote_size)) {
			vcr = commworld_ptr->vcr[lpids[i]];
			assert(vcr);
		}
		else {
			/* We must find the corresponding vcr for a given lpid */
			/* For now, this means iterating through the process groups */
			MPIDI_PG_t *pg;
			int j;

			pg = MPIDI_Process.my_pg->next; /* (skip comm_world) */

			do {
				assert(pg);

				for (j=0; j<pg->size; j++) {

					if(!pg->vcr[j]) continue;

					if (pg->vcr[j]->lpid == lpids[i]) {
						/* Found vc for current lpid in another pg! */
						vcr = pg->vcr[j];
						break;
					}
				}
				pg = pg->next;
			} while (!vcr);
		}

		/* Note that his will increment the ref count for the associate PG if necessary.  */
		newcomm_ptr->vcr[i] = MPIDI_VC_Dup(vcr);
	}

fn_exit:
	return mpi_errno;
fn_fail:
	goto fn_exit;
}



int MPID_PSP_comm_create_hook(MPIR_Comm * comm)
{
	pscom_connection_t *con1st;
	int i;

	if (comm->mapper_head) {
		MPID_PSP_comm_create_mapper(comm);
	}

	comm->is_disconnected = 0;
	comm->is_checked_as_host_local = 0;
	comm->group = NULL;

	if (comm->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
		/* do nothing on Intercomms */
		comm->pscom_socket = NULL;
		return MPI_SUCCESS;
	}

	/* Use pscom_socket from the rank 0 connection ... */
	con1st = MPID_PSCOM_rank2connection(comm, 0);
	comm->pscom_socket = con1st ? con1st->socket : NULL;

	/* ... and test if connections from different sockets are used ... */
	for (i = 0; i < comm->local_size; i++) {
		if (comm->pscom_socket && MPID_PSCOM_rank2connection(comm, i) &&
		    (MPID_PSCOM_rank2connection(comm, i)->socket != comm->pscom_socket)) {
			/* ... and disallow the usage of comm->pscom_socket in this case.
			   This will disallow ANY_SOURCE receives on that communicator for older pscoms
			   ... but should be fixed/handled within the pscom layer as of pscom 5.2.0 */
			comm->pscom_socket = NULL;
			break;
		}
	}

#ifdef HAVE_LIBHCOLL
	hcoll_comm_create(comm, NULL);
#endif

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
	if ((comm->hierarchy_kind == MPIR_COMM_HIERARCHY_KIND__NODE) && (MPIDI_Process.env.enable_msa_aware_collops > 1)) {

		int mpi_errno;
		MPIDI_PSP_topo_level_t *tl = MPIDI_Process.my_pg->topo_levels;

		while(tl && MPIDI_PSP_comm_is_flat_on_level(comm, tl)) {
			assert(tl->badge_table);
			tl = tl->next;
		}

		if(tl) { // This subcomm is not flat -> attach a further subcomm level: (to be handled in SMP-aware collectives)
			assert(comm->comm_kind == MPIR_COMM_KIND__INTRACOMM);
			mpi_errno = MPIR_Comm_dup_impl(comm, &comm->local_comm); // we "misuse" local_comm for this purpose
			assert(mpi_errno == MPI_SUCCESS);
		}
	}
#endif

	if (!MPIDI_Process.env.enable_collectives) return MPI_SUCCESS;

#ifdef MPIDI_PSP_WITH_PSCOM_COLLECTIVES
	MPID_PSP_group_init(comm);
#endif

	/*
	printf("%s (comm:%p(%s, id:%08x, size:%u))\n",
	       __func__, comm, comm->name, comm->context_id, comm->local_size););
	*/
	return MPI_SUCCESS;
}

int MPID_PSP_comm_destroy_hook(MPIR_Comm * comm)
{
	MPIDI_VCRT_Release(comm->vcrt, comm->is_disconnected);
	comm->vcr = NULL;

	if (comm->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
		MPIDI_VCRT_Release(comm->local_vcrt, comm->is_disconnected);
	}

#ifdef HAVE_LIBHCOLL
	hcoll_comm_destroy(comm, NULL);
#endif

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS
	if (comm->hierarchy_kind == MPIR_COMM_HIERARCHY_KIND__NODE) {
		if(comm->local_comm) {
			// Recursively release also further subcomm levels:
			assert(comm->comm_kind == MPIR_COMM_KIND__INTRACOMM);
			MPIR_Comm_release(comm->local_comm);
		}
	}
#endif

	if (!MPIDI_Process.env.enable_collectives) return MPI_SUCCESS;

#ifdef MPIDI_PSP_WITH_PSCOM_COLLECTIVES
	/* ToDo: Use comm Barrier before cleanup! */
	MPID_PSP_group_cleanup(comm);
#endif

	return MPI_SUCCESS;
}
