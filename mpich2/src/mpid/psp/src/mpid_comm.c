/*
 * ParaStation
 *
 * Copyright (C) 2006-2020 ParTec Cluster Competence Center GmbH, Munich
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

static
int get_my_shmem_split_color(MPIR_Comm * comm_ptr)
{
	int color = MPI_UNDEFINED;

	if(!MPIDI_Process.env.enable_smp_awareness) {
		return comm_ptr->rank;
	}

#if 1
	color =  MPIDI_Process.smp_node_id;
#else
	/* We now use MPIDI_Process.smp_node_id as the color (see above).
	   The former code for the color determination looked like this: */

	if(MPIDI_Process.env.enable_ondemand) {
		/* In the PSP_ONDEMAND=1 case, we cannot check reliably for CON_TYPE_SHM,
		   so we switch to the host_hash approach, accepting the possibility of
		   hash collisions that may lead to undefined situations... */
		return MPID_PSP_get_host_hash();
	}

	/* The following would not work if PSP_SHM=0 is set! */
	for(i=0; i<comm_ptr->local_size; i++) {
		if( (comm_ptr->vcr[i]->con->type == PSCOM_CON_TYPE_SHM) || (comm_ptr->rank == i) ) {
			color = i;
			break;
		}
	}
#endif

	return color;
}

int MPID_PSP_split_type(MPIR_Comm * comm_ptr, int split_type, int key,
			MPIR_Info * info_ptr, MPIR_Comm ** newcomm_ptr)
{
	int mpi_errno = MPI_SUCCESS;

	if(split_type == MPI_COMM_TYPE_SHARED) {
		int color;

		color = get_my_shmem_split_color(comm_ptr);

		mpi_errno = MPIR_Comm_split_impl(comm_ptr, color, key, newcomm_ptr);

		if(mpi_errno == MPI_SUCCESS) {
			mpi_errno = MPIR_Comm_set_attr_impl(*newcomm_ptr, MPIDI_Process.shm_attr_key, NULL, MPIR_ATTR_PTR);
		}
	} else if(split_type == MPIX_COMM_TYPE_MODULE) {
		int color;

		color = MPIDI_Process.msa_module_id;
		mpi_errno = MPIR_Comm_split_impl(comm_ptr, color, key, newcomm_ptr);

	} else {
		mpi_errno = MPIR_Comm_split_impl(comm_ptr,  MPI_UNDEFINED, key, newcomm_ptr);
	}

	return mpi_errno;
}

void MPIDI_PSP_pack_topology_badges(int** pack_msg, int* pack_size, MPIDI_PG_t *pg)
{
	int i;
	int* msg;
	MPIDI_PSP_topo_level_t *tl = pg->topo_levels;

	*pack_size = MPIDI_PSP_get_num_topology_levels(pg) * (pg->size + 3) * sizeof(int);
	*pack_msg = MPL_malloc(*pack_size * sizeof(int), MPL_MEM_OBJECT);

	msg = *pack_msg;
	while(tl) { // FIX ME: non-global badges need not to be exchanged!
		for(i=0; i<pg->size; i++, msg++) {
			*msg = tl->badge_table[i];
		}
		*msg = tl->degree;  msg++;
		*msg = tl->max_badge;  msg++;
		*msg = tl->badges_are_global;  msg++;
		tl=tl->next;
	}
}

void MPIDI_PSP_unpack_topology_badges(int* pack_msg, int pg_size, int num_levels, MPIDI_PSP_topo_level_t **levels)
{
	int i, j;
	int* msg;
	MPIDI_PSP_topo_level_t *level;

	*levels = NULL;

	msg = pack_msg;
	for(i=0; i<num_levels; i++) {

		level = MPL_malloc(sizeof(MPIDI_PSP_topo_level_t), MPL_MEM_OBJECT);

		level->badge_table = MPL_malloc(pg_size * sizeof(int), MPL_MEM_OBJECT);
		for(j=0; j<pg_size; j++) {
			level->badge_table[j] = msg[j];
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

static int MPIDI_PSP_comm_is_flat_on_level(MPIR_Comm *comm, MPIDI_PSP_topo_level_t *level)
{
	int i;
	int my_badge;

	assert(level->pg == MPIDI_Process.my_pg); // level must be local!
	my_badge = level->badge_table[MPIDI_Process.my_pg_rank];

	for(i=0; i<comm->local_size; i++) {

		if(likely(comm->vcr[i]->pg == MPIDI_Process.my_pg)) { // local process group

			assert(level->badge_table);
			if(my_badge != level->badge_table[comm->vcr[i]->pg_rank]) {
				return 0;
			}

		} else { // remote process group

			if(!level->badges_are_global) {
				return 0;
			} else {
				MPIDI_PSP_topo_level_t *ext_level = NULL;
				if(MPIDI_PSP_check_pg_for_level(level->degree, comm->vcr[i]->pg, &ext_level)) {

					assert(ext_level); // found remote level with identical degree
					assert(ext_level->badge_table);
					if(my_badge != ext_level->badge_table[comm->vcr[i]->pg_rank]) {
						return 0;
					}
				} else {
					return 0;
				}
			}
		}
	}
	return 1;
}

static
int MPIDI_PSP_get_max_badge_by_level(MPIDI_PSP_topo_level_t *level)
{
	MPIDI_PG_t *pg = MPIDI_Process.my_pg;
	int max_badge = level->max_badge;

	if(level->badges_are_global) { // check also the remote process groups

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
	}

	return max_badge;
}

static
int MPIDI_PSP_get_badge_by_level_and_comm_rank(MPIR_Comm *comm, MPIDI_PSP_topo_level_t *level, int rank)
{
	int badge;
	MPIDI_PSP_topo_level_t *ext_level = NULL;

	if(likely(comm->vcr[rank]->pg == MPIDI_Process.my_pg)) { // rank is in local process group

		assert(level->pg == MPIDI_Process.my_pg); // level must be local, too
		assert(level->badge_table);
		badge = level->badge_table[comm->vcr[rank]->pg_rank];

		if(badge == -1) {
			badge = MPIDI_PSP_get_max_badge_by_level(level) + 1; // plus 1 as wildcard for an unknown badge
		}
		return badge;
	}

	if(level->badges_are_global && MPIDI_PSP_check_pg_for_level(level->degree, comm->vcr[rank]->pg, &ext_level)) {

		assert(ext_level); // found remote level with identical degree
		assert(ext_level->badge_table);
		badge = ext_level->badge_table[comm->vcr[rank]->pg_rank];

		if(badge == -1) { // wildcard for an unknown badge
			badge = MPIDI_PSP_get_max_badge_by_level(level) + 1; // plus 1 as wildcard for an unknown badge
		}
		return badge;
	} else {
		return -1; // remote process group with unknown badge
	}
}

int MPID_Get_node_id(MPIR_Comm *comm, int rank, int *id_p)
{
	MPIDI_PSP_topo_level_t *tl = MPIDI_Process.my_pg->topo_levels;

	if(tl == NULL) {
		*id_p = -1;
		return MPI_ERR_OTHER;
	}

	while(tl->next && MPIDI_PSP_comm_is_flat_on_level(comm, tl)) {
		assert(tl->badge_table);
		tl = tl->next;
	}

	*id_p = MPIDI_PSP_get_badge_by_level_and_comm_rank(comm, tl, rank);

	return 0;
}

int MPID_Get_max_node_id(MPIR_Comm *comm, int *max_id_p)
{
	MPIDI_PSP_topo_level_t *tl = MPIDI_Process.my_pg->topo_levels;

	if(tl == NULL) {
		*max_id_p = 0;
		return MPI_ERR_OTHER;
	}

	while(tl->next && MPIDI_PSP_comm_is_flat_on_level(comm, tl)) {
		assert(tl->badge_table);
		tl = tl->next;
	}

	*max_id_p =  MPIDI_PSP_get_max_badge_by_level(tl) + 1; // plus 1 for the "unknown badge" wildcard

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

	return 0;
}

int MPID_PSP_get_host_hash(void)
{
       char host_name[MPI_MAX_PROCESSOR_NAME];
       int result_len;
       static int host_hash = 0;

       if(!host_hash) {
               MPID_Get_processor_name(host_name, MPI_MAX_PROCESSOR_NAME, &result_len);
               MPIDI_PG_Convert_id(host_name, &host_hash);
               assert(host_hash >= 0);
       }
       return host_hash;
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

#ifdef MPID_PSP_TOPOLOGY_AWARE_COLLOPS

	if(MPIDI_Process.env.enable_msa_awareness) {

		if(MPIDI_Process.msa_module_id < 0) {
			MPIDI_Process.msa_module_id = 0;
		}

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
	}

	if(MPIDI_Process.env.enable_smp_awareness) {

		if(MPIDI_Process.smp_node_id < 0) {

			int psp_shm = 1;
			pscom_env_get_uint(&psp_shm, "PSP_SHM");
			if (!MPIDI_Process.env.enable_ondemand && psp_shm) {
				/* In the PSP_ONDEMAND=0 case, we can (if SHM is not disabled) check the pscom connection types: */
				for (grank = 0; grank < pg_size; grank++) {
					pscom_connection_t *con = MPIDI_Process.grank2con[grank];
					if( (con->type == PSCOM_CON_TYPE_SHM) || (pg_rank == grank) ) {
						MPIDI_Process.smp_node_id = grank;
						break;
					}
				}
			} else {
				/* In the PSP_ONDEMAND=1 case, we have to use a hash of the host name: */
				MPIDI_Process.smp_node_id = MPID_PSP_get_host_hash();
				/* ...accepting the possibility of hash collisions that may lead to undefined situations! */
			}
		}

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
	}
#endif

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

#ifdef MPID_PSP_MSA_AWARENESS
	if ((comm->hierarchy_kind == MPIR_COMM_HIERARCHY_KIND__NODE) && (MPIDI_Process.env.enable_msa_awareness > 1)) {

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

	MPID_PSP_group_init(comm);

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

#ifdef MPID_PSP_MSA_AWARENESS
	if (comm->hierarchy_kind == MPIR_COMM_HIERARCHY_KIND__NODE) {
		if(comm->local_comm) {
			// Recursively release also further subcomm levels:
			assert(comm->comm_kind == MPIR_COMM_KIND__INTRACOMM);
			MPIR_Comm_release(comm->local_comm);
		}
	}
#endif

	if (!MPIDI_Process.env.enable_collectives) return MPI_SUCCESS;

	/* ToDo: Use comm Barrier before cleanup! */

	MPID_PSP_group_cleanup(comm);

	return MPI_SUCCESS;
}
