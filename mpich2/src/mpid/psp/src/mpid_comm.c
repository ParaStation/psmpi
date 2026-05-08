/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <unistd.h>
#include "mpidimpl.h"
#include "mpid_psp_topo.h"
#include "mpi-ext.h"
#include "mpl.h"
#include "errno.h"

struct MPIR_Commops MPIR_PSP_Comm_fns;
extern struct MPIR_Commops *MPIR_Comm_fns;

int MPID_PSP_split_type(MPIR_Comm * comm_ptr, int split_type, int key,
                        MPIR_Info * info_ptr, MPIR_Comm ** newcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    if (split_type == MPI_COMM_TYPE_SHARED) {
        int color;

        if (MPIDI_Process.smp_node_id == MPIDI_PSP_NODE_ID_NO_LOCAL) {
            /* Pretend all ranks live on their own node for debugging purposes. */
            color = comm_ptr->rank;
        } else {
            color = MPIDI_Process.smp_node_id;
        }
        mpi_errno = MPIR_Comm_split_impl(comm_ptr, color, key, newcomm_ptr);

    } else if (split_type == MPIX_COMM_TYPE_MODULE) {
        int color;

        if (!MPIDI_Process.env.enable_msa_awareness) {
            // assume that all ranks live in the same module:
            color = 0;
        } else {
            color = MPIDI_Process.msa_module_id;
        }

        mpi_errno = MPIR_Comm_split_impl(comm_ptr, color, key, newcomm_ptr);

    } else if ((split_type == MPIX_COMM_TYPE_NEIGHBORHOOD) ||
               (split_type == MPI_COMM_TYPE_HW_GUIDED) ||
               (split_type == MPI_COMM_TYPE_HW_UNGUIDED)) {
        // we don't know how to handle this split types -> so hand it back to the upper MPICH layer:
        mpi_errno = MPIR_Comm_split_type(comm_ptr, split_type, key, info_ptr, newcomm_ptr);
    } else {
        mpi_errno = MPIR_Comm_split_impl(comm_ptr, MPI_UNDEFINED, key, newcomm_ptr);
    }

    return mpi_errno;
}


int MPID_PSP_comm_init(int has_parent)
{
    char *parent_ep_str;
    int mpi_errno = MPI_SUCCESS;

    /* Initialize and overload Comm_ops (currently merely used for comm_split_type) */
    memset(&MPIR_PSP_Comm_fns, 0, sizeof(MPIR_PSP_Comm_fns));
    MPIR_Comm_fns = &MPIR_PSP_Comm_fns;
    MPIR_Comm_fns->split_type = MPID_PSP_split_type;

    if (has_parent) {
        MPID_PSP_Get_parent_ep_str(&parent_ep_str);
        if (parent_ep_str == NULL) {
            /* We might be a process spawned via MPIX_Spawn or MPIX_Ispawn,
             * in this case it is ok to not have the parent port in the KVS.
             * We do not have a parent comm and proceed without it. */
            MPIR_Process.comm_parent = NULL;
        } else {
            MPIR_Comm *comm_parent;
            mpi_errno =
                MPID_Comm_connect(parent_ep_str, NULL, 0, MPIR_Process.comm_world, &comm_parent);
            MPIR_ERR_CHKANDJUMP1(mpi_errno != MPI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                                 "**psp|spawn_child", "**psp|spawn_child %s",
                                 "MPI_Comm_connect(parent) failed");

            MPIR_Assert(comm_parent != NULL);
            MPL_strncpy(comm_parent->name, "MPI_COMM_PARENT", MPI_MAX_OBJECT_NAME);
            MPIR_Process.comm_parent = comm_parent;
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


int MPID_Create_intercomm_from_lpids(MPIR_Comm * newcomm_ptr, int size, const MPIR_Lpid lpids[])
{
    int mpi_errno = MPI_SUCCESS;
    /* Nothing to do here */
    return mpi_errno;
}

int MPIDI_PSP_comm_get_con(MPIR_Comm * comm, int rank, pscom_connection_t ** con)
{
    int mpi_errno = MPI_SUCCESS;

    if ((rank == MPI_PROC_NULL) || (rank == MPI_ANY_SOURCE) || (rank == MPI_ROOT)) {
        /* rank is one of the allowed negative values */
        *con = NULL;
    } else {
        MPIR_ERR_CHKANDJUMP1(rank < 0, mpi_errno, MPI_ERR_OTHER, "**psp|comminvalidrank",
                             "**psp|comminvalidrank %d", rank);

        if (comm->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
            MPIR_ERR_CHKANDJUMP1(rank >= comm->remote_size, mpi_errno, MPI_ERR_OTHER,
                                 "**psp|comminvalidrank", "**psp|comminvalidrank %d", rank);
            *con = comm->remote_vcr[rank]->con;
        } else {
            MPIR_ERR_CHKANDJUMP1(rank >= comm->local_size, mpi_errno, MPI_ERR_OTHER,
                                 "**psp|comminvalidrank", "**psp|comminvalidrank %d", rank);
            *con = comm->local_vcr[rank]->con;
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int MPIDI_PSP_comm_set_socket(MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;
    if (comm->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
        comm->pscom_socket = NULL;
    } else {
        /* Use pscom_socket from the rank 0 connection ... */
        pscom_connection_t *con = NULL;
        mpi_errno = MPIDI_PSP_comm_get_con(comm, 0, &con);
        MPIR_ERR_CHECK(mpi_errno);

        comm->pscom_socket = con ? con->socket : NULL;

        /* ... and test if connections from different sockets are used ... */
        for (int i = 0; i < comm->local_size; i++) {
            mpi_errno = MPIDI_PSP_comm_get_con(comm, i, &con);
            MPIR_ERR_CHECK(mpi_errno);
            if (comm->pscom_socket && con && (con->socket != comm->pscom_socket)) {
                /* ... and disallow the usage of comm->pscom_socket in this case.
                 * This will disallow ANY_SOURCE receives on that communicator for older pscoms
                 * ... but should be fixed/handled within the pscom layer as of pscom 5.2.0 */
                comm->pscom_socket = NULL;
                break;
            }
        }
    }

    /* Set sockets of subcomms - if any */
    if (comm->node_comm) {
        comm->node_comm->pscom_socket = comm->pscom_socket;
    }
    if (comm->node_roots_comm) {
        comm->node_roots_comm->pscom_socket = comm->pscom_socket;
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_PSP_Comm_commit_pre_hook(MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_ENTER;

    comm->pscom_socket = NULL;
    comm->local_vcrt = NULL;
    comm->remote_vcrt = NULL;
    comm->is_disconnected = 0;
    comm->is_checked_as_host_local = 0;
    comm->group = NULL;
    comm->msa = 0;

    if (comm->attr & MPIR_COMM_ATTR__SUBCOMM) {
        /* Subcomms (node_comm and node_root_comm) are created prior to
         * the commit of their parent comm (MPIR_Comm_create_subcomms),
         * so we end up here for subcomms before the required connections
         * may have been created for the parent comm.
         *
         * Hence we do nothing here for the subcomms and instead we create
         * the VCRT for the subcomms during the pre-commit hook of the parent
         * comm. */
        goto fn_exit;
    }

    if (comm->comm_kind == MPIR_COMM_KIND__INTRACOMM) {
        /* Create any missing connections in my_pg */
        mpi_errno = MPIDI_PSP_connection_init(comm);
        MPIR_ERR_CHECK(mpi_errno);
#if 0
        if (MPIDI_Process.env.enable_msa_aware_collops) {
            /* Update the hierarchical topology information of the comm as used for MSA-aware collectives. */
            mpi_errno = MPIDI_PSP_update_topo_level(comm);
            MPIR_ERR_CHECK(mpi_errno);
        }
#endif
    }

    mpi_errno = MPIDI_PSP_comm_set_vcrts(comm);
    MPIR_ERR_CHECK(mpi_errno);

    mpi_errno = MPIDI_PSP_comm_set_socket(comm);
    MPIR_ERR_CHECK(mpi_errno);

#ifdef HAVE_HCOLL
    hcoll_comm_create(comm, NULL);
#endif
#if 0
    if ((comm->hierarchy_kind == MPIR_COMM_HIERARCHY_KIND__NODE) &&
        (MPIDI_Process.env.enable_msa_aware_collops > 1)) {

        MPIDI_PSP_topo_level_t *tl = MPIDI_Process.my_pg->topo_levels;

        while (tl && MPIDI_PSP_comm_is_flat_on_level(comm, tl)) {
            MPIR_Assert(tl->badge_table);
            tl = tl->next;
        }

        if (tl) {       // This subcomm is not flat -> attach a further subcomm level: (to be handled in SMP-aware collectives)
            MPIR_Assert(comm->comm_kind == MPIR_COMM_KIND__INTRACOMM);
            mpi_errno = MPIR_Comm_dup_impl(comm, &comm->local_comm);    // we "misuse" local_comm for this purpose
            MPIR_Assert(mpi_errno == MPI_SUCCESS);
        }
        comm->msa = 1;
    }
#endif
#ifdef MPIDI_PSP_WITH_PSCOM_COLLECTIVES
    if (MPIDI_Process.env.enable_collectives && (comm->comm_kind == MPIR_COMM_KIND__INTRACOMM)) {
        MPID_PSP_group_init(comm);
    }
#endif

    /*
     * printf("%s (comm:%p(%s, id:%08x, size:%u))\n",
     * __func__, comm, comm->name, comm->context_id, comm->local_size););
     */
  fn_exit:
    MPIR_FUNC_EXIT;
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_PSP_Comm_commit_post_hook(MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

#ifdef HAVE_UCC
    MPIDI_common_ucc_comm_create_hook(comm);
#endif

    MPIR_FUNC_EXIT;
    return mpi_errno;
}


int MPIDI_PSP_Comm_destroy_hook(MPIR_Comm * comm)
{
#ifdef HAVE_UCC
    MPIDI_common_ucc_comm_destroy_hook(comm);
#endif

    MPIDI_VCRT_Release(comm->local_vcrt, comm->is_disconnected);
    comm->local_vcr = NULL;

    if (comm->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
        MPIDI_VCRT_Release(comm->remote_vcrt, comm->is_disconnected);
        comm->remote_vcr = NULL;
    }
#ifdef HAVE_HCOLL
    hcoll_comm_destroy(comm, NULL);
#endif

#if 0
    if (comm->hierarchy_kind == MPIR_COMM_HIERARCHY_KIND__NODE) {
        if (comm->local_comm) {
            // Recursively release also further subcomm levels:
            MPIR_Assert(comm->comm_kind == MPIR_COMM_KIND__INTRACOMM);
            MPIR_Comm_release(comm->local_comm);
        }
    }
#endif

    if (!MPIDI_Process.env.enable_collectives)
        return MPI_SUCCESS;

#ifdef MPIDI_PSP_WITH_PSCOM_COLLECTIVES
    /* ToDo: Use comm Barrier before cleanup! */
    MPID_PSP_group_cleanup(comm);
#endif

    return MPI_SUCCESS;
}

int MPID_Group_init_hook(MPIR_Group * group_ptr)
{
    group_ptr->psp_vcrt = NULL;
    return MPI_SUCCESS;
}

int MPID_Group_free_hook(MPIR_Group * group_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    if (group_ptr->psp_vcrt) {
        mpi_errno = MPIDI_VCRT_Release(group_ptr->psp_vcrt, FALSE);
        group_ptr->psp_vcrt = NULL;
    }
    return mpi_errno;
}

int MPIDI_PSP_Comm_set_hints(MPIR_Comm * comm_ptr, MPIR_Info * info_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

/* Get all group ranks (granks) in comm which belong to my_pg; also provide the size
 * of the granks array and the index of the calling process in the array (rank within
 * granks array).
 *
 * For merged comms (MPI_INTERCOMM_MERGE) it can happen that there are granks in a comm
 * that do not belong to my_pg. This function excludes those granks based on their world idx
 * and provides a grank array for only those granks that belong to my_pg.
 *
 * If comm is NULL or there is no local group in the comm: comm == MPI_COMM_WORLD. In
 * this case, only size and idx are set to my_pg size and rank, but granks will be NULL
 * to allow for shortcut code paths for the world comm. */
int MPIDI_PSP_comm_get_granks(MPIR_Comm * comm, int **granks, int *size, int *idx)
{
    int mpi_errno = MPI_SUCCESS;
    int *_granks = NULL;
    int i, _size = 0, _idx = -1;

    if (comm && comm->local_group) {
        _granks = MPL_malloc(MPIDI_Process.my_pg_size * sizeof(int), MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!_granks, mpi_errno, MPI_ERR_OTHER, "**nomem");

        MPIR_Group *group = comm->local_group;
        for (i = 0; i < group->size; i++) {
            MPIR_Lpid lpid = MPIR_Group_rank_to_lpid(group, i);
            int world_idx = MPIR_LPID_WORLD_INDEX(lpid);
            int grank = MPIR_LPID_WORLD_RANK(lpid);
            if (world_idx == 0) {
                /* Save granks that belong to my_pg and remember own idx (rank) within array */
                _granks[_size] = grank;
                if (_granks[_size] == MPIDI_Process.my_pg_rank) {
                    _idx = _size;
                }
                _size++;
            }
        }

        MPIR_Assert(_size > 0);
        MPIR_Assert(_idx >= 0);

        /* Shrink the granks array in size if required */
        if (_size < MPIDI_Process.my_pg_size) {
            _granks = MPL_realloc(_granks, _size * sizeof(int), MPL_MEM_OTHER);
            MPIR_ERR_CHKANDJUMP(!_granks, mpi_errno, MPI_ERR_OTHER, "**nomem");
        }

        *size = _size;
        *idx = _idx;
        *granks = _granks;
    } else {
        /* world comm */
        *size = MPIDI_Process.my_pg_size;
        *idx = MPIDI_Process.my_pg_rank;
        *granks = NULL;
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Provide a blob of data that contains infos about all worlds in comm_ptr */
static int pack_world_data(MPIR_Comm * comm_ptr, char **data_out, int *data_size_out)
{
    int mpi_errno = MPI_SUCCESS;
    int i;
    char *data = NULL;
    int len = 0;
    int num_worlds = 0;
    int local_size = comm_ptr->local_size;
    int *worlds_hash = NULL;
    int *worlds_idx = NULL;
    int *ranks = NULL;

#if 0
    int have_topo = 0;
    int *num_topo_levels = NULL;
    int **topo_badges = NULL;
    int *topo_msglen = NULL;
#endif

    worlds_hash = (int *) MPL_malloc(sizeof(int), MPL_MEM_OBJECT);
    MPIR_ERR_CHKANDJUMP(!worlds_hash, mpi_errno, MPI_ERR_OTHER, "**nomem");

    worlds_idx = (int *) MPL_malloc(local_size * sizeof(int), MPL_MEM_OBJECT);
    MPIR_ERR_CHKANDJUMP(!worlds_idx, mpi_errno, MPI_ERR_OTHER, "**nomem");

    ranks = (int *) MPL_malloc(local_size * sizeof(int), MPL_MEM_OBJECT);
    MPIR_ERR_CHKANDJUMP(!ranks, mpi_errno, MPI_ERR_OTHER, "**nomem");

    for (i = 0; i < local_size; i++) {
        MPIR_Lpid lpid = MPIR_comm_rank_to_lpid(comm_ptr, i);
        int world_idx = MPIR_LPID_WORLD_INDEX(lpid);
        int rank = MPIR_LPID_WORLD_RANK(lpid);
        worlds_idx[i] = world_idx;
        ranks[i] = rank;

        bool found = false;
        for (int j = 0; j < num_worlds; j++) {
            if (worlds_hash[j] == world_idx) {
                found = true;
                break;
            }
        }
        if (!found) {
            /* realloc hash for one more world */
            num_worlds++;
            worlds_hash = MPL_realloc(worlds_hash, num_worlds * sizeof(int), MPL_MEM_OBJECT);
            worlds_hash[num_worlds - 1] = world_idx;
        }
#if 0
        if (!have_topo) {
            /* Check if we have topo information available for this world/PG */
            MPIDI_PG_t *pg = NULL;
            MPIDI_PG_get(world_idx, &pg);
            MPIR_Assert(pg != NULL);
            have_topo = (MPIDI_PSP_get_num_topology_levels(pg) > 0);
        }
#endif
    }

    MPIR_Assert(num_worlds > 0);
#if 0
    if (have_topo) {
        /* Pack topology information */
        num_topo_levels = MPL_malloc(num_worlds * sizeof(int), MPL_MEM_OBJECT);
        MPIR_ERR_CHKANDJUMP(!num_topo_levels, mpi_errno, MPI_ERR_OTHER, "**nomem");

        topo_badges = MPL_malloc(num_worlds * sizeof(int *), MPL_MEM_OBJECT);
        MPIR_ERR_CHKANDJUMP(!topo_badges, mpi_errno, MPI_ERR_OTHER, "**nomem");

        topo_msglen = MPL_malloc(num_worlds * sizeof(int), MPL_MEM_OBJECT);
        MPIR_ERR_CHKANDJUMP(!topo_msglen, mpi_errno, MPI_ERR_OTHER, "**nomem");

        for (i = 0; i < num_worlds; i++) {
            MPIDI_PG_t *pg = NULL;
            MPIDI_PG_get(worlds_idx[i], &pg);
            MPIR_Assert(pg != NULL);
            num_topo_levels[i] = MPIDI_PSP_get_num_topology_levels(pg);
            MPIDI_PSP_pack_topology_badges(&topo_badges[i], &topo_msglen[i], pg);
        }
    }
#endif

    /* data layout:
     * - num_worlds
     * - world_sizes[num_worlds]
     * - worlds_hash[num_worlds] (local indices for mapping)
     * - world_namespace[num_worlds][MPIR_NAMESPACE_MAX]
     * - worlds_indices[local_size]
     * - world_ranks[local_size]
     * - have_topo
     * - num_topo_levels[num_worlds] (if have_topo)
     * - topo_msglen[num_worlds] (if have_topo)
     * - topo_badges[num_worlds][topo_msglen[i]] (if have_topo)
     */
    len = sizeof(int);
    len += num_worlds * sizeof(int);
    len += num_worlds * sizeof(int);
    len += num_worlds * sizeof(char) * MPIR_NAMESPACE_MAX;
    len += sizeof(int) * local_size;
    len += sizeof(int) * local_size;
#if 0
    len += sizeof(int); /* have_topo */
    if (have_topo) {
        len += num_worlds * sizeof(int);        /* levels */
        len += num_worlds * sizeof(int);        /* msg_lens */
        for (i = 0; i < num_worlds; i++) {
            len += topo_msglen[i];      /* badges */
        }
    }
#endif

    data = MPL_malloc(len, MPL_MEM_OTHER);
    MPIR_ERR_CHKANDJUMP(!data, mpi_errno, MPI_ERR_OTHER, "**nomem");

    char *s = data;

    /* num_worlds */
    *(int *) (s) = num_worlds;
    s += sizeof(int);

    /* world sizes */
    for (i = 0; i < num_worlds; i++) {
        *(int *) (s) = MPIR_Worlds[worlds_hash[i]].num_procs;
        s += sizeof(int);
    }

    /* world hash */
    for (i = 0; i < num_worlds; i++) {
        *(int *) (s) = worlds_hash[i];
        s += sizeof(int);
    }

    /* world namespaces */
    for (i = 0; i < num_worlds; i++) {
        strncpy(s, MPIR_Worlds[worlds_hash[i]].namespace, MPIR_NAMESPACE_MAX);
        s += MPIR_NAMESPACE_MAX;
    }

    /* world indices per local process */
    for (i = 0; i < local_size; i++) {
        *(int *) (s) = worlds_idx[i];
        s += sizeof(int);
    }

    /* world ranks per local process */
    for (i = 0; i < local_size; i++) {
        *(int *) (s) = ranks[i];
        s += sizeof(int);
    }

#if 0
    /* have topo */
    *(int *) (s) = have_topo;
    s += sizeof(int);

    if (have_topo) {
        /* num topo levels */
        for (i = 0; i < num_worlds; i++) {
            *(int *) (s) = num_topo_levels[i];
            s += sizeof(int);
        }

        /* topo msg_lens */
        for (i = 0; i < num_worlds; i++) {
            *(int *) (s) = topo_msglen[i];
            s += sizeof(int);
        }

        /* topo badges */
        for (i = 0; i < num_worlds; i++) {
            memcpy((int *) s, topo_badges[i], topo_msglen[i]);
            s += topo_msglen[i];
        }
    }
#endif

    *data_size_out = len;
    *data_out = data;

  fn_exit:
    MPL_free(worlds_hash);
    MPL_free(worlds_idx);
    MPL_free(ranks);
#if 0
    if (have_topo) {
        MPL_free(num_topo_levels);
        MPL_free(topo_msglen);
        for (i = 0; i < num_worlds; i++) {
            MPL_free(topo_badges[i]);
        }
        MPL_free(topo_badges);
    }
#endif
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Get the lpids of remote processes from a blob of world data about the remote worlds.
 * Any detected new world is added to the global MPIR_Worlds array. */
static int unpack_world_data(int remote_size, char *data, MPIR_Lpid * remote_lpids)
{
    int mpi_errno = MPI_SUCCESS;
    char *s = data;
    int i, j;
    int *p_world_hash_local = NULL;

    /* num remote worlds */
    int num_worlds = *(int *) s;
    s += sizeof(int);

    /* remote world sizes */
    int *p_world_sizes = (int *) s;
    s += num_worlds * sizeof(int);

    /* remote world hash */
    int *p_world_hash_remote = (int *) s;
    s += num_worlds * sizeof(int);

    /* remote world namespaces */
    char *p_worlds = s;
    s += num_worlds * MPIR_NAMESPACE_MAX;

    /* remote world indices per process */
    int *p_worlds_idx = (int *) s;
    s += remote_size * sizeof(int);

    /* remote world ranks per process */
    int *p_world_ranks = (int *) s;
    s += remote_size * sizeof(int);

#if 0
    /* have topo */
    int have_topo = *(int *) s;
    s += sizeof(int);

    /* topo information */
    int *p_num_topo_levels = NULL;
    int *p_topo_msglen = NULL;
    int **p_topo_bages = NULL;
    if (have_topo) {
        p_topo_bages = MPL_malloc(num_worlds * sizeof(int *), MPL_MEM_OBJECT);
        MPIR_ERR_CHKANDJUMP(!p_topo_bages, mpi_errno, MPI_ERR_OTHER, "**nomem");

        /* num topo levels */
        p_num_topo_levels = (int *) s;
        s += num_worlds * sizeof(int);
        MPIR_Assert(s != NULL);

        /* topo msglens */
        p_topo_msglen = (int *) s;
        s += num_worlds * sizeof(int);
        MPIR_Assert(s != NULL);

        /* topo badges */
        for (i = 0; i < num_worlds; i++) {
            p_topo_bages[i] = (int *) s;
            s += p_topo_msglen[i];
        }
    }
#endif

    p_world_hash_local = MPL_malloc(num_worlds * sizeof(int), MPL_MEM_OBJECT);
    MPIR_ERR_CHKANDJUMP(!p_world_hash_local, mpi_errno, MPI_ERR_OTHER, "**nomem");

    /* Find or add new worlds, create new PG for new world */
    for (i = 0; i < num_worlds; i++) {
        char *namespace = p_worlds + i * MPIR_NAMESPACE_MAX;
        MPIDI_PSP_topo_level_t *levels = NULL;

        int world_idx = MPIR_find_world(namespace);
        if (world_idx == -1) {
            world_idx = MPIR_add_world(namespace, p_world_sizes[i]);
#if 0
            if (have_topo) {
                /* unpack topo data */
                MPIDI_PSP_unpack_topology_badges(p_topo_bages[i], remote_size, p_num_topo_levels[i],
                                                 &levels);
            }
#endif
            /* Create a process group for the newly detected world (including topo information - if any) */
            mpi_errno = MPIDI_PG_Create(world_idx, levels, NULL);
            MPIR_ERR_CHECK(mpi_errno);
        } else {
            /* Check if there is a mismatch in known and received world sizes.
             * This can happen if we learned about this world via PMIx Pset. */
            if (MPIR_Worlds[world_idx].num_procs < p_world_sizes[i]) {
                MPIR_Worlds[world_idx].num_procs = p_world_sizes[i];
            }

            /* Check if we already have a PG for this world.
             * If we learned about the world via some other way then the group
             * in the device might be missing or have wrong size */
            MPIDI_PG_t *pg = NULL;
            mpi_errno = MPIDI_PG_get(world_idx, &pg);
            MPIR_ERR_CHECK(mpi_errno);
            if (!pg) {
                mpi_errno = MPIDI_PG_Create(world_idx, NULL);
                MPIR_ERR_CHECK(mpi_errno);
            } else if (pg->size < p_world_sizes[i]) {
                /* need to resize the PG */
                mpi_errno = MPIDI_PG_Resize(pg, p_world_sizes[i]);
                MPIR_ERR_CHECK(mpi_errno);
            }
#if 0
            if (have_topo) {
                /* Add topo information to existing world/ PG */
                MPIDI_PG_t *pg = NULL;
                MPIDI_PG_get(world_idx, &pg);
                MPIR_Assert(pg != NULL);

                if (!pg->topo_levels) {
                    /* Add only of this PG does not have any topo infos yet
                     * TODO: make sure that we update topo infos received here */

                    /* unpack topo data */
                    MPIDI_PSP_unpack_topology_badges(p_topo_bages[i], remote_size,
                                                     p_num_topo_levels[i], &levels);

                    mpi_errno = MPIDI_PSP_add_topo_levels_to_pg(pg, levels);
                    MPIR_ERR_CHECK(mpi_errno);
                }
            }
#endif
        }
        /* Map the remote world hash to the local world index */
        p_world_hash_local[i] = world_idx;
    }

    /* Map remote world indices + ranks to lpids */
    for (i = 0; i < remote_size; i++) {
        int found = 0;
        for (j = 0; j < num_worlds; j++) {
            if (p_world_hash_remote[j] == p_worlds_idx[i]) {
                remote_lpids[i] = MPIR_LPID_FROM(p_world_hash_local[j], p_world_ranks[i]);
                found = 1;
                break;
            }
        }

        if (!found) {
            MPIR_ERR_CHKANDJUMP1(!found, mpi_errno, MPI_ERR_OTHER, "**procnotfound",
                                 "**procnotfound %d", i);
        }
    }

  fn_exit:
    MPL_free(p_world_hash_local);
#if 0
    MPL_free(p_topo_bages);
#endif
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/*@
  MPID_Intercomm_exchange - Exchange address mapping for intercomm creation.
 @*/
int MPID_Intercomm_exchange(MPIR_Comm * local_comm, int local_leader,
                            MPIR_Comm * peer_comm, int remote_leader, int peer_tag,
                            int context_id, int *remote_context_id_out,
                            int *remote_size_out, MPIR_Lpid ** remote_lpids_out, int timeout)
{
    int mpi_errno = MPI_SUCCESS;
    int coll_attr = MPIR_COLL_ATTR_SYNC;

    int local_size = local_comm->local_size;
    int remote_size = 0;

    int local_context_id = context_id;
    int remote_context_id = MPIR_INVALID_CONTEXT_ID;

    MPIR_Lpid *remote_lpids = NULL;

    char *local_worlds_data = NULL;     /* local leader only */
    int local_worlds_data_size = 0;     /* local leader only */
    char *remote_worlds_data = NULL;
    int remote_worlds_data_size = 0;

    MPIR_CHKLMEM_DECL();

    /* Parameter 'timeout' currently not used in PSP device! */

    /* Exchange local/ remote size and context id with remote leader */
    if (local_comm->rank == local_leader) {
        mpi_errno = MPIC_Sendrecv(&local_size, 1, MPIR_INT_INTERNAL,
                                  remote_leader, peer_tag,
                                  &remote_size, 1, MPIR_INT_INTERNAL,
                                  remote_leader, peer_tag, peer_comm, MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);
        mpi_errno = MPIC_Sendrecv(&local_context_id, 1, MPIR_INT_INTERNAL,
                                  remote_leader, peer_tag,
                                  &remote_context_id, 1, MPIR_INT_INTERNAL,
                                  remote_leader, peer_tag, peer_comm, MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);
    }

    /* Bcast remote size and context id in local comm */
    mpi_errno = MPIR_Bcast(&remote_size, 1, MPIR_INT_INTERNAL, local_leader, local_comm, coll_attr);
    MPIR_Assert(mpi_errno == MPI_SUCCESS);

    mpi_errno =
        MPIR_Bcast(&remote_context_id, 1, MPIR_INT_INTERNAL, local_leader, local_comm, coll_attr);
    MPIR_Assert(mpi_errno == MPI_SUCCESS);

    if (local_comm->rank == local_leader) {
        /* Get world data for local comm (contains num_worlds, world sizes and world namespaces etc.) */
        mpi_errno = pack_world_data(local_comm, &local_worlds_data, &local_worlds_data_size);
        MPIR_ERR_CHECK(mpi_errno);
        MPIR_CHKLMEM_REGISTER(local_worlds_data);

        /* Exchange information on worlds with other leader */
        mpi_errno = MPIC_Sendrecv(&local_worlds_data_size, 1, MPIR_INT_INTERNAL,
                                  remote_leader, peer_tag,
                                  &remote_worlds_data_size, 1, MPIR_INT_INTERNAL,
                                  remote_leader, peer_tag, peer_comm, MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);

        MPIR_Assert(remote_worlds_data_size > 0);
        MPIR_CHKLMEM_MALLOC(remote_worlds_data, remote_worlds_data_size);

        mpi_errno = MPIC_Sendrecv(local_worlds_data, local_worlds_data_size, MPIR_CHAR_INTERNAL,
                                  remote_leader, peer_tag,
                                  remote_worlds_data, remote_worlds_data_size, MPIR_CHAR_INTERNAL,
                                  remote_leader, peer_tag, peer_comm, MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);

    }

    /* Bcast remote worlds data to everybody in comm so that all can update MPIR_Worlds array */
    mpi_errno =
        MPIR_Bcast(&remote_worlds_data_size, 1, MPIR_INT_INTERNAL, local_leader, local_comm,
                   coll_attr);
    MPIR_ERR_CHECK(mpi_errno);

    if (local_comm->rank != local_leader) {
        MPIR_Assert(remote_worlds_data_size > 0);
        MPIR_CHKLMEM_MALLOC(remote_worlds_data, remote_worlds_data_size);
    }

    mpi_errno =
        MPIR_Bcast(remote_worlds_data, remote_worlds_data_size, MPIR_CHAR_INTERNAL, local_leader,
                   local_comm, coll_attr);
    MPIR_ERR_CHECK(mpi_errno);

    /* Update MPIR_Worlds array and extract lpids of remote processes from world data */
    remote_lpids = (MPIR_Lpid *) MPL_malloc(remote_size * sizeof(MPIR_Lpid), MPL_MEM_OTHER);
    mpi_errno = unpack_world_data(remote_size, remote_worlds_data, remote_lpids);
    MPIR_ERR_CHECK(mpi_errno);

    /* Check if we have any missing connections to remote lpids */
    int missing_cons = 0;
    mpi_errno =
        MPIDI_PG_check_missing_remote_cons(local_comm, peer_comm, local_leader, remote_leader,
                                           peer_tag, remote_size, remote_lpids, &missing_cons);
    MPIR_ERR_CHECK(mpi_errno);

    if (missing_cons) {
        /* Establish any missing connections to remote lpids */
        mpi_errno = MPIDI_PSP_connect_remote(peer_comm, local_comm, local_leader,
                                             remote_leader, peer_tag, remote_lpids);
        MPIR_ERR_CHECK(mpi_errno);
    }

    *remote_lpids_out = remote_lpids;
    *remote_context_id_out = remote_context_id;
    *remote_size_out = remote_size;

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
