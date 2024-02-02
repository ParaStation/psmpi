/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2024 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <assert.h>
#include <unistd.h>
#include "mpidimpl.h"
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

        if (!MPIDI_Process.env.enable_smp_awareness) {
            // pretend that all ranks live on their own nodes:
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


#ifdef MPID_PSP_MSA_AWARENESS

int MPIDI_PSP_check_pg_for_level(int degree, MPIDI_PG_t * pg, MPIDI_PSP_topo_level_t ** level)
{
    MPIDI_PSP_topo_level_t *tl = pg->topo_levels;

    while (tl) {
        if (tl->degree == degree) {
            if (level)
                *level = tl;
            return 1;
        }
        tl = tl->next;
    }
    if (level)
        *level = NULL;
    return 0;
}

static
int MPIDI_PSP_publish_badge(int my_pg_rank, int degree, int my_badge, int normalize)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_max_key_size = MPIR_pmi_max_key_size();
    int pmi_max_val_size = MPIR_pmi_max_val_size();
    char *key = NULL;
    char *val = NULL;

    key = MPL_malloc(pmi_max_key_size * sizeof(char), MPL_MEM_STRINGS);
    val = MPL_malloc(pmi_max_val_size * sizeof(char), MPL_MEM_STRINGS);

    snprintf(key, pmi_max_key_size, "badge:%d:%d:%d", my_pg_rank, degree, !normalize);
    snprintf(val, pmi_max_val_size, "%d", my_badge);

    mpi_errno = MPIR_pmi_kvs_put(key, val);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    MPL_free(key);
    MPL_free(val);

    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static
int MPIDI_PSP_lookup_badge(int pg_rank, int degree, int *badge, int normalize)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_max_key_size = MPIR_pmi_max_key_size();
    int pmi_max_val_size = MPIR_pmi_max_val_size();
    char *key = NULL;
    char *val = NULL;
    char *tmp = NULL;

    key = MPL_malloc(pmi_max_key_size * sizeof(char), MPL_MEM_STRINGS);
    val = MPL_malloc(pmi_max_val_size * sizeof(char), MPL_MEM_STRINGS);

    snprintf(key, pmi_max_key_size, "badge:%d:%d:%d", pg_rank, degree, !normalize);

    mpi_errno = MPIR_pmi_kvs_get(pg_rank, key, val, pmi_max_val_size);
    MPIR_ERR_CHECK(mpi_errno);

    if (mpi_errno == MPI_SUCCESS) {
        *badge = strtol(val, &tmp, 0);
    }
    if (!tmp || (*tmp != '\0')) {
        *badge = MPIDI_PSP_TOPO_BADGE__NULL;
    }

  fn_exit:
    MPL_free(key);
    MPL_free(val);

    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static
int MPIDI_PSP_create_badge_table(int degree, int my_badge, int my_pg_rank, int pg_size,
                                 int *max_badge, int **badge_table, int normalize)
{
    int mpi_errno = MPI_SUCCESS;
    int grank;

    if (*badge_table != NULL) {

        /* When we get here, it is already the second round for normalizing,
         * which itself must not be further normalized. (See assertion.) */
        assert(!normalize);

        for (grank = 0; grank < pg_size; grank++) {
            if (my_badge == (*badge_table)[grank]) {
                my_badge = grank;
                break;
            }
        }

        MPL_free(*badge_table);
    }

    *badge_table = MPL_malloc(pg_size * sizeof(int), MPL_MEM_OBJECT);

    if (MPIDI_Process.singleton_but_no_pm) {

        /* Use shortcut w/o badge exchange in the MPI singleton case: */
        MPIR_Assert(pg_size == 1);
        (*badge_table)[0] = my_badge;

    } else {

        /* The exchange of the badge information is done here via the key/value space (KVS) of PMI(x).
         * This way, no (perhaps later unnecessary) pscom connections are already established at this point. */
        MPIDI_PSP_publish_badge(my_pg_rank, degree, my_badge, normalize);

        mpi_errno = MPIR_pmi_barrier();
        MPIR_ERR_CHECK(mpi_errno);

        for (grank = 0; grank < pg_size; grank++) {
            MPIDI_PSP_lookup_badge(grank, degree, &(*badge_table)[grank], normalize);
        }
    }

    *max_badge = (*badge_table)[0];
    for (grank = 1; grank < pg_size; grank++) {
        if ((*badge_table)[grank] > *max_badge)
            *max_badge = (*badge_table)[grank];
    }

    if (*max_badge >= pg_size && normalize) {
        MPIDI_PSP_create_badge_table(degree, my_badge, my_pg_rank, pg_size, max_badge, badge_table,
                                     !normalize /* == 0 */);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static
int MPIDI_PSP_comm_is_global(MPIR_Comm * comm)
{
    int i;
    for (i = 0; i < comm->local_size; i++) {
        if (comm->vcr[i]->pg != MPIDI_Process.my_pg) {
            return 1;
        }
    }
    return 0;
}

static
int MPIDI_PSP_get_max_badge_by_level(MPIDI_PSP_topo_level_t * level)
{
    MPIDI_PG_t *pg = MPIDI_Process.my_pg;
    int max_badge = level->max_badge;

    // check also the remote process groups:
    while (pg->next) {
        MPIDI_PSP_topo_level_t *ext_level = NULL;
        if (MPIDI_PSP_check_pg_for_level(level->degree, pg->next, &ext_level)) {
            assert(ext_level);
            if (ext_level->max_badge > max_badge) {
                max_badge = ext_level->max_badge;
            }
        }
        pg = pg->next;
    }

    return max_badge;
}

static
int MPIDI_PSP_get_badge_by_level_and_comm_rank(MPIR_Comm * comm, MPIDI_PSP_topo_level_t * level,
                                               int rank)
{
    MPIDI_PSP_topo_level_t *ext_level = NULL;
    assert(level->pg == MPIDI_Process.my_pg);   // level must be local!

    if (likely(comm->vcr[rank]->pg == MPIDI_Process.my_pg)) {   // rank is in local process group

        if (unlikely(!level->badge_table)) {    // "dummy" level
            assert(level->max_badge == MPIDI_PSP_TOPO_BADGE__NULL);
            goto badge_unknown;
        }

        if (!level->badges_are_global) {

            if (unlikely(MPIDI_PSP_comm_is_global(comm))) {
                // if own badges are not global, these are treated as "unknown" by other PGs
                goto badge_unknown;
            }
        }

        return level->badge_table[comm->vcr[rank]->pg_rank];

    } else {    // rank is in a remote process group

        if (MPIDI_PSP_check_pg_for_level(level->degree, comm->vcr[rank]->pg, &ext_level)) {
            // found remote level with identical degree
            assert(ext_level);

            if (ext_level->badges_are_global) {
                assert(ext_level->badge_table); // <- "dummy" levels only valid on home PG!

                return ext_level->badge_table[comm->vcr[rank]->pg_rank];
            }
        }
    }

  badge_unknown:
    return MPIDI_PSP_TOPO_BADGE__UNKNOWN(level);
}

static
int MPIDI_PSP_comm_is_flat_on_level(MPIR_Comm * comm, MPIDI_PSP_topo_level_t * level)
{
    int i;
    int my_badge;

    assert(level->pg == MPIDI_Process.my_pg);   // level must be local!
    my_badge = MPIDI_PSP_get_badge_by_level_and_comm_rank(comm, level, comm->rank);

    for (i = 0; i < comm->local_size; i++) {
        if (MPIDI_PSP_get_badge_by_level_and_comm_rank(comm, level, i) != my_badge) {
            return 0;
        }
    }
    return 1;
}

static
int MPIDI_PSP_create_topo_level(int my_badge, int degree, int badges_are_global, int normalize,
                                MPIDI_PSP_topo_level_t ** topo_level)
{
    int *module_badge_table = NULL;
    int module_max_badge = 0;
    MPIDI_PSP_topo_level_t *level = NULL;

    int pg_rank = MPIDI_Process.my_pg_rank;
    int pg_size = MPIDI_Process.my_pg_size;

    // Normalized badges are not unique and thus cannot be global!
    MPIR_Assert(!normalize || (normalize && !badges_are_global));

    MPIDI_PSP_create_badge_table(degree, my_badge, pg_rank, pg_size, &module_max_badge,
                                 &module_badge_table, normalize);
    assert(module_badge_table);

    level = MPL_malloc(sizeof(MPIDI_PSP_topo_level_t), MPL_MEM_OBJECT);
    level->badge_table = module_badge_table;
    level->max_badge = module_max_badge;
    level->degree = degree;
    level->badges_are_global = badges_are_global;

    level->next = *topo_level;
    *topo_level = level;

    return MPI_SUCCESS;
}

int MPIDI_PSP_topo_init(void)
{
    MPIDI_Process.topo_levels = NULL;

    if (MPIDI_Process.env.enable_msa_awareness) {

        if (MPIDI_Process.msa_module_id < 0) {
            /* No module ID found: Let all these processes fall into module 0... */
            MPIDI_Process.msa_module_id = 0;
        }
#ifdef MPID_PSP_MSA_AWARE_COLLOPS
        if (MPIDI_Process.env.enable_msa_aware_collops) {
            MPIDI_PSP_create_topo_level(MPIDI_Process.msa_module_id, MPIDI_PSP_TOPO_LEVEL__MODULES,
                                        1 /*badges_are_global */ , 0 /*normalize */ ,
                                        &MPIDI_Process.topo_levels);
        }
#endif
    }

    if (MPIDI_Process.env.enable_smp_awareness) {

        if (MPIDI_Process.smp_node_id < 0) {
            /* If no smp_node_id is set explicitly, use the pscom's node_id for this:
             * (...which is an int and might be negative. However, since we know that it actually
             * corresponds to the IPv4 address of the node, it is safe to force the most significant
             * bit to be unset so that it is positive and can thus also be used as a split color.)
             */
            MPIDI_Process.smp_node_id =
                (int) ((unsigned) MPIDI_Process.socket->
                       local_con_info.node_id & (unsigned) 0x7fffffff);
        }
#ifdef MPID_PSP_MSA_AWARE_COLLOPS
        if (MPIDI_Process.env.enable_smp_aware_collops) {
            MPIDI_PSP_create_topo_level(MPIDI_Process.smp_node_id, MPIDI_PSP_TOPO_LEVEL__NODES,
                                        0 /*badges_are_global */ , 1 /*normalize */ ,
                                        &MPIDI_Process.topo_levels);
        }
#endif
    }

    return MPI_SUCCESS;
}

int MPID_Get_badge(MPIR_Comm * comm, int rank, int *badge_p)
{
    MPIDI_PSP_topo_level_t *tl = MPIDI_Process.my_pg->topo_levels;

    if (tl == NULL) {
        return MPID_Get_node_id(comm, rank, badge_p);
    }

    while (tl->next && MPIDI_PSP_comm_is_flat_on_level(comm, tl)) {
        assert(tl->badge_table);
        tl = tl->next;
    }

    *badge_p = MPIDI_PSP_get_badge_by_level_and_comm_rank(comm, tl, rank);

    return MPI_SUCCESS;
}

int MPID_Get_max_badge(MPIR_Comm * comm, int *max_badge_p)
{
    MPIDI_PSP_topo_level_t *tl = MPIDI_Process.my_pg->topo_levels;

    if (tl == NULL) {
        *max_badge_p = 0;
        return MPI_ERR_OTHER;
    }

    while (tl->next && MPIDI_PSP_comm_is_flat_on_level(comm, tl)) {
        assert(tl->badge_table);
        tl = tl->next;
    }

    /* The value we need to return here to the MPICH layer is the maximum badge of the
     * level plus 1, where the "plus 1" corresponds to the "unknown badge" wildcard.
     * (See also the definition of MPIDI_PSP_TOPO_BADGE__UNKNOWN.)
     */
    *max_badge_p = MPIDI_PSP_get_max_badge_by_level(tl) + 1;

    return MPI_SUCCESS;
}

#endif /* MPID_PSP_MSA_AWARENESS */


int MPID_Get_node_id(MPIR_Comm * comm, int rank, int *id_p)
{
    uint64_t lpid = comm->vcr[rank]->lpid;

#ifdef MPID_PSP_MSA_AWARE_COLLOPS
    /* In the case of enabled MSA awareness, we can use the badge table at the nodes level.
     * If a badge at this level cannot be found, we fall back to MPICH's node_map table...
     */
    MPIDI_PSP_topo_level_t *level;
    if (MPIDI_PSP_check_pg_for_level(MPIDI_PSP_TOPO_LEVEL__NODES, MPIDI_Process.my_pg, &level)) {
        /* A badge table on node level exists. Get badge by comm rank: */
        *id_p = MPIDI_PSP_get_badge_by_level_and_comm_rank(comm, level, rank);
        /* The badge we get here is less than or equal to the maximum node-level badge
         * plus 1, where the latter corresponds to the "unknown badge" wildcard.
         * (See also the definition of MPIDI_PSP_TOPO_BADGE__UNKNOWN.)
         */
        MPIR_Assert(*id_p <= MPIDI_PSP_get_max_badge_by_level(level) + 1);
        return MPI_SUCCESS;
    }
#endif
    if (comm->vcr[rank]->pg == MPIDI_Process.my_pg) {
        // rank is within the own MPI_COMM_WORLD -> use map
        *id_p = MPIR_Process.node_map[lpid];
    } else {
        // node ids of remote process groups are unknown...
        *id_p = -1;
    }

    return MPI_SUCCESS;
}

#if 0
/* It seems that this ADI3 function is no longer used in the higher MPICH layers and
   has been replaced by a direct access to MPIR_Process.num_nodes.
   Therefore, this function is commented out here so that it cannot be used by mistake.
   In the MSA case, however, we must continue to pay attention that MPID_Get_max_badge()
   (see above) is still used also in the higher layers.
*/
int MPID_Get_max_node_id(MPIR_Comm * comm, int *max_id_p)
{
    *max_id_p = MPIR_Process.num_nodes - 1;

    return MPI_SUCCESS;
}
#endif


int MPID_PSP_comm_init(int has_parent)
{
    char *parent_port;
    int mpi_errno = MPI_SUCCESS;

    /* Initialize and overload Comm_ops (currently merely used for comm_split_type) */
    memset(&MPIR_PSP_Comm_fns, 0, sizeof(MPIR_PSP_Comm_fns));
    MPIR_Comm_fns = &MPIR_PSP_Comm_fns;
    MPIR_Comm_fns->split_type = MPID_PSP_split_type;

    if (has_parent) {
        MPIR_Comm *comm_parent;

        mpi_errno = MPID_PSP_GetParentPort(&parent_port);
        MPIR_ERR_CHKANDJUMP1(mpi_errno != MPI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                             "**psp|spawn_child", "**psp|spawn_child %s", "get parent port failed");

        mpi_errno = MPID_Comm_connect(parent_port, NULL, 0, MPIR_Process.comm_world, &comm_parent);
        MPIR_ERR_CHKANDJUMP1(mpi_errno != MPI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                             "**psp|spawn_child", "**psp|spawn_child %s",
                             "MPI_Comm_connect(parent) failed");

        assert(comm_parent != NULL);
        MPL_strncpy(comm_parent->name, "MPI_COMM_PARENT", MPI_MAX_OBJECT_NAME);
        MPIR_Process.comm_parent = comm_parent;
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPID_Comm_get_lpid(MPIR_Comm * comm_ptr, int idx, uint64_t * lpid_ptr, bool is_remote)
{
    if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM || is_remote) {
        *lpid_ptr = comm_ptr->vcr[idx]->lpid;
    } else {
        *lpid_ptr = comm_ptr->local_vcr[idx]->lpid;
    }

    return MPI_SUCCESS;
}

int MPID_Create_intercomm_from_lpids(MPIR_Comm * newcomm_ptr, int size, const uint64_t lpids[])
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

    for (i = 0; i < size; i++) {
        MPIDI_VC_t *vcr = NULL;

        /* For rank i in the new communicator, find the corresponding
         * virtual connection.  For lpids less than the size of comm_world,
         * we can just take the corresponding entry from comm_world.
         * Otherwise, we need to search through the process groups.
         */
        /* printf("[%d] Remote rank %d has lpid %" PRIu64 "\n",
         * MPIR_Process.comm_world->rank, i, lpids[i]); */

        /* All LPIDs passed in the array must be valid, because otherwise we
         * cannot find the matching VC here. Therefore, we check this with
         * an assertion just to be safe...
         */
        MPIR_Assert(lpids[i] != MPIDI_PSP_INVALID_LPID);

        if (lpids[i] < commworld_ptr->remote_size) {
            vcr = commworld_ptr->vcr[lpids[i]];
            assert(vcr);
        } else {
            /* We must find the corresponding vcr for a given lpid */
            /* For now, this means iterating through the process groups */
            MPIDI_PG_t *pg;
            int j;

            pg = MPIDI_Process.my_pg->next;     /* (skip comm_world) */

            do {
                assert(pg);

                for (j = 0; j < pg->size; j++) {

                    if (!pg->vcr[j])
                        continue;

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

    return mpi_errno;
}



int MPIDI_PSP_Comm_commit_pre_hook(MPIR_Comm * comm)
{
    pscom_connection_t *con1st;
    int mpi_errno = MPI_SUCCESS;
    int i, grank;
    MPIDI_VCRT_t *vcrt;

    MPIR_FUNC_ENTER;

    if (comm->mapper_head) {
        MPID_PSP_comm_create_mapper(comm);
    }

    if (comm == MPIR_Process.comm_world || comm == MPIR_Process.comm_self) {
        /* comm->remote_size should be set before the pre commit hook is executed */

        vcrt = MPIDI_VCRT_Create(comm->remote_size);
        MPIR_Assert(vcrt);
        MPID_PSP_comm_set_vcrt(comm, vcrt);

        if (comm == MPIR_Process.comm_world) {
            for (grank = 0; grank < comm->remote_size; grank++) {
                comm->vcr[grank] = MPIDI_VC_Dup(MPIDI_Process.my_pg->vcr[grank]);
            }
        } else if (comm == MPIR_Process.comm_self) {
            comm->vcr[0] = MPIDI_VC_Dup(MPIDI_Process.my_pg->vcr[MPIDI_Process.my_pg_rank]);
        }
    } else if (comm->context_id == MPIR_COMM_TMP_SESSION_CTXID) {
        /* initialize communicator within MPI session, need to look into comm->remote_group */

        vcrt = MPIDI_VCRT_Create(comm->remote_group->size);
        MPIR_Assert(vcrt);
        MPID_PSP_comm_set_vcrt(comm, vcrt);

        for (i = 0; i < comm->remote_group->size; i++) {
            comm->vcr[i] =
                MPIDI_VC_Dup(MPIDI_Process.my_pg->vcr[comm->remote_group->lrank_to_lpid[i].lpid]);
        }
    }

    comm->is_disconnected = 0;
    comm->is_checked_as_host_local = 0;
    comm->group = NULL;

    if (comm->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
        /* do nothing on Intercomms */
        comm->pscom_socket = NULL;
        goto fn_exit;
    }

    /* Use pscom_socket from the rank 0 connection ... */
    con1st = MPID_PSCOM_rank2connection(comm, 0);
    comm->pscom_socket = con1st ? con1st->socket : NULL;

    /* ... and test if connections from different sockets are used ... */
    for (i = 0; i < comm->local_size; i++) {
        if (comm->pscom_socket && MPID_PSCOM_rank2connection(comm, i) &&
            (MPID_PSCOM_rank2connection(comm, i)->socket != comm->pscom_socket)) {
            /* ... and disallow the usage of comm->pscom_socket in this case.
             * This will disallow ANY_SOURCE receives on that communicator for older pscoms
             * ... but should be fixed/handled within the pscom layer as of pscom 5.2.0 */
            comm->pscom_socket = NULL;
            break;
        }
    }

#ifdef HAVE_HCOLL
    hcoll_comm_create(comm, NULL);
#endif

#ifdef MPID_PSP_MSA_AWARE_COLLOPS
    if ((comm->hierarchy_kind == MPIR_COMM_HIERARCHY_KIND__NODE) &&
        (MPIDI_Process.env.enable_msa_aware_collops > 1)) {

        MPIDI_PSP_topo_level_t *tl = MPIDI_Process.my_pg->topo_levels;

        while (tl && MPIDI_PSP_comm_is_flat_on_level(comm, tl)) {
            assert(tl->badge_table);
            tl = tl->next;
        }

        if (tl) {       // This subcomm is not flat -> attach a further subcomm level: (to be handled in SMP-aware collectives)
            assert(comm->comm_kind == MPIR_COMM_KIND__INTRACOMM);
            mpi_errno = MPIR_Comm_dup_impl(comm, &comm->local_comm);    // we "misuse" local_comm for this purpose
            assert(mpi_errno == MPI_SUCCESS);
        }
    }
#endif

    if (!MPIDI_Process.env.enable_collectives)
        return MPI_SUCCESS;

#ifdef MPIDI_PSP_WITH_PSCOM_COLLECTIVES
    MPID_PSP_group_init(comm);
#endif

    /*
     * printf("%s (comm:%p(%s, id:%08x, size:%u))\n",
     * __func__, comm, comm->name, comm->context_id, comm->local_size););
     */
  fn_exit:
    MPIR_FUNC_EXIT;
    return mpi_errno;
}

int MPIDI_PSP_Comm_commit_post_hook(MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    MPIR_FUNC_EXIT;
    return mpi_errno;
}


int MPIDI_PSP_Comm_destroy_hook(MPIR_Comm * comm)
{
    MPIDI_VCRT_Release(comm->vcrt, comm->is_disconnected);
    comm->vcr = NULL;

    if (comm->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
        MPIDI_VCRT_Release(comm->local_vcrt, comm->is_disconnected);
    }
#ifdef HAVE_HCOLL
    hcoll_comm_destroy(comm, NULL);
#endif

#ifdef MPID_PSP_MSA_AWARE_COLLOPS
    if (comm->hierarchy_kind == MPIR_COMM_HIERARCHY_KIND__NODE) {
        if (comm->local_comm) {
            // Recursively release also further subcomm levels:
            assert(comm->comm_kind == MPIR_COMM_KIND__INTRACOMM);
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


int MPIDI_PSP_Comm_set_hints(MPIR_Comm * comm_ptr, MPIR_Info * info_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    MPIR_FUNC_EXIT;
    return mpi_errno;
}
