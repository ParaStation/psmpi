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
int MPIDI_PSP_update_badge_table(int degree, int my_badge, MPIR_Comm * comm,
                                 int normalize, bool second_run)
{
    int mpi_errno = MPI_SUCCESS;
    int grank;
    int pg_rank = MPIDI_Process.my_pg_rank;
    int pg_size = MPIDI_Process.my_pg_size;
    MPIDI_PSP_topo_level_t *level = NULL;
    bool fast_path = true;

    int *granks = NULL;
    int size = 0, rank = -1;    /* rank not required here */
    mpi_errno = MPIDI_PSP_comm_get_granks(comm, &granks, &size, &rank);
    MPIR_ERR_CHECK(mpi_errno);

    /* Find topo level for degree in the global list */
    if (!MPIDI_PSP_check_pg_for_level(degree, MPIDI_Process.my_pg, &level)) {
        goto fn_exit;
    }

    /* Sanity check */
    MPIR_Assert(level);

    /* Normalized badges are not unique and thus cannot be global! */
    MPIR_Assert(!normalize || (normalize && !level->badges_are_global));

    if (!second_run) {
        /* Check if there are badges missing for comm */
        for (int i = 0; i < size; i++) {
            int dest = granks ? granks[i] : i;
            if (level->badge_table[dest] == MPIDI_PSP_TOPO_BADGE__NULL) {
                fast_path = false;      /* There is at least one badge missing */
                break;
            }
        }

        if (fast_path) {
            /* Nothing to be done here */
            goto fn_exit;
        }
    } else {
        /* When we get here, it is already the second round for normalizing,
         * which itself must not be further normalized. (See assertion.) */
        MPIR_Assert(!normalize);

        for (grank = 0; grank < pg_size; grank++) {
            if (my_badge == level->badge_table[grank]) {
                my_badge = grank;
                break;
            }
        }
    }

    if (pg_size == 1) {

        /* Use shortcut w/o badge exchange in the MPI singleton case: */
        level->badge_table[0] = my_badge;

    } else {

        /* The exchange of the badge information is done here via the key/value space (KVS) of PMI(x).
         * This way, no (perhaps later unnecessary) pscom connections are already established at this point. */
        mpi_errno = MPIDI_PSP_publish_badge(pg_rank, degree, my_badge, normalize);
        MPIR_ERR_CHECK(mpi_errno);

        if (granks && (size < MPIDI_Process.my_pg_size)) {
            mpi_errno = MPIR_pmi_barrier_group(granks, size, comm->stringtag);
        } else {
            /* Use world barrier for world comm and comms that have size of world comm */
            mpi_errno = MPIR_pmi_barrier();
        }
        MPIR_ERR_CHECK(mpi_errno);

        /* Lookup the badges of processes in comm and save them in badge table */
        for (int i = 0; i < size; i++) {
            int dest = granks ? granks[i] : i;
            mpi_errno =
                MPIDI_PSP_lookup_badge(dest, degree, &(level->badge_table)[dest], normalize);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

    level->max_badge = level->badge_table[0];
    for (grank = 1; grank < pg_size; grank++) {
        if (level->badge_table[grank] > level->max_badge) {
            level->max_badge = level->badge_table[grank];
        }
    }

    if (level->max_badge >= pg_size && normalize) {
        /* For normalization do a second run */
        mpi_errno =
            MPIDI_PSP_update_badge_table(degree, my_badge, comm, !normalize /* == 0 */ , true);
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    MPL_free(granks);
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
            MPIR_Assert(ext_level);
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
    MPIR_Assert(level->pg == MPIDI_Process.my_pg);      // level must be local!

    if (likely(comm->vcr[rank]->pg == MPIDI_Process.my_pg)) {   // rank is in local process group

        if (unlikely(!level->badge_table)) {    // "dummy" level
            MPIR_Assert(level->max_badge == MPIDI_PSP_TOPO_BADGE__NULL);
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
            MPIR_Assert(ext_level);

            if (ext_level->badges_are_global) {
                MPIR_Assert(ext_level->badge_table);    // <- "dummy" levels only valid on home PG!

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

    MPIR_Assert(level->pg == MPIDI_Process.my_pg);      // level must be local!
    my_badge = MPIDI_PSP_get_badge_by_level_and_comm_rank(comm, level, comm->rank);

    for (i = 0; i < comm->local_size; i++) {
        if (MPIDI_PSP_get_badge_by_level_and_comm_rank(comm, level, i) != my_badge) {
            return 0;
        }
    }
    return 1;
}

#ifdef MPID_PSP_MSA_AWARE_COLLOPS
static
int MPIDI_PSP_init_topo_level(int degree, int badges_are_global,
                              MPIDI_PSP_topo_level_t ** topo_level)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_PSP_topo_level_t *level = NULL;
    int pg_size = MPIDI_Process.my_pg_size;

    level = MPL_malloc(sizeof(MPIDI_PSP_topo_level_t), MPL_MEM_OBJECT);
    MPIR_ERR_CHKANDJUMP(!level, mpi_errno, MPI_ERR_NO_MEM, "**nomem");
    level->badge_table = MPL_malloc(pg_size * sizeof(int), MPL_MEM_OBJECT);
    MPIR_ERR_CHKANDJUMP(!(level->badge_table), mpi_errno, MPI_ERR_NO_MEM, "**nomem");
    for (int i = 0; i < pg_size; i++) {
        level->badge_table[i] = MPIDI_PSP_TOPO_BADGE__NULL;
    }
    level->max_badge = -1;
    level->degree = degree;
    level->badges_are_global = badges_are_global;

    level->next = *topo_level;
    *topo_level = level;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static
int MPIDI_PSP_update_topo_level(MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;

    if (MPIDI_Process.env.enable_msa_awareness && MPIDI_Process.env.enable_msa_aware_collops) {
        mpi_errno = MPIDI_PSP_update_badge_table(MPIDI_PSP_TOPO_LEVEL__MODULES,
                                                 MPIDI_Process.msa_module_id,
                                                 comm, 0 /*normalize */ , false);
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
#endif

int MPID_Get_badge(MPIR_Comm * comm, int rank, int *badge_p)
{
    MPIDI_PSP_topo_level_t *tl = MPIDI_Process.my_pg->topo_levels;

    if (tl == NULL) {
        return MPID_Get_node_id(comm, rank, badge_p);
    }

    while (tl->next && MPIDI_PSP_comm_is_flat_on_level(comm, tl)) {
        MPIR_Assert(tl->badge_table);
        tl = tl->next;
    }

    *badge_p = MPIDI_PSP_get_badge_by_level_and_comm_rank(comm, tl, rank);

    return MPI_SUCCESS;
}

int MPID_Get_max_badge(MPIR_Comm * comm, int *max_badge_p)
{
    MPIDI_PSP_topo_level_t *tl = MPIDI_Process.my_pg->topo_levels;

    if (tl == NULL) {
        if ((MPIDI_Process.env.enable_msa_awareness && MPIDI_Process.env.enable_msa_aware_collops)) {
            *max_badge_p = 0;
            return MPI_ERR_OTHER;
        } else {
            /* No topo levels because no MSA features activated at runtime,
             * use fallback solution of MPIR layer */
            *max_badge_p = MPIR_Process.num_nodes;
        }
    } else {
        while (tl->next && MPIDI_PSP_comm_is_flat_on_level(comm, tl)) {
            MPIR_Assert(tl->badge_table);
            tl = tl->next;
        }

        /* The value we need to return here to the MPICH layer is the maximum badge of the
         * level plus 1, where the "plus 1" corresponds to the "unknown badge" wildcard.
         * (See also the definition of MPIDI_PSP_TOPO_BADGE__UNKNOWN.)
         */
        *max_badge_p = MPIDI_PSP_get_max_badge_by_level(tl) + 1;
    }

    return MPI_SUCCESS;
}

#endif /* MPID_PSP_MSA_AWARENESS */

int MPIDI_PSP_topo_init(MPIDI_PSP_topo_level_t ** topo_levels)
{
    int mpi_errno = MPI_SUCCESS;
#ifdef MPID_PSP_MSA_AWARENESS
    if (MPIDI_Process.env.enable_msa_awareness) {

        if (MPIDI_Process.msa_module_id < 0) {
            /* No module ID found: Let all these processes fall into module 0... */
            MPIDI_Process.msa_module_id = 0;
        }
#ifdef MPID_PSP_MSA_AWARE_COLLOPS
        if (MPIDI_Process.env.enable_msa_aware_collops) {
            mpi_errno =
                MPIDI_PSP_init_topo_level(MPIDI_PSP_TOPO_LEVEL__MODULES, 1 /*badges_are_global */ ,
                                          topo_levels);
            MPIR_ERR_CHECK(mpi_errno);
        }
#endif
    }
#endif

    if (MPIDI_Process.smp_node_id <= MPIDI_PSP_NODE_ID_UNDEFINED) {
        /* If no smp_node_id is set explicitly, use the pscom's node_id for this:
         * (...which is an int and might be negative. However, since we know that it actually
         * corresponds to the IPv4 address of the node, it is safe to force the most significant
         * bit to be unset so that it is positive and can thus also be used as a split color.)
         */
        MPIDI_Process.smp_node_id =
            (int) ((unsigned) MPIDI_Process.socket->local_con_info.node_id & (unsigned) 0x7fffffff);
    }
#ifdef MPID_PSP_MSA_AWARE_COLLOPS
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
#else
    return mpi_errno;
#endif
}

int MPID_Get_node_id(MPIR_Comm * comm, int rank, int *id_p)
{
    MPIR_Lpid lpid = MPIR_comm_rank_to_lpid(comm, rank);

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

/* Provide a pointer to the vcr for a given lpid (no vcr ref counting!) */
static
int get_vcr_for_lpid(MPIR_Lpid lpid, MPIDI_VC_t ** vcr)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_Assert(lpid != MPIR_LPID_INVALID);

    int world_idx = MPIR_LPID_WORLD_INDEX(lpid);        /* world index to which the process belongs */
    int world_rank = MPIR_LPID_WORLD_RANK(lpid);        /* rank of the process in its world */

    MPIDI_PG_t *pg = NULL;
    mpi_errno = MPIDI_PG_get(world_idx, &pg);
    MPIR_ERR_CHECK(mpi_errno);
    MPIR_Assert(pg != NULL);

    if (!pg->vcr || !pg->vcr[world_rank]) {
        /* Either the connection (table) is not set - we cannot provide the vcr */
        MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**procnotfound", "**procnotfound %d", lpid);
    }

    *vcr = pg->vcr[world_rank];

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

static int create_vcrt_from_group(MPIR_Group * group, struct MPIDI_VCRT **vcrt_out)
{
    int mpi_errno = MPI_SUCCESS;

    if (group->psp_vcrt) {
        *vcrt_out = MPIDI_VCRT_Dup(group->psp_vcrt);
        goto fn_exit;
    }

    struct MPIDI_VCRT *vcrt;
    vcrt = MPIDI_VCRT_Create(group->size);
    MPIR_ERR_CHKANDJUMP1(!vcrt, mpi_errno, MPI_ERR_OTHER, "**dev|vcrt_create",
                         "**dev|vcrt_create %s", "GROUP");

    *vcrt_out = vcrt;

    for (int i = 0; i < group->size; i++) {
        MPIR_Lpid lpid = MPIR_Group_rank_to_lpid(group, i);
        MPIDI_VC_t *vcr = NULL;

        mpi_errno = get_vcr_for_lpid(lpid, &vcr);
        MPIR_ERR_CHECK(mpi_errno);

        vcrt->vcr[i] = MPIDI_VC_Dup(vcr);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static
int create_subcomm_vcrts(MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_VCRT_t *vcrt = NULL;

    if (comm->node_comm) {
        mpi_errno = create_vcrt_from_group(comm->node_comm->local_group, &vcrt);
        MPIR_ERR_CHECK(mpi_errno);
        MPID_PSP_comm_set_vcrt(comm->node_comm, vcrt);
        comm->node_comm->pscom_socket = comm->pscom_socket;
    }

    if (comm->node_roots_comm) {
        mpi_errno = create_vcrt_from_group(comm->node_roots_comm->local_group, &vcrt);
        MPIR_ERR_CHECK(mpi_errno);
        MPID_PSP_comm_set_vcrt(comm->node_roots_comm, vcrt);
        comm->node_roots_comm->pscom_socket = comm->pscom_socket;
    }

    /* TODO: Do we need to add the VCRTs to the local_group of the subcomm? */

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_PSP_Comm_commit_pre_hook(MPIR_Comm * comm)
{
    pscom_connection_t *con1st;
    int mpi_errno = MPI_SUCCESS;
    int i;
    MPIDI_VCRT_t *vcrt;

    MPIR_FUNC_ENTER;

    comm->pscom_socket = NULL;
    comm->vcrt = NULL;
    comm->is_disconnected = 0;
    comm->is_checked_as_host_local = 0;
    comm->group = NULL;

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

#ifdef MPID_PSP_MSA_AWARE_COLLOPS
        /* Update the hierarchical topology information of the comm as used for MSA-aware collectives. */
        mpi_errno = MPIDI_PSP_update_topo_level(comm);
        MPIR_ERR_CHECK(mpi_errno);
#endif
    }

    if (comm == MPIR_Process.comm_world) {
        /* MPI_COMM_WORLD */
        comm->rank = MPIR_Process.rank;
        comm->remote_size = MPIR_Process.size;
        comm->local_size = MPIR_Process.size;

        vcrt = MPIDI_VCRT_Create(comm->remote_size);
        MPIR_ERR_CHKANDJUMP1(!vcrt, mpi_errno, MPI_ERR_OTHER, "**dev|vcrt_create",
                             "**dev|vcrt_create %s", "MPI_COMM_WORLD");
        MPID_PSP_comm_set_vcrt(comm, vcrt);

        for (i = 0; i < comm->remote_size; i++) {
            comm->vcr[i] = MPIDI_VC_Dup(MPIDI_Process.my_pg->vcr[i]);
        }
    } else if (comm == MPIR_Process.comm_self) {
        /* MPI_COMM_SELF */
        comm->rank = 0;
        comm->remote_size = 1;
        comm->local_size = 1;

        vcrt = MPIDI_VCRT_Create(comm->remote_size);
        MPIR_ERR_CHKANDJUMP1(!vcrt, mpi_errno, MPI_ERR_OTHER, "**dev|vcrt_create",
                             "**dev|vcrt_create %s", "MPI_COMM_SELF");
        MPID_PSP_comm_set_vcrt(comm, vcrt);

        comm->vcr[0] = MPIDI_VC_Dup(MPIDI_Process.my_pg->vcr[MPIR_Process.rank]);
    } else {
        /* Any other comm: Create VCRT from group */
        if (comm->comm_kind == MPIR_COMM_KIND__INTRACOMM) {
            mpi_errno = create_vcrt_from_group(comm->local_group, &vcrt);
            MPIR_ERR_CHECK(mpi_errno);
            MPID_PSP_comm_set_vcrt(comm, vcrt);
        } else {
            mpi_errno = create_vcrt_from_group(comm->local_group, &vcrt);
            MPIR_ERR_CHECK(mpi_errno);
            MPID_PSP_comm_set_local_vcrt(comm, vcrt);

            mpi_errno = create_vcrt_from_group(comm->remote_group, &vcrt);
            MPIR_ERR_CHECK(mpi_errno);
            MPID_PSP_comm_set_vcrt(comm, vcrt);
        }
    }

    /* add vcrt to the comm groups if they are not there */
    if (comm->comm_kind == MPIR_COMM_KIND__INTRACOMM) {
        if (comm->local_group->psp_vcrt == NULL) {
            comm->local_group->psp_vcrt = MPIDI_VCRT_Dup(comm->vcrt);
        }
    } else {
        if (comm->local_group->psp_vcrt == NULL) {
            comm->local_group->psp_vcrt = MPIDI_VCRT_Dup(comm->local_vcrt);
        }
        if (comm->remote_group->psp_vcrt == NULL) {
            comm->remote_group->psp_vcrt = MPIDI_VCRT_Dup(comm->vcrt);
        }
    }

    mpi_errno = create_subcomm_vcrts(comm);
    MPIR_ERR_CHECK(mpi_errno);

    if (comm->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
        /* setup the vcrt for the local_comm in the intercomm */
        if (comm->local_comm) {
            comm->local_comm->vcrt = MPIDI_VCRT_Dup(comm->local_vcrt);
        }
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
            MPIR_Assert(tl->badge_table);
            tl = tl->next;
        }

        if (tl) {       // This subcomm is not flat -> attach a further subcomm level: (to be handled in SMP-aware collectives)
            MPIR_Assert(comm->comm_kind == MPIR_COMM_KIND__INTRACOMM);
            mpi_errno = MPIR_Comm_dup_impl(comm, &comm->local_comm);    // we "misuse" local_comm for this purpose
            MPIR_Assert(mpi_errno == MPI_SUCCESS);
        }
    }
#endif

#ifdef MPIDI_PSP_WITH_PSCOM_COLLECTIVES
    if (MPIDI_Process.env.enable_collectives) {
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
    }

    MPIR_Assert(num_worlds > 0);

    /* data layout:
     * - num_worlds
     * - world_sizes[num_worlds]
     * - worlds_hash[num_worlds] (local indices for mapping)
     * - world_namespace[num_worlds][MPIR_NAMESPACE_MAX]
     * - worlds_indices[local_size]
     * - world_ranks[local_size]
     */
    len = sizeof(int);
    len += num_worlds * sizeof(int);
    len += num_worlds * sizeof(int);
    len += num_worlds * sizeof(char) * MPIR_NAMESPACE_MAX;
    len += sizeof(int) * local_size;
    len += sizeof(int) * local_size;

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

    /* TODO Pack topology information */

    *data_size_out = len;
    *data_out = data;

  fn_exit:
    MPL_free(worlds_hash);
    MPL_free(worlds_idx);
    MPL_free(ranks);
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

    /* TODO: unpack topo information */

    int *p_world_hash_local = MPL_malloc(num_worlds * sizeof(int), MPL_MEM_OBJECT);

    /* Find or add new worlds, create new PG for new world */
    for (i = 0; i < num_worlds; i++) {
        char *namespace = p_worlds + i * MPIR_NAMESPACE_MAX;
        int world_idx = MPIR_find_world(namespace);
        if (world_idx == -1) {
            world_idx = MPIR_add_world(namespace, p_world_sizes[i]);

            /* Create a process group for the newly detected world */
            /* TODO: replace first NULL with topo information */
            mpi_errno = MPIDI_PG_Create(world_idx, NULL, NULL);
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
