/*
 * ParaStation
 *
 * Copyright (C) 2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpid_psp_topo.h"

#if 0
static int MPIDI_PSP_init_topo_level(int degree, int badges_are_global,
                                     MPIDI_PSP_topo_level_t ** topo_level);
static int MPIDI_PSP_update_badge_table(int degree, int my_badge, MPIR_Comm * comm,
                                        int normalize, bool second_run);
static int MPIDI_PSP_publish_badge(int my_pg_rank, int degree, int my_badge, int normalize);
static int MPIDI_PSP_lookup_badge(int pg_rank, int degree, int *badge, int normalize);
static int MPIDI_PSP_add_topo_level_to_pg(MPIDI_PG_t * pg, MPIDI_PSP_topo_level_t * level);
static int MPIDI_PSP_comm_is_global(MPIR_Comm * comm);
static int MPIDI_PSP_get_max_badge_by_level(MPIDI_PSP_topo_level_t * level);
static int MPIDI_PSP_get_badge_by_level_and_comm_rank(MPIR_Comm * comm,
                                                      MPIDI_PSP_topo_level_t * level, int rank);

#define MPIDI_PSP_TOPO_BADGE__UNKNOWN(level) (MPIDI_PSP_get_max_badge_by_level(level) + 1)
#define MPIDI_PSP_TOPO_BADGE__NULL -1
#define MPIDI_PSP_TOPO_LEVEL__MODULES 4096
/* #define MPIDI_PSP_TOPO_LEVEL__NODES   1024
 * Removed because MPIR layer provides SMP awareness for collectives */

int MPIDI_PSP_topo_init(MPIDI_PSP_topo_level_t ** topo_levels)
{
    int mpi_errno = MPI_SUCCESS;
    if (MPIDI_Process.env.enable_msa_awareness) {

        if (MPIDI_Process.msa_module_id < 0) {
            /* No module ID found: Let all these processes fall into module 0... */
            MPIDI_Process.msa_module_id = 0;
        }
        if (MPIDI_Process.env.enable_msa_aware_collops) {
            mpi_errno =
                MPIDI_PSP_init_topo_level(MPIDI_PSP_TOPO_LEVEL__MODULES, 1 /*badges_are_global */ ,
                                          topo_levels);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

    if (MPIDI_Process.smp_node_id <= MPIDI_PSP_NODE_ID_UNDEFINED) {
        /* If no smp_node_id is set explicitly, use the pscom's node_id for this:
         * (...which is an int and might be negative. However, since we know that it actually
         * corresponds to the IPv4 address of the node, it is safe to force the most significant
         * bit to be unset so that it is positive and can thus also be used as a split color.)
         */
        MPIDI_Process.smp_node_id =
            (int) ((unsigned) MPIDI_Process.socket->local_con_info.node_id & (unsigned) 0x7fffffff);
    }
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Initialize the topology level for a specific topology degree */
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

/* Update the badge table for all processes in comm */
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

int MPIDI_PSP_get_num_topology_levels(MPIDI_PG_t * pg)
{
    int level_count = 0;
    MPIDI_PSP_topo_level_t *tl = pg->topo_levels;

    while (tl) {
        level_count++;
        tl = tl->next;
    }
    return level_count;
}


void MPIDI_PSP_pack_topology_badges(int **pack_msg, int *pack_size, MPIDI_PG_t * pg)
{
    int i;
    int *msg;
    MPIDI_PSP_topo_level_t *tl = pg->topo_levels;

    *pack_size = MPIDI_PSP_get_num_topology_levels(pg) * (pg->size + 3) * sizeof(int);
    *pack_msg = MPL_malloc(*pack_size, MPL_MEM_OBJECT);
    MPIR_Assert(*pack_msg != NULL);

    msg = *pack_msg;
    while (tl) {        // FIX ME: non-global badges and "dummy" tables need not to be exchanged!
        for (i = 0; i < pg->size; i++, msg++) {
            if (tl->badge_table) {
                *msg = tl->badge_table[i];
            } else {
                *msg = MPIDI_PSP_TOPO_BADGE__NULL;
            }
        }
        *msg = tl->degree;
        msg++;
        *msg = tl->max_badge;
        msg++;
        *msg = tl->badges_are_global;
        msg++;
        tl = tl->next;
    }
}

void MPIDI_PSP_unpack_topology_badges(int *pack_msg, int pg_size, int num_levels,
                                      MPIDI_PSP_topo_level_t ** levels)
{
    int i, j;
    int *msg;
    MPIDI_PSP_topo_level_t *level;

    *levels = NULL;

    msg = pack_msg;
    for (i = 0; i < num_levels; i++) {

        level = MPL_malloc(sizeof(MPIDI_PSP_topo_level_t), MPL_MEM_OBJECT);

        if (*msg != MPIDI_PSP_TOPO_BADGE__NULL) {
            level->badge_table = MPL_malloc(pg_size * sizeof(int), MPL_MEM_OBJECT);
            for (j = 0; j < pg_size; j++) {
                level->badge_table[j] = msg[j];
            }
        } else {        // just a "dummy" table
            MPIR_Assert(msg[pg_size + 1] == MPIDI_PSP_TOPO_BADGE__NULL);
            level->badge_table = NULL;
        }
        level->degree = msg[pg_size];
        level->max_badge = msg[pg_size + 1];
        level->badges_are_global = msg[pg_size + 2];
        msg += (pg_size + 3);

        level->next = *levels;
        *levels = level;
    }
    /* pack_msg must be freed by caller */
}

/* This function adds a new topology level to the list associated with the given
 * process group (pg). In doing so, the order within the list is defined by the
 * degree values of the topology levels, starting with the highest degree.
 */
static
int MPIDI_PSP_add_topo_level_to_pg(MPIDI_PG_t * pg, MPIDI_PSP_topo_level_t * level)
{
    MPIDI_PSP_topo_level_t *tl = pg->topo_levels;

    if (!tl || tl->degree < level->degree) {
        /* add level at the beginning of the list */
        level->next = tl;
        pg->topo_levels = level;
    } else {
        MPIR_Assert(tl->degree != level->degree);
        while (tl->next && tl->next->degree > level->degree) {
            /* iterate through the list until matching position is found */
            tl = tl->next;
        }
        /* add new level */
        level->next = tl->next;
        tl->next = level;
    }
    level->pg = pg;

    return MPI_SUCCESS;
}

int MPIDI_PSP_add_topo_levels_to_pg(MPIDI_PG_t * pg, MPIDI_PSP_topo_level_t * levels)
{
    while (levels) {
        MPIDI_PSP_topo_level_t *level_next = levels->next;
        MPIDI_PSP_add_topo_level_to_pg(pg, levels);
        levels = level_next;
    }

    return MPI_SUCCESS;
}

int MPIDI_PSP_add_flat_level_to_pg(MPIDI_PG_t * pg, int degree)
{
    MPIDI_PSP_topo_level_t *level = MPL_malloc(sizeof(MPIDI_PSP_topo_level_t), MPL_MEM_OBJECT);

    level->degree = degree;
    level->badges_are_global = 1;
    level->max_badge = MPIDI_PSP_TOPO_BADGE__NULL;
    level->badge_table = NULL;

    return MPIDI_PSP_add_topo_level_to_pg(pg, level);
}


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
#endif

int MPID_Get_node_id(MPIR_Comm * comm, int rank, int *id_p)
{
    MPIR_Lpid lpid = MPIR_comm_rank_to_lpid(comm, rank);
    int world_idx = MPIR_LPID_WORLD_INDEX(lpid);

    if (world_idx == 0) {
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

#if 0
/* Return 1 if the comm contains processes from a pg different than my_pg,
 * return 0 otherwise (local comm) */
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

/* Return maximum badge for a given topology level across all PGs */
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

/* Return the badge for a rank in a given comm and a specific topology level */
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
#endif
