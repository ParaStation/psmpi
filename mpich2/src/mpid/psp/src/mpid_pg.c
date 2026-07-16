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

#include "mpidimpl.h"

#include "mpid_psp_topo.h"

/* Get the process group for a given world index */
int MPIDI_PG_get(int world_idx, MPIDI_PG_t ** pg_out)
{

    int mpi_errno = MPI_SUCCESS;

    MPIDI_PG_t *pg = MPIDI_Process.my_pg;

    while (pg) {
        if (pg->world_idx == world_idx) {
            break;
        }
        pg = pg->next;
    }

    *pg_out = pg;       /* NULL is set here if world_idx was not found */

    return mpi_errno;
}

static void MPIDI_PG_Convert_id(char *pg_id_name, int *pg_id_num);

int MPIDI_PG_Create(int world_idx, MPIDI_PSP_topo_level_t * levels, MPIDI_PG_t ** pg_ptr)
{
    MPIDI_PG_t *pg = NULL, *pgnext;
    int i;
    int mpi_errno = MPI_SUCCESS;
    MPIR_CHKPMEM_DECL();

    MPIR_FUNC_ENTER;

    int pg_size = MPIR_Worlds[world_idx].num_procs;

    MPIR_CHKPMEM_MALLOC(pg, sizeof(MPIDI_PG_t), MPL_MEM_OBJECT);
    MPIR_CHKPMEM_MALLOC(pg->vcr, sizeof(MPIDI_VC_t) * pg_size, MPL_MEM_OBJECT);
    MPIR_CHKPMEM_MALLOC(pg->lpids, sizeof(MPIR_Lpid) * pg_size, MPL_MEM_OBJECT);
    MPIR_CHKPMEM_MALLOC(pg->cons, sizeof(pscom_connection_t *) * pg_size, MPL_MEM_OBJECT);

    pg->size = pg_size;
    MPIDI_PG_Convert_id(MPIR_Worlds[world_idx].namespace, &(pg->id_num));
    pg->world_idx = world_idx;
    pg->refcnt = 0;
#ifdef MPID_PSP_MSA_AWARE_COLLOPS
    pg->topo_levels = NULL;
#endif
    for (i = 0; i < pg_size; i++) {
        pg->vcr[i] = NULL;
        pg->lpids[i] = MPIR_LPID_INVALID;
        pg->cons[i] = NULL;
    }

    /* Add pg's at the tail so that comm world is always the first pg */
    pg->next = NULL;

    if (!MPIDI_Process.my_pg) {
        /* The first process group is always the world group */
        MPIDI_Process.my_pg = pg;
    } else {
        pgnext = MPIDI_Process.my_pg;
        while (pgnext->next) {
            pgnext = pgnext->next;
        }
        pgnext->next = pg;
    }

#ifdef MPID_PSP_MSA_AWARE_COLLOPS
    MPIDI_PSP_add_topo_levels_to_pg(pg, levels);

    if (pg != MPIDI_Process.my_pg) {    // This is for the rare case that joined PGs do not feature the same set of level degrees!

        MPIDI_PSP_topo_level_t *level = pg->topo_levels;
        while (level) { // If not known, add a flat badge table (as a "dummy") with same the degree to the home PG:
            MPIDI_PSP_topo_level_t *level_next = level->next;
            if (level->badges_are_global &&
                !MPIDI_PSP_check_pg_for_level(level->degree, MPIDI_Process.my_pg, NULL)) {
                MPIDI_PSP_add_flat_level_to_pg(MPIDI_Process.my_pg, level->degree);
            }
            level = level_next;
        }
    }
#endif

    if (pg_ptr)
        *pg_ptr = pg;

  fn_exit:
    MPIR_FUNC_EXIT;
    return mpi_errno;

  fn_fail:
    MPIR_CHKPMEM_REAP();
    goto fn_exit;
}

MPIDI_PG_t *MPIDI_PG_Destroy(MPIDI_PG_t * pg_ptr)
{
    int j;
    MPIDI_PG_t *pg_next = pg_ptr->next;

    /* Check if this is the PG of the local COMM_WORLD.
     * If not, ensure that the list of PGs does not get broken: */
    if (pg_ptr != MPIDI_Process.my_pg) {
        MPIDI_PG_t *pg_run = MPIDI_Process.my_pg;
        MPIR_Assert(pg_run);
        while (pg_run->next != pg_ptr) {
            pg_run = pg_run->next;
            MPIR_Assert(pg_run);
        }
        pg_run->next = pg_next;
    }


    for (j = 0; j < pg_ptr->size; j++) {

        MPIR_Assert((pg_ptr->refcnt > 0) || ((pg_ptr->refcnt == 0) && (!pg_ptr->vcr[j])));

        if (pg_ptr->vcr[j]) {
            /* If MPIDI_PG_Destroy() is called with still existing connections,
             * then this PG has not been disconnected before. Hence, this is most
             * likely the common case, where an MPI_Finalize() is tearing down the
             * current session. Therefore, we just close the still open connections
             * and free the related VCR without any decreasing of reference counters:
             */
            if (!MPIDI_Process.env.enable_keep_connections && (pg_ptr->vcr[j]->con != NULL)) {
                pscom_close_connection(pg_ptr->vcr[j]->con);
            }
            MPL_free(pg_ptr->vcr[j]);

        } else {

            if (pg_ptr->cons[j]) {
                /* If we come here, this rank has already been disconnected in an
                 * MPI sense but due to the 'lazy disconnect' feature, the old
                 * pscom connection is still open. Hence, close it right here:
                 */
                pscom_close_connection(pg_ptr->cons[j]);
            }
        }
    }

#ifdef MPID_PSP_MSA_AWARE_COLLOPS
    while (pg_ptr->topo_levels) {
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

int MPIDI_PG_Resize(MPIDI_PG_t * pg, int new_size)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_Assert(pg != NULL);
    MPIR_Assert(pg->size > 0);

    if (new_size <= pg->size) {
        /* never make a PG smaller */
        goto fn_fail;
    }

    MPL_realloc(pg->cons, new_size * sizeof(pscom_connection_t *), MPL_MEM_OTHER);
    MPIR_ERR_CHKANDJUMP(!pg->cons, mpi_errno, MPI_ERR_OTHER, "**nomem");
    MPL_realloc(pg->vcr, new_size * sizeof(MPIDI_VC_t), MPL_MEM_OTHER);
    MPIR_ERR_CHKANDJUMP(!pg->vcr, mpi_errno, MPI_ERR_OTHER, "**nomem");
    MPL_realloc(pg->lpids, new_size * sizeof(MPIR_Lpid), MPL_MEM_OTHER);
    MPIR_ERR_CHKANDJUMP(!pg->lpids, mpi_errno, MPI_ERR_OTHER, "**nomem");

    /* Init new part */
    for (int i = pg->size; i < new_size; i++) {
        pg->vcr[i] = NULL;
        pg->lpids[i] = MPIR_LPID_INVALID;
        pg->cons[i] = NULL;
    }

    pg->size = new_size;

  fn_fail:
    return mpi_errno;
}

/* Taken from MPIDI_PG_IdToNum() of CH3: */
static
void MPIDI_PG_Convert_id(char *pg_id_name, int *pg_id_num)
{
    const char *p = (const char *) pg_id_name;
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

int MPIDI_PSP_PG_init(void)
{
    int pg_size = MPIDI_Process.my_pg_size;
    int mpi_errno = MPI_SUCCESS;
    int grank;
    MPIDI_PG_t *pg_ptr;
    MPIDI_PSP_topo_level_t *topo_levels = NULL;
    int world_idx = 0;          /* my_pg is always world_idx 0 */

    if (MPIDI_Process.my_pg != NULL) {
        goto fn_exit;
    }

    /* Initialize the hierarchical topology information as used for MSA-aware collectives. */
    mpi_errno = MPIDI_PSP_topo_init(&topo_levels);
    MPIR_ERR_CHECK(mpi_errno);
#ifdef MPID_PSP_MSA_AWARE_COLLOPS
    if (MPIDI_Process.env.enable_msa_awareness && MPIDI_Process.env.enable_msa_aware_collops) {
        /* If MSA aware collops are enabled topo_levels MUST be initialized at this point */
        MPIR_Assert(topo_levels != NULL);
    }
#endif

    /* Create and set MPIDI_Process.my_pg including all processes */
    MPIR_Assert(pg_size == MPIR_Worlds[world_idx].num_procs);
    mpi_errno = MPIDI_PG_Create(world_idx, topo_levels, &pg_ptr);
    MPIR_ERR_CHECK(mpi_errno);

    MPIR_Assert(pg_ptr == MPIDI_Process.my_pg);

    for (grank = 0; grank < pg_size; grank++) {
        /* Init connections with NULL, initialized during comm creation in grank2con_set */
        MPIR_Lpid lpid = MPIR_LPID_FROM(world_idx, grank);
        pg_ptr->vcr[grank] = MPIDI_VC_Create(pg_ptr, grank, NULL, lpid);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

void MPIDI_PSP_PG_finalize(void)
{
    MPIDI_PG_t *pg_ptr;

    if (MPIDI_Process.my_pg) {
        pg_ptr = MPIDI_Process.my_pg->next;
        while (pg_ptr) {
            pg_ptr = MPIDI_PG_Destroy(pg_ptr);
        }
        MPIDI_PG_Destroy(MPIDI_Process.my_pg);
        MPIDI_Process.my_pg = NULL;
    }
    /* for re-init */
    //MPIDI_Process.next_lpid = 0;

    if (!MPIDI_Process.env.enable_keep_connections) {
        MPL_free(MPIDI_Process.grank2con);
        MPIDI_Process.grank2con = NULL;
    }

    MPL_free(MPIDI_Process.pg_id_name);
    MPIDI_Process.pg_id_name = NULL;
}
