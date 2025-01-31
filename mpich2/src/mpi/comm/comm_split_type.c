/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"
#include "mpicomm.h"

static int split_type_by_node(MPIR_Comm * comm_ptr, int key, MPIR_Comm ** newcomm_ptr);
static int node_split(MPIR_Comm * comm_ptr, int key, const char *hint_str,
                      MPIR_Comm ** newcomm_ptr);
static int split_type_pset_name(MPIR_Comm * comm_ptr, int key, const char *pset_name,
                                MPIR_Comm ** newcomm_ptr);
static int split_type_hw_guided(MPIR_Comm * comm_ptr, int key, const char *resource_type,
                                MPIR_Comm ** newcomm_ptr);
static int split_type_hw_unguided(MPIR_Comm * comm_ptr, int key, MPIR_Info * info_ptr,
                                  MPIR_Comm ** newcomm_ptr);

int MPIR_Comm_split_type(MPIR_Comm * user_comm_ptr, int split_type, int key,
                         MPIR_Info * info_ptr, MPIR_Comm ** newcomm_ptr)
{
    MPIR_Comm *comm_ptr = NULL;
    int mpi_errno = MPI_SUCCESS;

    /* split out the undefined processes */
    mpi_errno = MPIR_Comm_split_impl(user_comm_ptr, split_type == MPI_UNDEFINED ? MPI_UNDEFINED : 0,
                                     key, &comm_ptr);
    MPIR_ERR_CHECK(mpi_errno);

    if (split_type == MPI_UNDEFINED) {
        *newcomm_ptr = NULL;
        goto fn_exit;
    }

    if (split_type == MPI_COMM_TYPE_SHARED) {
        /* NOTE: MPIR_Comm_split_impl will typically call device layer function.
         * Currently ch4 calls MPIR_Comm_split_type_node_topo, thus doesn't run
         * the fallback code here.
         * On the otherhand, ch3:sock will directly execute code here. */
        mpi_errno = MPIR_Comm_split_type_self(comm_ptr, key, newcomm_ptr);
        MPIR_ERR_CHECK(mpi_errno);
    } else if (split_type == MPI_COMM_TYPE_HW_GUIDED) {
        const char *resource_type;
        mpi_errno = MPII_collect_info_key(comm_ptr, info_ptr, "mpi_hw_resource_type",
                                          &resource_type);
        MPIR_ERR_CHECK(mpi_errno);

        if (resource_type) {
            mpi_errno = split_type_hw_guided(comm_ptr, key, resource_type, newcomm_ptr);
            MPIR_ERR_CHECK(mpi_errno);
        } else {
            *newcomm_ptr = NULL;
            goto fn_exit;
        }
    } else if (split_type == MPI_COMM_TYPE_HW_UNGUIDED) {
        mpi_errno = split_type_hw_unguided(comm_ptr, key, info_ptr, newcomm_ptr);
        MPIR_ERR_CHECK(mpi_errno);
    } else if (split_type == MPI_COMM_TYPE_RESOURCE_GUIDED) {
        /* similar to MPI_COMM_TYPE_HW_GUIDED, but may also use process set name to split */
        /* first make sure we have consistent info keys */
        const char *resource_type = NULL;
        const char *pset_name = NULL;
        mpi_errno = MPII_collect_info_key(comm_ptr, info_ptr, "mpi_hw_resource_type",
                                          &resource_type);
        MPIR_ERR_CHECK(mpi_errno);
        mpi_errno = MPII_collect_info_key(comm_ptr, info_ptr, "mpi_pset_name", &pset_name);
        MPIR_ERR_CHECK(mpi_errno);
        if (resource_type) {
            mpi_errno = split_type_hw_guided(comm_ptr, key, resource_type, newcomm_ptr);
            MPIR_ERR_CHECK(mpi_errno);
        } else if (pset_name) {
            mpi_errno = split_type_pset_name(comm_ptr, key, pset_name, newcomm_ptr);
            MPIR_ERR_CHECK(mpi_errno);
        } else {
            *newcomm_ptr = NULL;
            goto fn_exit;
        }
    } else if (split_type == MPIX_COMM_TYPE_NEIGHBORHOOD) {
        mpi_errno =
            MPIR_Comm_split_type_neighborhood(comm_ptr, split_type, key, info_ptr, newcomm_ptr);
        MPIR_ERR_CHECK(mpi_errno);
    } else {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_ARG, "**arg");
    }

  fn_exit:
    if (comm_ptr)
        MPIR_Comm_free_impl(comm_ptr);
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

int MPIR_Comm_split_type_impl(MPIR_Comm * comm_ptr, int split_type, int key,
                              MPIR_Info * info_ptr, MPIR_Comm ** newcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    if (MPIR_Comm_fns != NULL && MPIR_Comm_fns->split_type != NULL) {
        mpi_errno = MPIR_Comm_fns->split_type(comm_ptr, split_type, key, info_ptr, newcomm_ptr);
    } else {
        mpi_errno = MPIR_Comm_split_type(comm_ptr, split_type, key, info_ptr, newcomm_ptr);
    }
    MPIR_ERR_CHECK(mpi_errno);

    mpi_errno = MPIR_Comm_set_info_impl(*newcomm_ptr, info_ptr);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_Comm_split_type_self(MPIR_Comm * comm_ptr, int key, MPIR_Comm ** newcomm_ptr)
{
    MPIR_Comm *comm_self_ptr;
    int mpi_errno = MPI_SUCCESS;

    MPIR_Comm_get_ptr(MPI_COMM_SELF, comm_self_ptr);
    mpi_errno = MPIR_Comm_dup_impl(comm_self_ptr, newcomm_ptr);

    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

static int split_type_pset_name(MPIR_Comm * comm_ptr, int key, const char *pset_name,
                                MPIR_Comm ** newcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    /* TODO: add implementation */
    *newcomm_ptr = NULL;

    return mpi_errno;
}

static int split_type_hw_guided(MPIR_Comm * comm_ptr, int key, const char *resource_type,
                                MPIR_Comm ** newcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm *node_comm = NULL;

    if (strcmp(resource_type, "mpi_shared_memory") == 0) {
        /* resource_type value "mpi_shared_memory" is equivalent to split_type
         * MPI_COMM_TYPE_SHARED */
        mpi_errno = MPIR_Comm_split_type_impl(comm_ptr, MPI_COMM_TYPE_SHARED, key, NULL,
                                              newcomm_ptr);
        MPIR_ERR_CHECK(mpi_errno);
        goto fn_exit;
    }

    /* now we should proceed */
    mpi_errno = split_type_by_node(comm_ptr, key, &node_comm);
    MPIR_ERR_CHECK(mpi_errno);

    if (comm_ptr == NULL) {
        /* it is possible with intercomm split */
        goto fn_exit;
    }

    if (!MPIR_hwtopo_is_initialized()) {
        /* if hwtopo is not available, return MPI_COMM_NULL */
        *newcomm_ptr = NULL;
        goto fn_exit;
    }

    /* only proceed when we have a proper gid, i.e. bindset belongs to a
     * single instance of given resource_type */
    MPIR_hwtopo_gid_t gid = MPIR_hwtopo_get_obj_by_name(resource_type);
    mpi_errno = MPIR_Comm_split_impl(node_comm, gid, key, newcomm_ptr);
    MPIR_ERR_CHECK(mpi_errno);

    if ((*newcomm_ptr)->remote_size == node_comm->remote_size) {
        /* failed to result in a proper split */
        MPIR_Comm_free_impl(*newcomm_ptr);
        *newcomm_ptr = NULL;
    }

  fn_exit:
    if (node_comm) {
        MPIR_Comm_free_impl(node_comm);
    }
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int split_type_hw_unguided(MPIR_Comm * comm_ptr, int key, MPIR_Info * info_ptr,
                                  MPIR_Comm ** newcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm *subcomm = NULL;
    const char *resource_type = NULL;

    int orig_size = MPIR_Comm_size(comm_ptr);

    /* With HW_UNDGUIDED, we need find the most top-level topology that result in a proper
     * subset of original comm_ptr. We'll try top-down until we hit upon a subset */

    /* TODO: Once we added network topology, we should try network topology first before
     * try splitting at node level.
     */
    mpi_errno = split_type_by_node(comm_ptr, key, &subcomm);
    MPIR_ERR_CHECK(mpi_errno);

    if (comm_ptr == NULL) {
        /* it is possible with intercomm split */
        goto fn_exit;
    }

    if (MPIR_Comm_size(subcomm) < orig_size) {
        resource_type = "node";
        *newcomm_ptr = subcomm;
        goto fn_exit;
    } else {
        MPIR_Comm_free_impl(subcomm);
    }

    /* TODO: determine the "proper" hierarchy */
    const char *topolist[] = {
        "package",
        "numanode",
        "cpu",
        "core",
        "hwthread",
        "bindset",
    };

    for (int i = 0; i < sizeof(topolist) / sizeof(topolist[0]); i++) {
        mpi_errno = node_split(comm_ptr, key, topolist[i], &subcomm);
        MPIR_ERR_CHECK(mpi_errno);

        if (MPIR_Comm_size(subcomm) < orig_size) {
            /* found the first true sub-comm, return it */
            resource_type = topolist[i];
            *newcomm_ptr = subcomm;
            goto fn_exit;
        } else {
            MPIR_Comm_free_impl(subcomm);
        }
    }

    /* no strict subset, return MPI_COMM_NULL */
    *newcomm_ptr = NULL;

  fn_exit:
    if (info_ptr && *newcomm_ptr && resource_type) {
        MPIR_Info_set_impl(info_ptr, "mpi_hw_resource_type", resource_type);
    }
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int split_type_by_node(MPIR_Comm * comm_ptr, int key, MPIR_Comm ** newcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    int color;

    mpi_errno = MPID_Get_node_id(comm_ptr, comm_ptr->rank, &color);
    MPIR_ERR_CHECK(mpi_errno);

    mpi_errno = MPIR_Comm_split_impl(comm_ptr, color, key, newcomm_ptr);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

int MPIR_Comm_split_type_node_topo(MPIR_Comm * user_comm_ptr, int key,
                                   MPIR_Info * info_ptr, MPIR_Comm ** newcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    *newcomm_ptr = NULL;

    MPIR_Comm *comm_ptr;
    mpi_errno = split_type_by_node(user_comm_ptr, key, &comm_ptr);
    MPIR_ERR_CHECK(mpi_errno);

    *newcomm_ptr = comm_ptr;

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

static int node_split(MPIR_Comm * comm_ptr, int key, const char *hint_str, MPIR_Comm ** newcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_hwtopo_gid_t gid;

    gid = MPIR_hwtopo_get_obj_by_name(hint_str);

    mpi_errno = MPIR_Comm_split_impl(comm_ptr, gid, key, newcomm_ptr);

    return mpi_errno;
}
