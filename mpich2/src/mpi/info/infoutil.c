/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"
#include "mpir_info.h"
/*#include <stdio.h>*/

/* This is the utility file for info that contains the basic info items
   and storage management */

/* Preallocated info objects */
MPIR_Info MPIR_Info_builtin[MPIR_INFO_N_BUILTIN];
MPIR_Info MPIR_Info_direct[MPIR_INFO_PREALLOC];

MPIR_Object_alloc_t MPIR_Info_mem = { 0, 0, 0, 0, 0, 0, MPIR_INFO,
    sizeof(MPIR_Info), MPIR_Info_direct,
    MPIR_INFO_PREALLOC,
    NULL, {0}
};

/* Free an info structure.  In the multithreaded case, this routine
   relies on the SINGLE_CS in the info routines (particularly MPI_Info_free) */
int MPIR_Info_free_impl(MPIR_Info * info_ptr)
{
    for (int i = 0; i < info_ptr->size; i++) {
        /* MPI_Info objects are allocated by normal MPL_direct_xxx() functions, so
         * they need to be freed by MPL_direct_free(), not MPL_free(). */
        MPL_direct_free(info_ptr->entries[i].key);
        MPL_direct_free(info_ptr->entries[i].value);
    }
    for (int i = 0; i < info_ptr->array_size; i++) {
        MPL_direct_free(info_ptr->array_entries[i].key);
        for (int j = 0; j < info_ptr->array_entries[i].num_values; j++) {
            MPL_direct_free(info_ptr->array_entries[i].values[j]);
        }
        MPL_direct_free(info_ptr->array_entries[i].values);
    }
    MPL_direct_free(info_ptr->array_entries);
    MPL_direct_free(info_ptr->entries);
    if (!HANDLE_IS_BUILTIN(info_ptr->handle)) {
        MPIR_Info_handle_obj_free(&MPIR_Info_mem, info_ptr);
    }

    return MPI_SUCCESS;
}

/* Allocate and initialize an MPIR_Info object. */

static void info_init(MPIR_Info * info_ptr)
{
    info_ptr->capacity = 0;
    info_ptr->size = 0;
    info_ptr->entries = NULL;
    info_ptr->array_capacity = 0;
    info_ptr->array_size = 0;
    info_ptr->array_entries = NULL;
}

int MPIR_Info_alloc(MPIR_Info ** info_p_p)
{
    int mpi_errno = MPI_SUCCESS;
    *info_p_p = (MPIR_Info *) MPIR_Info_handle_obj_alloc(&MPIR_Info_mem);
    MPIR_ERR_CHKANDJUMP1(!*info_p_p, mpi_errno, MPI_ERR_OTHER, "**nomem", "**nomem %s", "MPI_Info");

    MPIR_Object_set_ref(*info_p_p, 0);
    info_init(*info_p_p);

  fn_fail:
    return mpi_errno;
}

/* Set up MPI_INFO_ENV.  This may be called before initialization and after
 * finalization by MPI_Info_create_env().  This routine must be thread-safe. */
void MPIR_Info_setup_env(MPIR_Info * info_ptr)
{
    info_init(info_ptr);
    /* FIXME: Currently this info object is missing some data as
     * defined by the standard. */
}

#define INFO_INITIAL_SIZE 10
int MPIR_Info_push(MPIR_Info * info_ptr, const char *key, const char *val)
{
    int mpi_errno = MPI_SUCCESS;

    /* potentially grow the copacity */
    if (info_ptr->capacity == 0) {
        info_ptr->entries = MPL_direct_malloc(sizeof(*(info_ptr->entries)) * INFO_INITIAL_SIZE);
        MPIR_ERR_CHKANDJUMP(!info_ptr->entries, mpi_errno, MPI_ERR_OTHER, "**nomem");
        info_ptr->capacity = INFO_INITIAL_SIZE;
    } else if (info_ptr->size == info_ptr->capacity) {
        int n = info_ptr->capacity * 5 / 3;     /* arbitrary grow ratio */
        info_ptr->entries = MPL_direct_realloc(info_ptr->entries, sizeof(*(info_ptr->entries)) * n);
        info_ptr->capacity = n;

    }

    /* add new entry */
    int i = info_ptr->size;
    info_ptr->entries[i].key = MPL_direct_strdup(key);
    MPIR_ERR_CHKANDJUMP(!info_ptr->entries[i].key, mpi_errno, MPI_ERR_OTHER, "**nomem");
    info_ptr->entries[i].value = MPL_direct_strdup(val);
    MPIR_ERR_CHKANDJUMP(!info_ptr->entries[i].value, mpi_errno, MPI_ERR_OTHER, "**nomem");

    info_ptr->size++;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_Info_push_array(MPIR_Info * info_ptr, int index, int count, const char *key,
                         const char *val)
{
    int mpi_errno = MPI_SUCCESS;

    /* potentially grow the capacity */
    if (info_ptr->array_capacity == 0) {
        info_ptr->array_entries =
            MPL_direct_malloc(sizeof(*(info_ptr->array_entries)) * INFO_INITIAL_SIZE);
        MPIR_ERR_CHKANDJUMP(!info_ptr->array_entries, mpi_errno, MPI_ERR_OTHER, "**nomem");
        info_ptr->array_capacity = INFO_INITIAL_SIZE;
    } else if (info_ptr->array_size == info_ptr->array_capacity) {
        int n = info_ptr->array_capacity * 5 / 3;       /* arbitrary grow ratio */
        info_ptr->array_entries =
            MPL_direct_realloc(info_ptr->array_entries, sizeof(*(info_ptr->array_entries)) * n);
        MPIR_ERR_CHKANDJUMP(!info_ptr->array_entries, mpi_errno, MPI_ERR_OTHER, "**nomem");
        info_ptr->array_capacity = n;
    }

    /* add new entry */
    int i = info_ptr->array_size;
    info_ptr->array_entries[i].values =
        MPL_direct_malloc(sizeof(*(info_ptr->array_entries[i].values)) * count);
    MPIR_ERR_CHKANDJUMP(!info_ptr->array_entries[i].values, mpi_errno, MPI_ERR_OTHER, "**nomem");
    for (int j = 0; j < count; j++) {
        if (j != index) {
            info_ptr->array_entries[i].values[j] = MPL_direct_strdup("none");
            MPIR_ERR_CHKANDJUMP(!info_ptr->array_entries[i].values[j], mpi_errno, MPI_ERR_OTHER,
                                "**nomem");
        }
    }
    info_ptr->array_entries[i].key = MPL_direct_strdup(key);
    MPIR_ERR_CHKANDJUMP(!info_ptr->array_entries[i].key, mpi_errno, MPI_ERR_OTHER, "**nomem");
    info_ptr->array_entries[i].values[index] = MPL_direct_strdup(val);
    MPIR_ERR_CHKANDJUMP(!info_ptr->array_entries[i].values[index], mpi_errno, MPI_ERR_OTHER,
                        "**nomem");
    info_ptr->array_entries[i].num_values = count;

    info_ptr->array_size++;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_Info_set_array(MPIR_Info * info_ptr, int index, const char *key, const char *val)
{
    int mpi_errno = MPI_SUCCESS;
    int i, j = -1;
    int found = 0;

    for (i = 0; i < info_ptr->array_size; i++) {
        if (strncmp(info_ptr->array_entries[i].key, key, MPI_MAX_INFO_KEY) == 0) {
            found++;
            j = i;
        }
    }

    MPIR_Assertp((found == 1) && (j >= 0) && (index < info_ptr->array_entries[j].num_values));
    MPL_direct_free(info_ptr->array_entries[j].values[index]);
    info_ptr->array_entries[j].values[index] = MPL_direct_strdup(val);
    MPIR_ERR_CHKANDJUMP(!info_ptr->array_entries[j].values[index], mpi_errno, MPI_ERR_OTHER,
                        "**nomem");

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
