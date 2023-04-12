/*
 * ParaStation
 *
 * Copyright (C) 2023 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpir_pset.h"
#include "mpir_session.h"


/* Protect ciritial sections in session and process set management

   This is required independent of multi-threaded support
   because the PM event thread and MPI worker process/thread
   may access the global PM pset array concurrently -
   MPID_THREAD_CS_ENTER and MPID_THREAD_CS_EXIT cannot be used here.*/

/**
 * @brief Enter crititcal section with mutex (multi-threading support not required)
 */
#define PSET_CS_ENTER(mutex_ptr)                     \
    if (mutex_ptr) {                                    \
        int thr_err;                                    \
        MPID_Thread_mutex_lock(mutex_ptr, &thr_err);    \
        MPIR_Assert(thr_err == MPI_SUCCESS);            \
    }                                                   \

/**
 * @brief Exit crititcal section with mutex (multi-threading support not required)
 */
#define PSET_CS_EXIT(mutex_ptr)                      \
    if (mutex_ptr) {                                    \
        int thr_err;                                    \
        MPID_Thread_mutex_unlock(mutex_ptr, &thr_err);  \
        MPIR_Assert(thr_err == MPI_SUCCESS);            \
    }                                                   \

/**
 * @brief Create a mutex (multi-threading support not required)
 */
#define PSET_MUTEX_CREATE(mutex_ptr)                 \
        int thr_err;                                    \
        MPID_Thread_mutex_create(mutex_ptr, &thr_err);  \
        MPIR_Assert(thr_err == MPI_SUCCESS);            \

/**
 * @brief Destroy a mutex (multi-threading support not required)
 */
#define PSET_MUTEX_DESTROY(mutex_ptr)                \
        int thr_err;                                    \
        MPID_Thread_mutex_destroy(mutex_ptr, &thr_err); \
        MPIR_Assert(thr_err == MPI_SUCCESS);            \


/**
 * @brief Enum for indexes of session's array of pset arrays
 */
typedef enum Pset_source {
    DEFAULT_PSET_IDX = 0,
    PM_PSET_IDX,
    NUM_PSET_SOURCES
} Pset_source_t;

/**
 * @brief Copy function for MPIR_Psets in UT_arrays
 *
 * @param _dst Destination for copy operation
 * @param _src Source of copy operation
 */
static
void pset_copy(void *_dst, const void *_src)
{
    MPIR_Pset *dst = (MPIR_Pset *) _dst;
    MPIR_Pset *src = (MPIR_Pset *) _src;

    dst->is_valid = src->is_valid;
    dst->size = src->size;
    dst->uri = src->uri ? MPL_strdup(src->uri) : NULL;
    dst->members = MPL_malloc(sizeof(int) * src->size, MPL_MEM_SESSION);
    memcpy(dst->members, src->members, sizeof(int) * src->size);
}

/**
 * @brief Destructor for MPIR_Psets in UT_arrays.
 *
 * @param _elt Pointer to MPIR_Pset object to be destroyed
 */
static
void pset_dtor(void *_elt)
{
    MPIR_Pset *elt = (MPIR_Pset *) _elt;
    if (elt->uri != NULL)
        MPL_free(elt->uri);

    if (elt->members != NULL)
        MPL_free(elt->members);
}

/**
 * @brief Structure used to work with MPIR_Psets in UT_arrays. Configures init, copy and destructor methods for MPIR_Psets.
 */
static const UT_icd pset_array_icd = { sizeof(MPIR_Pset), NULL, pset_copy, pset_dtor };

/**
 * @brief   Find pset by its name in UT_array (not thread-safe).
 *
 * @param   parray      UT_array in which the pset shall be found
 * @param   pset_name   Name of the pset
 * @param   pset        Returns the found pset
 * @return  int         MPI_SUCCESS or MPI_ERR_OTHER (not found)
 */
static
int pset_find_by_name(UT_array * parray, const char *pset_name, MPIR_Pset ** pset)
{
    int ret = MPI_ERR_OTHER;
    MPIR_Pset *p = NULL;
    *pset = NULL;
    for (unsigned i = 0; i < utarray_len(parray); i++) {
        p = (MPIR_Pset *) utarray_eltptr(parray, i);
        if (strncasecmp(pset_name, p->uri, MAX(strlen(pset_name), strlen(p->uri))) == 0) {
            ret = MPI_SUCCESS;
            *pset = p;
            break;
        }
    }
    return ret;
}

/**
 * @brief   Get pset from pset array based on its name.
 *
 * @param   parray      Pset array in which the pset shall be found
 * @param   pset_name   Name of the pset
 * @param   pset        Returns the found pset
 * @return  int         MPI_SUCCESS or MPI_ERR_OTHER
 */
static
int pset_by_name(MPIR_Pset_array * pset_array, const char *pset_name, MPIR_Pset ** pset)
{
    int ret;

    PSET_CS_ENTER(pset_array->mutex);

    if (pset_name != NULL) {
        ret = pset_find_by_name(pset_array->parray, pset_name, pset);
    } else {
        ret = MPI_ERR_OTHER;
    }

    PSET_CS_EXIT(pset_array->mutex);

    return ret;
}

/**
 * @brief   Get pset from pset array based on its index.
 *
 * @param   parray  Pset array in which the pset shall be found
 * @param   idx     Index of the pset
 * @param   pset    Returns the found pset
 * @return  int     MPI_SUCCESS or MPI_ERR_OTHER
 */
static
int pset_by_idx(MPIR_Pset_array * pset_array, int idx, MPIR_Pset ** pset)
{
    int ret = MPI_SUCCESS;

    PSET_CS_ENTER(pset_array->mutex);

    unsigned len = utarray_len(pset_array->parray);

    /* Overflow check between unsigned and int data types */
    MPIR_Assert(len <= INT_MAX);

    if (idx >= 0 && idx < (int) len) {
        *pset = (MPIR_Pset *) utarray_eltptr(pset_array->parray, idx);
    } else {
        ret = MPI_ERR_OTHER;
    }

    PSET_CS_EXIT(pset_array->mutex);

    return ret;
}

void MPIR_Pset_array_init(MPIR_Pset_array ** pset_array, MPID_Thread_mutex_t * mutex)
{
    (*pset_array) = MPL_malloc(sizeof(MPIR_Pset_array), MPL_MEM_SESSION);

    utarray_new((*pset_array)->parray, &pset_array_icd, MPL_MEM_SESSION);

    (*pset_array)->mutex = mutex;
    if ((*pset_array)->mutex) {
        /* Init the mutex to protect the pset array from concurrent accesses */
        PSET_MUTEX_CREATE((*pset_array)->mutex);
    }

    MPIR_Object_set_ref(*pset_array, 1);
}

void MPIR_Pset_array_destroy(MPIR_Pset_array * pset_array)
{
    int in_use;

    MPIR_Pset_array_release_ref(pset_array, &in_use);

    if (in_use == 0) {
        /* Destroy only if all refs are gone */
        utarray_free(pset_array->parray);
        if (pset_array->mutex) {
            PSET_MUTEX_DESTROY(pset_array->mutex);
        }
        MPL_free(pset_array);
    }
}

int MPIR_Pset_array_add(MPIR_Pset_array * pset_array, MPIR_Pset * pset)
{
    int ret;
    MPIR_Pset *p;

    PSET_CS_ENTER(pset_array->mutex);

    if (pset_find_by_name(pset_array->parray, pset->uri, &p) == MPI_ERR_OTHER) {
        /* Pset with uri NOT found in the parray */
        utarray_push_back(pset_array->parray, pset, MPL_MEM_SESSION);
        ret = MPI_SUCCESS;
    } else {
        ret = MPI_ERR_OTHER;
    }

    PSET_CS_EXIT(pset_array->mutex);

    return ret;
}

int MPIR_Pset_array_invalidate(MPIR_Pset_array * pset_array, char *pset_name)
{
    int ret;
    MPIR_Pset *p = NULL;

    PSET_CS_ENTER(pset_array->mutex);

    ret = pset_find_by_name(pset_array->parray, pset_name, &p);
    if (ret == MPI_SUCCESS) {
        p->is_valid = false;    /* Set to invalid */
    }

    PSET_CS_EXIT(pset_array->mutex);

    return ret;
}

/**
 * @brief   Initialize and add default MPI process sets to a session
 *
 * @param   session_ptr     Session for which default psets shall be initialized and added
 * @return  int             MPI_SUCCESS or error code
 */
static
int init_add_default_psets(MPIR_Session * session_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    char world_name[] = "mpi://WORLD";
    char self_name[] = "mpi://SELF";
    int *world_members = NULL;
    int *self_member = NULL;

    /* Init array of MPI default psets (freed in MPIR_Session_psets_destroy), no mutex required */
    MPIR_Pset_array_init(&(session_ptr->default_pset_array), NULL);

    world_members = MPL_malloc(MPIR_Process.size * sizeof(int), MPL_MEM_SESSION);
    for (int p = 0; p < MPIR_Process.size; p++) {
        world_members[p] = p;
    }
    MPIR_Pset pset_world = {
        .uri = world_name,
        .size = MPIR_Process.size,
        .is_valid = true,
        .members = world_members
    };

    mpi_errno = MPIR_Pset_array_add(session_ptr->default_pset_array, &pset_world);
    MPL_free(world_members);
    if (mpi_errno == MPI_ERR_OTHER) {
        mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE, __func__, __LINE__,
                                         MPI_ERR_OTHER, "**sessioninit", "**sessioninit %s",
                                         "could not add default mpi pset to session");
        return mpi_errno;
    }

    self_member = MPL_malloc(sizeof(int), MPL_MEM_SESSION);
    self_member[0] = MPIR_Process.rank;
    MPIR_Pset pset_self = {
        .uri = self_name,
        .size = 1,
        .is_valid = true,
        .members = self_member
    };

    mpi_errno = MPIR_Pset_array_add(session_ptr->default_pset_array, &pset_self);
    MPL_free(self_member);
    if (mpi_errno == MPI_ERR_OTHER) {
        mpi_errno = MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE, __func__, __LINE__,
                                         MPI_ERR_OTHER, "**sessioninit", "**sessioninit %s",
                                         "could not add default mpi pset to session");
        return mpi_errno;
    }

    /* Add default pset array to session's array of pset arrays */
    utarray_push_back(session_ptr->psets, (void *) &(session_ptr->default_pset_array),
                      MPL_MEM_SESSION);

    return mpi_errno;
}

/**
 * @brief   Add process manager's process sets to a session
 *
 * @param   session_ptr     Session to which process manager's process sets shall be added
 * @return  int             MPI_SUCCESS
 */
static
int add_pm_psets(MPIR_Session * session_ptr)
{
    /* Add pointer to global PM pset array and increment its ref counter */
    utarray_push_back(session_ptr->psets, (void *) &(MPIR_Process.pm_pset_array), MPL_MEM_SESSION);
    MPIR_Pset_array_add_ref(MPIR_Process.pm_pset_array);

    return MPI_SUCCESS;
}

int MPIR_Finalize_pm_pset_cb(void *param ATTRIBUTE((unused)))
{
    /* Deregister PM event handler for pset define and delete events */
    MPIR_pmi_deregister_process_set_event_handlers();

    /* Free memory allocated for PM pset array and destroy the mutex */
    MPIR_Pset_array_destroy(MPIR_Process.pm_pset_array);
    return 0;
}

int MPIR_Session_psets_init(MPIR_Session * session_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    /* Initialize array of pset arrays */
    utarray_new(session_ptr->psets, &ut_ptr_icd, MPL_MEM_SESSION);

    for (Pset_source_t pset_src = 0; pset_src < NUM_PSET_SOURCES; pset_src++) {
        switch (pset_src) {
            case DEFAULT_PSET_IDX:{
                    mpi_errno = init_add_default_psets(session_ptr);
                    MPIR_ERR_CHECK(mpi_errno);
                    break;
                }
            case PM_PSET_IDX:{
                    mpi_errno = add_pm_psets(session_ptr);
                    MPIR_ERR_CHECK(mpi_errno);
                    break;
                }
            default:{
                    MPIR_ERR_SETANDSTMT1(mpi_errno, MPI_ERR_OTHER, goto fn_fail,
                                         "**sessioninit", "**sessioninit %s",
                                         "unsupported process set source");
                    break;
                }
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_Session_psets_destroy(MPIR_Session * session_ptr)
{
    if (session_ptr->psets != NULL) {
        for (unsigned i = 0; i < utarray_len(session_ptr->psets); i++) {
            MPIR_Pset_array **pset_array =
                (MPIR_Pset_array **) utarray_eltptr(session_ptr->psets, i);
            MPIR_Pset_array_destroy(*pset_array);
        }

        /* Free pset array of session */
        utarray_free(session_ptr->psets);
    }

    return MPI_SUCCESS;
}

int MPIR_Session_psets_get_num(MPIR_Session * session_ptr)
{
    unsigned count = 0;
    for (unsigned i = 0; i < utarray_len(session_ptr->psets); i++) {
        MPIR_Pset_array **pset_array = (MPIR_Pset_array **) utarray_eltptr(session_ptr->psets, i);
        count += utarray_len((*pset_array)->parray);
    }

    /* Overflow check between unsigned and int data types */
    MPIR_Assert(count <= INT_MAX);

    return (int) count;
}

int MPIR_Session_pset_by_idx(MPIR_Session * session_ptr, int idx, MPIR_Pset ** pset)
{
    int mpi_errno = MPI_SUCCESS;
    unsigned upper = 0;

    for (unsigned i = 0; i < utarray_len(session_ptr->psets); i++) {
        MPIR_Pset_array **pset_array = (MPIR_Pset_array **) utarray_eltptr(session_ptr->psets, i);
        unsigned len = utarray_len((*pset_array)->parray);

        /* Overflow check between unsigned and int data types */
        MPIR_Assert(upper + len <= INT_MAX);

        if (idx < upper + len) {
            /* The pset is in pset_arary */
            mpi_errno = pset_by_idx(*pset_array, idx - (int) upper, pset);
            break;
        }
        upper += len;
    }

    return mpi_errno;
}


int MPIR_Session_pset_by_name(MPIR_Session * session_ptr, const char *pset_name, MPIR_Pset ** pset)
{
    int mpi_errno = MPI_SUCCESS;
    for (unsigned i = 0; i < utarray_len(session_ptr->psets); i++) {
        MPIR_Pset_array **pset_array = (MPIR_Pset_array **) utarray_eltptr(session_ptr->psets, i);
        mpi_errno = pset_by_name(*pset_array, pset_name, pset);
        if (mpi_errno == MPI_SUCCESS) {
            /* Found pset_name in pset_array, stop search */
            break;
        }
    }

    return mpi_errno;
}
