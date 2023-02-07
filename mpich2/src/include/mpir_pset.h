/*
 * ParaStation
 *
 * Copyright (C) 2023 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpiimpl.h"

/**
 * @brief Process set structure
 */
struct MPIR_Pset {
    char *uri;                  /* Name of the process set */
    int size;                   /* Number of processes in the set */
    bool is_valid;              /* True if process set is valid, false if not */
    int *members;               /* Array of members of the process set (ranks) sorted in ascending order */
};

/**
 * @brief Process set array structure
 */
struct MPIR_Pset_array {
    MPIR_OBJECT_HEADER;
    UT_array *parray;           /* Array of psets */
    MPID_Thread_mutex_t *mutex; /* Optional mutex to protect parray from concurrent accesses */
};

/**
 * @brief Increase ref counter of pset array by one
 */
#define MPIR_Pset_array_add_ref(_pset_array) \
    do { MPIR_Object_add_ref(_pset_array); } while (0)

/**
 * @brief Decrease ref counter of pset array by one and return number of remaining refs
 */
#define MPIR_Pset_array_release_ref(_pset_array, _inuse) \
    do { MPIR_Object_release_ref(_pset_array, _inuse); } while (0)

/**
 * @brief Initialize a pset array
 *
 * @param pset_array    Pset array to initialize
 * @param mutex         Optional pointer to mutex (mutex created if not NULL)
 */
void MPIR_Pset_array_init(MPIR_Pset_array ** pset_array, MPID_Thread_mutex_t * mutex);

/**
 * @brief Destroy a pset array
 *
 * @param parray    Pset array to be destroyed
 */
void MPIR_Pset_array_destroy(MPIR_Pset_array * pset_array);

/**
 * @brief   Add a pset to a pset array.
 *
 * @param   parray  Pset array to which the pset shall be added
 * @param   pset    Pset to be added
 * @return  int     MPI_SUCCESS or MPI_ERR_OTHER
 */
int MPIR_Pset_array_add(MPIR_Pset_array * pset_array, MPIR_Pset * pset);

/**
 * @brief   Invalidate a pset in pset array.
 *
 * @param   parray      Pset array in which the pset shall be invalidated
 * @param   pset_name   Name of the pset
 * @return  int         MPI_SUCCESS or MPI_ERR_OTHER (not found)
 */
int MPIR_Pset_array_invalidate(MPIR_Pset_array * pset_array, char *pset_name);

/**
 * @brief   Initialize the process set array of a session
 *
 * @param   session_ptr Session pointer
 * @return  int         MPI_SUCCESS or MPI_ERR_OTHER
 */
int MPIR_Session_psets_init(MPIR_Session * session_ptr);

/**
 * @brief   Destroy the process set array of a session
 *
 * @param   session_ptr Session pointer
 * @return  int         MPI_SUCCESS
 */
int MPIR_Session_psets_destroy(MPIR_Session * session_ptr);

/**
 * @brief   Get the number of psets known in a session
 *
 * @param   session_ptr Session for which number of psets is requested
 * @return  int         Number of psets
 */
int MPIR_Session_psets_get_num(MPIR_Session * session_ptr);

/**
 * @brief   Get a pset of a session based on the index
 *
 * @param   session_ptr Session pointer
 * @param   idx         Valid index
 * @param   pset        Pset to return
 * @return  int         MPI_SUCCESS or MPI_ERR_OTHER
 */
int MPIR_Session_pset_by_idx(MPIR_Session * session_ptr, int idx, MPIR_Pset ** pset);

/**
 * @brief   Get a pset of a session based on the pset name
 *
 * @param   session_ptr Session pointer
 * @param   pset_name   Name of the pset
 * @param   pset        Pset to return
 * @return  int         MPI_SUCCESS or MPI_ERR_OTHER
 */
int MPIR_Session_pset_by_name(MPIR_Session * session_ptr, const char *pset_name, MPIR_Pset ** pset);
