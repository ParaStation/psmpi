/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpidimpl.h"
#include "ch4_win.h"

int MPID_Win_set_info(MPIR_Win * win, MPIR_Info * info)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPID_WIN_SET_INFO);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPID_WIN_SET_INFO);

#ifdef MPIDI_CH4_DIRECT_NETMOD
    mpi_errno = MPIDI_NM_mpi_win_set_info(win, info);
    MPIR_ERR_CHECK(mpi_errno);
#else
    mpi_errno = MPIDIG_mpi_win_set_info(win, info);
    MPIR_ERR_CHECK(mpi_errno);
#endif

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPID_WIN_SET_INFO);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPID_Win_get_info(MPIR_Win * win, MPIR_Info ** info_p_p)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPID_WIN_GET_INFO);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPID_WIN_GET_INFO);

#ifdef MPIDI_CH4_DIRECT_NETMOD
    mpi_errno = MPIDI_NM_mpi_win_get_info(win, info_p_p);
    MPIR_ERR_CHECK(mpi_errno);
#else
    mpi_errno = MPIDIG_mpi_win_get_info(win, info_p_p);
    MPIR_ERR_CHECK(mpi_errno);
#endif

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPID_WIN_GET_INFO);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPID_Win_free(MPIR_Win ** win_ptr)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPID_WIN_FREE);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPID_WIN_FREE);

#ifdef MPIDI_CH4_DIRECT_NETMOD
    mpi_errno = MPIDI_NM_mpi_win_free(win_ptr);
    MPIR_ERR_CHECK(mpi_errno);
#else
    mpi_errno = MPIDIG_mpi_win_free(win_ptr);
    MPIR_ERR_CHECK(mpi_errno);
#endif

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPID_WIN_FREE);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPID_Win_create(void *base, MPI_Aint length, int disp_unit, MPIR_Info * info,
                    MPIR_Comm * comm_ptr, MPIR_Win ** win_ptr)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPID_WIN_CREATE);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPID_WIN_CREATE);
#ifdef MPIDI_CH4_DIRECT_NETMOD
    mpi_errno = MPIDI_NM_mpi_win_create(base, length, disp_unit, info, comm_ptr, win_ptr);
    MPIR_ERR_CHECK(mpi_errno);
#else
    mpi_errno = MPIDIG_mpi_win_create(base, length, disp_unit, info, comm_ptr, win_ptr);
    MPIR_ERR_CHECK(mpi_errno);
#endif

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPID_WIN_CREATE);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPID_Win_attach(MPIR_Win * win, void *base, MPI_Aint size)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPID_WIN_ATTACH);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPID_WIN_ATTACH);

#ifdef MPIDI_CH4_DIRECT_NETMOD
    mpi_errno = MPIDI_NM_mpi_win_attach(win, base, size);
    MPIR_ERR_CHECK(mpi_errno);
#else
    mpi_errno = MPIDIG_mpi_win_attach(win, base, size);
    MPIR_ERR_CHECK(mpi_errno);
#endif

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPID_WIN_ATTACH);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPID_Win_allocate_shared(MPI_Aint size, int disp_unit, MPIR_Info * info_ptr,
                             MPIR_Comm * comm_ptr, void **base_ptr, MPIR_Win ** win_ptr)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPID_WIN_ALLOCATE_SHARED);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPID_WIN_ALLOCATE_SHARED);
#ifdef MPIDI_CH4_DIRECT_NETMOD
    mpi_errno = MPIDI_NM_mpi_win_allocate_shared(size, disp_unit,
                                                 info_ptr, comm_ptr, base_ptr, win_ptr);
    MPIR_ERR_CHECK(mpi_errno);
#else
    mpi_errno = MPIDIG_mpi_win_allocate_shared(size, disp_unit, info_ptr, comm_ptr, base_ptr,
                                               win_ptr);
    MPIR_ERR_CHECK(mpi_errno);
#endif
  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPID_WIN_ALLOCATE_SHARED);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPID_Win_detach(MPIR_Win * win, const void *base)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPID_WIN_DETACH);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPID_WIN_DETACH);

#ifdef MPIDI_CH4_DIRECT_NETMOD
    mpi_errno = MPIDI_NM_mpi_win_detach(win, base);
    MPIR_ERR_CHECK(mpi_errno);
#else
    mpi_errno = MPIDIG_mpi_win_detach(win, base);
    MPIR_ERR_CHECK(mpi_errno);
#endif
  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPID_WIN_DETACH);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPID_Win_allocate(MPI_Aint size, int disp_unit, MPIR_Info * info, MPIR_Comm * comm,
                      void *baseptr, MPIR_Win ** win)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPID_WIN_ALLOCATE);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPID_WIN_ALLOCATE);
#ifdef MPIDI_CH4_DIRECT_NETMOD
    mpi_errno = MPIDI_NM_mpi_win_allocate(size, disp_unit, info, comm, baseptr, win);
    MPIR_ERR_CHECK(mpi_errno);
#else
    mpi_errno = MPIDIG_mpi_win_allocate(size, disp_unit, info, comm, baseptr, win);
    MPIR_ERR_CHECK(mpi_errno);
#endif
  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPID_WIN_ALLOCATE);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPID_Win_create_dynamic(MPIR_Info * info, MPIR_Comm * comm, MPIR_Win ** win)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPID_WIN_CREATE_DYNAMIC);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPID_WIN_CREATE_DYNAMIC);
#ifdef MPIDI_CH4_DIRECT_NETMOD
    mpi_errno = MPIDI_NM_mpi_win_create_dynamic(info, comm, win);
    MPIR_ERR_CHECK(mpi_errno);
#else
    mpi_errno = MPIDIG_mpi_win_create_dynamic(info, comm, win);
    MPIR_ERR_CHECK(mpi_errno);
#endif
  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPID_WIN_CREATE_DYNAMIC);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
