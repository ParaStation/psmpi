/*
 * ParaStation
 *
 * Copyright (C) 2025-2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"
#include "mpid_win_info.h"



int MPID_Win_set_info(MPIR_Win * win, MPIR_Info * info)
{
    int mpi_errno = MPI_SUCCESS;
    int info_flag = 0;
    char info_value[MPI_MAX_INFO_VAL + 1];

    MPIR_FUNC_ENTER;

    if (info == NULL) {
        goto fn_exit;
    }

    /* check for info key "no_locks" */
    MPIDI_PSP_WIN_INFO_GET_ARG(win->info_args, info, no_locks, true, false, info_value, info_flag);

    MPIDI_PSP_INFO_GET(info, "accumulate_ordering", info_value, info_flag);
    if (info_flag) {
        if (strcmp(info_value, "none") == 0) {
            win->info_args.accumulate_ordering = 0;
        } else {
            char *token, *save_ptr;
            int ordering = 0;

            token = (char *) strtok_r(info_value, ",", &save_ptr);
            while (token) {
                if (strcmp(token, "rar") == 0) {
                    ordering |= MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_rar;
                } else if (strcmp(token, "raw") == 0) {
                    ordering |= MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_raw;
                } else if (strcmp(token, "war") == 0) {
                    ordering |= MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_war;
                } else if (strcmp(token, "waw") == 0) {
                    ordering |= MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_waw;
                } else {
                    ordering = MPIDI_PSP_WIN_INFO_ARG_invalid;
                    break;
                }
                token = (char *) strtok_r(NULL, ",", &save_ptr);
            }
            win->info_args.accumulate_ordering = ordering;
        }
    }

    /* check for info key "accumulate_ops" */
    MPIDI_PSP_WIN_INFO_GET_ARG(win->info_args, info, accumulate_ops, same_op, same_op_no_op,
                               info_value, info_flag);

    /* check for info key "(mpi_)accumualte_granularity" */
    MPIDI_PSP_WIN_INFO_GET_ARG_INT(win->info_args, info, mpi_accumulate_granularity, info_value,
                                   info_flag);

    /* check for info key "same_size" */
    MPIDI_PSP_WIN_INFO_GET_ARG(win->info_args, info, same_size, true, false, info_value, info_flag);

    /* check for info key "same_disp_unit" */
    MPIDI_PSP_WIN_INFO_GET_ARG(win->info_args, info, same_disp_unit, true, false, info_value,
                               info_flag);

    /* check for info key "alloc_shared_noncontig" */
    MPIDI_PSP_WIN_INFO_GET_ARG(win->info_args, info, alloc_shared_noncontig, true, false,
                               info_value, info_flag);

    /* check for info key "wait_on_passive_side" (PSP/psmpi-specific) */
    MPIDI_PSP_WIN_INFO_GET_ARG(win->info_args, info, wait_on_passive_side, explicit, none,
                               info_value, info_flag);

    /* apply updates to current window configuration */
    if (MPIDI_PSP_WIN_INFO_APPLY_ARG(win->info_args, accumulate_ordering, none, 0)) {
        win->enable_rma_accumulate_ordering = 0;
    }
    if (MPIDI_PSP_WIN_INFO_APPLY_ARG(win->info_args, wait_on_passive_side, none, 0)) {
        win->enable_explicit_wait_on_passive_side = 0;
    }

  fn_exit:
    MPIR_FUNC_EXIT;
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPID_Win_get_info(MPIR_Win * win, MPIR_Info ** info_p_p)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Info *info_used = NULL;

    MPIR_FUNC_ENTER;

    *info_p_p = NULL;

    /* Allocate an empty info object */
    mpi_errno = MPIR_Info_alloc(&info_used);
    MPIR_ERR_CHECK(mpi_errno);

    /* check standardized keys: explicitly set if supported (often only the default)
     * and also set/overwrite if an invalid value has been set by the user. */

    /* check for info key "no_locks" */
    MPIDI_PSP_WIN_INFO_SET_ARG_DEFAULT(win->info_args, info_used, no_locks, false);

    /* check for info key "accumulate_ordering" */
    MPIDI_PSP_WIN_INFO_SET_ARG_DEFAULT(win->info_args, info_used, accumulate_ordering, all,
                                       MPIDI_PSP_WIN_INFO_ARG_DIFFERENT("rar,raw,war,waw"));
    MPIDI_PSP_WIN_INFO_SET_ARG_SPECIFIC(win->info_args, info_used, accumulate_ordering, all,
                                        win->enable_rma_accumulate_ordering,
                                        MPIDI_PSP_WIN_INFO_ARG_DIFFERENT("rar,raw,war,waw"));
    MPIDI_PSP_WIN_INFO_SET_ARG_SPECIFIC(win->info_args, info_used, accumulate_ordering, none,
                                        !win->enable_rma_accumulate_ordering);

    /* check for info key "accumulate_ops" */
    MPIDI_PSP_WIN_INFO_SET_ARG_DEFAULT(win->info_args, info_used, accumulate_ops, same_op_no_op);

    /* check for info key "mpi_accumualte_granularity" */
    MPIDI_PSP_WIN_INFO_SET_ARG_INT_DEFAULT(win->info_args, info_used, mpi_accumulate_granularity,
                                           0);
    /* check for info key "same_size" */
    MPIDI_PSP_WIN_INFO_SET_ARG_DEFAULT(win->info_args, info_used, same_size, false);

    /* check for info key "same_disp_unit" */
    MPIDI_PSP_WIN_INFO_SET_ARG_DEFAULT(win->info_args, info_used, same_disp_unit, false);

    /* check for info key "alloc_shared_noncontig" */
    MPIDI_PSP_WIN_INFO_SET_ARG_DEFAULT(win->info_args, info_used, alloc_shared_noncontig, false);
    MPIDI_PSP_WIN_INFO_SET_ARG_SPECIFIC(win->info_args, info_used, alloc_shared_noncontig, true,
                                        win->is_shared_noncontig);
    MPIDI_PSP_WIN_INFO_SET_ARG_SPECIFIC(win->info_args, info_used, alloc_shared_noncontig, false,
                                        !win->is_shared_noncontig);

    /* check for info wait_on_passive_side (PSP/psmpi-specific) */
    MPIDI_PSP_WIN_INFO_SET_ARG_DEFAULT(win->info_args, info_used, wait_on_passive_side, explicit);
    MPIDI_PSP_WIN_INFO_SET_ARG_SPECIFIC(win->info_args, info_used, wait_on_passive_side, explicit,
                                        win->enable_explicit_wait_on_passive_side);
    MPIDI_PSP_WIN_INFO_SET_ARG_SPECIFIC(win->info_args, info_used, wait_on_passive_side, none,
                                        !win->enable_explicit_wait_on_passive_side);

    /* check for "mpi_memory_alloc_kinds" */
    if (win->comm_ptr) {
        char *memory_alloc_kinds;
        MPIR_get_memory_kinds_from_comm(win->comm_ptr, &memory_alloc_kinds);
        mpi_errno = MPIR_Info_set_impl(info_used, "mpi_memory_alloc_kinds", memory_alloc_kinds);
        MPIR_ERR_CHECK(mpi_errno);
    }

    *info_p_p = info_used;

  fn_exit:
    MPIR_FUNC_EXIT;
    return mpi_errno;
  fn_fail:
    if (info_used)
        MPIR_Info_free_impl(info_used);
    goto fn_exit;
}
