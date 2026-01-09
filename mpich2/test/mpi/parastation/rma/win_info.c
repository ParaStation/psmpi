/*
 * ParaStation
 *
 * Copyright (C) 2025-2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include "mpitest.h"

/*
  MPI-4.1 says: "An MPI implementation is required to return all hints
  that are supported by the implementation and have default values
  specified; any user-supplied hints that were not ignored by the
  implementation; and any additional hints that were set by the
  implementation. If no such hints exist, a handle to a newly created
  info object is returned that contains no key/value pair.”

  This test checks the desired behavior of MPI_Win_get/set_info() with
  regard to setting (and also not setting) info keys for RMA windows.

  It does so in the following 12 different sub-tests:
  - Test #1: check that default values are set after MPI_Win_allocate().
  - Test #2: check setting and getting an invalid key (MPI_Win_allocate).
  - Test #3: check setting and getting an invalid key (MPI_Win_set_info).
  - Test #4: check setting an getting "accumulate_ordering" (MPI_Win_allocate).
  - Test #5: check setting an getting "accumulate_ordering" (MPI_Win_set_info).
  - Test #6: check setting an getting "wait_on_passive_side" (MPI_Win_allocate).
  - Test #7: check setting an getting "wait_on_passive_side" (MPI_Win_set_info).
  - Test #8: check setting and getting unsupported values (MPI_Win_allocate).
  - Test #9: check setting and getting unsupported values (MPI_Win_set_info).
  - Test #10: check setting and getting invalid values (MPI_Win_allocate).
  - Test #11: check setting and getting invalid values (MPI_Win_set_info).
  - Test #12: check setting and getting "alloc_shared_noncontig" (MPI_Win_allocate_shared).
*/

int world_rank;
MPI_Comm comm;

int testnum = 0;
#define TEST(_x, _y)                                                   \
    do {                                                               \
        testnum = _x;                                                  \
        MTestPrintfMsg(1, "(%d) Test #%d: %s\n", world_rank, _x, _y);  \
    } while (0);

#define ERRMSG(_x, ...)                                                \
    do {                                                               \
        printf("(%d) Test #%d: " _x " (ERROR)\n", world_rank, testnum, \
               __VA_ARGS__);                                           \
    } while (0);

#define OKMSG(_x, ...)                                                 \
    do {                                                               \
        MTestPrintfMsg(1, "(%d) Test #%d: " _x " (OK)\n", world_rank,  \
                       testnum, __VA_ARGS__);                          \
    } while (0);

static int check_win_info_get(MPI_Win win, const char *key, const char *exp_val)
{
    int flag = 0;
    MPI_Info info = MPI_INFO_NULL;
    char val[MPI_MAX_INFO_VAL];
    int errors = 0;

    MPI_Win_get_info(win, &info);
    MPI_Info_get(info, key, MPI_MAX_INFO_VAL, val, &flag);
    if (flag && exp_val && strncmp(val, exp_val, strlen(exp_val)) != 0) {
        ERRMSG("%s expected \"%s\" but got \"%s\".", key, exp_val, val);
        errors++;
    } else if (!flag && exp_val) {
        ERRMSG("%s not defined but should.", key);
        errors++;
    } else if (flag && !exp_val) {
        ERRMSG("%s defined (\"%s\") but should not.", key, val);
    } else {
        if (flag) {
            OKMSG("%s set to \"%s\".", key, val);
        } else {
            OKMSG("%s not defined.", key);
        }
    }

    MPI_Info_free(&info);

    return errors;
}

static void info_set(MPI_Info * info, int num_keys, ...)
{
    va_list args;
    va_start(args, num_keys);

    if (*info != MPI_INFO_NULL)
        MPI_Info_free(info);
    MPI_Info_create(info);

    for (int i = 0; i < num_keys; i++) {
        char *key = va_arg(args, char *);
        char *val = va_arg(args, char *);
        MPI_Info_set(*info, key, val);
    }

    va_end(args);
}

static void win_info_set(MPI_Win win, int num_keys, ...)
{
    va_list args;
    va_start(args, num_keys);

    MPI_Info info = MPI_INFO_NULL;
    MPI_Info_create(&info);

    for (int i = 0; i < num_keys; i++) {
        char *key = va_arg(args, char *);
        char *val = va_arg(args, char *);
        MPI_Info_set(info, key, val);
    }

    MPI_Win_set_info(win, info);
    MPI_Info_free(&info);
}

static void win_allocate(MPI_Info info, MPI_Win * win)
{
    void *base;
    if (*win != MPI_WIN_NULL)
        MPI_Win_free(win);
    MPI_Win_allocate(sizeof(int), sizeof(int), info, comm, &base, win);
}

static void win_allocate_shared(MPI_Info info, MPI_Win * win)
{
    void *base;
    if (*win != MPI_WIN_NULL)
        MPI_Win_free(win);
    MPI_Win_allocate_shared(sizeof(int), sizeof(int), info, comm, &base, win);
}

static int check_envar_is_set(const char *name, int value)
{
    char *str;
    int strval;
    char *endptr;

    str = getenv(name);
    if (str == NULL)
        return 0;

    errno = 0;
    strval = strtol(str, &endptr, 10);
    if (errno != 0 || *endptr != '\0')
        return 0;

    if (value != strval)
        return 0;

    return 1;
}


int main(int argc, char **argv)
{
    int errors = 0;
    MPI_Info info = MPI_INFO_NULL;
    MPI_Win win = MPI_WIN_NULL;;

    MTest_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm);

    /* Test #1: check that default values are set after MPI_Win_allocate(). */
    TEST(1, "check that default values are set after MPI_Win_allocate().");

    win_allocate(info, &win);

    MPI_Win_get_info(win, &info);

    /* standardized keys */
    errors += check_win_info_get(win, "no_locks", "false");
    errors += check_win_info_get(win, "accumulate_ops", "same_op_no_op");
    errors += check_win_info_get(win, "mpi_accumulate_granularity", "0");
    errors += check_win_info_get(win, "same_size", "false");
    errors += check_win_info_get(win, "same_disp_unit", "false");
    errors += check_win_info_get(win, "alloc_shared_noncontig", "false");
    if (check_envar_is_set("PSP_ACCUMULATE_ORDERING", 0)) {
        /* disabled explicitly */
        errors += check_win_info_get(win, "accumulate_ordering", "none");
    } else {
        /* default */
        errors += check_win_info_get(win, "accumulate_ordering", "rar,raw,war,waw");
    }

    /* PSP/psmpi-specific: */
    if (check_envar_is_set("PSP_RMA_EXPLICIT_WAIT", 0)) {
        /* disabled explicitly */
        errors += check_win_info_get(win, "wait_on_passive_side", "none");
    } else {
        /* default */
        errors += check_win_info_get(win, "wait_on_passive_side", "explicit");
    }


    /* Test #2: check setting and getting an invalid key (MPI_Win_allocate). */
    TEST(2, "check setting and getting an invalid key (MPI_Win_allocate).");

    info_set(&info, 1, "invalid_key", "true");

    win_allocate(info, &win);

    errors += check_win_info_get(win, "invalid_key", NULL);


    /* Test #3: check setting and getting an invalid key (MPI_Win_set_info). */
    TEST(3, "check setting and getting an invalid key (MPI_Win_set_info).");

    win_allocate(MPI_INFO_NULL, &win);

    win_info_set(win, 1, "invalid_key", "true");

    errors += check_win_info_get(win, "invalid_key", NULL);


    /* Test #4: check setting an getting "accumulate_ordering" (MPI_Win_allocate). */
    TEST(4, "check setting an getting \"accumulate_ordering\" (MPI_Win_allocate).");

    info_set(&info, 1, "accumulate_ordering", "none");

    win_allocate(info, &win);

    errors += check_win_info_get(win, "accumulate_ordering", "none");


    /* Test #5: check setting an getting "accumulate_ordering" (MPI_Win_set_info). */
    TEST(5, "check setting an getting \"accumulate_ordering\" (MPI_Win_set_info).");

    win_allocate(MPI_INFO_NULL, &win);

    win_info_set(win, 1, "accumulate_ordering", "none");

    errors += check_win_info_get(win, "accumulate_ordering", "none");


    /* Test #6: check setting an getting "wait_on_passive_side" (MPI_Win_allocate.) */
    TEST(6, "check setting an getting \"wait_on_passive_side\" (MPI_Win_allocate).");

    info_set(&info, 1, "wait_on_passive_side", "none");

    win_allocate(info, &win);

    errors += check_win_info_get(win, "wait_on_passive_side", "none");


    /* Test #7: check setting an getting "wait_on_passive_side" (MPI_Win_set_info). */
    TEST(7, "check setting an getting \"wait_on_passive_side\" (MPI_Win_set_info).");

    win_allocate(MPI_INFO_NULL, &win);

    win_info_set(win, 1, "wait_on_passive_side", "none");

    errors += check_win_info_get(win, "wait_on_passive_side", "none");


    /* Test #8: check setting and getting unsupported values (MPI_Win_allocate). */
    TEST(8, "check setting and getting unsupported values (MPI_Win_allocate).");

    /* Note that these values are specifically unsupported for PSP/psmpi! */
    info_set(&info, 6, "no_locks", "true",
             "accumulate_ops", "same_op",
             "accumulate_ordering", "raw,waw",
             "mpi_accumulate_granularity", "4", "same_size", "true", "same_disp_unit", "true");

    win_allocate(info, &win);

    /* check that they all are not set */
    errors += check_win_info_get(win, "no_locks", NULL);
    errors += check_win_info_get(win, "accumulate_ops", NULL);
    errors += check_win_info_get(win, "accumulate_ordering", NULL);
    errors += check_win_info_get(win, "mpi_accumulate_granularity", NULL);
    errors += check_win_info_get(win, "same_size", NULL);
    errors += check_win_info_get(win, "same_disp_unit", NULL);


    /* Test #9: check setting and getting unsupported values (MPI_Win_set_info). */
    TEST(9, "check setting and getting unsupported values (MPI_Win_set_info).");

    win_allocate(MPI_INFO_NULL, &win);

    /* Note that these values are specifically unsupported for PSP/psmpi! */
    win_info_set(win, 6, "no_locks", "true",
                 "accumulate_ops", "same_op",
                 "accumulate_ordering", "raw,waw",
                 "mpi_accumulate_granularity", "4", "same_size", "true", "same_disp_unit", "true");

    /* check that they all are not set */
    errors += check_win_info_get(win, "no_locks", NULL);
    errors += check_win_info_get(win, "accumulate_ops", NULL);
    errors += check_win_info_get(win, "mpi_accumulate_granularity", NULL);
    errors += check_win_info_get(win, "same_size", NULL);
    errors += check_win_info_get(win, "same_disp_unit", NULL);


    /* Test #10: check setting and getting invalid values (MPI_Win_allocate). */
    TEST(10, "check setting and getting invalid values (MPI_Win_allocate).");

    /* Note that these "foo" values are assumed to be unsupported for all MPI libraries... */
    info_set(&info, 8, "no_locks", "foo",
             "accumulate_ops", "foo",
             "accumulate_ordering", "foo",
             "mpi_accumulate_granularity", "foo", "same_size", "foo", "same_disp_unit", "foo",
             "alloc_shared_noncontig", "foo", "wait_on_passive_side", "foo");

    win_allocate(info, &win);

    /* check that they all are not set */
    errors += check_win_info_get(win, "no_locks", NULL);
    errors += check_win_info_get(win, "accumulate_ops", NULL);
    errors += check_win_info_get(win, "accumulate_ordering", NULL);
    errors += check_win_info_get(win, "mpi_accumulate_granularity", NULL);
    errors += check_win_info_get(win, "same_size", NULL);
    errors += check_win_info_get(win, "same_disp_unit", NULL);
    errors += check_win_info_get(win, "alloc_shared_noncontig", NULL);
    errors += check_win_info_get(win, "wait_on_passive_side", NULL);


    /* Test #11: check setting and getting invalid values (MPI_Win_set_info). */
    TEST(11, "check setting and getting invalid values (MPI_Win_set_info).");

    win_allocate(MPI_INFO_NULL, &win);

    /* Note that these "foo" values are assumed to be unsupported for all MPI libraries... */
    win_info_set(win, 8, "no_locks", "foo",
                 "accumulate_ops", "foo",
                 "accumulate_ordering", "foo",
                 "mpi_accumulate_granularity", "foo", "same_size", "foo", "same_disp_unit", "foo",
                 "alloc_shared_noncontig", "foo", "wait_on_passive_side", "foo");

    /* check that they all are not set */
    errors += check_win_info_get(win, "no_locks", NULL);
    errors += check_win_info_get(win, "accumulate_ops", NULL);
    errors += check_win_info_get(win, "mpi_accumulate_granularity", NULL);
    errors += check_win_info_get(win, "same_size", NULL);
    errors += check_win_info_get(win, "same_disp_unit", NULL);
    errors += check_win_info_get(win, "alloc_shared_noncontig", NULL);
    errors += check_win_info_get(win, "wait_on_passive_side", NULL);


    /* Test #12: check setting and getting "alloc_shared_noncontig" (MPI_Win_allocate_shared). */
    TEST(12, "check setting and getting \"alloc_shared_noncontig\" (MPI_Win_allocate_shared).");

    /* request for a non-contiguous window (but note that it depends on the MPI lib if
     * this is actuallay supported) */
    info_set(&info, 1, "alloc_shared_noncontig", "true");

    win_allocate_shared(info, &win);

    /* check that it is a non-contiguous window */
    errors += check_win_info_get(win, "alloc_shared_noncontig", "true");

    /* try to change this to a contiguous one */
    info_set(&info, 1, "alloc_shared_noncontig", "false");

    /* check that changing cannot (and did not) work */
    errors += check_win_info_get(win, "alloc_shared_noncontig", "true");


    MPI_Info_free(&info);

    MPI_Win_free(&win);

    MPI_Comm_free(&comm);

    MTest_Finalize(errors);

    return MTestReturnValue(errors);
}
