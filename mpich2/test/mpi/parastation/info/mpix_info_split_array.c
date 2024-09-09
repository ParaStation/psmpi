/*
 * ParaStation
 *
 * Copyright (C) 2024 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpitest.h"
#include "mpitestconf.h"
#ifdef HAVE_STRING_H
#include <string.h>
#endif

const char *test_function = NULL;

#define test_handle(function)   \
    test_function = #function ; \
    do {                        \
        function;               \
    } while (0);

#define test_assert(expression, string, ...)                            \
    if (!(expression)) {                                                \
        fprintf(stderr, "ERROR: %s " string " (%s:%d)\n",               \
                test_function,                                          \
                __VA_ARGS__, __FILE__, __LINE__);                       \
        errs++;                                                         \
    }

#define test_assert_nkeys(nkeys1, nkeys2)                               \
    test_assert((nkeys1 == nkeys2), "returned unexpected number of"     \
                " keys in info: " #nkeys1 " = %d vs. " #nkeys2 " = %d", \
                nkeys1, nkeys2);

#define test_assert_count(count1, count2)                               \
    test_assert((count1 == count2), "returned unexpected number of"     \
                " array entries: " #count1 " = %d vs. " #count2 " ="    \
                " %d", count1, count2);

#define test_assert_none_values_count(count1, count2, idx)              \
    test_assert((count1 == count2), "(i=%d) returned unexpected number" \
                " of \"none\" entries: " #count1 " = %d vs. " #count2   \
                " = %d", idx, count1, count2);

#define test_assert_no_array_type_entries(count, idx)                   \
    test_assert((!count), "(i=%d) has found \"mpix_info_array_type\""   \
                " (%d times) but shouldn't have", idx, count)


static int count_key_entries_with_same_values(MPI_Info info, const char *chkval)
{
    int count = 0;
    int nkeys = 0;
    MPI_Info_get_nkeys(info, &nkeys);
    for (int i = 0; i < nkeys; i++) {
        int flag;
        int buflen = MPI_MAX_INFO_VAL;
        char key[MPI_MAX_INFO_KEY];
        char value[MPI_MAX_INFO_VAL];
        MPI_Info_get_nthkey(info, i, key);
        MPI_Info_get_string(info, key, &buflen, value, &flag);
        if (flag && (strcmp(value, chkval) == 0)) {
            count++;
        }
    }
    return count;
}

static int count_array_type_entries(MPI_Info info)
{
    return count_key_entries_with_same_values(info, "mpix_info_array_type");
}

static int count_none_type_entries(MPI_Info info)
{
    return count_key_entries_with_same_values(info, "none");
}


/* This test program checks the envisioned functionality of the proposed API
 * extension of the MPI interface by `MPIX_Info_split_into_array()`.
 *
 * For this, the test splits an initially merged info object back into an array
 * of info objects by using `MPIX_Info_split_into_array()` and checks for the
 * correct behavior if different sizes for this array are used.
 *
 * In addition, the test also checks if cases where more info objects are given
 * in the array than actually needed are still handled correctly. Finally, it
 * also checks what happens if no objects are given in the array.
 */

int main(int argc, char *argv[])
{
    int errs = 0;
    MPI_Info info_array[4 + 2], info_merge;
    int count;
    int nkeys;
    int count_expected;
    int nkeys_expected;
    int none_values_found;
    int none_values_expected[4];
    int count_array_entries;

    MTest_Init(&argc, &argv);

    MPI_Info_create(&info_array[0]);
    MPI_Info_create(&info_array[1]);
    MPI_Info_create(&info_array[2]);
    MPI_Info_create(&info_array[3]);

    MPI_Info_set(info_array[0], (char *) "x", (char *) "x0");
    MPI_Info_set(info_array[0], (char *) "y", (char *) "y0");

    MPI_Info_set(info_array[1], (char *) "b", (char *) "b1");
    MPI_Info_set(info_array[1], (char *) "y", (char *) "y1");

    MPI_Info_set(info_array[2], (char *) "z", (char *) "z2");
    MPI_Info_set(info_array[2], (char *) "a", (char *) "a2");
    MPI_Info_set(info_array[2], (char *) "x", (char *) "x2");

    MPI_Info_set(info_array[3], (char *) "z", (char *) "z3");
    MPI_Info_set(info_array[3], (char *) "x", (char *) "x3");
    MPI_Info_set(info_array[3], (char *) "y", (char *) "y3");

    MPIX_Info_merge_from_array(4, info_array, &info_merge);

    MPI_Info_free(&info_array[0]);
    MPI_Info_free(&info_array[1]);
    MPI_Info_free(&info_array[2]);
    MPI_Info_free(&info_array[3]);

    /* Create clean set of array entries and split info_merge back into the array: */

    MPI_Info_create(&info_array[0]);
    MPI_Info_create(&info_array[1]);
    MPI_Info_create(&info_array[2]);
    MPI_Info_create(&info_array[3]);

    count = 4;
    count_expected = 4;
    test_handle(MPIX_Info_split_into_array(&count, info_array, info_merge));

    /* Do some validity checks: */

    test_assert_count(count, count_expected);

    nkeys_expected = 5; // "x", "y", "z", "a", "b"
    none_values_expected[0] = 1;        // "z"
    none_values_expected[1] = 2;        // "x", "y"
    none_values_expected[2] = 1;        // "y", "z"
    none_values_expected[3] = 0;        // --

    for (int i = 0; i < 4; i++) {

        test_handle(MPI_Info_get_nkeys(info_array[i], &nkeys));
        test_assert_nkeys(nkeys, nkeys_expected);

        test_handle(none_values_found = count_none_type_entries(info_array[i]));
        test_assert_none_values_count(none_values_found, none_values_expected[i], i);

        test_handle(count_array_entries = count_array_type_entries(info_array[i]));
        test_assert_no_array_type_entries(count_array_entries, i);
    }

    MPI_Info_free(&info_array[0]);
    MPI_Info_free(&info_array[1]);
    MPI_Info_free(&info_array[2]);
    MPI_Info_free(&info_array[3]);

    /* Now let's do it again, but with an increasing number of array items in a loop: */

    for (int k = 0; k < 4; k++) {

        MPI_Info_create(&info_array[k]);

        count = k + 1;
        count_expected = 4;
        test_handle(MPIX_Info_split_into_array(&count, info_array, info_merge));

        /* Do again some validity checks: */

        test_assert_count(count, count_expected);

        nkeys_expected = 5;     // "x", "y", "z", "a", "b"
        none_values_expected[0] = 1;    // "z"
        none_values_expected[1] = 2;    // "x", "y"
        none_values_expected[2] = 1;    // "y", "z"
        none_values_expected[3] = 0;    // --

        for (int i = 0; i < k + 1; i++) {

            test_handle(MPI_Info_get_nkeys(info_array[i], &nkeys));
            test_assert_nkeys(nkeys, nkeys_expected);

            test_handle(none_values_found = count_none_type_entries(info_array[i]));
            test_assert_none_values_count(none_values_found, none_values_expected[i], i);

            test_handle(count_array_entries = count_array_type_entries(info_array[i]));
            test_assert_no_array_type_entries(count_array_entries, i);
        }
    }

    MPI_Info_free(&info_array[0]);
    MPI_Info_free(&info_array[1]);
    MPI_Info_free(&info_array[2]);
    MPI_Info_free(&info_array[3]);

    /* Now also check what happens if there are more info objects in the array than actually needed: */

    MPI_Info_create(&info_array[0]);
    MPI_Info_create(&info_array[1]);
    MPI_Info_create(&info_array[2]);
    MPI_Info_create(&info_array[3]);
    MPI_Info_create(&info_array[4]);
    MPI_Info_create(&info_array[5]);

    count = 6;
    count_expected = 4;
    test_handle(MPIX_Info_split_into_array(&count, info_array, info_merge));

    /* Do some validity checks: */

    test_assert_count(count, count_expected);

    nkeys_expected = 5; // "x", "y", "z", "a", "b"
    for (int i = 0; i < 4; i++) {

        test_handle(MPI_Info_get_nkeys(info_array[i], &nkeys));
        test_assert_nkeys(nkeys, nkeys_expected);
    }

    nkeys_expected = 2; // "a", "b"
    for (int i = 4; i < 6; i++) {

        test_handle(MPI_Info_get_nkeys(info_array[i], &nkeys));
        test_assert_nkeys(nkeys, nkeys_expected);
    }

    MPI_Info_free(&info_array[0]);
    MPI_Info_free(&info_array[1]);
    MPI_Info_free(&info_array[2]);
    MPI_Info_free(&info_array[3]);
    MPI_Info_free(&info_array[4]);
    MPI_Info_free(&info_array[5]);

    /* Finally, also test what happens if there is no info object given: */

    count = 0;
    count_expected = 4;
    test_handle(MPIX_Info_split_into_array(&count, NULL, info_merge));
    test_assert_count(count, count_expected);

    MPI_Info_free(&info_merge);

    MTest_Finalize(errs);
    return MTestReturnValue(errs);
}
