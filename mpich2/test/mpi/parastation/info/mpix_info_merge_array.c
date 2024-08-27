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

const char *test_function;

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
                nkeys1, nkeys2);                                        \

#define test_assert_count(count1, count2)                               \
    test_assert((count1 == count2), "called in a loop returned"         \
                " unexpected number of array entries: " #count1 " = %d" \
                " vs. " #count2 " = %d", count1, count2);


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


/* This test program checks the envisioned functionality of the proposed API
 * extension of the MPI interface by `MPIX_Info_merge_from_array()`.
 *
 * For this, the test merges 4 info objects in an array with partially identical
 * info keys into a single one by using `MPIX_Info_merge_from_array()`.
 * In doing so, only the multiple keys should be converted _internally_
 * into value arrays and their value is therefore replaced by the string
 * `mpix_info_type_array`. The test checks whether the number of normal
 * entries and the number of array entries correspond to the expected
 * values.
 *
 * In addition, the test also checks whether duplicating an info object
 * with array values already attached works correctly and checks what
 * happens if two of such info objects get merged again. Finally, it also
 * checks what happens if no objects are given in the array.
 */

int main(int argc, char *argv[])
{
    int errs = 0;
    MPI_Info info_array[4], info_merge;
    int nkeys;
    int nkeys_expected;
    int count_array_entries;
    int array_entries_expected;

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

    /* Do some validity checks: */

    nkeys_expected = 5; // "x", "y", "z", "a", "b"
    test_handle(MPI_Info_get_nkeys(info_merge, &nkeys));
    test_assert_nkeys(nkeys_expected, nkeys);

    count_array_entries = 0;
    array_entries_expected = 3; // "x", "y", "z"
    test_handle(count_array_entries = count_array_type_entries(info_merge));
    test_assert_count(count_array_entries, array_entries_expected);

    /* Duplicate an info object with array, overwrite one array entry and check the keys: */

    MPI_Info info_merge_array[2];
    MPI_Info info_merge_dup, info_merge_twice;

    MPI_Info_dup(info_merge, &info_merge_dup);
    MPI_Info_get_nkeys(info_merge, &nkeys);

    nkeys_expected = 5; // "x", "y", "z", "a", "b"
    test_handle(MPI_Info_get_nkeys(info_merge_dup, &nkeys));
    test_assert_nkeys(nkeys_expected, nkeys);

    MPI_Info_set(info_merge_dup, "z", "overwrite");     // This should remove the array property for "z".

    count_array_entries = 0;
    array_entries_expected = 2; // "x", "y"
    test_handle(count_array_entries = count_array_type_entries(info_merge_dup));
    test_assert_count(count_array_entries, array_entries_expected);

    /* Now merge two info objects with already attached arrays and check what happens: */

    info_merge_array[0] = info_merge;
    info_merge_array[1] = info_merge_dup;
    MPIX_Info_merge_from_array(2, info_merge_array, &info_merge_twice);
    MPI_Info_get_nkeys(info_merge_twice, &nkeys);

    nkeys_expected = 3; // "a", "b", "z" (the array entries for "x", "y" should not be further considered when merging again)
    test_handle(MPI_Info_get_nkeys(info_merge_twice, &nkeys));
    test_assert_nkeys(nkeys, nkeys_expected);

    count_array_entries = 0;
    array_entries_expected = 2; // "a", "b" (as both have been duplicated before, they've now become array entries)
    test_handle(count_array_entries = count_array_type_entries(info_merge_twice));
    test_assert_count(count_array_entries, array_entries_expected);

    MPI_Info_free(&info_array[0]);
    MPI_Info_free(&info_array[1]);
    MPI_Info_free(&info_array[2]);
    MPI_Info_free(&info_array[3]);

    MPI_Info_free(&info_merge);
    MPI_Info_free(&info_merge_twice);
    MPI_Info_free(&info_merge_dup);

    /* Finally, also test what happens if there is no info object given: */

    MPI_Info info_merge_empty;
    MPIX_Info_merge_from_array(0, info_merge_array, &info_merge_empty);

    nkeys_expected = 0; // empty
    test_handle(MPI_Info_get_nkeys(info_merge_empty, &nkeys));
    test_assert_nkeys(nkeys, nkeys_expected);

    MPI_Info_free(&info_merge_empty);

    MTest_Finalize(errs);
    return MTestReturnValue(errs);
}
