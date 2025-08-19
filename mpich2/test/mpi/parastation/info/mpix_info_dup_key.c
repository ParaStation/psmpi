/*
 * ParaStation
 *
 * Copyright (C) 2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpi.h"
#include "mpitest.h"
#include "mpitestconf.h"
#include <stdio.h>
#include <stdlib.h>
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

#define test_assert_key(key, flag)	                                \
    test_assert(flag, "did not find key \"%s\"", key);

#define test_assert_value(key, val, expval, maxlen)			\
    test_assert(!strncmp(val, expval, maxlen), "returned value \"%.*s"	\
		"\" for key \"%s\", which is not the expected \"%s\".",	\
		maxlen, val, key, expval);

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
 * extension of the MPI interface by `MPIX_Info_dup_key()`.
 *
 * For this, the test merges 2 info objects in an array with partially identical
 * info keys into a single one by using `MPIX_Info_merge_from_array()`.
 * It then copies the three key values (x, y, and z, with the latter as an array
 * type) individually and checks that they have been transferred correctly.
 * Finally, it checks whether an `MPI_ERR_INFO_NOKEY` error is thrown if the key
 * to be duplicated does not exist in the source object.
 */

int main(int argc, char *argv[])
{
    int errs = 0;
    MPI_Info info_array[2], info_merge, info_dup;
    int count;
    int count_expected;
    int nkeys;
    int nkeys_expected;
    int count_array_entries;
    int array_entries_expected;
    int flag;
    int buflen;
    char value[MPI_MAX_INFO_VAL];
    char *value_expected;
    char *key;
    int mpi_errno, errclass;

    MTest_Init(&argc, &argv);

    MPI_Info_create(&info_array[0]);
    MPI_Info_create(&info_array[1]);

    MPI_Info_set(info_array[0], (char *) "x", (char *) "x0");
    MPI_Info_set(info_array[0], (char *) "z", (char *) "z0");

    MPI_Info_set(info_array[1], (char *) "y", (char *) "y1");
    MPI_Info_set(info_array[1], (char *) "z", (char *) "z1");

    MPIX_Info_merge_from_array(2, info_array, &info_merge);

    /* Do some validity checks: */

    nkeys_expected = 3; // "x", "y", "z"
    test_handle(MPI_Info_get_nkeys(info_merge, &nkeys));
    test_assert_nkeys(nkeys_expected, nkeys);

    count_array_entries = 0;
    array_entries_expected = 1; // "z"
    test_handle(count_array_entries = count_array_type_entries(info_merge));
    test_assert_count(count_array_entries, array_entries_expected);

    /* Create an empty info object and duplicate the keys into it: */

    MPI_Info_create(&info_dup);

    MPIX_Info_dup_key(info_merge, "x", info_dup);
    nkeys_expected = 1; // "x"
    test_handle(MPI_Info_get_nkeys(info_dup, &nkeys));
    test_assert_nkeys(nkeys_expected, nkeys);

    MPIX_Info_dup_key(info_merge, "y", info_dup);
    nkeys_expected = 2; // "x", "y"
    test_handle(MPI_Info_get_nkeys(info_dup, &nkeys));
    test_assert_nkeys(nkeys_expected, nkeys);

    MPIX_Info_dup_key(info_merge, "z", info_dup);
    nkeys_expected = 3; // "x", "y", "z"
    test_handle(MPI_Info_get_nkeys(info_dup, &nkeys));
    test_assert_nkeys(nkeys_expected, nkeys);
    count_array_entries = 0;
    array_entries_expected = 1; // "z"
    test_handle(count_array_entries = count_array_type_entries(info_merge));
    test_assert_count(count_array_entries, array_entries_expected);

    MPI_Info_free(&info_array[0]);
    MPI_Info_free(&info_array[1]);

    /* Now split the new info object and check for the array entries: */

    MPI_Info_create(&info_array[0]);
    MPI_Info_create(&info_array[1]);

    count = 2;
    count_expected = 2;
    test_handle(MPIX_Info_split_into_array(&count, info_array, info_merge));
    test_assert_count(count, count_expected);

    key = "z";
    value_expected = "z0";
    buflen = MPI_MAX_INFO_VAL;
    test_handle(MPI_Info_get_string(info_array[0], key, &buflen, value, &flag));
    test_assert_key(key, flag);
    if (flag) {
        test_assert_value(key, value, value_expected, buflen);
    }

    key = "z";
    value_expected = "z1";
    buflen = MPI_MAX_INFO_VAL;
    test_handle(MPI_Info_get_string(info_array[1], key, &buflen, value, &flag));
    test_assert_key(key, flag);
    if (flag) {
        test_assert_value(key, value, value_expected, buflen);
    }

    MPI_Info_free(&info_array[0]);
    MPI_Info_free(&info_array[1]);

    /* Finally check for correct error handling with a non-existing key */

    MPI_Comm_set_errhandler(MPI_COMM_SELF, MPI_ERRORS_RETURN);
    test_handle(mpi_errno = MPIX_Info_dup_key(info_merge, "dummy", info_dup));
    MPI_Error_class(mpi_errno, &errclass);
    test_assert((mpi_errno != MPI_SUCCESS), "returned MPI_SUCCES (%d) for non existing info key",
                mpi_errno);
    test_assert((errclass == MPI_ERR_INFO_NOKEY),
                "returned wrong error class (%d) while MPI_ERR_INFO_NOKEY was expected", errclass);

    /* Release all remaining info objects */

    MPI_Info_free(&info_merge);
    MPI_Info_free(&info_dup);

    MTest_Finalize(errs);
    return MTestReturnValue(errs);
}
