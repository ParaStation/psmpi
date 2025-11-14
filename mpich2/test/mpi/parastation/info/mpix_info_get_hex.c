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

#define test_assert_key(key, flag)			                \
    test_assert(flag, "did not find key \"%s\"", key);

#define test_assert_value(key, val, expval, maxlen)			\
    test_assert(!strncmp(val, expval, maxlen), "returned value \"%.*s"	\
		"\" for key \"%s\", which is not the expected \"%s\".",	\
		maxlen, val, key, expval);

#define PATTERN "abcdefghijklmnopqrstuvwxyz"

/*
 * This test checks the correct behavior of `MPIX_Info_get_hex()` as the
 * counterpart to `MPIX_Info_set_hex()`. It does this by using a string as
 * a memory object which is first encoded using `MPIX_Info_set_hex()` and
 * then retrieved again and decoded via `MPIX_Info_get_hex()`.
 * The test is passed successfully if the reconstructed string matches
 * the original one.
 */

int main(int argc, char *argv[])
{
    int errs = 0;
    int flag = 0;
    MPI_Info info = MPI_INFO_NULL;
    char *key = "buffer";
    char buffer[] = PATTERN;

    MTest_Init(&argc, &argv);

    MPI_Info_create(&info);

    MPIX_Info_set_hex(info, key, buffer, sizeof(buffer));

    memset(&buffer, 0, sizeof(buffer));

    test_handle(MPIX_Info_get_hex(info, key, buffer, sizeof(buffer), &flag));
    test_assert_key(key, flag);
    if (flag) {
        test_assert_value(key, buffer, PATTERN, (int) sizeof(buffer));
    }

    MPI_Info_free(&info);

    MTest_Finalize(errs);
    return MTestReturnValue(errs);
}
