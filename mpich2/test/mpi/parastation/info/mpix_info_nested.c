/*
 * ParaStation
 *
 * Copyright (C) 2025-2026 ParTec AG, Munich
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

/*
 * This test provides a simple example that shows how info objects can
 * also be nested using the hex set/get functions.
 * However, a thing that needs to be taken into account is that only the
 * object _handle_ is stored as the hex value, so when accessing it, it
 * must be ensured that the linked info object still exists.
 */

const char *key1 = "info_key";
const char *value = "info_value";
const char *key2 = "info_object_key";

int func(MPI_Info info)
{
    int errs = 0;
    int flag = 0;
    int buflen = MPI_MAX_INFO_VAL;
    char buffer[MPI_MAX_INFO_VAL];
    MPI_Info info_nested = MPI_INFO_NULL;

    MPIX_Info_get_hex(info, key1, &info_nested, sizeof(info_nested), &flag);
    if (!flag) {
        fprintf(stderr,
                "ERROR: MPIX_Info_get_hex() did not return the info handle for key \"%s\".\n",
                key1);
        errs++;
    }

    MPI_Info_get_string(info_nested, key2, &buflen, buffer, &flag);
    if (!flag) {
        fprintf(stderr,
                "ERROR: MPI_Info_get_string() did not return a value for the key \"%s\".\n", key2);
        errs++;
    } else if (strncmp(value, buffer, MPI_MAX_INFO_VAL)) {
        fprintf(stderr,
                "ERROR: MPI_Info_get_string() returned \"%s\" instead of the expected value \"%s\" for the key \"%s\".\n",
                buffer, value, key2);
        errs++;
    }

    return errs;
}

int main(int argc, char *argv[])
{
    int errs = 0;
    MPI_Info info1 = MPI_INFO_NULL;
    MPI_Info info2 = MPI_INFO_NULL;

    MTest_Init(&argc, &argv);

    MPI_Info_create(&info1);
    MPI_Info_create(&info2);

    MPIX_Info_set_hex(info1, key1, &info2, sizeof(info2));
    MPI_Info_set(info2, key2, value);

    errs += func(info1);

    MPI_Info_free(&info1);
    MPI_Info_free(&info2);

    MTest_Finalize(errs);
    return MTestReturnValue(errs);
}
