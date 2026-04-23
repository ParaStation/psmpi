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
#include <stdlib.h>
#include <assert.h>
#ifndef WITH_COMPRESSOR_PLUGIN
#include "mpitest.h"
#include "mpitestconf.h"
#include <stdio.h>
#endif

#ifdef WITH_COMPRESSOR_PLUGIN
/* In this plugin mode, only the callbacks are compiled and the main
 * function is omitted. This allows to generate a compressor plugin
 * with these callbacks that can then be used by a separate test
 * (see mpix_compressor_plg.c).
 */
#define __WITHOUT_MAIN
#endif /* WITH_COMPRESSOR_PLUGIN */

#ifdef WITH_DERIVED_DTYPE
MPI_Datatype dtype_contig = MPI_DATATYPE_NULL;
#endif

int compressor_req_init(void *buf, int *partitions, MPI_Count * count, MPI_Datatype * dtype,
                        MPI_Info info, void *temp_buf, MPI_Aint * temp_buf_size,
                        void *extra_state, void *extra_req_state)
{
#ifdef WITH_DERIVED_DTYPE
    assert(*dtype == dtype_contig);
#else
    assert(*dtype == MPI_DOUBLE);
#endif

    *(void **) temp_buf = MPI_BUFFER_AUTOMATIC;

    return MPI_SUCCESS;
}

int compressor_deflate(void *buf, int partition, MPI_Count count, MPI_Datatype dtype,
                       void *compr_buf, MPI_Aint * size, void *extra_req_state)
{
    double *_buf = (double *) buf;
    int *_compr_buf = (int *) compr_buf;

    /* compress values by casting from double to int */
    for (int i = 0; i < count; i++) {
        _compr_buf[i] = (int) _buf[i];
    }

    /* return size of compressed partition buffer */
    *size = count * sizeof(int);

    return MPI_SUCCESS;
}

int compressor_inflate(void *buf, int partition, MPI_Count count, MPI_Datatype dtype,
                       void *compr_buf, MPI_Aint * size, void *extra_req_state)
{
    double *_buf = (double *) buf;
    int *_compr_buf = (int *) compr_buf;

    /* decompress values by casting back from int to double */
    for (int i = 0; i < count; i++) {
        _buf[i] = (double) _compr_buf[i];
    }

    /* return size of decompressed partition buffer */
    *size = count * sizeof(double);

    return MPI_SUCCESS;
}

#ifdef WITH_COMPRESSOR_PLUGIN
/* entry function for compressor plugin */
int compressor_register(const char *name, MPI_Info info)
{
    MPIX_Register_compressor(name,
                             compressor_req_init,
                             MPIX_COMPRESSOR_REQ_FREE_FN_NULL,
                             compressor_deflate,
                             compressor_inflate,
                             MPIX_COMPRESSOR_DEREGISTER_FN_NULL, MPI_INFO_NULL, NULL);

    return MPI_SUCCESS;
}
#endif


#ifndef __WITHOUT_MAIN

#define partitions 2
#define total_count 32
double buffer[total_count];
const double const_double_value = 3.14159265;

/* This test program is to check the envisioned functionality of the proposed API extension
 * of the MPI interface by `MPIX_Register_compressor()`.
 *
 * For this, two callbacks are registered for the compression/decompression routines, which
 * convert (i.e., cast with loss) the double values into integer values and back again.
 *
 * The test is passed successfully if the received double data matches the integer data that
 * was sent as compressed double values.
 */

int main(int argc, char *argv[])
{
    int errs = 0;
    MPI_Datatype dtype;
    MPI_Request request;
    MPI_Info info;
    int rank, size;

    MTest_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef WITH_DERIVED_DTYPE
    MPI_Type_contiguous(1, MPI_DOUBLE, &dtype_contig);
    MPI_Type_commit(&dtype_contig);
    dtype = dtype_contig;
#else
    dtype = MPI_DOUBLE;
#endif

    MPIX_Register_compressor("darexa-f",
                             compressor_req_init,
                             MPIX_COMPRESSOR_REQ_FREE_FN_NULL,
                             compressor_deflate, compressor_inflate,
                             MPIX_COMPRESSOR_DEREGISTER_FN_NULL, MPI_INFO_NULL, NULL);

    MPI_Info_create(&info);
    MPI_Info_set(info, "compressor", "darexa-f");

    if (rank == 0) {

        for (int j = 0; j < total_count; j++) {
            buffer[j] = const_double_value;
        }

        MPI_Psend_init(buffer, partitions, total_count / partitions, dtype, 1, 0, MPI_COMM_WORLD,
                       info, &request);

        MPI_Start(&request);

        for (int i = 0; i < partitions; i++) {
            MPI_Pready(i, request);
        }

        MPI_Wait(&request, MPI_STATUS_IGNORE);

    } else if (rank == 1) {

        MPI_Precv_init(buffer, partitions, total_count / partitions, dtype, 0, 0, MPI_COMM_WORLD,
                       info, &request);

        MPI_Start(&request);

        MPI_Wait(&request, MPI_STATUS_IGNORE);

        for (int j = 0; j < total_count; j++) {

            if (buffer[j] != (double) ((int) const_double_value)) {
                fprintf(stderr, "ERROR: Received %f at index %d but expected %f\n", buffer[j], j,
                        (double) ((int) const_double_value));
                errs++;
            }
        }
    }

    MPI_Request_free(&request);

    MPI_Info_free(&info);

#ifdef WITH_DERIVED_DTYPE
    MPI_Type_free(&dtype_contig);
#endif

    MTest_Finalize(errs);
    return MTestReturnValue(errs);
}

#endif /* !__WITHOUT_MAIN */
