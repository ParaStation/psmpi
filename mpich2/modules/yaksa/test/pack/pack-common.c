/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>
#include "yaksa_config.h"
#include "yaksa.h"
#include "dtpools.h"
#include "pack-common.h"

void pack_init_devices(void)
{
#ifdef HAVE_CUDA
    pack_cuda_init_devices();
#elif defined(HAVE_ZE)
    pack_ze_init_devices();
#endif
}

void pack_finalize_devices()
{
#ifdef HAVE_CUDA
    pack_cuda_finalize_devices();
#elif defined(HAVE_ZE)
    pack_ze_finalize_devices();
#endif
}

void pack_alloc_mem(size_t size, mem_type_e type, void **hostbuf, void **devicebuf)
{
    if (type == MEM_TYPE__UNREGISTERED_HOST) {
        *devicebuf = malloc(size);
        if (hostbuf)
            *hostbuf = *devicebuf;
    } else {
#ifdef HAVE_CUDA
        pack_cuda_alloc_mem(size, type, hostbuf, devicebuf);
#elif defined(HAVE_ZE)
        pack_ze_alloc_mem(size, type, hostbuf, devicebuf);
#else
        fprintf(stderr, "ERROR: no GPU device is supported\n");
        exit(1);
#endif
    }
}

void pack_free_mem(mem_type_e type, void *hostbuf, void *devicebuf)
{
    if (type == MEM_TYPE__UNREGISTERED_HOST) {
        free(hostbuf);
    } else {
#ifdef HAVE_CUDA
        pack_cuda_free_mem(type, hostbuf, devicebuf);
#elif defined(HAVE_ZE)
        pack_ze_free_mem(type, hostbuf, devicebuf);
#else
        fprintf(stderr, "ERROR: no GPU device is supported\n");
        exit(1);
#endif
    }
}

void pack_get_ptr_attr(const void *inbuf, void *outbuf, yaksa_info_t * info)
{
#ifdef HAVE_CUDA
    pack_cuda_get_ptr_attr(inbuf, outbuf, info);
#elif defined(HAVE_ZE)
    pack_ze_get_ptr_attr(inbuf, outbuf, info);
#else
    *info = NULL;
#endif
}

void pack_copy_content(const void *sbuf, void *dbuf, size_t size, mem_type_e type)
{
#ifdef HAVE_CUDA
    pack_cuda_copy_content(sbuf, dbuf, size, type);
#elif defined(HAVE_ZE)
    pack_ze_copy_content(sbuf, dbuf, size, type);
#endif
}
