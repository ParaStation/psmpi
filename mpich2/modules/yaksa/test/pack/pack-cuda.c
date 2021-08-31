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

#ifdef HAVE_CUDA

#include <cuda_runtime_api.h>

static int ndevices = -1;

void pack_cuda_init_devices(void)
{
    cudaGetDeviceCount(&ndevices);
    assert(ndevices != -1);
    cudaSetDevice(device_id);
}

void pack_cuda_finalize_devices()
{
}

void pack_cuda_alloc_mem(size_t size, mem_type_e type, void **hostbuf, void **devicebuf)
{
    if (type == MEM_TYPE__REGISTERED_HOST) {
        cudaMallocHost(devicebuf, size);
        if (hostbuf)
            *hostbuf = *devicebuf;
    } else if (type == MEM_TYPE__MANAGED) {
        cudaMallocManaged(devicebuf, size, cudaMemAttachGlobal);
        if (hostbuf)
            *hostbuf = *devicebuf;
    } else if (type == MEM_TYPE__DEVICE) {
        cudaSetDevice(device_id);
        cudaMalloc(devicebuf, size);
        if (hostbuf)
            cudaMallocHost(hostbuf, size);
        device_id += device_stride;
        device_id %= ndevices;
    } else {
        fprintf(stderr, "ERROR: unsupported memory type\n");
        exit(1);
    }
}

void pack_cuda_free_mem(mem_type_e type, void *hostbuf, void *devicebuf)
{
    if (type == MEM_TYPE__REGISTERED_HOST) {
        cudaFreeHost(devicebuf);
    } else if (type == MEM_TYPE__MANAGED) {
        cudaFree(devicebuf);
    } else if (type == MEM_TYPE__DEVICE) {
        cudaFree(devicebuf);
        if (hostbuf) {
            cudaFreeHost(hostbuf);
        }
    }
}

void pack_cuda_get_ptr_attr(const void *inbuf, void *outbuf, yaksa_info_t * info)
{
    static int count = 0;
    if ((++count) % 2 == 0) {
        int rc;

        rc = yaksa_info_create(info);
        assert(rc == YAKSA_SUCCESS);

        rc = yaksa_info_keyval_append(*info, "yaksa_gpu_driver", "cuda", strlen("cuda"));
        assert(rc == YAKSA_SUCCESS);

        struct cudaPointerAttributes attr;

        cudaPointerGetAttributes(&attr, inbuf);
        rc = yaksa_info_keyval_append(*info, "yaksa_cuda_inbuf_ptr_attr", &attr, sizeof(attr));
        assert(rc == YAKSA_SUCCESS);

        cudaPointerGetAttributes(&attr, outbuf);
        rc = yaksa_info_keyval_append(*info, "yaksa_cuda_outbuf_ptr_attr", &attr, sizeof(attr));
        assert(rc == YAKSA_SUCCESS);
    } else
        *info = NULL;
}

void pack_cuda_copy_content(const void *sbuf, void *dbuf, size_t size, mem_type_e type)
{
    if (type == MEM_TYPE__DEVICE) {
        cudaMemcpy(dbuf, sbuf, size, cudaMemcpyDefault);
    }
}

#endif /* HAVE_CUDA */
