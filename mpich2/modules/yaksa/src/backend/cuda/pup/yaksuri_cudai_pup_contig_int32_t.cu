/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * DO NOT EDIT: AUTOMATICALLY GENERATED FILE !!
 */

#include <string.h>
#include <stdint.h>
#include <wchar.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "yaksuri_cudai_base.h"
#include "yaksuri_cudai_pup.h"

__global__ void yaksuri_cudai_kernel_pack_contig_int32_t(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
{
    const char *__restrict__ sbuf = (const char *) inbuf;
    char *__restrict__ dbuf = (char *) outbuf;
    uintptr_t extent = md->extent;
    uintptr_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uintptr_t res = idx;
    uintptr_t inner_elements = md->num_elements;
    
    if (idx >= (count * inner_elements))
        return;
    
    uintptr_t x0 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.contig.count;
    
    uintptr_t x1 = res;
    
    intptr_t stride1 = md->u.contig.child->extent;
    *((int32_t *) (void *) (dbuf + idx * sizeof(int32_t))) = *((const int32_t *) (const void *) (sbuf + x0 * extent + x1 * stride1));
}

void yaksuri_cudai_pack_contig_int32_t(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_pack_contig_int32_t,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_unpack_contig_int32_t(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
{
    const char *__restrict__ sbuf = (const char *) inbuf;
    char *__restrict__ dbuf = (char *) outbuf;
    uintptr_t extent = md->extent;
    uintptr_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uintptr_t res = idx;
    uintptr_t inner_elements = md->num_elements;
    
    if (idx >= (count * inner_elements))
        return;
    
    uintptr_t x0 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.contig.count;
    
    uintptr_t x1 = res;
    
    intptr_t stride1 = md->u.contig.child->extent;
    *((int32_t *) (void *) (dbuf + x0 * extent + x1 * stride1)) = *((const int32_t *) (const void *) (sbuf + idx * sizeof(int32_t)));
}

void yaksuri_cudai_unpack_contig_int32_t(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_unpack_contig_int32_t,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

