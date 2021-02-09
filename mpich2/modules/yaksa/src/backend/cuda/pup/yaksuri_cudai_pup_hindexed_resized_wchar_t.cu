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

__global__ void yaksuri_cudai_kernel_pack_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.hindexed.count;
    
    uintptr_t x1;
    for (int i = 0; i < md->u.hindexed.count; i++) {
            uintptr_t in_elems = md->u.hindexed.array_of_blocklengths[i] *
                                 md->u.hindexed.child->num_elements;
            if (res < in_elems) {
                    x1 = i;
                    res %= in_elems;
                    inner_elements = md->u.hindexed.child->num_elements;
                    break;
            } else {
                    res -= in_elems;
            }
    }
    
    uintptr_t x2 = res;
    
    intptr_t *array_of_displs1 = md->u.hindexed.array_of_displs;
    uintptr_t extent2 = md->u.hindexed.child->extent;
    *((wchar_t *) (void *) (dbuf + idx * sizeof(wchar_t))) = *((const wchar_t *) (const void *) (sbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2));
}

void yaksuri_cudai_pack_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_pack_hindexed_resized_wchar_t,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_unpack_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.hindexed.count;
    
    uintptr_t x1;
    for (int i = 0; i < md->u.hindexed.count; i++) {
            uintptr_t in_elems = md->u.hindexed.array_of_blocklengths[i] *
                                 md->u.hindexed.child->num_elements;
            if (res < in_elems) {
                    x1 = i;
                    res %= in_elems;
                    inner_elements = md->u.hindexed.child->num_elements;
                    break;
            } else {
                    res -= in_elems;
            }
    }
    
    uintptr_t x2 = res;
    
    intptr_t *array_of_displs1 = md->u.hindexed.array_of_displs;
    uintptr_t extent2 = md->u.hindexed.child->extent;
    *((wchar_t *) (void *) (dbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2)) = *((const wchar_t *) (const void *) (sbuf + idx * sizeof(wchar_t)));
}

void yaksuri_cudai_unpack_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_unpack_hindexed_resized_wchar_t,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_pack_hvector_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.hvector.count;
    
    uintptr_t x1 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hvector.blocklength;
    uintptr_t x2 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hvector.child->u.hindexed.count;
    
    uintptr_t x3;
    for (int i = 0; i < md->u.hvector.child->u.hindexed.count; i++) {
            uintptr_t in_elems = md->u.hvector.child->u.hindexed.array_of_blocklengths[i] *
                                 md->u.hvector.child->u.hindexed.child->num_elements;
            if (res < in_elems) {
                    x3 = i;
                    res %= in_elems;
                    inner_elements = md->u.hvector.child->u.hindexed.child->num_elements;
                    break;
            } else {
                    res -= in_elems;
            }
    }
    
    uintptr_t x4 = res;
    
    intptr_t stride1 = md->u.hvector.stride;
    intptr_t *array_of_displs2 = md->u.hvector.child->u.hindexed.array_of_displs;
    uintptr_t extent2 = md->u.hvector.child->extent;
    uintptr_t extent3 = md->u.hvector.child->u.hindexed.child->extent;
    *((wchar_t *) (void *) (dbuf + idx * sizeof(wchar_t))) = *((const wchar_t *) (const void *) (sbuf + x0 * extent + x1 * stride1 + x2 * extent2 + array_of_displs2[x3] + x4 * extent3));
}

void yaksuri_cudai_pack_hvector_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_pack_hvector_hindexed_resized_wchar_t,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_unpack_hvector_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.hvector.count;
    
    uintptr_t x1 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hvector.blocklength;
    uintptr_t x2 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hvector.child->u.hindexed.count;
    
    uintptr_t x3;
    for (int i = 0; i < md->u.hvector.child->u.hindexed.count; i++) {
            uintptr_t in_elems = md->u.hvector.child->u.hindexed.array_of_blocklengths[i] *
                                 md->u.hvector.child->u.hindexed.child->num_elements;
            if (res < in_elems) {
                    x3 = i;
                    res %= in_elems;
                    inner_elements = md->u.hvector.child->u.hindexed.child->num_elements;
                    break;
            } else {
                    res -= in_elems;
            }
    }
    
    uintptr_t x4 = res;
    
    intptr_t stride1 = md->u.hvector.stride;
    intptr_t *array_of_displs2 = md->u.hvector.child->u.hindexed.array_of_displs;
    uintptr_t extent2 = md->u.hvector.child->extent;
    uintptr_t extent3 = md->u.hvector.child->u.hindexed.child->extent;
    *((wchar_t *) (void *) (dbuf + x0 * extent + x1 * stride1 + x2 * extent2 + array_of_displs2[x3] + x4 * extent3)) = *((const wchar_t *) (const void *) (sbuf + idx * sizeof(wchar_t)));
}

void yaksuri_cudai_unpack_hvector_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_unpack_hvector_hindexed_resized_wchar_t,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_pack_blkhindx_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.blkhindx.count;
    
    uintptr_t x1 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.blkhindx.blocklength;
    uintptr_t x2 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.blkhindx.child->u.hindexed.count;
    
    uintptr_t x3;
    for (int i = 0; i < md->u.blkhindx.child->u.hindexed.count; i++) {
            uintptr_t in_elems = md->u.blkhindx.child->u.hindexed.array_of_blocklengths[i] *
                                 md->u.blkhindx.child->u.hindexed.child->num_elements;
            if (res < in_elems) {
                    x3 = i;
                    res %= in_elems;
                    inner_elements = md->u.blkhindx.child->u.hindexed.child->num_elements;
                    break;
            } else {
                    res -= in_elems;
            }
    }
    
    uintptr_t x4 = res;
    
    intptr_t *array_of_displs1 = md->u.blkhindx.array_of_displs;
    intptr_t *array_of_displs2 = md->u.blkhindx.child->u.hindexed.array_of_displs;
    uintptr_t extent2 = md->u.blkhindx.child->extent;
    uintptr_t extent3 = md->u.blkhindx.child->u.hindexed.child->extent;
    *((wchar_t *) (void *) (dbuf + idx * sizeof(wchar_t))) = *((const wchar_t *) (const void *) (sbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2 + array_of_displs2[x3] + x4 * extent3));
}

void yaksuri_cudai_pack_blkhindx_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_pack_blkhindx_hindexed_resized_wchar_t,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_unpack_blkhindx_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.blkhindx.count;
    
    uintptr_t x1 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.blkhindx.blocklength;
    uintptr_t x2 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.blkhindx.child->u.hindexed.count;
    
    uintptr_t x3;
    for (int i = 0; i < md->u.blkhindx.child->u.hindexed.count; i++) {
            uintptr_t in_elems = md->u.blkhindx.child->u.hindexed.array_of_blocklengths[i] *
                                 md->u.blkhindx.child->u.hindexed.child->num_elements;
            if (res < in_elems) {
                    x3 = i;
                    res %= in_elems;
                    inner_elements = md->u.blkhindx.child->u.hindexed.child->num_elements;
                    break;
            } else {
                    res -= in_elems;
            }
    }
    
    uintptr_t x4 = res;
    
    intptr_t *array_of_displs1 = md->u.blkhindx.array_of_displs;
    intptr_t *array_of_displs2 = md->u.blkhindx.child->u.hindexed.array_of_displs;
    uintptr_t extent2 = md->u.blkhindx.child->extent;
    uintptr_t extent3 = md->u.blkhindx.child->u.hindexed.child->extent;
    *((wchar_t *) (void *) (dbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2 + array_of_displs2[x3] + x4 * extent3)) = *((const wchar_t *) (const void *) (sbuf + idx * sizeof(wchar_t)));
}

void yaksuri_cudai_unpack_blkhindx_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_unpack_blkhindx_hindexed_resized_wchar_t,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_pack_hindexed_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.hindexed.count;
    
    uintptr_t x1;
    for (int i = 0; i < md->u.hindexed.count; i++) {
            uintptr_t in_elems = md->u.hindexed.array_of_blocklengths[i] *
                                 md->u.hindexed.child->num_elements;
            if (res < in_elems) {
                    x1 = i;
                    res %= in_elems;
                    inner_elements = md->u.hindexed.child->num_elements;
                    break;
            } else {
                    res -= in_elems;
            }
    }
    
    uintptr_t x2 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hindexed.child->u.hindexed.count;
    
    uintptr_t x3;
    for (int i = 0; i < md->u.hindexed.child->u.hindexed.count; i++) {
            uintptr_t in_elems = md->u.hindexed.child->u.hindexed.array_of_blocklengths[i] *
                                 md->u.hindexed.child->u.hindexed.child->num_elements;
            if (res < in_elems) {
                    x3 = i;
                    res %= in_elems;
                    inner_elements = md->u.hindexed.child->u.hindexed.child->num_elements;
                    break;
            } else {
                    res -= in_elems;
            }
    }
    
    uintptr_t x4 = res;
    
    intptr_t *array_of_displs1 = md->u.hindexed.array_of_displs;
    intptr_t *array_of_displs2 = md->u.hindexed.child->u.hindexed.array_of_displs;
    uintptr_t extent2 = md->u.hindexed.child->extent;
    uintptr_t extent3 = md->u.hindexed.child->u.hindexed.child->extent;
    *((wchar_t *) (void *) (dbuf + idx * sizeof(wchar_t))) = *((const wchar_t *) (const void *) (sbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2 + array_of_displs2[x3] + x4 * extent3));
}

void yaksuri_cudai_pack_hindexed_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_pack_hindexed_hindexed_resized_wchar_t,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_unpack_hindexed_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.hindexed.count;
    
    uintptr_t x1;
    for (int i = 0; i < md->u.hindexed.count; i++) {
            uintptr_t in_elems = md->u.hindexed.array_of_blocklengths[i] *
                                 md->u.hindexed.child->num_elements;
            if (res < in_elems) {
                    x1 = i;
                    res %= in_elems;
                    inner_elements = md->u.hindexed.child->num_elements;
                    break;
            } else {
                    res -= in_elems;
            }
    }
    
    uintptr_t x2 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hindexed.child->u.hindexed.count;
    
    uintptr_t x3;
    for (int i = 0; i < md->u.hindexed.child->u.hindexed.count; i++) {
            uintptr_t in_elems = md->u.hindexed.child->u.hindexed.array_of_blocklengths[i] *
                                 md->u.hindexed.child->u.hindexed.child->num_elements;
            if (res < in_elems) {
                    x3 = i;
                    res %= in_elems;
                    inner_elements = md->u.hindexed.child->u.hindexed.child->num_elements;
                    break;
            } else {
                    res -= in_elems;
            }
    }
    
    uintptr_t x4 = res;
    
    intptr_t *array_of_displs1 = md->u.hindexed.array_of_displs;
    intptr_t *array_of_displs2 = md->u.hindexed.child->u.hindexed.array_of_displs;
    uintptr_t extent2 = md->u.hindexed.child->extent;
    uintptr_t extent3 = md->u.hindexed.child->u.hindexed.child->extent;
    *((wchar_t *) (void *) (dbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2 + array_of_displs2[x3] + x4 * extent3)) = *((const wchar_t *) (const void *) (sbuf + idx * sizeof(wchar_t)));
}

void yaksuri_cudai_unpack_hindexed_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_unpack_hindexed_hindexed_resized_wchar_t,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_pack_contig_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    
    uintptr_t x1 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.contig.child->u.hindexed.count;
    
    uintptr_t x2;
    for (int i = 0; i < md->u.contig.child->u.hindexed.count; i++) {
            uintptr_t in_elems = md->u.contig.child->u.hindexed.array_of_blocklengths[i] *
                                 md->u.contig.child->u.hindexed.child->num_elements;
            if (res < in_elems) {
                    x2 = i;
                    res %= in_elems;
                    inner_elements = md->u.contig.child->u.hindexed.child->num_elements;
                    break;
            } else {
                    res -= in_elems;
            }
    }
    
    uintptr_t x3 = res;
    
    intptr_t stride1 = md->u.contig.child->extent;
    intptr_t *array_of_displs2 = md->u.contig.child->u.hindexed.array_of_displs;
    uintptr_t extent3 = md->u.contig.child->u.hindexed.child->extent;
    *((wchar_t *) (void *) (dbuf + idx * sizeof(wchar_t))) = *((const wchar_t *) (const void *) (sbuf + x0 * extent + x1 * stride1 + array_of_displs2[x2] + x3 * extent3));
}

void yaksuri_cudai_pack_contig_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_pack_contig_hindexed_resized_wchar_t,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_unpack_contig_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    
    uintptr_t x1 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.contig.child->u.hindexed.count;
    
    uintptr_t x2;
    for (int i = 0; i < md->u.contig.child->u.hindexed.count; i++) {
            uintptr_t in_elems = md->u.contig.child->u.hindexed.array_of_blocklengths[i] *
                                 md->u.contig.child->u.hindexed.child->num_elements;
            if (res < in_elems) {
                    x2 = i;
                    res %= in_elems;
                    inner_elements = md->u.contig.child->u.hindexed.child->num_elements;
                    break;
            } else {
                    res -= in_elems;
            }
    }
    
    uintptr_t x3 = res;
    
    intptr_t stride1 = md->u.contig.child->extent;
    intptr_t *array_of_displs2 = md->u.contig.child->u.hindexed.array_of_displs;
    uintptr_t extent3 = md->u.contig.child->u.hindexed.child->extent;
    *((wchar_t *) (void *) (dbuf + x0 * extent + x1 * stride1 + array_of_displs2[x2] + x3 * extent3)) = *((const wchar_t *) (const void *) (sbuf + idx * sizeof(wchar_t)));
}

void yaksuri_cudai_unpack_contig_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_unpack_contig_hindexed_resized_wchar_t,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_pack_resized_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.resized.child->u.hindexed.count;
    
    uintptr_t x1;
    for (int i = 0; i < md->u.resized.child->u.hindexed.count; i++) {
            uintptr_t in_elems = md->u.resized.child->u.hindexed.array_of_blocklengths[i] *
                                 md->u.resized.child->u.hindexed.child->num_elements;
            if (res < in_elems) {
                    x1 = i;
                    res %= in_elems;
                    inner_elements = md->u.resized.child->u.hindexed.child->num_elements;
                    break;
            } else {
                    res -= in_elems;
            }
    }
    
    uintptr_t x2 = res;
    
    intptr_t *array_of_displs2 = md->u.resized.child->u.hindexed.array_of_displs;
    uintptr_t extent3 = md->u.resized.child->u.hindexed.child->extent;
    *((wchar_t *) (void *) (dbuf + idx * sizeof(wchar_t))) = *((const wchar_t *) (const void *) (sbuf + x0 * extent + array_of_displs2[x1] + x2 * extent3));
}

void yaksuri_cudai_pack_resized_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_pack_resized_hindexed_resized_wchar_t,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_unpack_resized_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.resized.child->u.hindexed.count;
    
    uintptr_t x1;
    for (int i = 0; i < md->u.resized.child->u.hindexed.count; i++) {
            uintptr_t in_elems = md->u.resized.child->u.hindexed.array_of_blocklengths[i] *
                                 md->u.resized.child->u.hindexed.child->num_elements;
            if (res < in_elems) {
                    x1 = i;
                    res %= in_elems;
                    inner_elements = md->u.resized.child->u.hindexed.child->num_elements;
                    break;
            } else {
                    res -= in_elems;
            }
    }
    
    uintptr_t x2 = res;
    
    intptr_t *array_of_displs2 = md->u.resized.child->u.hindexed.array_of_displs;
    uintptr_t extent3 = md->u.resized.child->u.hindexed.child->extent;
    *((wchar_t *) (void *) (dbuf + x0 * extent + array_of_displs2[x1] + x2 * extent3)) = *((const wchar_t *) (const void *) (sbuf + idx * sizeof(wchar_t)));
}

void yaksuri_cudai_unpack_resized_hindexed_resized_wchar_t(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_unpack_resized_hindexed_resized_wchar_t,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

