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

__global__ void yaksuri_cudai_kernel_pack_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.blkhindx.child->u.contig.count;
    
    uintptr_t x3 = res;
    
    intptr_t *array_of_displs1 = md->u.blkhindx.array_of_displs;
    intptr_t stride2 = md->u.blkhindx.child->u.contig.child->extent;
    uintptr_t extent2 = md->u.blkhindx.child->extent;
    *((float *) (void *) (dbuf + idx * sizeof(float))) = *((const float *) (const void *) (sbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2 + x3 * stride2));
}

void yaksuri_cudai_pack_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_pack_blkhindx_contig_float,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_unpack_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.blkhindx.child->u.contig.count;
    
    uintptr_t x3 = res;
    
    intptr_t *array_of_displs1 = md->u.blkhindx.array_of_displs;
    intptr_t stride2 = md->u.blkhindx.child->u.contig.child->extent;
    uintptr_t extent2 = md->u.blkhindx.child->extent;
    *((float *) (void *) (dbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2 + x3 * stride2)) = *((const float *) (const void *) (sbuf + idx * sizeof(float)));
}

void yaksuri_cudai_unpack_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_unpack_blkhindx_contig_float,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_pack_hvector_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.hvector.child->u.blkhindx.count;
    
    uintptr_t x3 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hvector.child->u.blkhindx.blocklength;
    uintptr_t x4 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hvector.child->u.blkhindx.child->u.contig.count;
    
    uintptr_t x5 = res;
    
    intptr_t stride1 = md->u.hvector.stride;
    intptr_t *array_of_displs2 = md->u.hvector.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 = md->u.hvector.child->extent;
    intptr_t stride3 = md->u.hvector.child->u.blkhindx.child->u.contig.child->extent;
    uintptr_t extent3 = md->u.hvector.child->u.blkhindx.child->extent;
    *((float *) (void *) (dbuf + idx * sizeof(float))) = *((const float *) (const void *) (sbuf + x0 * extent + x1 * stride1 + x2 * extent2 + array_of_displs2[x3] + x4 * extent3 + x5 * stride3));
}

void yaksuri_cudai_pack_hvector_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_pack_hvector_blkhindx_contig_float,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_unpack_hvector_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.hvector.child->u.blkhindx.count;
    
    uintptr_t x3 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hvector.child->u.blkhindx.blocklength;
    uintptr_t x4 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hvector.child->u.blkhindx.child->u.contig.count;
    
    uintptr_t x5 = res;
    
    intptr_t stride1 = md->u.hvector.stride;
    intptr_t *array_of_displs2 = md->u.hvector.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 = md->u.hvector.child->extent;
    intptr_t stride3 = md->u.hvector.child->u.blkhindx.child->u.contig.child->extent;
    uintptr_t extent3 = md->u.hvector.child->u.blkhindx.child->extent;
    *((float *) (void *) (dbuf + x0 * extent + x1 * stride1 + x2 * extent2 + array_of_displs2[x3] + x4 * extent3 + x5 * stride3)) = *((const float *) (const void *) (sbuf + idx * sizeof(float)));
}

void yaksuri_cudai_unpack_hvector_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_unpack_hvector_blkhindx_contig_float,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_pack_blkhindx_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.blkhindx.child->u.blkhindx.count;
    
    uintptr_t x3 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.blkhindx.child->u.blkhindx.blocklength;
    uintptr_t x4 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.blkhindx.child->u.blkhindx.child->u.contig.count;
    
    uintptr_t x5 = res;
    
    intptr_t *array_of_displs1 = md->u.blkhindx.array_of_displs;
    intptr_t *array_of_displs2 = md->u.blkhindx.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 = md->u.blkhindx.child->extent;
    intptr_t stride3 = md->u.blkhindx.child->u.blkhindx.child->u.contig.child->extent;
    uintptr_t extent3 = md->u.blkhindx.child->u.blkhindx.child->extent;
    *((float *) (void *) (dbuf + idx * sizeof(float))) = *((const float *) (const void *) (sbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2 + array_of_displs2[x3] + x4 * extent3 + x5 * stride3));
}

void yaksuri_cudai_pack_blkhindx_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_pack_blkhindx_blkhindx_contig_float,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_unpack_blkhindx_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.blkhindx.child->u.blkhindx.count;
    
    uintptr_t x3 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.blkhindx.child->u.blkhindx.blocklength;
    uintptr_t x4 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.blkhindx.child->u.blkhindx.child->u.contig.count;
    
    uintptr_t x5 = res;
    
    intptr_t *array_of_displs1 = md->u.blkhindx.array_of_displs;
    intptr_t *array_of_displs2 = md->u.blkhindx.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 = md->u.blkhindx.child->extent;
    intptr_t stride3 = md->u.blkhindx.child->u.blkhindx.child->u.contig.child->extent;
    uintptr_t extent3 = md->u.blkhindx.child->u.blkhindx.child->extent;
    *((float *) (void *) (dbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2 + array_of_displs2[x3] + x4 * extent3 + x5 * stride3)) = *((const float *) (const void *) (sbuf + idx * sizeof(float)));
}

void yaksuri_cudai_unpack_blkhindx_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_unpack_blkhindx_blkhindx_contig_float,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_pack_hindexed_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.hindexed.child->u.blkhindx.count;
    
    uintptr_t x3 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hindexed.child->u.blkhindx.blocklength;
    uintptr_t x4 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hindexed.child->u.blkhindx.child->u.contig.count;
    
    uintptr_t x5 = res;
    
    intptr_t *array_of_displs1 = md->u.hindexed.array_of_displs;
    intptr_t *array_of_displs2 = md->u.hindexed.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 = md->u.hindexed.child->extent;
    intptr_t stride3 = md->u.hindexed.child->u.blkhindx.child->u.contig.child->extent;
    uintptr_t extent3 = md->u.hindexed.child->u.blkhindx.child->extent;
    *((float *) (void *) (dbuf + idx * sizeof(float))) = *((const float *) (const void *) (sbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2 + array_of_displs2[x3] + x4 * extent3 + x5 * stride3));
}

void yaksuri_cudai_pack_hindexed_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_pack_hindexed_blkhindx_contig_float,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_unpack_hindexed_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.hindexed.child->u.blkhindx.count;
    
    uintptr_t x3 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hindexed.child->u.blkhindx.blocklength;
    uintptr_t x4 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hindexed.child->u.blkhindx.child->u.contig.count;
    
    uintptr_t x5 = res;
    
    intptr_t *array_of_displs1 = md->u.hindexed.array_of_displs;
    intptr_t *array_of_displs2 = md->u.hindexed.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 = md->u.hindexed.child->extent;
    intptr_t stride3 = md->u.hindexed.child->u.blkhindx.child->u.contig.child->extent;
    uintptr_t extent3 = md->u.hindexed.child->u.blkhindx.child->extent;
    *((float *) (void *) (dbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2 + array_of_displs2[x3] + x4 * extent3 + x5 * stride3)) = *((const float *) (const void *) (sbuf + idx * sizeof(float)));
}

void yaksuri_cudai_unpack_hindexed_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_unpack_hindexed_blkhindx_contig_float,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_pack_contig_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.contig.child->u.blkhindx.count;
    
    uintptr_t x2 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.contig.child->u.blkhindx.blocklength;
    uintptr_t x3 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.contig.child->u.blkhindx.child->u.contig.count;
    
    uintptr_t x4 = res;
    
    intptr_t stride1 = md->u.contig.child->extent;
    intptr_t *array_of_displs2 = md->u.contig.child->u.blkhindx.array_of_displs;
    intptr_t stride3 = md->u.contig.child->u.blkhindx.child->u.contig.child->extent;
    uintptr_t extent3 = md->u.contig.child->u.blkhindx.child->extent;
    *((float *) (void *) (dbuf + idx * sizeof(float))) = *((const float *) (const void *) (sbuf + x0 * extent + x1 * stride1 + array_of_displs2[x2] + x3 * extent3 + x4 * stride3));
}

void yaksuri_cudai_pack_contig_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_pack_contig_blkhindx_contig_float,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_unpack_contig_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.contig.child->u.blkhindx.count;
    
    uintptr_t x2 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.contig.child->u.blkhindx.blocklength;
    uintptr_t x3 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.contig.child->u.blkhindx.child->u.contig.count;
    
    uintptr_t x4 = res;
    
    intptr_t stride1 = md->u.contig.child->extent;
    intptr_t *array_of_displs2 = md->u.contig.child->u.blkhindx.array_of_displs;
    intptr_t stride3 = md->u.contig.child->u.blkhindx.child->u.contig.child->extent;
    uintptr_t extent3 = md->u.contig.child->u.blkhindx.child->extent;
    *((float *) (void *) (dbuf + x0 * extent + x1 * stride1 + array_of_displs2[x2] + x3 * extent3 + x4 * stride3)) = *((const float *) (const void *) (sbuf + idx * sizeof(float)));
}

void yaksuri_cudai_unpack_contig_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_unpack_contig_blkhindx_contig_float,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_pack_resized_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.resized.child->u.blkhindx.count;
    
    uintptr_t x1 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.resized.child->u.blkhindx.blocklength;
    uintptr_t x2 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.resized.child->u.blkhindx.child->u.contig.count;
    
    uintptr_t x3 = res;
    
    intptr_t *array_of_displs2 = md->u.resized.child->u.blkhindx.array_of_displs;
    intptr_t stride3 = md->u.resized.child->u.blkhindx.child->u.contig.child->extent;
    uintptr_t extent3 = md->u.resized.child->u.blkhindx.child->extent;
    *((float *) (void *) (dbuf + idx * sizeof(float))) = *((const float *) (const void *) (sbuf + x0 * extent + array_of_displs2[x1] + x2 * extent3 + x3 * stride3));
}

void yaksuri_cudai_pack_resized_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_pack_resized_blkhindx_contig_float,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

__global__ void yaksuri_cudai_kernel_unpack_resized_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)
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
    inner_elements /= md->u.resized.child->u.blkhindx.count;
    
    uintptr_t x1 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.resized.child->u.blkhindx.blocklength;
    uintptr_t x2 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.resized.child->u.blkhindx.child->u.contig.count;
    
    uintptr_t x3 = res;
    
    intptr_t *array_of_displs2 = md->u.resized.child->u.blkhindx.array_of_displs;
    intptr_t stride3 = md->u.resized.child->u.blkhindx.child->u.contig.child->extent;
    uintptr_t extent3 = md->u.resized.child->u.blkhindx.child->extent;
    *((float *) (void *) (dbuf + x0 * extent + array_of_displs2[x1] + x2 * extent3 + x3 * stride3)) = *((const float *) (const void *) (sbuf + idx * sizeof(float)));
}

void yaksuri_cudai_unpack_resized_blkhindx_contig_float(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)
{
void *args[] = { &inbuf, &outbuf, &count, &md };
    cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_unpack_resized_blkhindx_contig_float,
        dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

