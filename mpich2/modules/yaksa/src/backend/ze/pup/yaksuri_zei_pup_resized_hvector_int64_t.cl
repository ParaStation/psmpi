/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * DO NOT EDIT: AUTOMATICALLY GENERATED FILE !!
 */

typedef signed char int8_t;
typedef signed short int int16_t;
typedef signed int int32_t;
typedef signed long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;
#include "yaksuri_zei_md.h"

__kernel void yaksuri_zei_kernel_pack_resized_hvector_int64_t(__global const void *inbuf, __global void *outbuf, unsigned long count, __global const yaksuri_zei_md_s *__restrict__ md)
{
    __global const char *__restrict__ sbuf = (__global char *) inbuf;
    __global char *__restrict__ dbuf = (__global char *) outbuf;
    sbuf = (__global const char *) ((__global char *)sbuf - md->true_lb);
    uintptr_t extent = md->extent;
    uintptr_t idx = get_global_id(0);
    uintptr_t res = idx;
    uintptr_t inner_elements = md->num_elements;
    
    if (idx >= (count * inner_elements))
        return;
    
    uintptr_t x0 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.resized.child->u.hvector.count;
    
    uintptr_t x1 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.resized.child->u.hvector.blocklength;
    uintptr_t x2 = res;
    
    intptr_t stride2 = md->u.resized.child->u.hvector.stride;
    *((int64_t *) (void *) (dbuf + idx * sizeof(int64_t))) = *((const int64_t *) (const void *) (sbuf + x0 * extent + x1 * stride2 + x2 * sizeof(int64_t)));
}

__kernel void yaksuri_zei_kernel_unpack_resized_hvector_int64_t(__global const void *inbuf, __global void *outbuf, unsigned long count, __global const yaksuri_zei_md_s *__restrict__ md)
{
    __global const char *__restrict__ sbuf = (__global char *) inbuf;
    __global char *__restrict__ dbuf = (__global char *) outbuf;
    dbuf = dbuf - md->true_lb;
    uintptr_t extent = md->extent;
    uintptr_t idx = get_global_id(0);
    uintptr_t res = idx;
    uintptr_t inner_elements = md->num_elements;
    
    if (idx >= (count * inner_elements))
        return;
    
    uintptr_t x0 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.resized.child->u.hvector.count;
    
    uintptr_t x1 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.resized.child->u.hvector.blocklength;
    uintptr_t x2 = res;
    
    intptr_t stride2 = md->u.resized.child->u.hvector.stride;
    *((int64_t *) (void *) (dbuf + x0 * extent + x1 * stride2 + x2 * sizeof(int64_t))) = *((const int64_t *) (const void *) (sbuf + idx * sizeof(int64_t)));
}

__kernel void yaksuri_zei_kernel_pack_hvector_resized_hvector_int64_t(__global const void *inbuf, __global void *outbuf, unsigned long count, __global const yaksuri_zei_md_s *__restrict__ md)
{
    __global const char *__restrict__ sbuf = (__global char *) inbuf;
    __global char *__restrict__ dbuf = (__global char *) outbuf;
    sbuf = (__global const char *) ((__global char *)sbuf - md->true_lb);
    uintptr_t extent = md->extent;
    uintptr_t idx = get_global_id(0);
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
    inner_elements /= md->u.hvector.child->u.resized.child->u.hvector.count;
    
    uintptr_t x3 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hvector.child->u.resized.child->u.hvector.blocklength;
    uintptr_t x4 = res;
    
    intptr_t stride1 = md->u.hvector.stride;
    uintptr_t extent2 = md->u.hvector.child->extent;
    intptr_t stride3 = md->u.hvector.child->u.resized.child->u.hvector.stride;
    *((int64_t *) (void *) (dbuf + idx * sizeof(int64_t))) = *((const int64_t *) (const void *) (sbuf + x0 * extent + x1 * stride1 + x2 * extent2 + x3 * stride3 + x4 * sizeof(int64_t)));
}

__kernel void yaksuri_zei_kernel_unpack_hvector_resized_hvector_int64_t(__global const void *inbuf, __global void *outbuf, unsigned long count, __global const yaksuri_zei_md_s *__restrict__ md)
{
    __global const char *__restrict__ sbuf = (__global char *) inbuf;
    __global char *__restrict__ dbuf = (__global char *) outbuf;
    dbuf = dbuf - md->true_lb;
    uintptr_t extent = md->extent;
    uintptr_t idx = get_global_id(0);
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
    inner_elements /= md->u.hvector.child->u.resized.child->u.hvector.count;
    
    uintptr_t x3 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hvector.child->u.resized.child->u.hvector.blocklength;
    uintptr_t x4 = res;
    
    intptr_t stride1 = md->u.hvector.stride;
    uintptr_t extent2 = md->u.hvector.child->extent;
    intptr_t stride3 = md->u.hvector.child->u.resized.child->u.hvector.stride;
    *((int64_t *) (void *) (dbuf + x0 * extent + x1 * stride1 + x2 * extent2 + x3 * stride3 + x4 * sizeof(int64_t))) = *((const int64_t *) (const void *) (sbuf + idx * sizeof(int64_t)));
}

__kernel void yaksuri_zei_kernel_pack_blkhindx_resized_hvector_int64_t(__global const void *inbuf, __global void *outbuf, unsigned long count, __global const yaksuri_zei_md_s *__restrict__ md)
{
    __global const char *__restrict__ sbuf = (__global char *) inbuf;
    __global char *__restrict__ dbuf = (__global char *) outbuf;
    sbuf = (__global const char *) ((__global char *)sbuf - md->true_lb);
    uintptr_t extent = md->extent;
    uintptr_t idx = get_global_id(0);
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
    inner_elements /= md->u.blkhindx.child->u.resized.child->u.hvector.count;
    
    uintptr_t x3 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.blkhindx.child->u.resized.child->u.hvector.blocklength;
    uintptr_t x4 = res;
    
    intptr_t *array_of_displs1 = md->u.blkhindx.array_of_displs;
    uintptr_t extent2 = md->u.blkhindx.child->extent;
    intptr_t stride3 = md->u.blkhindx.child->u.resized.child->u.hvector.stride;
    *((int64_t *) (void *) (dbuf + idx * sizeof(int64_t))) = *((const int64_t *) (const void *) (sbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2 + x3 * stride3 + x4 * sizeof(int64_t)));
}

__kernel void yaksuri_zei_kernel_unpack_blkhindx_resized_hvector_int64_t(__global const void *inbuf, __global void *outbuf, unsigned long count, __global const yaksuri_zei_md_s *__restrict__ md)
{
    __global const char *__restrict__ sbuf = (__global char *) inbuf;
    __global char *__restrict__ dbuf = (__global char *) outbuf;
    dbuf = dbuf - md->true_lb;
    uintptr_t extent = md->extent;
    uintptr_t idx = get_global_id(0);
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
    inner_elements /= md->u.blkhindx.child->u.resized.child->u.hvector.count;
    
    uintptr_t x3 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.blkhindx.child->u.resized.child->u.hvector.blocklength;
    uintptr_t x4 = res;
    
    intptr_t *array_of_displs1 = md->u.blkhindx.array_of_displs;
    uintptr_t extent2 = md->u.blkhindx.child->extent;
    intptr_t stride3 = md->u.blkhindx.child->u.resized.child->u.hvector.stride;
    *((int64_t *) (void *) (dbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2 + x3 * stride3 + x4 * sizeof(int64_t))) = *((const int64_t *) (const void *) (sbuf + idx * sizeof(int64_t)));
}

__kernel void yaksuri_zei_kernel_pack_hindexed_resized_hvector_int64_t(__global const void *inbuf, __global void *outbuf, unsigned long count, __global const yaksuri_zei_md_s *__restrict__ md)
{
    __global const char *__restrict__ sbuf = (__global char *) inbuf;
    __global char *__restrict__ dbuf = (__global char *) outbuf;
    sbuf = (__global const char *) ((__global char *)sbuf - md->true_lb);
    uintptr_t extent = md->extent;
    uintptr_t idx = get_global_id(0);
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
    inner_elements /= md->u.hindexed.child->u.resized.child->u.hvector.count;
    
    uintptr_t x3 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hindexed.child->u.resized.child->u.hvector.blocklength;
    uintptr_t x4 = res;
    
    intptr_t *array_of_displs1 = md->u.hindexed.array_of_displs;
    uintptr_t extent2 = md->u.hindexed.child->extent;
    intptr_t stride3 = md->u.hindexed.child->u.resized.child->u.hvector.stride;
    *((int64_t *) (void *) (dbuf + idx * sizeof(int64_t))) = *((const int64_t *) (const void *) (sbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2 + x3 * stride3 + x4 * sizeof(int64_t)));
}

__kernel void yaksuri_zei_kernel_unpack_hindexed_resized_hvector_int64_t(__global const void *inbuf, __global void *outbuf, unsigned long count, __global const yaksuri_zei_md_s *__restrict__ md)
{
    __global const char *__restrict__ sbuf = (__global char *) inbuf;
    __global char *__restrict__ dbuf = (__global char *) outbuf;
    dbuf = dbuf - md->true_lb;
    uintptr_t extent = md->extent;
    uintptr_t idx = get_global_id(0);
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
    inner_elements /= md->u.hindexed.child->u.resized.child->u.hvector.count;
    
    uintptr_t x3 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.hindexed.child->u.resized.child->u.hvector.blocklength;
    uintptr_t x4 = res;
    
    intptr_t *array_of_displs1 = md->u.hindexed.array_of_displs;
    uintptr_t extent2 = md->u.hindexed.child->extent;
    intptr_t stride3 = md->u.hindexed.child->u.resized.child->u.hvector.stride;
    *((int64_t *) (void *) (dbuf + x0 * extent + array_of_displs1[x1] + x2 * extent2 + x3 * stride3 + x4 * sizeof(int64_t))) = *((const int64_t *) (const void *) (sbuf + idx * sizeof(int64_t)));
}

__kernel void yaksuri_zei_kernel_pack_contig_resized_hvector_int64_t(__global const void *inbuf, __global void *outbuf, unsigned long count, __global const yaksuri_zei_md_s *__restrict__ md)
{
    __global const char *__restrict__ sbuf = (__global char *) inbuf;
    __global char *__restrict__ dbuf = (__global char *) outbuf;
    sbuf = (__global const char *) ((__global char *)sbuf - md->true_lb);
    uintptr_t extent = md->extent;
    uintptr_t idx = get_global_id(0);
    uintptr_t res = idx;
    uintptr_t inner_elements = md->num_elements;
    
    if (idx >= (count * inner_elements))
        return;
    
    uintptr_t x0 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.contig.count;
    
    uintptr_t x1 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.contig.child->u.resized.child->u.hvector.count;
    
    uintptr_t x2 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.contig.child->u.resized.child->u.hvector.blocklength;
    uintptr_t x3 = res;
    
    intptr_t stride1 = md->u.contig.child->extent;
    intptr_t stride3 = md->u.contig.child->u.resized.child->u.hvector.stride;
    *((int64_t *) (void *) (dbuf + idx * sizeof(int64_t))) = *((const int64_t *) (const void *) (sbuf + x0 * extent + x1 * stride1 + x2 * stride3 + x3 * sizeof(int64_t)));
}

__kernel void yaksuri_zei_kernel_unpack_contig_resized_hvector_int64_t(__global const void *inbuf, __global void *outbuf, unsigned long count, __global const yaksuri_zei_md_s *__restrict__ md)
{
    __global const char *__restrict__ sbuf = (__global char *) inbuf;
    __global char *__restrict__ dbuf = (__global char *) outbuf;
    dbuf = dbuf - md->true_lb;
    uintptr_t extent = md->extent;
    uintptr_t idx = get_global_id(0);
    uintptr_t res = idx;
    uintptr_t inner_elements = md->num_elements;
    
    if (idx >= (count * inner_elements))
        return;
    
    uintptr_t x0 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.contig.count;
    
    uintptr_t x1 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.contig.child->u.resized.child->u.hvector.count;
    
    uintptr_t x2 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.contig.child->u.resized.child->u.hvector.blocklength;
    uintptr_t x3 = res;
    
    intptr_t stride1 = md->u.contig.child->extent;
    intptr_t stride3 = md->u.contig.child->u.resized.child->u.hvector.stride;
    *((int64_t *) (void *) (dbuf + x0 * extent + x1 * stride1 + x2 * stride3 + x3 * sizeof(int64_t))) = *((const int64_t *) (const void *) (sbuf + idx * sizeof(int64_t)));
}

__kernel void yaksuri_zei_kernel_pack_resized_resized_hvector_int64_t(__global const void *inbuf, __global void *outbuf, unsigned long count, __global const yaksuri_zei_md_s *__restrict__ md)
{
    __global const char *__restrict__ sbuf = (__global char *) inbuf;
    __global char *__restrict__ dbuf = (__global char *) outbuf;
    sbuf = (__global const char *) ((__global char *)sbuf - md->true_lb);
    uintptr_t extent = md->extent;
    uintptr_t idx = get_global_id(0);
    uintptr_t res = idx;
    uintptr_t inner_elements = md->num_elements;
    
    if (idx >= (count * inner_elements))
        return;
    
    uintptr_t x0 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.resized.child->u.resized.child->u.hvector.count;
    
    uintptr_t x1 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.resized.child->u.resized.child->u.hvector.blocklength;
    uintptr_t x2 = res;
    
    intptr_t stride3 = md->u.resized.child->u.resized.child->u.hvector.stride;
    *((int64_t *) (void *) (dbuf + idx * sizeof(int64_t))) = *((const int64_t *) (const void *) (sbuf + x0 * extent + x1 * stride3 + x2 * sizeof(int64_t)));
}

__kernel void yaksuri_zei_kernel_unpack_resized_resized_hvector_int64_t(__global const void *inbuf, __global void *outbuf, unsigned long count, __global const yaksuri_zei_md_s *__restrict__ md)
{
    __global const char *__restrict__ sbuf = (__global char *) inbuf;
    __global char *__restrict__ dbuf = (__global char *) outbuf;
    dbuf = dbuf - md->true_lb;
    uintptr_t extent = md->extent;
    uintptr_t idx = get_global_id(0);
    uintptr_t res = idx;
    uintptr_t inner_elements = md->num_elements;
    
    if (idx >= (count * inner_elements))
        return;
    
    uintptr_t x0 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.resized.child->u.resized.child->u.hvector.count;
    
    uintptr_t x1 = res / inner_elements;
    res %= inner_elements;
    inner_elements /= md->u.resized.child->u.resized.child->u.hvector.blocklength;
    uintptr_t x2 = res;
    
    intptr_t stride3 = md->u.resized.child->u.resized.child->u.hvector.stride;
    *((int64_t *) (void *) (dbuf + x0 * extent + x1 * stride3 + x2 * sizeof(int64_t))) = *((const int64_t *) (const void *) (sbuf + idx * sizeof(int64_t)));
}

