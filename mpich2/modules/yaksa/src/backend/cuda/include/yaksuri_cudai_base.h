/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_CUDAI_BASE_H_INCLUDED
#define YAKSURI_CUDAI_BASE_H_INCLUDED

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <cuda_runtime_api.h>

#define YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr)                              \
    do {                                                                \
        if (cerr != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA Error (%s:%s,%d): %s\n", __func__, __FILE__, __LINE__, cudaGetErrorString(cerr)); \
        }                                                               \
    } while (0)

typedef struct {
    int ndevices;
    cudaStream_t *stream;
    bool **p2p;
} yaksuri_cudai_global_s;
extern yaksuri_cudai_global_s yaksuri_cudai_global;

typedef struct yaksuri_cudai_md_s {
    union {
        struct {
            int count;
            intptr_t stride;
            struct yaksuri_cudai_md_s *child;
        } contig;
        struct {
            struct yaksuri_cudai_md_s *child;
        } resized;
        struct {
            int count;
            int blocklength;
            intptr_t stride;
            struct yaksuri_cudai_md_s *child;
        } hvector;
        struct {
            int count;
            int blocklength;
            intptr_t *array_of_displs;
            struct yaksuri_cudai_md_s *child;
        } blkhindx;
        struct {
            int count;
            int *array_of_blocklengths;
            intptr_t *array_of_displs;
            struct yaksuri_cudai_md_s *child;
        } hindexed;
    } u;

    uintptr_t extent;
    uintptr_t num_elements;
} yaksuri_cudai_md_s;

#endif /* YAKSURI_CUDAI_BASE_H_INCLUDED */
