/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef PACK_COMMON_H
#define PACK_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "yaksa_config.h"
#include "yaksa.h"
#include "dtpools.h"

/* *INDENT-OFF* */
#ifdef __cplusplus
extern "C" {
#endif
/* *INDENT-ON* */

typedef enum {
    MEM_TYPE__UNSET,
    MEM_TYPE__UNREGISTERED_HOST,
    MEM_TYPE__REGISTERED_HOST,
    MEM_TYPE__MANAGED,
    MEM_TYPE__DEVICE,
} mem_type_e;

extern int device_id;
extern int device_stride;

void pack_init_devices(void);
void pack_finalize_devices(void);
void pack_alloc_mem(size_t size, mem_type_e type, void **hostbuf, void **devicebuf);
void pack_free_mem(mem_type_e type, void *hostbuf, void *devicebuf);
void pack_get_ptr_attr(const void *inbuf, void *outbuf, yaksa_info_t * info);
void pack_copy_content(const void *sbuf, void *dbuf, size_t size, mem_type_e type);

#ifdef HAVE_CUDA
void pack_cuda_init_devices(void);
void pack_cuda_finalize_devices(void);
void pack_cuda_alloc_mem(size_t size, mem_type_e type, void **hostbuf, void **devicebuf);
void pack_cuda_free_mem(mem_type_e type, void *hostbuf, void *devicebuf);
void pack_cuda_get_ptr_attr(const void *inbuf, void *outbuf, yaksa_info_t * info);
void pack_cuda_copy_content(const void *sbuf, void *dbuf, size_t size, mem_type_e type);
#endif

#ifdef HAVE_ZE
void pack_ze_init_devices(void);
void pack_ze_finalize_devices(void);
void pack_ze_alloc_mem(size_t size, mem_type_e type, void **hostbuf, void **devicebuf);
void pack_ze_free_mem(mem_type_e type, void *hostbuf, void *devicebuf);
void pack_ze_get_ptr_attr(const void *inbuf, void *outbuf, yaksa_info_t * info);
void pack_ze_copy_content(const void *sbuf, void *dbuf, size_t size, mem_type_e type);
#endif

/* *INDENT-OFF* */
#ifdef __cplusplus
}
#endif
/* *INDENT-ON* */

#endif /* PACK_COMMON_H */
