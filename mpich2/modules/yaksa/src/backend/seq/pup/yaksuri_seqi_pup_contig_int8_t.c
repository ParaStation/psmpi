/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * DO NOT EDIT: AUTOMATICALLY GENERATED FILE !!
 */

#include <string.h>
#include <stdint.h>
#include <wchar.h>
#include "yaksuri_seqi_pup.h"

int yaksuri_seqi_pack_contig_int8_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            *((int8_t *) (void *) (dbuf + idx)) = *((const int8_t *) (const void *) (sbuf + i * extent + j1 * stride1));
            idx += sizeof(int8_t);
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_contig_int8_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            *((int8_t *) (void *) (dbuf + i * extent + j1 * stride1)) = *((const int8_t *) (const void *) (sbuf + idx));
            idx += sizeof(int8_t);
        }
    }
    
    return rc;
}

