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

int yaksuri_seqi_pack_contig_resized_char(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            *((char *) (void *) (dbuf + idx)) = *((const char *) (const void *) (sbuf + i * extent + j1 * stride1));
            idx += sizeof(char);
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_contig_resized_char(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            *((char *) (void *) (dbuf + i * extent + j1 * stride1)) = *((const char *) (const void *) (sbuf + idx));
            idx += sizeof(char);
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hvector_contig_resized_char(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.hvector.child->u.contig.count;
    intptr_t stride2 = type->u.hvector.child->u.contig.child->extent;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.contig.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j2 = 0; j2 < count2; j2++) {
                    *((char *) (void *) (dbuf + idx)) = *((const char *) (const void *) (sbuf + i * extent + j1 * stride1 + k1 * extent2 + j2 * stride2));
                    idx += sizeof(char);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hvector_contig_resized_char(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.hvector.child->u.contig.count;
    intptr_t stride2 = type->u.hvector.child->u.contig.child->extent;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.contig.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j2 = 0; j2 < count2; j2++) {
                    *((char *) (void *) (dbuf + i * extent + j1 * stride1 + k1 * extent2 + j2 * stride2)) = *((const char *) (const void *) (sbuf + idx));
                    idx += sizeof(char);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_contig_resized_char(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.blkhindx.child->u.contig.count;
    intptr_t stride2 = type->u.blkhindx.child->u.contig.child->extent;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.contig.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j2 = 0; j2 < count2; j2++) {
                    *((char *) (void *) (dbuf + idx)) = *((const char *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + j2 * stride2));
                    idx += sizeof(char);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_contig_resized_char(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.blkhindx.child->u.contig.count;
    intptr_t stride2 = type->u.blkhindx.child->u.contig.child->extent;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.contig.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j2 = 0; j2 < count2; j2++) {
                    *((char *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + j2 * stride2)) = *((const char *) (const void *) (sbuf + idx));
                    idx += sizeof(char);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hindexed_contig_resized_char(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.hindexed.child->u.contig.count;
    intptr_t stride2 = type->u.hindexed.child->u.contig.child->extent;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.contig.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j2 = 0; j2 < count2; j2++) {
                    *((char *) (void *) (dbuf + idx)) = *((const char *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + j2 * stride2));
                    idx += sizeof(char);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hindexed_contig_resized_char(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.hindexed.child->u.contig.count;
    intptr_t stride2 = type->u.hindexed.child->u.contig.child->extent;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.contig.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j2 = 0; j2 < count2; j2++) {
                    *((char *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + j2 * stride2)) = *((const char *) (const void *) (sbuf + idx));
                    idx += sizeof(char);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_contig_contig_resized_char(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.contig.child->u.contig.count;
    intptr_t stride2 = type->u.contig.child->u.contig.child->extent;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.contig.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j2 = 0; j2 < count2; j2++) {
                *((char *) (void *) (dbuf + idx)) = *((const char *) (const void *) (sbuf + i * extent + j1 * stride1 + j2 * stride2));
                idx += sizeof(char);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_contig_contig_resized_char(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.contig.child->u.contig.count;
    intptr_t stride2 = type->u.contig.child->u.contig.child->extent;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.contig.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j2 = 0; j2 < count2; j2++) {
                *((char *) (void *) (dbuf + i * extent + j1 * stride1 + j2 * stride2)) = *((const char *) (const void *) (sbuf + idx));
                idx += sizeof(char);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_contig_resized_char(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.contig.count;
    intptr_t stride2 = type->u.resized.child->u.contig.child->extent;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.contig.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            *((char *) (void *) (dbuf + idx)) = *((const char *) (const void *) (sbuf + i * extent + j2 * stride2));
            idx += sizeof(char);
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_contig_resized_char(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.contig.count;
    intptr_t stride2 = type->u.resized.child->u.contig.child->extent;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.contig.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            *((char *) (void *) (dbuf + i * extent + j2 * stride2)) = *((const char *) (const void *) (sbuf + idx));
            idx += sizeof(char);
        }
    }
    
    return rc;
}

