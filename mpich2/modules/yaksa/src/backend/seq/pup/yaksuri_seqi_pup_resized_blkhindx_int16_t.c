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

int yaksuri_seqi_pack_resized_blkhindx_blklen_1_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 1; k2++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_blkhindx_blklen_1_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 1; k2++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_blkhindx_blklen_2_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 2; k2++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_blkhindx_blklen_2_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 2; k2++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_blkhindx_blklen_3_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 3; k2++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_blkhindx_blklen_3_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 3; k2++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_blkhindx_blklen_4_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 4; k2++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_blkhindx_blklen_4_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 4; k2++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_blkhindx_blklen_5_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 5; k2++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_blkhindx_blklen_5_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 5; k2++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_blkhindx_blklen_6_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 6; k2++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_blkhindx_blklen_6_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 6; k2++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_blkhindx_blklen_7_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 7; k2++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_blkhindx_blklen_7_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 7; k2++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_blkhindx_blklen_8_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 8; k2++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_blkhindx_blklen_8_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < 8; k2++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_blkhindx_blklen_generic_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < blocklength2; k2++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_blkhindx_blklen_generic_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    int count2 = type->u.resized.child->u.blkhindx.count;
    int blocklength2 ATTRIBUTE((unused)) = type->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs2 = type->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j2 = 0; j2 < count2; j2++) {
            for (int k2 = 0; k2 < blocklength2; k2++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs2[j2] + k2 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hvector_resized_blkhindx_blklen_1_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 1; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hvector_resized_blkhindx_blklen_1_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 1; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hvector_resized_blkhindx_blklen_2_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 2; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hvector_resized_blkhindx_blklen_2_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 2; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hvector_resized_blkhindx_blklen_3_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 3; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hvector_resized_blkhindx_blklen_3_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 3; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hvector_resized_blkhindx_blklen_4_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 4; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hvector_resized_blkhindx_blklen_4_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 4; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hvector_resized_blkhindx_blklen_5_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 5; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hvector_resized_blkhindx_blklen_5_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 5; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hvector_resized_blkhindx_blklen_6_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 6; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hvector_resized_blkhindx_blklen_6_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 6; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hvector_resized_blkhindx_blklen_7_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 7; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hvector_resized_blkhindx_blklen_7_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 7; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hvector_resized_blkhindx_blklen_8_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 8; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hvector_resized_blkhindx_blklen_8_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 8; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hvector_resized_blkhindx_blklen_generic_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < blocklength3; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hvector_resized_blkhindx_blklen_generic_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hvector.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.hvector.blocklength;
    intptr_t stride1 = type->u.hvector.stride;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hvector.child->extent;
    
    int count3 = type->u.hvector.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hvector.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hvector.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < blocklength3; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_1_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 1; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_1_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 1; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_2_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 2; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_2_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 2; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_3_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 3; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_3_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 3; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_4_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 4; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_4_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 4; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_5_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 5; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_5_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 5; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_6_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 6; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_6_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 6; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_7_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 7; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_7_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 7; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_8_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 8; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_8_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 8; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_generic_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < blocklength3; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_generic_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.blkhindx.child->extent;
    
    int count3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.blkhindx.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.blkhindx.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < blocklength3; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hindexed_resized_blkhindx_blklen_1_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 1; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hindexed_resized_blkhindx_blklen_1_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 1; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hindexed_resized_blkhindx_blklen_2_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 2; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hindexed_resized_blkhindx_blklen_2_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 2; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hindexed_resized_blkhindx_blklen_3_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 3; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hindexed_resized_blkhindx_blklen_3_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 3; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hindexed_resized_blkhindx_blklen_4_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 4; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hindexed_resized_blkhindx_blklen_4_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 4; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hindexed_resized_blkhindx_blklen_5_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 5; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hindexed_resized_blkhindx_blklen_5_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 5; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hindexed_resized_blkhindx_blklen_6_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 6; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hindexed_resized_blkhindx_blklen_6_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 6; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hindexed_resized_blkhindx_blklen_7_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 7; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hindexed_resized_blkhindx_blklen_7_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 7; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hindexed_resized_blkhindx_blklen_8_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 8; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hindexed_resized_blkhindx_blklen_8_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < 8; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_hindexed_resized_blkhindx_blklen_generic_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < blocklength3; k3++) {
                        *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_hindexed_resized_blkhindx_blklen_generic_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.hindexed.count;
    int *restrict array_of_blocklengths1 = type->u.hindexed.array_of_blocklengths;
    intptr_t *restrict array_of_displs1 = type->u.hindexed.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.hindexed.child->extent;
    
    int count3 = type->u.hindexed.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.hindexed.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.hindexed.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < array_of_blocklengths1[j1]; k1++) {
                for (int j3 = 0; j3 < count3; j3++) {
                    for (int k3 = 0; k3 < blocklength3; k3++) {
                        *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * extent2 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                        idx += sizeof(int16_t);
                    }
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_contig_resized_blkhindx_blklen_1_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 1; k3++) {
                    *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_contig_resized_blkhindx_blklen_1_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 1; k3++) {
                    *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_contig_resized_blkhindx_blklen_2_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 2; k3++) {
                    *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_contig_resized_blkhindx_blklen_2_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 2; k3++) {
                    *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_contig_resized_blkhindx_blklen_3_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 3; k3++) {
                    *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_contig_resized_blkhindx_blklen_3_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 3; k3++) {
                    *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_contig_resized_blkhindx_blklen_4_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 4; k3++) {
                    *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_contig_resized_blkhindx_blklen_4_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 4; k3++) {
                    *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_contig_resized_blkhindx_blklen_5_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 5; k3++) {
                    *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_contig_resized_blkhindx_blklen_5_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 5; k3++) {
                    *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_contig_resized_blkhindx_blklen_6_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 6; k3++) {
                    *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_contig_resized_blkhindx_blklen_6_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 6; k3++) {
                    *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_contig_resized_blkhindx_blklen_7_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 7; k3++) {
                    *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_contig_resized_blkhindx_blklen_7_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 7; k3++) {
                    *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_contig_resized_blkhindx_blklen_8_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 8; k3++) {
                    *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_contig_resized_blkhindx_blklen_8_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < 8; k3++) {
                    *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_contig_resized_blkhindx_blklen_generic_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < blocklength3; k3++) {
                    *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_contig_resized_blkhindx_blklen_generic_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.contig.count;
    intptr_t stride1 = type->u.contig.child->extent;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.contig.child->extent;
    
    int count3 = type->u.contig.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.contig.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.contig.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int j3 = 0; j3 < count3; j3++) {
                for (int k3 = 0; k3 < blocklength3; k3++) {
                    *((int16_t *) (void *) (dbuf + i * extent + j1 * stride1 + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                    idx += sizeof(int16_t);
                }
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_resized_blkhindx_blklen_1_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 1; k3++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_resized_blkhindx_blklen_1_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 1; k3++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_resized_blkhindx_blklen_2_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 2; k3++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_resized_blkhindx_blklen_2_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 2; k3++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_resized_blkhindx_blklen_3_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 3; k3++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_resized_blkhindx_blklen_3_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 3; k3++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_resized_blkhindx_blklen_4_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 4; k3++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_resized_blkhindx_blklen_4_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 4; k3++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_resized_blkhindx_blklen_5_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 5; k3++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_resized_blkhindx_blklen_5_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 5; k3++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_resized_blkhindx_blklen_6_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 6; k3++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_resized_blkhindx_blklen_6_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 6; k3++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_resized_blkhindx_blklen_7_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 7; k3++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_resized_blkhindx_blklen_7_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 7; k3++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_resized_blkhindx_blklen_8_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 8; k3++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_resized_blkhindx_blklen_8_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < 8; k3++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_resized_resized_blkhindx_blklen_generic_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < blocklength3; k3++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_resized_resized_blkhindx_blklen_generic_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t extent2 ATTRIBUTE((unused)) = type->u.resized.child->extent;
    
    int count3 = type->u.resized.child->u.resized.child->u.blkhindx.count;
    int blocklength3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs3 = type->u.resized.child->u.resized.child->u.blkhindx.array_of_displs;
    uintptr_t extent3 ATTRIBUTE((unused)) = type->u.resized.child->u.resized.child->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j3 = 0; j3 < count3; j3++) {
            for (int k3 = 0; k3 < blocklength3; k3++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs3[j3] + k3 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

