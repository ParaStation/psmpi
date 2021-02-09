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

int yaksuri_seqi_pack_blkhindx_blklen_1_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 1; k1++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_blklen_1_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 1; k1++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_blklen_2_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 2; k1++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_blklen_2_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 2; k1++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_blklen_3_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 3; k1++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_blklen_3_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 3; k1++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_blklen_4_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 4; k1++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_blklen_4_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 4; k1++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_blklen_5_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 5; k1++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_blklen_5_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 5; k1++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_blklen_6_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 6; k1++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_blklen_6_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 6; k1++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_blklen_7_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 7; k1++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_blklen_7_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 7; k1++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_blklen_8_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 8; k1++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_blklen_8_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < 8; k1++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_pack_blkhindx_blklen_generic_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                *((int16_t *) (void *) (dbuf + idx)) = *((const int16_t *) (const void *) (sbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t)));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

int yaksuri_seqi_unpack_blkhindx_blklen_generic_int16_t(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    const char *restrict sbuf = (const char *) inbuf;
    char *restrict dbuf = (char *) outbuf;
    uintptr_t extent ATTRIBUTE((unused)) = type->extent;
    
    int count1 = type->u.blkhindx.count;
    int blocklength1 ATTRIBUTE((unused)) = type->u.blkhindx.blocklength;
    intptr_t *restrict array_of_displs1 = type->u.blkhindx.array_of_displs;
    uintptr_t extent1 ATTRIBUTE((unused)) = type->extent;
    
    uintptr_t idx = 0;
    for (int i = 0; i < count; i++) {
        for (int j1 = 0; j1 < count1; j1++) {
            for (int k1 = 0; k1 < blocklength1; k1++) {
                *((int16_t *) (void *) (dbuf + i * extent + array_of_displs1[j1] + k1 * sizeof(int16_t))) = *((const int16_t *) (const void *) (sbuf + idx));
                idx += sizeof(int16_t);
            }
        }
    }
    
    return rc;
}

