/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * DO NOT EDIT: AUTOMATICALLY GENERATED FILE !!
 */

#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri_cudai.h"
#include "yaksuri_cudai_populate_pupfns.h"
#include "yaksuri_cudai_pup.h"

int yaksuri_cudai_populate_pupfns_blkhindx_builtin(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_type_s *cuda = (yaksuri_cudai_type_s *) type->backend.cuda.priv;
    
    char *str = getenv("YAKSA_ENV_MAX_NESTING_LEVEL");
    int max_nesting_level;
    if (str) {
        max_nesting_level = atoi(str);
    } else {
        max_nesting_level = YAKSI_ENV_DEFAULT_NESTING_LEVEL;
    }
    
    switch (type->u.blkhindx.child->u.builtin.handle) {
        case YAKSA_TYPE___BOOL:
        if (max_nesting_level >= 1) {
            cuda->pack = yaksuri_cudai_pack_blkhindx__Bool;
            cuda->unpack = yaksuri_cudai_unpack_blkhindx__Bool;
        }
        break;
        case YAKSA_TYPE__CHAR:
        if (max_nesting_level >= 1) {
            cuda->pack = yaksuri_cudai_pack_blkhindx_char;
            cuda->unpack = yaksuri_cudai_unpack_blkhindx_char;
        }
        break;
        case YAKSA_TYPE__WCHAR_T:
        if (max_nesting_level >= 1) {
            cuda->pack = yaksuri_cudai_pack_blkhindx_wchar_t;
            cuda->unpack = yaksuri_cudai_unpack_blkhindx_wchar_t;
        }
        break;
        case YAKSA_TYPE__INT8_T:
        if (max_nesting_level >= 1) {
            cuda->pack = yaksuri_cudai_pack_blkhindx_int8_t;
            cuda->unpack = yaksuri_cudai_unpack_blkhindx_int8_t;
        }
        break;
        case YAKSA_TYPE__INT16_T:
        if (max_nesting_level >= 1) {
            cuda->pack = yaksuri_cudai_pack_blkhindx_int16_t;
            cuda->unpack = yaksuri_cudai_unpack_blkhindx_int16_t;
        }
        break;
        case YAKSA_TYPE__INT32_T:
        if (max_nesting_level >= 1) {
            cuda->pack = yaksuri_cudai_pack_blkhindx_int32_t;
            cuda->unpack = yaksuri_cudai_unpack_blkhindx_int32_t;
        }
        break;
        case YAKSA_TYPE__INT64_T:
        if (max_nesting_level >= 1) {
            cuda->pack = yaksuri_cudai_pack_blkhindx_int64_t;
            cuda->unpack = yaksuri_cudai_unpack_blkhindx_int64_t;
        }
        break;
        case YAKSA_TYPE__FLOAT:
        if (max_nesting_level >= 1) {
            cuda->pack = yaksuri_cudai_pack_blkhindx_float;
            cuda->unpack = yaksuri_cudai_unpack_blkhindx_float;
        }
        break;
        case YAKSA_TYPE__DOUBLE:
        if (max_nesting_level >= 1) {
            cuda->pack = yaksuri_cudai_pack_blkhindx_double;
            cuda->unpack = yaksuri_cudai_unpack_blkhindx_double;
        }
        break;
        case YAKSA_TYPE__LONG_DOUBLE:
        if (max_nesting_level >= 1) {
            cuda->pack = yaksuri_cudai_pack_blkhindx_double;
            cuda->unpack = yaksuri_cudai_unpack_blkhindx_double;
        }
        break;
        default:
            break;
    }
    
    return rc;
}
