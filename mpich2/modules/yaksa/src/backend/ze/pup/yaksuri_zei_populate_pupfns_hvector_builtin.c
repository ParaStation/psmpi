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
#include "yaksuri_zei.h"
#include "yaksuri_zei_populate_pupfns.h"
#include "yaksuri_zei_pup.h"

int yaksuri_zei_populate_pupfns_hvector_builtin(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_zei_type_s *ze = (yaksuri_zei_type_s *) type->backend.ze.priv;
    
    char *str = getenv("YAKSA_ENV_MAX_NESTING_LEVEL");
    int max_nesting_level;
    if (str) {
        max_nesting_level = atoi(str);
    } else {
        max_nesting_level = YAKSI_ENV_DEFAULT_NESTING_LEVEL;
    }
    
    switch (type->u.hvector.child->u.builtin.handle) {
        case YAKSA_TYPE__CHAR:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_char;
            ze->unpack = yaksuri_zei_unpack_hvector_char;
        }
        break;
        case YAKSA_TYPE__INT:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_int;
            ze->unpack = yaksuri_zei_unpack_hvector_int;
        }
        break;
        case YAKSA_TYPE__SHORT:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_short;
            ze->unpack = yaksuri_zei_unpack_hvector_short;
        }
        break;
        case YAKSA_TYPE__LONG:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_long;
            ze->unpack = yaksuri_zei_unpack_hvector_long;
        }
        break;
        case YAKSA_TYPE__INT8_T:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_int8_t;
            ze->unpack = yaksuri_zei_unpack_hvector_int8_t;
        }
        break;
        case YAKSA_TYPE__INT16_T:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_int16_t;
            ze->unpack = yaksuri_zei_unpack_hvector_int16_t;
        }
        break;
        case YAKSA_TYPE__INT32_T:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_int32_t;
            ze->unpack = yaksuri_zei_unpack_hvector_int32_t;
        }
        break;
        case YAKSA_TYPE__INT64_T:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_int64_t;
            ze->unpack = yaksuri_zei_unpack_hvector_int64_t;
        }
        break;
        case YAKSA_TYPE__FLOAT:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_float;
            ze->unpack = yaksuri_zei_unpack_hvector_float;
        }
        break;
        case YAKSA_TYPE__DOUBLE:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_double;
            ze->unpack = yaksuri_zei_unpack_hvector_double;
        }
        break;
        case YAKSA_TYPE__UNSIGNED_CHAR:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_char;
            ze->unpack = yaksuri_zei_unpack_hvector_char;
        }
        break;
        case YAKSA_TYPE__UNSIGNED:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_int;
            ze->unpack = yaksuri_zei_unpack_hvector_int;
        }
        break;
        case YAKSA_TYPE__UNSIGNED_SHORT:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_short;
            ze->unpack = yaksuri_zei_unpack_hvector_short;
        }
        break;
        case YAKSA_TYPE__UNSIGNED_LONG:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_long;
            ze->unpack = yaksuri_zei_unpack_hvector_long;
        }
        break;
        case YAKSA_TYPE__LONG_DOUBLE:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_double;
            ze->unpack = yaksuri_zei_unpack_hvector_double;
        }
        break;
        case YAKSA_TYPE__UNSIGNED_LONG_LONG:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_long;
            ze->unpack = yaksuri_zei_unpack_hvector_long;
        }
        break;
        case YAKSA_TYPE__UINT8_T:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_int8_t;
            ze->unpack = yaksuri_zei_unpack_hvector_int8_t;
        }
        break;
        case YAKSA_TYPE__UINT16_T:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_int16_t;
            ze->unpack = yaksuri_zei_unpack_hvector_int16_t;
        }
        break;
        case YAKSA_TYPE__UINT32_T:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_int32_t;
            ze->unpack = yaksuri_zei_unpack_hvector_int32_t;
        }
        break;
        case YAKSA_TYPE__UINT64_T:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_int64_t;
            ze->unpack = yaksuri_zei_unpack_hvector_int64_t;
        }
        break;
        case YAKSA_TYPE__C_COMPLEX:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_float;
            ze->unpack = yaksuri_zei_unpack_hvector_float;
        }
        break;
        case YAKSA_TYPE__C_DOUBLE_COMPLEX:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_double;
            ze->unpack = yaksuri_zei_unpack_hvector_double;
        }
        break;
        case YAKSA_TYPE__C_LONG_DOUBLE_COMPLEX:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_double;
            ze->unpack = yaksuri_zei_unpack_hvector_double;
        }
        break;
        case YAKSA_TYPE__BYTE:
        if (max_nesting_level >= 1) {
            ze->pack = yaksuri_zei_pack_hvector_int8_t;
            ze->unpack = yaksuri_zei_unpack_hvector_int8_t;
        }
        break;
        default:
            break;
    }
    
    return rc;
}
