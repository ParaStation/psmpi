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
#include "yaksuri_seqi.h"
#include "yaksuri_seqi_populate_pupfns.h"
#include "yaksuri_seqi_pup.h"

int yaksuri_seqi_populate_pupfns_resized_builtin(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_seqi_type_s *seq = (yaksuri_seqi_type_s *) type->backend.seq.priv;
    
    char *str = getenv("YAKSA_ENV_MAX_NESTING_LEVEL");
    int max_nesting_level;
    if (str) {
        max_nesting_level = atoi(str);
    } else {
        max_nesting_level = YAKSI_ENV_DEFAULT_NESTING_LEVEL;
    }
    
    switch (type->u.resized.child->u.builtin.handle) {
        case YAKSA_TYPE___BOOL:
        if (max_nesting_level >= 1) {
            seq->pack = yaksuri_seqi_pack_resized__Bool;
            seq->unpack = yaksuri_seqi_unpack_resized__Bool;
        }
        break;
        case YAKSA_TYPE__CHAR:
        if (max_nesting_level >= 1) {
            seq->pack = yaksuri_seqi_pack_resized_char;
            seq->unpack = yaksuri_seqi_unpack_resized_char;
        }
        break;
        case YAKSA_TYPE__WCHAR_T:
        if (max_nesting_level >= 1) {
            seq->pack = yaksuri_seqi_pack_resized_wchar_t;
            seq->unpack = yaksuri_seqi_unpack_resized_wchar_t;
        }
        break;
        case YAKSA_TYPE__INT8_T:
        if (max_nesting_level >= 1) {
            seq->pack = yaksuri_seqi_pack_resized_int8_t;
            seq->unpack = yaksuri_seqi_unpack_resized_int8_t;
        }
        break;
        case YAKSA_TYPE__INT16_T:
        if (max_nesting_level >= 1) {
            seq->pack = yaksuri_seqi_pack_resized_int16_t;
            seq->unpack = yaksuri_seqi_unpack_resized_int16_t;
        }
        break;
        case YAKSA_TYPE__INT32_T:
        if (max_nesting_level >= 1) {
            seq->pack = yaksuri_seqi_pack_resized_int32_t;
            seq->unpack = yaksuri_seqi_unpack_resized_int32_t;
        }
        break;
        case YAKSA_TYPE__INT64_T:
        if (max_nesting_level >= 1) {
            seq->pack = yaksuri_seqi_pack_resized_int64_t;
            seq->unpack = yaksuri_seqi_unpack_resized_int64_t;
        }
        break;
        case YAKSA_TYPE__FLOAT:
        if (max_nesting_level >= 1) {
            seq->pack = yaksuri_seqi_pack_resized_float;
            seq->unpack = yaksuri_seqi_unpack_resized_float;
        }
        break;
        case YAKSA_TYPE__DOUBLE:
        if (max_nesting_level >= 1) {
            seq->pack = yaksuri_seqi_pack_resized_double;
            seq->unpack = yaksuri_seqi_unpack_resized_double;
        }
        break;
        case YAKSA_TYPE__LONG_DOUBLE:
        if (max_nesting_level >= 1) {
            seq->pack = yaksuri_seqi_pack_resized_long_double;
            seq->unpack = yaksuri_seqi_unpack_resized_long_double;
        }
        break;
        default:
            break;
    }
    
    return rc;
}
