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

int yaksuri_zei_populate_pupfns_blkhindx_resized(yaksi_type_s * type)
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
    
    switch (type->u.blkhindx.child->u.resized.child->kind) {
        case YAKSI_TYPE_KIND__HVECTOR:
        switch (type->u.blkhindx.child->u.resized.child->u.hvector.child->kind) {
            case YAKSI_TYPE_KIND__BUILTIN:
            switch (type->u.blkhindx.child->u.resized.child->u.hvector.child->u.builtin.handle) {
                case YAKSA_TYPE__CHAR:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_char;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_char;
                }
                break;
                case YAKSA_TYPE__INT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_int;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_int;
                }
                break;
                case YAKSA_TYPE__SHORT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_short;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_short;
                }
                break;
                case YAKSA_TYPE__LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_long;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_int8_t;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_int16_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_int16_t;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_int32_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_int32_t;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_int64_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_int64_t;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_float;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_float;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_double;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_CHAR:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_char;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_char;
                }
                break;
                case YAKSA_TYPE__UNSIGNED:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_int;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_int;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_SHORT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_short;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_short;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_long;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_double;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_LONG_LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_long;
                }
                break;
                case YAKSA_TYPE__UINT8_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_int8_t;
                }
                break;
                case YAKSA_TYPE__UINT16_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_int16_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_int16_t;
                }
                break;
                case YAKSA_TYPE__UINT32_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_int32_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_int32_t;
                }
                break;
                case YAKSA_TYPE__UINT64_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_int64_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_int64_t;
                }
                break;
                case YAKSA_TYPE__C_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_float;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_float;
                }
                break;
                case YAKSA_TYPE__C_DOUBLE_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_double;
                }
                break;
                case YAKSA_TYPE__C_LONG_DOUBLE_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_double;
                }
                break;
                case YAKSA_TYPE__BYTE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hvector_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hvector_int8_t;
                }
                break;
                default:
                    break;
            }
            break;
            default:
                break;
        }
        break;
        case YAKSI_TYPE_KIND__BLKHINDX:
        switch (type->u.blkhindx.child->u.resized.child->u.blkhindx.child->kind) {
            case YAKSI_TYPE_KIND__BUILTIN:
            switch (type->u.blkhindx.child->u.resized.child->u.blkhindx.child->u.builtin.handle) {
                case YAKSA_TYPE__CHAR:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_char;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_char;
                }
                break;
                case YAKSA_TYPE__INT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_int;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_int;
                }
                break;
                case YAKSA_TYPE__SHORT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_short;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_short;
                }
                break;
                case YAKSA_TYPE__LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_long;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_int8_t;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_int16_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_int16_t;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_int32_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_int32_t;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_int64_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_int64_t;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_float;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_float;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_double;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_CHAR:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_char;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_char;
                }
                break;
                case YAKSA_TYPE__UNSIGNED:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_int;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_int;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_SHORT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_short;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_short;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_long;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_double;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_LONG_LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_long;
                }
                break;
                case YAKSA_TYPE__UINT8_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_int8_t;
                }
                break;
                case YAKSA_TYPE__UINT16_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_int16_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_int16_t;
                }
                break;
                case YAKSA_TYPE__UINT32_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_int32_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_int32_t;
                }
                break;
                case YAKSA_TYPE__UINT64_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_int64_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_int64_t;
                }
                break;
                case YAKSA_TYPE__C_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_float;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_float;
                }
                break;
                case YAKSA_TYPE__C_DOUBLE_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_double;
                }
                break;
                case YAKSA_TYPE__C_LONG_DOUBLE_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_double;
                }
                break;
                case YAKSA_TYPE__BYTE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_blkhindx_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_blkhindx_int8_t;
                }
                break;
                default:
                    break;
            }
            break;
            default:
                break;
        }
        break;
        case YAKSI_TYPE_KIND__HINDEXED:
        switch (type->u.blkhindx.child->u.resized.child->u.hindexed.child->kind) {
            case YAKSI_TYPE_KIND__BUILTIN:
            switch (type->u.blkhindx.child->u.resized.child->u.hindexed.child->u.builtin.handle) {
                case YAKSA_TYPE__CHAR:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_char;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_char;
                }
                break;
                case YAKSA_TYPE__INT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_int;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_int;
                }
                break;
                case YAKSA_TYPE__SHORT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_short;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_short;
                }
                break;
                case YAKSA_TYPE__LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_long;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_int8_t;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_int16_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_int16_t;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_int32_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_int32_t;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_int64_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_int64_t;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_float;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_float;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_double;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_CHAR:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_char;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_char;
                }
                break;
                case YAKSA_TYPE__UNSIGNED:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_int;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_int;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_SHORT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_short;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_short;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_long;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_double;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_LONG_LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_long;
                }
                break;
                case YAKSA_TYPE__UINT8_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_int8_t;
                }
                break;
                case YAKSA_TYPE__UINT16_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_int16_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_int16_t;
                }
                break;
                case YAKSA_TYPE__UINT32_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_int32_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_int32_t;
                }
                break;
                case YAKSA_TYPE__UINT64_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_int64_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_int64_t;
                }
                break;
                case YAKSA_TYPE__C_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_float;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_float;
                }
                break;
                case YAKSA_TYPE__C_DOUBLE_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_double;
                }
                break;
                case YAKSA_TYPE__C_LONG_DOUBLE_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_double;
                }
                break;
                case YAKSA_TYPE__BYTE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_hindexed_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_hindexed_int8_t;
                }
                break;
                default:
                    break;
            }
            break;
            default:
                break;
        }
        break;
        case YAKSI_TYPE_KIND__CONTIG:
        switch (type->u.blkhindx.child->u.resized.child->u.contig.child->kind) {
            case YAKSI_TYPE_KIND__BUILTIN:
            switch (type->u.blkhindx.child->u.resized.child->u.contig.child->u.builtin.handle) {
                case YAKSA_TYPE__CHAR:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_char;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_char;
                }
                break;
                case YAKSA_TYPE__INT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_int;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_int;
                }
                break;
                case YAKSA_TYPE__SHORT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_short;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_short;
                }
                break;
                case YAKSA_TYPE__LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_long;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_int8_t;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_int16_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_int16_t;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_int32_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_int32_t;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_int64_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_int64_t;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_float;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_float;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_double;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_CHAR:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_char;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_char;
                }
                break;
                case YAKSA_TYPE__UNSIGNED:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_int;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_int;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_SHORT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_short;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_short;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_long;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_double;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_LONG_LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_long;
                }
                break;
                case YAKSA_TYPE__UINT8_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_int8_t;
                }
                break;
                case YAKSA_TYPE__UINT16_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_int16_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_int16_t;
                }
                break;
                case YAKSA_TYPE__UINT32_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_int32_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_int32_t;
                }
                break;
                case YAKSA_TYPE__UINT64_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_int64_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_int64_t;
                }
                break;
                case YAKSA_TYPE__C_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_float;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_float;
                }
                break;
                case YAKSA_TYPE__C_DOUBLE_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_double;
                }
                break;
                case YAKSA_TYPE__C_LONG_DOUBLE_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_double;
                }
                break;
                case YAKSA_TYPE__BYTE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_contig_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_contig_int8_t;
                }
                break;
                default:
                    break;
            }
            break;
            default:
                break;
        }
        break;
        case YAKSI_TYPE_KIND__RESIZED:
        switch (type->u.blkhindx.child->u.resized.child->u.resized.child->kind) {
            case YAKSI_TYPE_KIND__BUILTIN:
            switch (type->u.blkhindx.child->u.resized.child->u.resized.child->u.builtin.handle) {
                case YAKSA_TYPE__CHAR:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_char;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_char;
                }
                break;
                case YAKSA_TYPE__INT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_int;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_int;
                }
                break;
                case YAKSA_TYPE__SHORT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_short;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_short;
                }
                break;
                case YAKSA_TYPE__LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_long;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_int8_t;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_int16_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_int16_t;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_int32_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_int32_t;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_int64_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_int64_t;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_float;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_float;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_double;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_CHAR:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_char;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_char;
                }
                break;
                case YAKSA_TYPE__UNSIGNED:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_int;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_int;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_SHORT:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_short;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_short;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_long;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_double;
                }
                break;
                case YAKSA_TYPE__UNSIGNED_LONG_LONG:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_long;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_long;
                }
                break;
                case YAKSA_TYPE__UINT8_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_int8_t;
                }
                break;
                case YAKSA_TYPE__UINT16_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_int16_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_int16_t;
                }
                break;
                case YAKSA_TYPE__UINT32_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_int32_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_int32_t;
                }
                break;
                case YAKSA_TYPE__UINT64_T:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_int64_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_int64_t;
                }
                break;
                case YAKSA_TYPE__C_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_float;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_float;
                }
                break;
                case YAKSA_TYPE__C_DOUBLE_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_double;
                }
                break;
                case YAKSA_TYPE__C_LONG_DOUBLE_COMPLEX:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_double;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_double;
                }
                break;
                case YAKSA_TYPE__BYTE:
                if (max_nesting_level >= 3) {
                    ze->pack = yaksuri_zei_pack_blkhindx_resized_resized_int8_t;
                    ze->unpack = yaksuri_zei_unpack_blkhindx_resized_resized_int8_t;
                }
                break;
                default:
                    break;
            }
            break;
            default:
                break;
        }
        break;
        case YAKSI_TYPE_KIND__BUILTIN:
        switch (type->u.blkhindx.child->u.resized.child->u.builtin.handle) {
            case YAKSA_TYPE__CHAR:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_char;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_char;
            }
            break;
            case YAKSA_TYPE__INT:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_int;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_int;
            }
            break;
            case YAKSA_TYPE__SHORT:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_short;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_short;
            }
            break;
            case YAKSA_TYPE__LONG:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_long;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_long;
            }
            break;
            case YAKSA_TYPE__INT8_T:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_int8_t;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_int8_t;
            }
            break;
            case YAKSA_TYPE__INT16_T:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_int16_t;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_int16_t;
            }
            break;
            case YAKSA_TYPE__INT32_T:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_int32_t;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_int32_t;
            }
            break;
            case YAKSA_TYPE__INT64_T:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_int64_t;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_int64_t;
            }
            break;
            case YAKSA_TYPE__FLOAT:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_float;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_float;
            }
            break;
            case YAKSA_TYPE__DOUBLE:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_double;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_double;
            }
            break;
            case YAKSA_TYPE__UNSIGNED_CHAR:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_char;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_char;
            }
            break;
            case YAKSA_TYPE__UNSIGNED:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_int;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_int;
            }
            break;
            case YAKSA_TYPE__UNSIGNED_SHORT:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_short;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_short;
            }
            break;
            case YAKSA_TYPE__UNSIGNED_LONG:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_long;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_long;
            }
            break;
            case YAKSA_TYPE__LONG_DOUBLE:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_double;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_double;
            }
            break;
            case YAKSA_TYPE__UNSIGNED_LONG_LONG:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_long;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_long;
            }
            break;
            case YAKSA_TYPE__UINT8_T:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_int8_t;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_int8_t;
            }
            break;
            case YAKSA_TYPE__UINT16_T:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_int16_t;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_int16_t;
            }
            break;
            case YAKSA_TYPE__UINT32_T:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_int32_t;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_int32_t;
            }
            break;
            case YAKSA_TYPE__UINT64_T:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_int64_t;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_int64_t;
            }
            break;
            case YAKSA_TYPE__C_COMPLEX:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_float;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_float;
            }
            break;
            case YAKSA_TYPE__C_DOUBLE_COMPLEX:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_double;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_double;
            }
            break;
            case YAKSA_TYPE__C_LONG_DOUBLE_COMPLEX:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_double;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_double;
            }
            break;
            case YAKSA_TYPE__BYTE:
            if (max_nesting_level >= 2) {
                ze->pack = yaksuri_zei_pack_blkhindx_resized_int8_t;
                ze->unpack = yaksuri_zei_unpack_blkhindx_resized_int8_t;
            }
            break;
            default:
                break;
        }
        break;
        default:
            break;
    }
    
    return rc;
}
