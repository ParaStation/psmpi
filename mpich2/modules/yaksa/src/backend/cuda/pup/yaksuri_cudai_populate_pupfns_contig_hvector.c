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

int yaksuri_cudai_populate_pupfns_contig_hvector(yaksi_type_s * type)
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
    
    switch (type->u.contig.child->u.hvector.child->kind) {
        case YAKSI_TYPE_KIND__HVECTOR:
        switch (type->u.contig.child->u.hvector.child->u.hvector.child->kind) {
            case YAKSI_TYPE_KIND__BUILTIN:
            switch (type->u.contig.child->u.hvector.child->u.hvector.child->u.builtin.handle) {
                case YAKSA_TYPE___BOOL:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hvector__Bool;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hvector__Bool;
                }
                break;
                case YAKSA_TYPE__CHAR:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hvector_char;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hvector_char;
                }
                break;
                case YAKSA_TYPE__WCHAR_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hvector_wchar_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hvector_wchar_t;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hvector_int8_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hvector_int8_t;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hvector_int16_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hvector_int16_t;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hvector_int32_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hvector_int32_t;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hvector_int64_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hvector_int64_t;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hvector_float;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hvector_float;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hvector_double;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hvector_double;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hvector_double;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hvector_double;
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
        switch (type->u.contig.child->u.hvector.child->u.blkhindx.child->kind) {
            case YAKSI_TYPE_KIND__BUILTIN:
            switch (type->u.contig.child->u.hvector.child->u.blkhindx.child->u.builtin.handle) {
                case YAKSA_TYPE___BOOL:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_blkhindx__Bool;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_blkhindx__Bool;
                }
                break;
                case YAKSA_TYPE__CHAR:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_blkhindx_char;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_blkhindx_char;
                }
                break;
                case YAKSA_TYPE__WCHAR_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_blkhindx_wchar_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_blkhindx_wchar_t;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_blkhindx_int8_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_blkhindx_int8_t;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_blkhindx_int16_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_blkhindx_int16_t;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_blkhindx_int32_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_blkhindx_int32_t;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_blkhindx_int64_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_blkhindx_int64_t;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_blkhindx_float;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_blkhindx_float;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_blkhindx_double;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_blkhindx_double;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_blkhindx_double;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_blkhindx_double;
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
        switch (type->u.contig.child->u.hvector.child->u.hindexed.child->kind) {
            case YAKSI_TYPE_KIND__BUILTIN:
            switch (type->u.contig.child->u.hvector.child->u.hindexed.child->u.builtin.handle) {
                case YAKSA_TYPE___BOOL:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hindexed__Bool;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hindexed__Bool;
                }
                break;
                case YAKSA_TYPE__CHAR:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hindexed_char;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hindexed_char;
                }
                break;
                case YAKSA_TYPE__WCHAR_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hindexed_wchar_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hindexed_wchar_t;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hindexed_int8_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hindexed_int8_t;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hindexed_int16_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hindexed_int16_t;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hindexed_int32_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hindexed_int32_t;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hindexed_int64_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hindexed_int64_t;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hindexed_float;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hindexed_float;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hindexed_double;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hindexed_double;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_hindexed_double;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_hindexed_double;
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
        switch (type->u.contig.child->u.hvector.child->u.contig.child->kind) {
            case YAKSI_TYPE_KIND__BUILTIN:
            switch (type->u.contig.child->u.hvector.child->u.contig.child->u.builtin.handle) {
                case YAKSA_TYPE___BOOL:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_contig__Bool;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_contig__Bool;
                }
                break;
                case YAKSA_TYPE__CHAR:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_contig_char;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_contig_char;
                }
                break;
                case YAKSA_TYPE__WCHAR_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_contig_wchar_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_contig_wchar_t;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_contig_int8_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_contig_int8_t;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_contig_int16_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_contig_int16_t;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_contig_int32_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_contig_int32_t;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_contig_int64_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_contig_int64_t;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_contig_float;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_contig_float;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_contig_double;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_contig_double;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_contig_double;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_contig_double;
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
        switch (type->u.contig.child->u.hvector.child->u.resized.child->kind) {
            case YAKSI_TYPE_KIND__BUILTIN:
            switch (type->u.contig.child->u.hvector.child->u.resized.child->u.builtin.handle) {
                case YAKSA_TYPE___BOOL:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_resized__Bool;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_resized__Bool;
                }
                break;
                case YAKSA_TYPE__CHAR:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_resized_char;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_resized_char;
                }
                break;
                case YAKSA_TYPE__WCHAR_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_resized_wchar_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_resized_wchar_t;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_resized_int8_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_resized_int8_t;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_resized_int16_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_resized_int16_t;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_resized_int32_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_resized_int32_t;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_resized_int64_t;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_resized_int64_t;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_resized_float;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_resized_float;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_resized_double;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_resized_double;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                if (max_nesting_level >= 3) {
                    cuda->pack = yaksuri_cudai_pack_contig_hvector_resized_double;
                    cuda->unpack = yaksuri_cudai_unpack_contig_hvector_resized_double;
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
        switch (type->u.contig.child->u.hvector.child->u.builtin.handle) {
            case YAKSA_TYPE___BOOL:
            if (max_nesting_level >= 2) {
                cuda->pack = yaksuri_cudai_pack_contig_hvector__Bool;
                cuda->unpack = yaksuri_cudai_unpack_contig_hvector__Bool;
            }
            break;
            case YAKSA_TYPE__CHAR:
            if (max_nesting_level >= 2) {
                cuda->pack = yaksuri_cudai_pack_contig_hvector_char;
                cuda->unpack = yaksuri_cudai_unpack_contig_hvector_char;
            }
            break;
            case YAKSA_TYPE__WCHAR_T:
            if (max_nesting_level >= 2) {
                cuda->pack = yaksuri_cudai_pack_contig_hvector_wchar_t;
                cuda->unpack = yaksuri_cudai_unpack_contig_hvector_wchar_t;
            }
            break;
            case YAKSA_TYPE__INT8_T:
            if (max_nesting_level >= 2) {
                cuda->pack = yaksuri_cudai_pack_contig_hvector_int8_t;
                cuda->unpack = yaksuri_cudai_unpack_contig_hvector_int8_t;
            }
            break;
            case YAKSA_TYPE__INT16_T:
            if (max_nesting_level >= 2) {
                cuda->pack = yaksuri_cudai_pack_contig_hvector_int16_t;
                cuda->unpack = yaksuri_cudai_unpack_contig_hvector_int16_t;
            }
            break;
            case YAKSA_TYPE__INT32_T:
            if (max_nesting_level >= 2) {
                cuda->pack = yaksuri_cudai_pack_contig_hvector_int32_t;
                cuda->unpack = yaksuri_cudai_unpack_contig_hvector_int32_t;
            }
            break;
            case YAKSA_TYPE__INT64_T:
            if (max_nesting_level >= 2) {
                cuda->pack = yaksuri_cudai_pack_contig_hvector_int64_t;
                cuda->unpack = yaksuri_cudai_unpack_contig_hvector_int64_t;
            }
            break;
            case YAKSA_TYPE__FLOAT:
            if (max_nesting_level >= 2) {
                cuda->pack = yaksuri_cudai_pack_contig_hvector_float;
                cuda->unpack = yaksuri_cudai_unpack_contig_hvector_float;
            }
            break;
            case YAKSA_TYPE__DOUBLE:
            if (max_nesting_level >= 2) {
                cuda->pack = yaksuri_cudai_pack_contig_hvector_double;
                cuda->unpack = yaksuri_cudai_unpack_contig_hvector_double;
            }
            break;
            case YAKSA_TYPE__LONG_DOUBLE:
            if (max_nesting_level >= 2) {
                cuda->pack = yaksuri_cudai_pack_contig_hvector_double;
                cuda->unpack = yaksuri_cudai_unpack_contig_hvector_double;
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
