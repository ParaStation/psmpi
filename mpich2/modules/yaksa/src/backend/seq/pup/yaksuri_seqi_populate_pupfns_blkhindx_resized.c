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

int yaksuri_seqi_populate_pupfns_blkhindx_resized(yaksi_type_s * type)
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
    
    switch (type->u.blkhindx.child->u.resized.child->kind) {
        case YAKSI_TYPE_KIND__HVECTOR:
        switch (type->u.blkhindx.child->u.resized.child->u.hvector.child->kind) {
            case YAKSI_TYPE_KIND__BUILTIN:
            switch (type->u.blkhindx.child->u.resized.child->u.hvector.child->u.builtin.handle) {
                case YAKSA_TYPE___BOOL:
                switch (type->u.blkhindx.child->u.resized.child->u.hvector.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_1__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_1__Bool;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_2__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_2__Bool;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_3__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_3__Bool;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_4__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_4__Bool;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_5__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_5__Bool;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_6__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_6__Bool;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_7__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_7__Bool;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_8__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_8__Bool;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_generic__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_generic__Bool;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__CHAR:
                switch (type->u.blkhindx.child->u.resized.child->u.hvector.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_1_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_1_char;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_2_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_2_char;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_3_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_3_char;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_4_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_4_char;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_5_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_5_char;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_6_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_6_char;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_7_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_7_char;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_8_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_8_char;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_generic_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_generic_char;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__WCHAR_T:
                switch (type->u.blkhindx.child->u.resized.child->u.hvector.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_1_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_1_wchar_t;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_2_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_2_wchar_t;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_3_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_3_wchar_t;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_4_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_4_wchar_t;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_5_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_5_wchar_t;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_6_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_6_wchar_t;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_7_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_7_wchar_t;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_8_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_8_wchar_t;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_generic_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_generic_wchar_t;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                switch (type->u.blkhindx.child->u.resized.child->u.hvector.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_1_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_1_int8_t;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_2_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_2_int8_t;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_3_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_3_int8_t;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_4_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_4_int8_t;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_5_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_5_int8_t;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_6_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_6_int8_t;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_7_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_7_int8_t;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_8_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_8_int8_t;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_generic_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_generic_int8_t;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                switch (type->u.blkhindx.child->u.resized.child->u.hvector.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_1_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_1_int16_t;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_2_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_2_int16_t;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_3_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_3_int16_t;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_4_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_4_int16_t;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_5_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_5_int16_t;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_6_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_6_int16_t;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_7_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_7_int16_t;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_8_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_8_int16_t;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_generic_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_generic_int16_t;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                switch (type->u.blkhindx.child->u.resized.child->u.hvector.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_1_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_1_int32_t;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_2_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_2_int32_t;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_3_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_3_int32_t;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_4_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_4_int32_t;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_5_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_5_int32_t;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_6_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_6_int32_t;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_7_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_7_int32_t;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_8_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_8_int32_t;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_generic_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_generic_int32_t;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                switch (type->u.blkhindx.child->u.resized.child->u.hvector.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_1_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_1_int64_t;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_2_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_2_int64_t;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_3_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_3_int64_t;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_4_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_4_int64_t;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_5_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_5_int64_t;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_6_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_6_int64_t;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_7_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_7_int64_t;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_8_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_8_int64_t;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_generic_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_generic_int64_t;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                switch (type->u.blkhindx.child->u.resized.child->u.hvector.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_1_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_1_float;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_2_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_2_float;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_3_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_3_float;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_4_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_4_float;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_5_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_5_float;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_6_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_6_float;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_7_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_7_float;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_8_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_8_float;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_generic_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_generic_float;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                switch (type->u.blkhindx.child->u.resized.child->u.hvector.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_1_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_1_double;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_2_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_2_double;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_3_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_3_double;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_4_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_4_double;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_5_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_5_double;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_6_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_6_double;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_7_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_7_double;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_8_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_8_double;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_generic_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_generic_double;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                switch (type->u.blkhindx.child->u.resized.child->u.hvector.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_1_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_1_long_double;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_2_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_2_long_double;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_3_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_3_long_double;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_4_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_4_long_double;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_5_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_5_long_double;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_6_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_6_long_double;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_7_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_7_long_double;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_8_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_8_long_double;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_hvector_blklen_generic_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hvector_blklen_generic_long_double;
                    }
                    break;
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
                case YAKSA_TYPE___BOOL:
                switch (type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_1__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_1__Bool;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_2__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_2__Bool;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_3__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_3__Bool;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_4__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_4__Bool;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_5__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_5__Bool;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_6__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_6__Bool;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_7__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_7__Bool;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_8__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_8__Bool;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_generic__Bool;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_generic__Bool;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__CHAR:
                switch (type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_1_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_1_char;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_2_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_2_char;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_3_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_3_char;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_4_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_4_char;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_5_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_5_char;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_6_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_6_char;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_7_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_7_char;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_8_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_8_char;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_generic_char;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_generic_char;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__WCHAR_T:
                switch (type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_1_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_1_wchar_t;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_2_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_2_wchar_t;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_3_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_3_wchar_t;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_4_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_4_wchar_t;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_5_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_5_wchar_t;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_6_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_6_wchar_t;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_7_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_7_wchar_t;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_8_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_8_wchar_t;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_generic_wchar_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_generic_wchar_t;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                switch (type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_1_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_1_int8_t;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_2_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_2_int8_t;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_3_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_3_int8_t;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_4_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_4_int8_t;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_5_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_5_int8_t;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_6_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_6_int8_t;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_7_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_7_int8_t;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_8_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_8_int8_t;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_generic_int8_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_generic_int8_t;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                switch (type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_1_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_1_int16_t;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_2_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_2_int16_t;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_3_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_3_int16_t;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_4_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_4_int16_t;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_5_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_5_int16_t;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_6_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_6_int16_t;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_7_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_7_int16_t;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_8_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_8_int16_t;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_generic_int16_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_generic_int16_t;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                switch (type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_1_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_1_int32_t;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_2_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_2_int32_t;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_3_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_3_int32_t;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_4_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_4_int32_t;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_5_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_5_int32_t;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_6_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_6_int32_t;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_7_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_7_int32_t;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_8_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_8_int32_t;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_generic_int32_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_generic_int32_t;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                switch (type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_1_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_1_int64_t;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_2_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_2_int64_t;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_3_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_3_int64_t;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_4_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_4_int64_t;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_5_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_5_int64_t;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_6_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_6_int64_t;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_7_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_7_int64_t;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_8_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_8_int64_t;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_generic_int64_t;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_generic_int64_t;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                switch (type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_1_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_1_float;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_2_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_2_float;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_3_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_3_float;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_4_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_4_float;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_5_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_5_float;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_6_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_6_float;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_7_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_7_float;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_8_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_8_float;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_generic_float;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_generic_float;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                switch (type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_1_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_1_double;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_2_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_2_double;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_3_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_3_double;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_4_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_4_double;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_5_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_5_double;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_6_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_6_double;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_7_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_7_double;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_8_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_8_double;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_generic_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_generic_double;
                    }
                    break;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                switch (type->u.blkhindx.child->u.resized.child->u.blkhindx.blocklength) {
                    case 1:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_1_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_1_long_double;
                    }
                    break;
                    case 2:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_2_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_2_long_double;
                    }
                    break;
                    case 3:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_3_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_3_long_double;
                    }
                    break;
                    case 4:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_4_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_4_long_double;
                    }
                    break;
                    case 5:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_5_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_5_long_double;
                    }
                    break;
                    case 6:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_6_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_6_long_double;
                    }
                    break;
                    case 7:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_7_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_7_long_double;
                    }
                    break;
                    case 8:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_8_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_8_long_double;
                    }
                    break;
                    default:
                    if (max_nesting_level >= 3) {
                        seq->pack = yaksuri_seqi_pack_blkhindx_resized_blkhindx_blklen_generic_long_double;
                        seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_blkhindx_blklen_generic_long_double;
                    }
                    break;
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
                case YAKSA_TYPE___BOOL:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_hindexed__Bool;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hindexed__Bool;
                }
                break;
                case YAKSA_TYPE__CHAR:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_hindexed_char;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hindexed_char;
                }
                break;
                case YAKSA_TYPE__WCHAR_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_hindexed_wchar_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hindexed_wchar_t;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_hindexed_int8_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hindexed_int8_t;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_hindexed_int16_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hindexed_int16_t;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_hindexed_int32_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hindexed_int32_t;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_hindexed_int64_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hindexed_int64_t;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_hindexed_float;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hindexed_float;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_hindexed_double;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hindexed_double;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_hindexed_long_double;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_hindexed_long_double;
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
                case YAKSA_TYPE___BOOL:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_contig__Bool;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_contig__Bool;
                }
                break;
                case YAKSA_TYPE__CHAR:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_contig_char;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_contig_char;
                }
                break;
                case YAKSA_TYPE__WCHAR_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_contig_wchar_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_contig_wchar_t;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_contig_int8_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_contig_int8_t;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_contig_int16_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_contig_int16_t;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_contig_int32_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_contig_int32_t;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_contig_int64_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_contig_int64_t;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_contig_float;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_contig_float;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_contig_double;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_contig_double;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_contig_long_double;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_contig_long_double;
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
                case YAKSA_TYPE___BOOL:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_resized__Bool;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_resized__Bool;
                }
                break;
                case YAKSA_TYPE__CHAR:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_resized_char;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_resized_char;
                }
                break;
                case YAKSA_TYPE__WCHAR_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_resized_wchar_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_resized_wchar_t;
                }
                break;
                case YAKSA_TYPE__INT8_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_resized_int8_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_resized_int8_t;
                }
                break;
                case YAKSA_TYPE__INT16_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_resized_int16_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_resized_int16_t;
                }
                break;
                case YAKSA_TYPE__INT32_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_resized_int32_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_resized_int32_t;
                }
                break;
                case YAKSA_TYPE__INT64_T:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_resized_int64_t;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_resized_int64_t;
                }
                break;
                case YAKSA_TYPE__FLOAT:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_resized_float;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_resized_float;
                }
                break;
                case YAKSA_TYPE__DOUBLE:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_resized_double;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_resized_double;
                }
                break;
                case YAKSA_TYPE__LONG_DOUBLE:
                if (max_nesting_level >= 3) {
                    seq->pack = yaksuri_seqi_pack_blkhindx_resized_resized_long_double;
                    seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_resized_long_double;
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
            case YAKSA_TYPE___BOOL:
            if (max_nesting_level >= 2) {
                seq->pack = yaksuri_seqi_pack_blkhindx_resized__Bool;
                seq->unpack = yaksuri_seqi_unpack_blkhindx_resized__Bool;
            }
            break;
            case YAKSA_TYPE__CHAR:
            if (max_nesting_level >= 2) {
                seq->pack = yaksuri_seqi_pack_blkhindx_resized_char;
                seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_char;
            }
            break;
            case YAKSA_TYPE__WCHAR_T:
            if (max_nesting_level >= 2) {
                seq->pack = yaksuri_seqi_pack_blkhindx_resized_wchar_t;
                seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_wchar_t;
            }
            break;
            case YAKSA_TYPE__INT8_T:
            if (max_nesting_level >= 2) {
                seq->pack = yaksuri_seqi_pack_blkhindx_resized_int8_t;
                seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_int8_t;
            }
            break;
            case YAKSA_TYPE__INT16_T:
            if (max_nesting_level >= 2) {
                seq->pack = yaksuri_seqi_pack_blkhindx_resized_int16_t;
                seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_int16_t;
            }
            break;
            case YAKSA_TYPE__INT32_T:
            if (max_nesting_level >= 2) {
                seq->pack = yaksuri_seqi_pack_blkhindx_resized_int32_t;
                seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_int32_t;
            }
            break;
            case YAKSA_TYPE__INT64_T:
            if (max_nesting_level >= 2) {
                seq->pack = yaksuri_seqi_pack_blkhindx_resized_int64_t;
                seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_int64_t;
            }
            break;
            case YAKSA_TYPE__FLOAT:
            if (max_nesting_level >= 2) {
                seq->pack = yaksuri_seqi_pack_blkhindx_resized_float;
                seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_float;
            }
            break;
            case YAKSA_TYPE__DOUBLE:
            if (max_nesting_level >= 2) {
                seq->pack = yaksuri_seqi_pack_blkhindx_resized_double;
                seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_double;
            }
            break;
            case YAKSA_TYPE__LONG_DOUBLE:
            if (max_nesting_level >= 2) {
                seq->pack = yaksuri_seqi_pack_blkhindx_resized_long_double;
                seq->unpack = yaksuri_seqi_unpack_blkhindx_resized_long_double;
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
