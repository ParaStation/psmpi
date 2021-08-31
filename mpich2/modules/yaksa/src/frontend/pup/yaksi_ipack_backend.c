/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <assert.h>

#define BUILTIN_PAIRTYPE_PACK(type, TYPE1, TYPE2, inbuf, count, outbuf, info, request) \
    do {                                                                \
        type z;                                                         \
        uintptr_t offset = (char *) &z.y - (char *) &z;                 \
        const char *sbuf = (const char *) inbuf;                        \
        char *dbuf = (char *) outbuf;                                   \
                                                                        \
        yaksi_type_s *type1;                                            \
        rc = yaksi_type_get(YAKSA_TYPE__ ## TYPE1, &type1);             \
        YAKSU_ERR_CHECK(rc, fn_fail);                                   \
                                                                        \
        yaksi_type_s *type2;                                            \
        rc = yaksi_type_get(YAKSA_TYPE__ ## TYPE2, &type2);             \
        YAKSU_ERR_CHECK(rc, fn_fail);                                   \
                                                                        \
        for (int i = 0; i < count; i++) {                               \
            rc = yaksi_ipack_backend(sbuf, dbuf, 1, type1, info, request); \
            YAKSU_ERR_CHECK(rc, fn_fail);                               \
            dbuf += type1->size;                                        \
                                                                        \
            rc = yaksi_ipack_backend(sbuf + offset, dbuf, 1, type2, info, request); \
            YAKSU_ERR_CHECK(rc, fn_fail);                               \
            dbuf += type2->size;                                        \
                                                                        \
            sbuf += sizeof(type);                                       \
        }                                                               \
    } while (0)

static inline int pack_backend(const void *inbuf, void *outbuf, uintptr_t count,
                               yaksi_type_s * type, yaksi_info_s * info, yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

    switch (type->kind) {
        case YAKSI_TYPE_KIND__BUILTIN:
            assert(!type->is_contig);
            switch (type->u.builtin.handle) {
                case YAKSA_TYPE__FLOAT_INT:
                    BUILTIN_PAIRTYPE_PACK(yaksi_float_int_s, FLOAT, INT, inbuf, count, outbuf,
                                          info, request);
                    break;

                case YAKSA_TYPE__DOUBLE_INT:
                    BUILTIN_PAIRTYPE_PACK(yaksi_double_int_s, DOUBLE, INT, inbuf, count, outbuf,
                                          info, request);
                    break;

                case YAKSA_TYPE__LONG_INT:
                    BUILTIN_PAIRTYPE_PACK(yaksi_long_int_s, LONG, INT, inbuf, count, outbuf,
                                          info, request);
                    break;

                case YAKSA_TYPE__SHORT_INT:
                    BUILTIN_PAIRTYPE_PACK(yaksi_short_int_s, SHORT, INT, inbuf, count, outbuf,
                                          info, request);
                    break;

                case YAKSA_TYPE__LONG_DOUBLE_INT:
                    BUILTIN_PAIRTYPE_PACK(yaksi_long_double_int_s, LONG_DOUBLE, INT, inbuf, count,
                                          outbuf, info, request);
                    break;

                default:
                    assert(0);
            }
            break;

        case YAKSI_TYPE_KIND__HVECTOR:
            {
                const char *sbuf = (const char *) inbuf;
                char *dbuf = (char *) outbuf;
                const char *type_start = (const char *) inbuf;
                uintptr_t size = type->u.hvector.blocklength * type->u.hvector.child->size;

                for (int i = 0; i < count; i++) {
                    for (int j = 0; j < type->u.hvector.count; j++) {
                        rc = yaksi_ipack_backend(sbuf, dbuf, type->u.hvector.blocklength,
                                                 type->u.hvector.child, info, request);
                        YAKSU_ERR_CHECK(rc, fn_fail);
                        sbuf += type->u.hvector.stride;
                        dbuf += size;
                    }

                    type_start += type->extent;
                    sbuf = type_start;
                }
            }
            break;

        case YAKSI_TYPE_KIND__BLKHINDX:
            {
                const char *sbuf;
                char *dbuf = (char *) outbuf;
                uintptr_t size = type->u.blkhindx.blocklength * type->u.blkhindx.child->size;

                for (int i = 0; i < count; i++) {
                    for (int j = 0; j < type->u.blkhindx.count; j++) {
                        sbuf =
                            (const char *) inbuf + i * type->extent +
                            type->u.blkhindx.array_of_displs[j];
                        rc = yaksi_ipack_backend(sbuf, dbuf, type->u.blkhindx.blocklength,
                                                 type->u.blkhindx.child, info, request);
                        YAKSU_ERR_CHECK(rc, fn_fail);
                        dbuf += size;
                    }
                }
            }
            break;

        case YAKSI_TYPE_KIND__HINDEXED:
            {
                const char *sbuf;
                char *dbuf = (char *) outbuf;

                for (int i = 0; i < count; i++) {
                    for (int j = 0; j < type->u.hindexed.count; j++) {
                        if (type->u.hindexed.array_of_blocklengths[j] == 0)
                            continue;

                        sbuf =
                            (const char *) inbuf + i * type->extent +
                            type->u.hindexed.array_of_displs[j];
                        rc = yaksi_ipack_backend(sbuf, dbuf,
                                                 type->u.hindexed.array_of_blocklengths[j],
                                                 type->u.hindexed.child, info, request);
                        YAKSU_ERR_CHECK(rc, fn_fail);
                        dbuf +=
                            type->u.hindexed.array_of_blocklengths[j] *
                            type->u.hindexed.child->size;
                    }
                }
            }
            break;

        case YAKSI_TYPE_KIND__STRUCT:
            {
                const char *sbuf;
                char *dbuf = (char *) outbuf;

                for (int i = 0; i < count; i++) {
                    for (int j = 0; j < type->u.str.count; j++) {
                        if (type->u.str.array_of_blocklengths[j] == 0)
                            continue;

                        sbuf =
                            (const char *) inbuf + i * type->extent +
                            type->u.str.array_of_displs[j];
                        rc = yaksi_ipack_backend(sbuf, dbuf, type->u.str.array_of_blocklengths[j],
                                                 type->u.str.array_of_types[j], info, request);
                        YAKSU_ERR_CHECK(rc, fn_fail);
                        dbuf +=
                            type->u.str.array_of_blocklengths[j] *
                            type->u.str.array_of_types[j]->size;
                    }
                }
                break;
            }

        case YAKSI_TYPE_KIND__RESIZED:
            {
                const char *sbuf = (const char *) inbuf;
                char *dbuf = (char *) outbuf;
                for (int i = 0; i < count; i++) {
                    rc = yaksi_ipack_backend(sbuf, dbuf, 1, type->u.resized.child, info, request);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    sbuf += type->extent;
                    dbuf += type->u.resized.child->size;
                }
            }
            break;

        case YAKSI_TYPE_KIND__CONTIG:
            {
                rc = yaksi_ipack_backend(inbuf, outbuf, count * type->u.contig.count,
                                         type->u.contig.child, info, request);
                YAKSU_ERR_CHECK(rc, fn_fail);
            }
            break;

        case YAKSI_TYPE_KIND__SUBARRAY:
            {
                const char *sbuf =
                    (const char *) inbuf + type->true_lb - type->u.subarray.primary->true_lb;
                rc = yaksi_ipack_backend(sbuf, outbuf, count, type->u.subarray.primary, info,
                                         request);
                YAKSU_ERR_CHECK(rc, fn_fail);
            }
            break;

        default:
            assert(0);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksi_ipack_backend(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                        yaksi_info_s * info, yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksur_ipack(inbuf, outbuf, count, type, info, request);
    if (rc == YAKSA_ERR__NOT_SUPPORTED) {
        rc = pack_backend(inbuf, outbuf, count, type, info, request);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
