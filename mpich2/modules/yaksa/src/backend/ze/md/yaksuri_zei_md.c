/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri_zei.h"
#include <stdlib.h>
#include <assert.h>
#include <level_zero/ze_api.h>
#include <string.h>

static yaksuri_zei_md_s *type_to_md(yaksi_type_s * type, int dev_id)
{
    yaksuri_zei_type_s *ze = type->backend.ze.priv;

    return ze->md[dev_id];
}

int yaksuri_zei_md_alloc(yaksi_type_s * type, int dev_id)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_zei_type_s *ze = (yaksuri_zei_type_s *) type->backend.ze.priv;
    ze_result_t zerr;

    ze_host_mem_alloc_desc_t host_desc = {
        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
        .pNext = NULL,
        .flags = 0,
    };

    ze_device_mem_alloc_desc_t device_desc = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        .pNext = NULL,
        .flags = 0,
        .ordinal = 0,
    };

    pthread_mutex_lock(&ze->mdmutex);

    assert(type->kind != YAKSI_TYPE_KIND__STRUCT);
    assert(type->kind != YAKSI_TYPE_KIND__SUBARRAY);

    /* if the metadata is already allocated, return */
    if (ze->md && ze->md[dev_id]) {
        goto fn_exit;
    } else {
        if (ze->md == NULL)
            ze->md =
                (yaksuri_zei_md_s **) calloc(yaksuri_zei_global.ndevices,
                                             sizeof(yaksuri_zei_md_s *));
        zerr =
            zeMemAllocShared(yaksuri_zei_global.context, &device_desc, &host_desc,
                             sizeof(yaksuri_zei_md_s), 1, yaksuri_zei_global.device[dev_id],
                             (void **) &ze->md[dev_id]);
        YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);
    }

    switch (type->kind) {
        case YAKSI_TYPE_KIND__BUILTIN:
            break;

        case YAKSI_TYPE_KIND__DUP:
            rc = yaksuri_zei_md_alloc(type->u.dup.child, dev_id);
            YAKSU_ERR_CHECK(rc, fn_fail);
            ze->md[dev_id]->u.dup.child = type_to_md(type->u.dup.child, dev_id);
            break;

        case YAKSI_TYPE_KIND__RESIZED:
            rc = yaksuri_zei_md_alloc(type->u.resized.child, dev_id);
            YAKSU_ERR_CHECK(rc, fn_fail);
            ze->md[dev_id]->u.resized.child = type_to_md(type->u.resized.child, dev_id);
            break;

        case YAKSI_TYPE_KIND__HVECTOR:
            ze->md[dev_id]->u.hvector.count = type->u.hvector.count;
            ze->md[dev_id]->u.hvector.blocklength = type->u.hvector.blocklength;
            ze->md[dev_id]->u.hvector.stride = type->u.hvector.stride;

            rc = yaksuri_zei_md_alloc(type->u.hvector.child, dev_id);
            YAKSU_ERR_CHECK(rc, fn_fail);
            ze->md[dev_id]->u.hvector.child = type_to_md(type->u.hvector.child, dev_id);
            break;

        case YAKSI_TYPE_KIND__BLKHINDX:
            ze->md[dev_id]->u.blkhindx.count = type->u.blkhindx.count;
            ze->md[dev_id]->u.blkhindx.blocklength = type->u.blkhindx.blocklength;

            zerr = zeMemAllocShared(yaksuri_zei_global.context, &device_desc, &host_desc,
                                    type->u.blkhindx.count * sizeof(intptr_t), 1,
                                    yaksuri_zei_global.device[dev_id],
                                    (void **) &ze->md[dev_id]->u.blkhindx.array_of_displs);
            YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);

            memcpy(ze->md[dev_id]->u.blkhindx.array_of_displs, type->u.blkhindx.array_of_displs,
                   type->u.blkhindx.count * sizeof(intptr_t));

            rc = yaksuri_zei_md_alloc(type->u.blkhindx.child, dev_id);
            YAKSU_ERR_CHECK(rc, fn_fail);
            ze->md[dev_id]->u.blkhindx.child = type_to_md(type->u.blkhindx.child, dev_id);
            break;

        case YAKSI_TYPE_KIND__HINDEXED:
            ze->md[dev_id]->u.hindexed.count = type->u.hindexed.count;

            zerr = zeMemAllocShared(yaksuri_zei_global.context, &device_desc, &host_desc,
                                    type->u.hindexed.count * sizeof(intptr_t), 1,
                                    yaksuri_zei_global.device[dev_id],
                                    (void **) &ze->md[dev_id]->u.hindexed.array_of_displs);
            YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);

            memcpy(ze->md[dev_id]->u.hindexed.array_of_displs, type->u.hindexed.array_of_displs,
                   type->u.hindexed.count * sizeof(intptr_t));

            zerr = zeMemAllocShared(yaksuri_zei_global.context, &device_desc, &host_desc,
                                    type->u.hindexed.count * sizeof(int), 1,
                                    yaksuri_zei_global.device[dev_id],
                                    (void **) &ze->md[dev_id]->u.hindexed.array_of_blocklengths);
            YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);

            memcpy(ze->md[dev_id]->u.hindexed.array_of_blocklengths,
                   type->u.hindexed.array_of_blocklengths, type->u.hindexed.count * sizeof(int));

            rc = yaksuri_zei_md_alloc(type->u.hindexed.child, dev_id);
            YAKSU_ERR_CHECK(rc, fn_fail);
            ze->md[dev_id]->u.hindexed.child = type_to_md(type->u.hindexed.child, dev_id);
            break;

        case YAKSI_TYPE_KIND__CONTIG:
            ze->md[dev_id]->u.contig.count = type->u.contig.count;
            ze->md[dev_id]->u.contig.stride = type->u.contig.child->extent;

            rc = yaksuri_zei_md_alloc(type->u.contig.child, dev_id);
            YAKSU_ERR_CHECK(rc, fn_fail);
            ze->md[dev_id]->u.contig.child = type_to_md(type->u.contig.child, dev_id);
            break;

        default:
            assert(0);
    }

    ze->md[dev_id]->extent = type->extent;
    ze->md[dev_id]->num_elements = ze->num_elements;
    ze->md[dev_id]->true_lb = type->true_lb;

  fn_exit:
    pthread_mutex_unlock(&ze->mdmutex);
    return rc;
  fn_fail:
    goto fn_exit;
}
