/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <assert.h>
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri_zei.h"
#include <level_zero/ze_api.h>

static inline void attr_convert(ze_memory_allocation_properties_t prop, ze_device_handle_t device,
                                yaksur_ptr_attr_s * attr)
{
    if (prop.type == ZE_MEMORY_TYPE_UNKNOWN) {
        attr->type = YAKSUR_PTR_TYPE__UNREGISTERED_HOST;
        attr->device = -1;
    } else if (prop.type == ZE_MEMORY_TYPE_HOST) {
        attr->type = YAKSUR_PTR_TYPE__REGISTERED_HOST;
        attr->device = -1;
    } else if (prop.type == ZE_MEMORY_TYPE_SHARED) {
        attr->type = YAKSUR_PTR_TYPE__MANAGED;
        attr->device = -1;
        for (int i = 0; i < yaksuri_zei_global.ndevices; i++)
            if (yaksuri_zei_global.device[i] == device)
                attr->device = i;
    } else if (prop.type == ZE_MEMORY_TYPE_DEVICE) {
        attr->type = YAKSUR_PTR_TYPE__GPU;
        attr->device = -1;
        for (int i = 0; i < yaksuri_zei_global.ndevices; i++)
            if (yaksuri_zei_global.device[i] == device)
                attr->device = i;
    } else
        assert(0);
}

int yaksuri_zei_get_ptr_attr(const void *inbuf, void *outbuf, yaksi_info_s * info,
                             yaksur_ptr_attr_s * inattr, yaksur_ptr_attr_s * outattr)
{
    int rc = YAKSA_SUCCESS;
    ze_result_t zerr;
    yaksuri_zei_info_s *infopriv = NULL;
    ze_memory_allocation_properties_t prop;
    ze_device_handle_t device;

    if (info) {
        infopriv = (yaksuri_zei_info_s *) info->backend.ze.priv;
    }

    if (infopriv && infopriv->inbuf.is_valid) {
        attr_convert(infopriv->inbuf.attr.prop, infopriv->inbuf.attr.device, inattr);
    } else {
        zerr = zeMemGetAllocProperties(yaksuri_zei_global.context, inbuf, &prop, &device);
        YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);
        attr_convert(prop, device, inattr);
    }

    if (infopriv && infopriv->outbuf.is_valid) {
        attr_convert(infopriv->outbuf.attr.prop, infopriv->outbuf.attr.device, outattr);
    } else {
        zerr = zeMemGetAllocProperties(yaksuri_zei_global.context, outbuf, &prop, &device);
        YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);
        attr_convert(prop, device, outattr);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
