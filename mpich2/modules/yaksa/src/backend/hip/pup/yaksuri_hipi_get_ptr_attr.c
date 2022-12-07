/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri_hipi.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

static void attr_convert(struct hipPointerAttribute_t cattr, yaksur_ptr_attr_s * attr)
{
    if (cattr.memoryType == hipMemoryTypeHost) {
        attr->type = YAKSUR_PTR_TYPE__REGISTERED_HOST;
        attr->device = -1;
    } else if (cattr.isManaged) {
        attr->type = YAKSUR_PTR_TYPE__MANAGED;
        attr->device = -1;
    } else if (cattr.memoryType == hipMemoryTypeDevice) {
        attr->type = YAKSUR_PTR_TYPE__GPU;
        attr->device = cattr.device;
    } else {
        attr->type = YAKSUR_PTR_TYPE__UNREGISTERED_HOST;
        attr->device = -1;
    }
}

int yaksuri_hipi_get_ptr_attr(const void *inbuf, void *outbuf, yaksi_info_s * info,
                              yaksur_ptr_attr_s * inattr, yaksur_ptr_attr_s * outattr)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_hipi_info_s *infopriv;

    if (info) {
        infopriv = (yaksuri_hipi_info_s *) info->backend.hip.priv;
    } else {
        infopriv = NULL;
    }

    if (infopriv && infopriv->inbuf.is_valid) {
        attr_convert(infopriv->inbuf.attr, inattr);
    } else {
        struct hipPointerAttribute_t attr;
        hipError_t cerr = hipPointerGetAttributes(&attr, inbuf);
        if (cerr == hipErrorInvalidValue) {
            /* attr.memoryType = hipMemoryTypeUnregistered;  */
            /* HIP does not seem to have something corresponding to cudaMemoryTypeUnregistered */
            attr.memoryType = -1;
            attr.device = -1;
            cerr = hipSuccess;
        }
        YAKSURI_HIPI_HIP_ERR_CHKANDJUMP(cerr, rc, fn_fail);
        attr_convert(attr, inattr);
    }

    if (infopriv && infopriv->outbuf.is_valid) {
        attr_convert(infopriv->outbuf.attr, outattr);
    } else {
        struct hipPointerAttribute_t attr;
        hipError_t cerr = hipPointerGetAttributes(&attr, outbuf);
        if (cerr == hipErrorInvalidValue) {
            /* attr.memoryType = hipMemoryTypeUnregistered; */
            /* HIP does not seem to have something corresponding to cudaMemoryTypeUnregistered */
            attr.memoryType = -1;
            attr.device = -1;
            cerr = hipSuccess;
        }
        YAKSURI_HIPI_HIP_ERR_CHKANDJUMP(cerr, rc, fn_fail);
        attr_convert(attr, outattr);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
