/*
 * Copyright (c) Amazon.com, Inc. or its affiliates.
 * All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "efa_rdm_util.h"

/**
 * @brief Fetch the initial value for use_device_rdma
 *
 * This function fetches the value of FI_EFA_USE_DEVICE_RDMA from the
 * user's environment.  If not set, it uses the API version and the EFA
 * hardware's version and capabilities to decide when to enable or
 * disable use_device_rdma during endpoint initialization.
 *
 * This value can be modified per-endpoint by using fi_setopt if the
 * application uses API>=1.18.
 *
 * The application may abort() during this method for two reasons:
 *  - If the Environment variable is set with non-boolean-like value
 *  - If the Environment variable requests RDMA but no hardware support
 *    is available.
 *
 * @return	bool: use_device_rdma default or environment setting.
 */
bool efa_rdm_get_use_device_rdma(uint32_t fabric_api_version)
{
	int ret;
	int param_val;
	bool hw_support;
	bool default_val;
	uint32_t vendor_part_id;

	vendor_part_id = g_device_list[0].ibv_attr.vendor_part_id;
	hw_support = efa_device_support_rdma_read();

	if (FI_VERSION_GE(fabric_api_version, FI_VERSION(1,18))) {
		default_val = hw_support;
	} else {
		if (vendor_part_id == 0xefa0 || vendor_part_id == 0xefa1) {
			default_val = false;
		} else {
			default_val = true;
		}

		if (default_val && !hw_support) {
			fprintf(stderr, "EFA device with vendor id %x unexpectedly has "
				"no RDMA support. Application will abort().\n",
				vendor_part_id);
			abort();
		}
	}
	param_val = default_val;

	/* Fetch the value of environment variable set by the user if any. */
	ret = fi_param_get_bool(&efa_prov, "use_device_rdma", &param_val);
	if (ret == -EINVAL) {
		fprintf(stderr, "FI_EFA_USE_DEVICE_RDMA was set to an invalid value by the user."
			" FI_EFA_USE_DEVICE_RDMA is boolean and can be set to only 0/false/no/off or"
			" 1/true/yes/on.  Application will abort().\n");
		abort();
	}
	if (ret < 0) return default_val;

	/* When the user requests use device RDMA but the device does not
	   support RDMA, exit the run. */
	if (param_val && !hw_support) {
		fprintf(stderr, "FI_EFA_USE_DEVICE_RDMA=1 was set by user, but "
			"EFA device has no rdma-read capability.  Application "
			"will abort().\n");
		abort();
	}

	return param_val;
}
