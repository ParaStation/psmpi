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

#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

#ifndef _EFA_CNTR_H_
#define _EFA_CNTR_H_

struct efa_cntr {
	struct util_cntr util_cntr;
	struct fid_cntr *shm_cntr;
};

int efa_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr_fid, void *context);

void efa_cntr_report_tx_completion(struct util_ep *ep, uint64_t flags);

void efa_cntr_report_rx_completion(struct util_ep *ep, uint64_t flags);

void efa_cntr_report_error(struct util_ep *ep, uint64_t flags);

static inline
void *efa_cntr_get_srx_ctx(struct fid_cntr *cntr_fid)
{
	struct efa_cntr *efa_cntr;
	struct fid_peer_srx *srx = NULL;

	efa_cntr = container_of(cntr_fid, struct efa_cntr, util_cntr.cntr_fid);

	srx = efa_cntr->util_cntr.domain->srx;
	if (!srx)
		return NULL;

	return srx->ep_fid.fid.context;
}

#endif

