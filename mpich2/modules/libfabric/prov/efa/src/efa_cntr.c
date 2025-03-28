/*
 * Copyright (c) 2013-2018 Intel Corporation. All rights reserved.
 * Copyright (c) 2019 Amazon.com, Inc. or its affiliates.
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

#include "ofi_util.h"
#include "efa.h"
#include "efa_cntr.h"

static int efa_cntr_wait(struct fid_cntr *cntr_fid, uint64_t threshold, int timeout)
{
	struct util_cntr *cntr;
	uint64_t start, errcnt;
	int ret;
	int numtry = 5;
	int tryid = 0;
	int waitim = 1;
	struct util_srx_ctx *srx_ctx;

	srx_ctx = efa_cntr_get_srx_ctx(cntr_fid);

	if (srx_ctx)
		ofi_genlock_lock(srx_ctx->lock);

	cntr = container_of(cntr_fid, struct util_cntr, cntr_fid);
	assert(cntr->wait);
	errcnt = ofi_atomic_get64(&cntr->err);
	start = (timeout >= 0) ? ofi_gettime_ms() : 0;

	for (tryid = 0; tryid < numtry; ++tryid) {
		cntr->progress(cntr);
		if (threshold <= ofi_atomic_get64(&cntr->cnt)) {
		        ret = FI_SUCCESS;
			goto unlock;
		}

		if (errcnt != ofi_atomic_get64(&cntr->err)) {
			ret = -FI_EAVAIL;
			goto unlock;
		}

		if (timeout >= 0) {
			timeout -= (int)(ofi_gettime_ms() - start);
			if (timeout <= 0) {
				ret = -FI_ETIMEDOUT;
				goto unlock;
			}
		}

		ret = fi_wait(&cntr->wait->wait_fid, waitim);
		if (ret == -FI_ETIMEDOUT)
			ret = 0;

		waitim *= 2;
	}

unlock:
	if (srx_ctx)
		ofi_genlock_unlock(srx_ctx->lock);
	return ret;
}

static uint64_t efa_cntr_read(struct fid_cntr *cntr_fid)
{
	struct util_srx_ctx *srx_ctx;
	struct efa_cntr *efa_cntr;
	uint64_t ret;

	efa_cntr = container_of(cntr_fid, struct efa_cntr, util_cntr.cntr_fid);

	srx_ctx = efa_cntr_get_srx_ctx(cntr_fid);

	if (srx_ctx)
		ofi_genlock_lock(srx_ctx->lock);

	if (efa_cntr->shm_cntr)
		fi_cntr_read(efa_cntr->shm_cntr);
	ret = ofi_cntr_read(cntr_fid);

	if (srx_ctx)
		ofi_genlock_unlock(srx_ctx->lock);

	return ret;
}

static uint64_t efa_cntr_readerr(struct fid_cntr *cntr_fid)
{
	struct util_srx_ctx *srx_ctx;
	struct efa_cntr *efa_cntr;
	uint64_t ret;

	efa_cntr = container_of(cntr_fid, struct efa_cntr, util_cntr.cntr_fid);

	srx_ctx = efa_cntr_get_srx_ctx(cntr_fid);

	if (srx_ctx)
		ofi_genlock_lock(srx_ctx->lock);
	if (efa_cntr->shm_cntr)
		fi_cntr_read(efa_cntr->shm_cntr);
	ret = ofi_cntr_readerr(cntr_fid);

	if (srx_ctx)
		ofi_genlock_unlock(srx_ctx->lock);

	return ret;
}

static struct fi_ops_cntr efa_cntr_ops = {
	.size = sizeof(struct fi_ops_cntr),
	.read = efa_cntr_read,
	.readerr = efa_cntr_readerr,
	.add = ofi_cntr_add,
	.adderr = ofi_cntr_adderr,
	.set = ofi_cntr_set,
	.seterr = ofi_cntr_seterr,
	.wait = efa_cntr_wait
};

static int efa_cntr_close(struct fid *fid)
{
	struct efa_cntr *cntr;
	int ret, retv;

	retv = 0;
	cntr = container_of(fid, struct efa_cntr, util_cntr.cntr_fid.fid);

	if (cntr->shm_cntr) {
		ret = fi_close(&cntr->shm_cntr->fid);
		if (ret) {
			EFA_WARN(FI_LOG_CNTR, "Unable to close shm cntr: %s\n", fi_strerror(-ret));
			retv = ret;
		}
	}

	ret = ofi_cntr_cleanup(&cntr->util_cntr);
	if (ret)
		return ret;
	free(cntr);
	return retv;
}

static struct fi_ops efa_cntr_fi_ops = {
	.size = sizeof(efa_cntr_fi_ops),
	.close = efa_cntr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int efa_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr_fid, void *context)
{
	int ret;
	struct efa_cntr *cntr;
	struct efa_domain *efa_domain;
	struct fi_cntr_attr shm_cntr_attr = {0};
	struct fi_peer_cntr_context peer_cntr_context = {0};

	cntr = calloc(1, sizeof(*cntr));
	if (!cntr)
		return -FI_ENOMEM;

	efa_domain = container_of(domain, struct efa_domain,
				  util_domain.domain_fid);

	ret = ofi_cntr_init(&efa_prov, domain, attr, &cntr->util_cntr,
			    &ofi_cntr_progress, context);
	if (ret)
		goto free;

	*cntr_fid = &cntr->util_cntr.cntr_fid;
	cntr->util_cntr.cntr_fid.ops = &efa_cntr_ops;
	cntr->util_cntr.cntr_fid.fid.ops = &efa_cntr_fi_ops;

	/* open shm cntr as peer cntr */
	if (efa_domain->shm_domain) {
		memcpy(&shm_cntr_attr, attr, sizeof(*attr));
		shm_cntr_attr.flags |= FI_PEER;
		peer_cntr_context.size = sizeof(peer_cntr_context);
		peer_cntr_context.cntr = cntr->util_cntr.peer_cntr;
		ret = fi_cntr_open(efa_domain->shm_domain, &shm_cntr_attr,
				   &cntr->shm_cntr, &peer_cntr_context);
		if (ret) {
			EFA_WARN(FI_LOG_CNTR, "Unable to open shm cntr, err: %s\n", fi_strerror(-ret));
			goto free;
		}
	}

	return FI_SUCCESS;

free:
	free(cntr);
	return ret;
}

void efa_cntr_report_tx_completion(struct util_ep *ep, uint64_t flags)
{
	struct util_cntr *cntr;

	flags &= (FI_SEND | FI_WRITE | FI_READ);
	assert(flags == FI_SEND || flags == FI_WRITE || flags == FI_READ);

	if (flags == FI_SEND)
		cntr = ep->cntrs[CNTR_TX];
	else if (flags == FI_WRITE)
		cntr = ep->cntrs[CNTR_WR];
	else if (flags == FI_READ)
		cntr = ep->cntrs[CNTR_RD];
	else
		cntr = NULL;

	if (cntr)
		cntr->cntr_fid.ops->add(&cntr->cntr_fid, 1);
}

void efa_cntr_report_rx_completion(struct util_ep *ep, uint64_t flags)
{
	struct util_cntr *cntr;

	flags &= (FI_RECV | FI_REMOTE_WRITE | FI_REMOTE_READ);
	assert(flags == FI_RECV || flags == FI_REMOTE_WRITE || flags == FI_REMOTE_READ);

	if (flags == FI_RECV)
		cntr = ep->cntrs[CNTR_RX];
	else if (flags == FI_REMOTE_READ)
		cntr = ep->cntrs[CNTR_REM_RD];
	else if (flags == FI_REMOTE_WRITE)
		cntr = ep->cntrs[CNTR_REM_WR];
	else
		cntr = NULL;

	if (cntr)
		cntr->cntr_fid.ops->add(&cntr->cntr_fid, 1);
}

void efa_cntr_report_error(struct util_ep *ep, uint64_t flags)
{
	flags = flags & (FI_SEND | FI_READ | FI_WRITE | FI_ATOMIC |
			 FI_RECV | FI_REMOTE_READ | FI_REMOTE_WRITE);

	struct util_cntr *cntr;

	if (flags == FI_WRITE || flags == FI_ATOMIC)
		cntr = ep->cntrs[CNTR_WR];
	else if (flags == FI_READ)
		cntr = ep->cntrs[CNTR_RD];
	else if (flags == FI_SEND)
		cntr = ep->cntrs[CNTR_TX];
	else if (flags == FI_RECV)
		cntr = ep->cntrs[CNTR_RX];
	else if (flags == FI_REMOTE_READ)
		cntr = ep->cntrs[CNTR_REM_RD];
	else if (flags == FI_REMOTE_WRITE)
		cntr = ep->cntrs[CNTR_REM_WR];
	else
		cntr = NULL;

	if (cntr)
		cntr->cntr_fid.ops->adderr(&cntr->cntr_fid, 1);
}

