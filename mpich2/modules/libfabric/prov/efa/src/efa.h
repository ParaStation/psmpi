/*
 * Copyright (c) 2018-2022 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#ifndef EFA_H
#define EFA_H

#include "config.h"

#include <asm/types.h>
#include <errno.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <sys/epoll.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#include <infiniband/verbs.h>
#include <infiniband/efadv.h>

#include "ofi.h"
#include "ofi_enosys.h"
#include "ofi_list.h"
#include "ofi_util.h"
#include "ofi_file.h"

#include "efa_base_ep.h"
#include "efa_mr.h"
#include "efa_shm.h"
#include "efa_hmem.h"
#include "efa_device.h"
#include "efa_domain.h"
#include "efa_errno.h"
#include "efa_user_info.h"
#include "efa_fork_support.h"
#include "rdm/efa_rdm_peer.h"
#include "rdm/efa_rdm_util.h"
#include "rdm/rxr.h"

#define EFA_ABI_VER_MAX_LEN 8

#define EFA_EP_TYPE_IS_RDM(_info) \
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_RDM))

#define EFA_EP_TYPE_IS_DGRAM(_info) \
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_DGRAM))

#define EFA_DGRAM_CONNID (0x0)

#define EFA_DEF_POOL_ALIGNMENT (8)
#define EFA_MEM_ALIGNMENT (64)

#define EFA_DEF_CQ_SIZE 1024


#define EFA_DEFAULT_RUNT_SIZE (307200)
#define EFA_DEFAULT_INTER_MAX_MEDIUM_MESSAGE_SIZE (65536)
#define EFA_DEFAULT_INTER_MIN_READ_MESSAGE_SIZE (1048576)
#define EFA_DEFAULT_INTER_MIN_READ_WRITE_SIZE (65536)
#define EFA_DEFAULT_INTRA_MAX_GDRCOPY_FROM_DEV_SIZE (3072)

struct efa_fabric {
	struct util_fabric	util_fabric;
	struct fid_fabric *shm_fabric;
#ifdef EFA_PERF_ENABLED
	struct ofi_perfset perf_set;
#endif
};

static inline
int efa_str_to_ep_addr(const char *node, const char *service, struct efa_ep_addr *addr)
{
	int ret;

	if (!node)
		return -FI_EINVAL;

	memset(addr, 0, sizeof(*addr));

	ret = inet_pton(AF_INET6, node, addr->raw);
	if (ret != 1)
		return -FI_EINVAL;
	if (service)
		addr->qpn = atoi(service);

	return 0;
}

static inline
bool efa_is_same_addr(struct efa_ep_addr *lhs, struct efa_ep_addr *rhs)
{
	return !memcmp(lhs->raw, rhs->raw, sizeof(lhs->raw)) &&
	       lhs->qpn == rhs->qpn && lhs->qkey == rhs->qkey;
}

int efa_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric_fid,
	       void *context);

/* Performance counter declarations */
#ifdef EFA_PERF_ENABLED
#define EFA_PERF_FOREACH(DECL)	\
	DECL(perf_efa_tx),	\
	DECL(perf_efa_recv),	\
	DECL(efa_perf_size)	\

enum efa_perf_counters {
	EFA_PERF_FOREACH(OFI_ENUM_VAL)
};

extern const char *efa_perf_counters_str[];

static inline void efa_perfset_start(struct rxr_ep *ep, size_t index)
{
	struct efa_domain *domain = rxr_ep_domain(ep);
	struct efa_fabric *fabric = container_of(domain->util_domain.fabric,
						 struct efa_fabric,
						 util_fabric);
	ofi_perfset_start(&fabric->perf_set, index);
}

static inline void efa_perfset_end(struct rxr_ep *ep, size_t index)
{
	struct efa_domain *domain = rxr_ep_domain(ep);
	struct efa_fabric *fabric = container_of(domain->util_domain.fabric,
						 struct efa_fabric,
						 util_fabric);
	ofi_perfset_end(&fabric->perf_set, index);
}
#else
#define efa_perfset_start(ep, index) do {} while (0)
#define efa_perfset_end(ep, index) do {} while (0)
#endif

#endif /* EFA_H */
