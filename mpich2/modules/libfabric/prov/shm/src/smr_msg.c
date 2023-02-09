/*
 * Copyright (c) 2013-2021 Intel Corporation. All rights reserved
 * (C) Copyright 2021 Amazon.com, Inc. or its affiliates.
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

#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>

#include "ofi_iov.h"
#include "smr.h"


static inline uint16_t smr_convert_rx_flags(uint64_t fi_flags)
{
	uint16_t flags = 0;

	if (fi_flags & FI_COMPLETION)
		flags |= SMR_RX_COMPLETION;
	if (fi_flags & FI_MULTI_RECV)
		flags |= SMR_MULTI_RECV;

	return flags;
}

static struct smr_rx_entry *smr_get_recv_entry(struct smr_ep *ep,
		const struct iovec *iov, void **desc, size_t count, fi_addr_t addr,
		void *context, uint64_t tag, uint64_t ignore, uint64_t flags)
{
	struct smr_rx_entry *entry;

	if (ofi_cirque_isfull(ep->util_ep.rx_cq->cirq) ||
	    ofi_freestack_isempty(ep->recv_fs)) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"not enough space to post recv\n");
		return NULL;
	}

	entry = ofi_freestack_pop(ep->recv_fs);

	memcpy(&entry->iov, iov, sizeof(*iov) * count);
	entry->iov_count = count;
	entry->context = context;
	entry->err = 0;
	entry->flags = smr_convert_rx_flags(flags);
	entry->peer_id = ep->util_ep.caps & FI_DIRECTED_RECV &&
				addr != FI_ADDR_UNSPEC ?
				smr_addr_lookup(ep->util_ep.av, addr) : -1;
	entry->tag = tag;
	entry->ignore = ignore;

	entry->iface = smr_get_mr_hmem_iface(ep->util_ep.domain, desc,
					     &entry->device);

	return entry;
}

ssize_t smr_generic_recv(struct smr_ep *ep, const struct iovec *iov, void **desc,
			 size_t iov_count, fi_addr_t addr, void *context,
			 uint64_t tag, uint64_t ignore, uint64_t flags,
			 struct smr_queue *recv_queue,
			 struct smr_queue *unexp_queue)
{
	struct smr_rx_entry *entry;
	ssize_t ret = -FI_EAGAIN;

	assert(iov_count <= SMR_IOV_LIMIT);
	assert(!(flags & FI_MULTI_RECV) || iov_count == 1);

	pthread_spin_lock(&ep->region->lock);
	ofi_genlock_lock(&ep->util_ep.rx_cq->cq_lock);

	entry = smr_get_recv_entry(ep, iov, desc, iov_count, addr, context, tag,
				   ignore, flags);
	if (!entry)
		goto out;

	dlist_insert_tail(&entry->entry, &recv_queue->list);
	ret = smr_progress_unexp_queue(ep, entry, unexp_queue);
out:
	ofi_genlock_unlock(&ep->util_ep.rx_cq->cq_lock);
	pthread_spin_unlock(&ep->region->lock);
	return ret;
}

ssize_t smr_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		    uint64_t flags)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_recv(ep, msg->msg_iov, msg->desc, msg->iov_count,
				msg->addr, msg->context, 0, 0,
				flags | ep->util_ep.rx_msg_flags,
				&ep->recv_queue, &ep->unexp_msg_queue);
}

ssize_t smr_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t src_addr, void *context)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_recv(ep, iov, desc, count, src_addr, context, 0, 0,
				smr_ep_rx_flags(ep), &ep->recv_queue,
				&ep->unexp_msg_queue);
}

ssize_t smr_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
		fi_addr_t src_addr, void *context)
{
	struct iovec iov;
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	iov.iov_base = buf;
	iov.iov_len = len;

	return smr_generic_recv(ep, &iov, &desc, 1, src_addr, context, 0, 0,
				smr_ep_rx_flags(ep), &ep->recv_queue,
				&ep->unexp_msg_queue);
}

static ssize_t smr_generic_sendmsg(struct smr_ep *ep, const struct iovec *iov,
				   void **desc, size_t iov_count, fi_addr_t addr,
				   uint64_t tag, uint64_t data, void *context,
				   uint32_t op, uint64_t op_flags)
{
	struct smr_region *peer_smr;
	enum fi_hmem_iface iface;
	uint64_t device;
	int64_t id, peer_id;
	ssize_t ret = 0;
	size_t total_len;
	bool use_ipc;
	int proto;

	assert(iov_count <= SMR_IOV_LIMIT);

	id = smr_verify_peer(ep, addr);
	if (id < 0)
		return -FI_EAGAIN;

	peer_id = smr_peer_data(ep->region)[id].addr.id;
	peer_smr = smr_peer_region(ep->region, id);

	pthread_spin_lock(&peer_smr->lock);
	if (!peer_smr->cmd_cnt || smr_peer_data(ep->region)[id].sar_status) {
		ret = -FI_EAGAIN;
		goto unlock_region;
	}

	ofi_genlock_lock(&ep->util_ep.tx_cq->cq_lock);
	if (ofi_cirque_isfull(ep->util_ep.tx_cq->cirq)) {
		ret = -FI_EAGAIN;
		goto unlock_cq;
	}

	iface = smr_get_mr_hmem_iface(ep->util_ep.domain, desc, &device);

	total_len = ofi_total_iov_len(iov, iov_count);
	assert(!(op_flags & FI_INJECT) || total_len <= SMR_INJECT_SIZE);

	/* Do not inline/inject if IPC is available so device to device
	 * transfer may occur if possible. */
	use_ipc = ofi_hmem_is_ipc_enabled(iface) && (iov_count == 1) &&
		  desc && (smr_get_mr_flags(desc) & FI_HMEM_DEVICE_ONLY) &&
		  !(op_flags & FI_INJECT);

	proto = smr_select_proto(use_ipc, smr_cma_enabled(ep, peer_smr), iface,
				 op, total_len, op_flags);

	ret = smr_proto_ops[proto](ep, peer_smr, id, peer_id, op, tag, data, op_flags,
				   iface, device, iov, iov_count, total_len, context);
	if (ret)
		goto unlock_cq;

	smr_signal(peer_smr);

	if (proto != smr_src_inline && proto != smr_src_inject)
		goto unlock_cq;

	ret = smr_complete_tx(ep, context, op, op_flags, 0);
	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unable to process tx completion\n");
		goto unlock_cq;
	}

unlock_cq:
	ofi_genlock_unlock(&ep->util_ep.tx_cq->cq_lock);
unlock_region:
	pthread_spin_unlock(&peer_smr->lock);
	return ret;
}

ssize_t smr_send(struct fid_ep *ep_fid, const void *buf, size_t len, void *desc,
		fi_addr_t dest_addr, void *context)
{
	struct smr_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return smr_generic_sendmsg(ep, &msg_iov, &desc, 1, dest_addr, 0,
				   0, context, ofi_op_msg, smr_ep_tx_flags(ep));
}


ssize_t smr_sendv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t dest_addr, void *context)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_sendmsg(ep, iov, desc, count, dest_addr, 0,
				   0, context, ofi_op_msg, smr_ep_tx_flags(ep));
}

ssize_t smr_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		    uint64_t flags)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_sendmsg(ep, msg->msg_iov, msg->desc, msg->iov_count,
				   msg->addr, 0, msg->data, msg->context,
				   ofi_op_msg, flags | ep->util_ep.tx_msg_flags);
}

static ssize_t smr_generic_inject(struct fid_ep *ep_fid, const void *buf,
				  size_t len, fi_addr_t dest_addr, uint64_t tag,
				  uint64_t data, uint32_t op, uint64_t op_flags)
{
	struct smr_ep *ep;
	struct smr_region *peer_smr;
	int64_t id, peer_id;
	ssize_t ret = 0;
	struct iovec msg_iov;
	int proto;

	assert(len <= SMR_INJECT_SIZE);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	id = smr_verify_peer(ep, dest_addr);
	if (id < 0)
		return -FI_EAGAIN;

	peer_id = smr_peer_data(ep->region)[id].addr.id;
	peer_smr = smr_peer_region(ep->region, id);

	pthread_spin_lock(&peer_smr->lock);
	if (!peer_smr->cmd_cnt || smr_peer_data(ep->region)[id].sar_status) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	proto = len <= SMR_MSG_DATA_LEN ? smr_src_inline : smr_src_inject;
	ret = smr_proto_ops[proto](ep, peer_smr, id, peer_id, op, tag, data,
			op_flags, FI_HMEM_SYSTEM, 0, &msg_iov, 1, len, NULL);

	assert(!ret);
	ofi_ep_tx_cntr_inc_func(&ep->util_ep, op);

	smr_signal(peer_smr);
unlock:
	pthread_spin_unlock(&peer_smr->lock);

	return ret;
}

ssize_t smr_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
		fi_addr_t dest_addr)
{
	return smr_generic_inject(ep_fid, buf, len, dest_addr, 0, 0,
				  ofi_op_msg, 0);
}

ssize_t smr_senddata(struct fid_ep *ep_fid, const void *buf, size_t len,
		     void *desc, uint64_t data, fi_addr_t dest_addr,
		     void *context)
{
	struct smr_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return smr_generic_sendmsg(ep, &iov, &desc, 1, dest_addr, 0, data,
				   context, ofi_op_msg,
				   FI_REMOTE_CQ_DATA | smr_ep_tx_flags(ep));
}

ssize_t smr_injectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
		       uint64_t data, fi_addr_t dest_addr)
{
	return smr_generic_inject(ep_fid, buf, len, dest_addr, 0, data,
				  ofi_op_msg, FI_REMOTE_CQ_DATA);
}

struct fi_ops_msg smr_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = smr_recv,
	.recvv = smr_recvv,
	.recvmsg = smr_recvmsg,
	.send = smr_send,
	.sendv = smr_sendv,
	.sendmsg = smr_sendmsg,
	.inject = smr_inject,
	.senddata = smr_senddata,
	.injectdata = smr_injectdata,
};

ssize_t smr_trecv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context)
{
	struct iovec iov;
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	iov.iov_base = buf;
	iov.iov_len = len;

	return smr_generic_recv(ep, &iov, &desc, 1, src_addr, context, tag, ignore,
				smr_ep_rx_flags(ep), &ep->trecv_queue,
				&ep->unexp_tagged_queue);
}

ssize_t smr_trecvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
	size_t count, fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
	void *context)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_recv(ep, iov, desc, count, src_addr, context, tag, ignore,
				smr_ep_rx_flags(ep), &ep->trecv_queue,
				&ep->unexp_tagged_queue);
}

ssize_t smr_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
	uint64_t flags)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_recv(ep, msg->msg_iov, msg->desc, msg->iov_count,
				msg->addr, msg->context, msg->tag, msg->ignore,
				flags | ep->util_ep.rx_msg_flags,
				&ep->trecv_queue, &ep->unexp_tagged_queue);
}

ssize_t smr_tsend(struct fid_ep *ep_fid, const void *buf, size_t len,
	void *desc, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	struct smr_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return smr_generic_sendmsg(ep, &msg_iov, &desc, 1, dest_addr, tag,
				   0, context, ofi_op_tagged,
				   smr_ep_tx_flags(ep));
}

ssize_t smr_tsendv(struct fid_ep *ep_fid, const struct iovec *iov,
	void **desc, size_t count, fi_addr_t dest_addr, uint64_t tag,
	void *context)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_sendmsg(ep, iov, desc, count, dest_addr, tag,
				   0, context, ofi_op_tagged,
				   smr_ep_tx_flags(ep));
}

ssize_t smr_tsendmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
	uint64_t flags)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_sendmsg(ep, msg->msg_iov, msg->desc, msg->iov_count,
				   msg->addr, msg->tag, msg->data, msg->context,
				   ofi_op_tagged, flags | ep->util_ep.tx_msg_flags);
}

ssize_t smr_tinject(struct fid_ep *ep_fid, const void *buf, size_t len,
		    fi_addr_t dest_addr, uint64_t tag)
{
	return smr_generic_inject(ep_fid, buf, len, dest_addr, tag, 0,
				  ofi_op_tagged, 0);
}

ssize_t smr_tsenddata(struct fid_ep *ep_fid, const void *buf, size_t len,
		      void *desc, uint64_t data, fi_addr_t dest_addr,
		      uint64_t tag, void *context)
{
	struct smr_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return smr_generic_sendmsg(ep, &iov, &desc, 1, dest_addr, tag, data,
				   context, ofi_op_tagged,
				   FI_REMOTE_CQ_DATA | smr_ep_tx_flags(ep));
}

ssize_t smr_tinjectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
			uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	return smr_generic_inject(ep_fid, buf, len, dest_addr, tag, data,
				  ofi_op_tagged, FI_REMOTE_CQ_DATA);
}

struct fi_ops_tagged smr_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = smr_trecv,
	.recvv = smr_trecvv,
	.recvmsg = smr_trecvmsg,
	.send = smr_tsend,
	.sendv = smr_tsendv,
	.sendmsg = smr_tsendmsg,
	.inject = smr_tinject,
	.senddata = smr_tsenddata,
	.injectdata = smr_tinjectdata,
};
