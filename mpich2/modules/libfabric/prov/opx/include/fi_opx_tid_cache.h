/*
 * Copyright (C) 2022-2023 Cornelis Networks.
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
#ifndef _FI_PROV_OPX_TID_CACHE_H_
#define _FI_PROV_OPX_TID_CACHE_H_

#include "config.h"
#include <ofi_util.h>
struct opx_tid_domain;


/* @brief Setup the MR cache.
 *
 * This function enables the MR cache using the util MR cache code.
 *
 * @param cache		The ofi_mr_cache that is to be set up.
 * @param domain	The EFA domain where cache will be used.
 * @return 0 on success, fi_errno on failure.
 */
int opx_tid_cache_setup(struct ofi_mr_cache **cache,
			struct opx_tid_domain *domain);

int opx_tid_cache_add_abort();
void opx_tid_cache_delete_abort();

#define OPX_ENTRY_FOUND 0
#define OPX_ENTRY_OVERLAP 1
#define OPX_ENTRY_NOT_FOUND 2
#define OPX_ENTRY_IN_USE 3

/* Flush cache entries internal entry point */
bool opx_tid_cache_flush(struct ofi_mr_cache *cache, bool flush_lru);

/* Cleanup the cache at exit/finalize */
void opx_tid_cache_cleanup(struct ofi_mr_cache *cache);

/* De-register (lazy) a memory region on TID rendezvous completion */
void opx_deregister_for_rzv(struct fi_opx_ep *opx_ep, const uint64_t tid_vaddr,
			    const int64_t tid_length);

/* forward declaration of parameter structure */
struct fi_opx_hfi1_rx_rzv_rts_params;

/* Register a memory region for TID rendezvous,
 * return 0 on success
 * returns non-zero on failure (fallback to Eager rendezvous)
 */
int opx_register_for_rzv(struct fi_opx_hfi1_rx_rzv_rts_params *params,
			 const uint64_t tid_vaddr, const uint64_t tid_length);

#endif /* _FI_PROV_OPX_TID_CACHE_H_ */
