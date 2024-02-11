/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef PMISERV_PMI_H_INCLUDED
#define PMISERV_PMI_H_INCLUDED

#include "hydra.h"
#include "demux.h"
#include "pmiserv_common.h"
#include "uthash.h"
#include "utarray.h"

struct HYD_pmcd_kvs {
    char *key;
    char *val;
    UT_hash_handle hh;
};

struct HYD_pmcd_pmi_pg_scratch {
    /* PMI-1's PMI_Barrier is blocking, thus a single barrier_count works */
    int barrier_count;
    /* PMI-2's PMI2_KVS_Fence is non-blocking, thus need track epoch */
    int epoch;
    int fence_count;
    struct HYD_pmcd_pmi_ecount {
        int process_fd;
        int epoch;
    } *ecount;

    int pmi_listen_fd;

    char *dead_processes;
    int dead_process_count;

    char kvsname[PMI_MAXKVSLEN];
    struct HYD_pmcd_kvs *kvs;
    /* an array of kvs entries (keys) that needs to be pushed to proxies at barrier_out */
    UT_array *kvs_batch;
};

HYD_status HYD_pmcd_pmi_finalize(void);

HYD_status HYD_pmiserv_pmi_reply(struct HYD_proxy *proxy, int process_fd, struct PMIU_cmd *pmi);

HYD_status HYD_pmiserv_add_kvs(struct HYD_pmcd_pmi_pg_scratch *pg_scratch,
                               const char *key, const char *val);
HYD_status HYD_pmiserv_bcast_keyvals(struct HYD_proxy *proxy, int process_fd);
HYD_status HYD_pmiserv_epoch_init(struct HYD_pg *pg);
HYD_status HYD_pmiserv_epoch_free(struct HYD_pg *pg);

HYD_status HYD_pmiserv_kvs_get(struct HYD_proxy *proxy, int process_fd, int pgid,
                               struct PMIU_cmd *pmi, bool sync);
HYD_status HYD_pmiserv_kvs_put(struct HYD_proxy *proxy, int process_fd, int pgid,
                               struct PMIU_cmd *pmi);
HYD_status HYD_pmiserv_kvs_mput(struct HYD_proxy *proxy, int process_fd, int pgid,
                                struct PMIU_cmd *pmi);
HYD_status HYD_pmiserv_kvs_fence(struct HYD_proxy *proxy, int process_fd, int pgid,
                                 struct PMIU_cmd *pmi);

HYD_status HYD_pmiserv_barrier(struct HYD_proxy *proxy, int process_fd, int pgid,
                               struct PMIU_cmd *pmi);
HYD_status HYD_pmiserv_abort(struct HYD_proxy *proxy, int process_fd, int pgid,
                             struct PMIU_cmd *pmi);

HYD_status HYD_pmiserv_spawn(struct HYD_proxy *proxy, int process_fd, int pgid,
                             struct PMIU_cmd *pmi);
HYD_status HYD_pmiserv_publish(struct HYD_proxy *proxy, int process_fd, int pgid,
                               struct PMIU_cmd *pmi);
HYD_status HYD_pmiserv_unpublish(struct HYD_proxy *proxy, int process_fd, int pgid,
                                 struct PMIU_cmd *pmi);
HYD_status HYD_pmiserv_lookup(struct HYD_proxy *proxy, int process_fd, int pgid,
                              struct PMIU_cmd *pmi);

#endif /* PMISERV_PMI_H_INCLUDED */
