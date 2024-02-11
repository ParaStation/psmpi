/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "hydra_server.h"
#include "pmiserv_pmi.h"
#include "pmiserv_utils.h"

HYD_status HYD_pmcd_pmi_finalize(void)
{
    HYD_status status = HYD_SUCCESS;

    HYDU_FUNC_ENTER();

    HYDU_FUNC_EXIT();
    return status;
}

HYD_status HYD_pmiserv_pmi_reply(struct HYD_proxy * proxy, int process_fd, struct PMIU_cmd * pmi)
{
    struct HYD_pmcd_hdr hdr;
    HYD_pmcd_init_header(&hdr);
    hdr.cmd = CMD_PMI_RESPONSE;
    hdr.pgid = proxy->pgid;
    hdr.proxy_id = proxy->proxy_id;
    hdr.u.pmi.process_fd = process_fd;
    return HYD_pmcd_pmi_send(proxy->control_fd, pmi, &hdr, HYD_server_info.user_global.debug);
}

HYD_status HYD_pmiserv_add_kvs(struct HYD_pmcd_pmi_pg_scratch * pg_scratch,
                               const char *key, const char *val)
{
    HYD_status status = HYD_SUCCESS;

    struct HYD_pmcd_kvs *s;
    HASH_FIND_STR(pg_scratch->kvs, key, s);
    if (s) {
        if (HYD_server_info.user_global.debug) {
            HYDU_dump(stdout, "add to kvs: duplicate key %s - replacing value %s with %s.\n",
                      key, s->val, val);
        }
        MPL_free(s->val);
        s->val = MPL_strdup(val);
    } else {
        s = MPL_malloc(sizeof(*s), MPL_MEM_OTHER);
        s->key = MPL_strdup(key);
        s->val = MPL_strdup(val);
        HASH_ADD_KEYPTR(hh, pg_scratch->kvs, s->key, strlen(s->key), s, MPL_MEM_OTHER);
    }
    utarray_push_back(pg_scratch->kvs_batch, &(s->key), MPL_MEM_OTHER);

    return status;
}

HYD_status HYD_pmiserv_bcast_keyvals(struct HYD_proxy * proxy, int process_fd)
{
    HYD_status status = HYD_SUCCESS;
    HYDU_FUNC_ENTER();

    struct HYD_pg *pg;
    struct HYD_pmcd_pmi_pg_scratch *pg_scratch;
    pg = PMISERV_pg_by_id(proxy->pgid);
    pg_scratch = (struct HYD_pmcd_pmi_pg_scratch *) pg->pg_scratch;

    struct PMIU_cmd pmi;
    int arg_count = 0, num_total = 0;

    const char **p = NULL;
    while (true) {
        p = (const char **) utarray_next(pg_scratch->kvs_batch, p);
        if (p) {
            struct HYD_pmcd_kvs *s;
            HASH_FIND_STR(pg_scratch->kvs, *p, s);
            HYDU_ASSERT(s, status);

            if (arg_count == 0) {
                PMIU_msg_set_query(&pmi, PMIU_WIRE_V1, PMIU_CMD_KVSCACHE, false /* not static */);
                arg_count = 1;
            }
            PMIU_cmd_add_str(&pmi, s->key, s->val);
            arg_count++;
            num_total++;
        }

        if (arg_count >= MAX_PMI_ARGS || (!p && arg_count > 1)) {
            for (int i = 0; i < pg->proxy_count; i++) {
                status = HYD_pmiserv_pmi_reply(&pg->proxy_list[i], process_fd, &pmi);
                HYDU_ERR_POP(status, "error writing PMI line\n");
            }
            arg_count = 0;
        }

        if (!p) {
            break;
        }
    }
    utarray_clear(pg_scratch->kvs_batch);

    if (num_total > 0) {
        PMIU_cmd_free_buf(&pmi);
    }

  fn_exit:
    HYDU_FUNC_EXIT();
    return status;

  fn_fail:
    goto fn_exit;
}
