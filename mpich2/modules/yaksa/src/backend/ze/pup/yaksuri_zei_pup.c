/*
* Copyright (C) by Argonne National Laboratory
*     See COPYRIGHT in top-level directory
*/

#include <string.h>
#include <assert.h>
#include <level_zero/ze_api.h>
#include "yaksi.h"
#include "yaksuri_zei.h"
#include <stdlib.h>

#define THREAD_BLOCK_SIZE  (256)
#define MAX_GRIDSZ_X       ((1ULL << 31) - 1)
#define MAX_GRIDSZ_Y       (65535)
#define MAX_GRIDSZ_Z       (65535)

#define MAX_IOV_LENGTH     (8192)

void recycle_command_list(ze_command_list_handle_t cl, int dev_id)
{
    yaksuri_zei_device_state_s *device_state = yaksuri_zei_global.device_states + dev_id;
    pthread_mutex_lock(&device_state->cl_mutex);
    if (device_state->cl_pool_cnt + 1 == device_state->cl_pool_size) {
        device_state->cl_pool = realloc(device_state->cl_pool,
                                        device_state->cl_pool_size * 2 *
                                        sizeof(ze_command_list_handle_t));
        device_state->cl_pool_size *= 2;
    }
    device_state->cl_pool[device_state->cl_pool_cnt++] = cl;
    pthread_mutex_unlock(&device_state->cl_mutex);
}

static int create_command_list(int dev_id, ze_command_list_handle_t * p_h_command_list)
{
    int rc = YAKSA_SUCCESS;
    ze_result_t zerr = ZE_RESULT_SUCCESS;
    ze_device_handle_t h_device = yaksuri_zei_global.device[dev_id];
    yaksuri_zei_device_state_s *device_state = yaksuri_zei_global.device_states + dev_id;

    /* check cl pool first */
    pthread_mutex_lock(&device_state->cl_mutex);
    if (device_state->cl_pool_cnt > 0) {
        *p_h_command_list = device_state->cl_pool[--device_state->cl_pool_cnt];
        pthread_mutex_unlock(&device_state->cl_mutex);
        goto fn_exit;
    }
    pthread_mutex_unlock(&device_state->cl_mutex);

    ze_command_queue_desc_t cmdQueueDesc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .pNext = NULL,
        .ordinal = -1,
        .index = 0,
        .flags = 0,
        .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };

    if (device_state->queueProperties == NULL) {
        ze_command_queue_group_properties_t *queueProperties;
        int numQueueGroups = 0;
        zerr = zeDeviceGetCommandQueueGroupProperties(h_device, &numQueueGroups, NULL);
        assert(zerr == ZE_RESULT_SUCCESS && numQueueGroups);
        queueProperties = (ze_command_queue_group_properties_t *)
            malloc(sizeof(ze_command_queue_group_properties_t) * numQueueGroups);
        zerr = zeDeviceGetCommandQueueGroupProperties(h_device, &numQueueGroups, queueProperties);
        assert(zerr == ZE_RESULT_SUCCESS);
        device_state->queueProperties = queueProperties;
        device_state->numQueueGroups = numQueueGroups;
    }
    /* use compute engine */
    for (int i = 0; i < device_state->numQueueGroups; i++) {
        if (device_state->queueProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            cmdQueueDesc.ordinal = i;
            break;
        }
    }

    zerr =
        zeCommandListCreateImmediate(yaksuri_zei_global.context, h_device, &cmdQueueDesc,
                                     p_h_command_list);
    YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);

  fn_exit:
    /* add to device state */
    pthread_mutex_lock(&device_state->mutex);
    if (device_state->num_cl + 1 == device_state->cl_cap) {
        device_state->cl =
            realloc(device_state->cl, device_state->cl_cap * 2 * sizeof(ze_command_list_handle_t));
        device_state->cl_cap *= 2;
    }
    device_state->cl[device_state->num_cl++] = *p_h_command_list;
    pthread_mutex_unlock(&device_state->mutex);
    return rc;
  fn_fail:
    goto fn_exit;
}

static int throttle_ze(int dev_id)
{
    int rc = YAKSA_SUCCESS;

    /* wait for the last event to finish */
    rc = yaksuri_zei_add_dependency(dev_id, dev_id);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int get_thread_block_dims(uint64_t count, yaksi_type_s * type, int *n_threads,
                                 int *n_blocks_x, int *n_blocks_y, int *n_blocks_z)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_zei_type_s *ze_type = (yaksuri_zei_type_s *) type->backend.ze.priv;

    *n_threads = THREAD_BLOCK_SIZE;
    uint64_t n_blocks = count * ze_type->num_elements / THREAD_BLOCK_SIZE;
    n_blocks += ! !(count * ze_type->num_elements % THREAD_BLOCK_SIZE);

    if (n_blocks <= MAX_GRIDSZ_X) {
        *n_blocks_x = (int) n_blocks;
        *n_blocks_y = 1;
        *n_blocks_z = 1;
    } else if (n_blocks <= MAX_GRIDSZ_X * MAX_GRIDSZ_Y) {
        *n_blocks_x = YAKSU_CEIL(n_blocks, MAX_GRIDSZ_Y);
        *n_blocks_y = YAKSU_CEIL(n_blocks, (*n_blocks_x));
        *n_blocks_z = 1;
    } else {
        int n_blocks_xy = YAKSU_CEIL(n_blocks, MAX_GRIDSZ_Z);
        *n_blocks_x = YAKSU_CEIL(n_blocks_xy, MAX_GRIDSZ_Y);
        *n_blocks_y = YAKSU_CEIL(n_blocks_xy, (*n_blocks_x));
        *n_blocks_z = YAKSU_CEIL(n_blocks, (uintptr_t) (*n_blocks_x) * (*n_blocks_y));
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_zei_pup_is_supported(yaksi_type_s * type, bool * is_supported)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_zei_type_s *ze_type = (yaksuri_zei_type_s *) type->backend.ze.priv;

    if (type->is_contig || (ze_type->pack != YAKSURI_KERNEL_NULL && ze_type->pack_kernels))
        *is_supported = true;
    else
        *is_supported = false;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static inline int yaksuri_zei_prep_kernel(ze_kernel_handle_t kernel, const void *inbuf,
                                          void *outbuf, uintptr_t count, yaksuri_zei_md_s * md)
{
    ze_result_t zerr;
    int rc = YAKSA_SUCCESS;

    zerr = zeKernelSetGroupSize(kernel, THREAD_BLOCK_SIZE, 1, 1);
    YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);

    zerr = zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &inbuf);
    YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);

    zerr = zeKernelSetArgumentValue(kernel, 1, sizeof(void *), &outbuf);
    YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);

    zerr = zeKernelSetArgumentValue(kernel, 2, sizeof(uintptr_t), &count);
    YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);

    zerr = zeKernelSetArgumentValue(kernel, 3, sizeof(yaksuri_zei_md_s *), &md);
    YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

uintptr_t yaksuri_zei_get_iov_pack_threshold(yaksi_info_s * info)
{
    uintptr_t iov_pack_threshold = YAKSURI_ZEI_INFO__DEFAULT_IOV_PUP_THRESHOLD;
    if (info) {
        yaksuri_zei_info_s *ze_info = (yaksuri_zei_info_s *) info->backend.ze.priv;
        iov_pack_threshold = ze_info->iov_pack_threshold;
    }

    return iov_pack_threshold;
}

uintptr_t yaksuri_zei_get_iov_unpack_threshold(yaksi_info_s * info)
{
    uintptr_t iov_unpack_threshold = YAKSURI_ZEI_INFO__DEFAULT_IOV_PUP_THRESHOLD;
    if (info) {
        yaksuri_zei_info_s *ze_info = (yaksuri_zei_info_s *) info->backend.ze.priv;
        iov_unpack_threshold = ze_info->iov_unpack_threshold;
    }

    return iov_unpack_threshold;
}

int yaksuri_zei_ipack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                      yaksi_info_s * info, int target)
{
    int rc = YAKSA_SUCCESS;
    ze_event_handle_t ze_event;
    int ze_event_idx;
    yaksuri_zei_type_s *ze_type = (yaksuri_zei_type_s *) type->backend.ze.priv;
    ze_result_t zerr;

    uintptr_t iov_pack_threshold = yaksuri_zei_get_iov_pack_threshold(info);

    yaksuri_zei_device_state_s *device_state = yaksuri_zei_global.device_states + target;
    ze_event_handle_t prev_event =
        device_state->last_event_idx ==
        -1 ? NULL : device_state->events[device_state->last_event_idx];

    /* create an immediate command list */
    ze_command_list_handle_t command_list;
    rc = create_command_list(target, &command_list);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* shortcut for contiguous types */
    if (type->is_contig) {
        rc = create_ze_event(target, &ze_event, &ze_event_idx);
        YAKSU_ERR_CHECK(rc, fn_fail);

        zerr =
            zeCommandListAppendMemoryCopy(command_list, outbuf,
                                          (const char *) inbuf + type->true_lb, count * type->size,
                                          ze_event, prev_event ? 1 : 0, &prev_event);
        YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);
    } else if (type->size / type->num_contig >= iov_pack_threshold) {
        struct iovec iov[MAX_IOV_LENGTH];
        char *dbuf = (char *) outbuf;
        uintptr_t offset = 0;

        while (offset < type->num_contig * count) {
            uintptr_t actual_iov_len;
            rc = yaksi_iov(inbuf, count, type, offset, iov, MAX_IOV_LENGTH, &actual_iov_len);
            YAKSU_ERR_CHECK(rc, fn_fail);

            for (uintptr_t i = 0; i < actual_iov_len; i++) {
                rc = create_ze_event(target, &ze_event, &ze_event_idx);
                YAKSU_ERR_CHECK(rc, fn_fail);
                zerr =
                    zeCommandListAppendMemoryCopy(command_list, dbuf, iov[i].iov_base,
                                                  iov[i].iov_len, ze_event, prev_event ? 1 : 0,
                                                  &prev_event);
                YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);
                dbuf += iov[i].iov_len;
                prev_event = ze_event;
            }
            offset += actual_iov_len;
        }
    } else {
        rc = yaksuri_zei_md_alloc(type, target);
        YAKSU_ERR_CHECK(rc, fn_fail);
        int n_threads;
        int n_blocks_x, n_blocks_y, n_blocks_z;
        rc = get_thread_block_dims(count, type, &n_threads, &n_blocks_x, &n_blocks_y, &n_blocks_z);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = create_ze_event(target, &ze_event, &ze_event_idx);
        YAKSU_ERR_CHECK(rc, fn_fail);

        ze_group_count_t launchArgs = { n_blocks_x, n_blocks_y, n_blocks_z };

        void *in_addr = (char *) inbuf + type->true_lb;
        zerr = yaksuri_zei_prep_kernel(ze_type->pack_kernels[target], in_addr, outbuf, count,
                                       ze_type->md[target]);
        YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);

        zerr =
            zeCommandListAppendLaunchKernel(command_list, ze_type->pack_kernels[target],
                                            &launchArgs, ze_event, prev_event ? 1 : 0, &prev_event);
        YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);
    }

    if (device_state->num_cl >= yaksuri_zei_global.throttle_threshold) {
        rc = throttle_ze(target);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    device_state->last_event_idx = ze_event_idx;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_zei_iunpack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                        yaksi_info_s * info, int target)
{
    int rc = YAKSA_SUCCESS;
    ze_event_handle_t ze_event;
    int ze_event_idx;
    yaksuri_zei_type_s *ze_type = (yaksuri_zei_type_s *) type->backend.ze.priv;
    ze_result_t zerr;

    uintptr_t iov_unpack_threshold = yaksuri_zei_get_iov_unpack_threshold(info);

    yaksuri_zei_device_state_s *device_state = yaksuri_zei_global.device_states + target;
    ze_event_handle_t prev_event =
        device_state->last_event_idx ==
        -1 ? NULL : device_state->events[device_state->last_event_idx];

    /* create an immediate command list */
    ze_command_list_handle_t command_list;
    rc = create_command_list(target, &command_list);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* shortcut for contiguous types */
    if (type->is_contig) {
        rc = create_ze_event(target, &ze_event, &ze_event_idx);
        YAKSU_ERR_CHECK(rc, fn_fail);
        /* ze performance is optimized when we synchronize on the
         * source buffer's GPU */
        zerr =
            zeCommandListAppendMemoryCopy(command_list, (char *) outbuf + type->true_lb, inbuf,
                                          count * type->size, ze_event, prev_event ? 1 : 0,
                                          &prev_event);
        YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);
    } else if (type->size / type->num_contig >= iov_unpack_threshold) {
        struct iovec iov[MAX_IOV_LENGTH];
        const char *sbuf = (const char *) inbuf;
        uintptr_t offset = 0;

        while (offset < type->num_contig * count) {
            uintptr_t actual_iov_len;
            rc = yaksi_iov(outbuf, count, type, offset, iov, MAX_IOV_LENGTH, &actual_iov_len);
            YAKSU_ERR_CHECK(rc, fn_fail);

            for (uintptr_t i = 0; i < actual_iov_len; i++) {
                rc = create_ze_event(target, &ze_event, &ze_event_idx);
                YAKSU_ERR_CHECK(rc, fn_fail);

                zerr =
                    zeCommandListAppendMemoryCopy(command_list, iov[i].iov_base, sbuf,
                                                  iov[i].iov_len, ze_event, prev_event ? 1 : 0,
                                                  &prev_event);
                sbuf += iov[i].iov_len;
                prev_event = ze_event;
            }
            offset += actual_iov_len;
        }
    } else {
        rc = yaksuri_zei_md_alloc(type, target);
        YAKSU_ERR_CHECK(rc, fn_fail);

        int n_threads;
        int n_blocks_x, n_blocks_y, n_blocks_z;
        rc = get_thread_block_dims(count, type, &n_threads, &n_blocks_x, &n_blocks_y, &n_blocks_z);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = create_ze_event(target, &ze_event, &ze_event_idx);
        YAKSU_ERR_CHECK(rc, fn_fail);

        ze_group_count_t launchArgs = { n_blocks_x, n_blocks_y, n_blocks_z };

        void *out_addr = (char *) outbuf + type->true_lb;
        rc = yaksuri_zei_prep_kernel(ze_type->unpack_kernels[target], inbuf, out_addr, count,
                                     ze_type->md[target]);
        YAKSU_ERR_CHECK(rc, fn_fail);

        zerr =
            zeCommandListAppendLaunchKernel(command_list, ze_type->unpack_kernels[target],
                                            &launchArgs, ze_event, prev_event ? 1 : 0, &prev_event);
        YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);
    }

    if (device_state->num_cl >= yaksuri_zei_global.throttle_threshold) {
        rc = throttle_ze(target);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    device_state->last_event_idx = ze_event_idx;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
