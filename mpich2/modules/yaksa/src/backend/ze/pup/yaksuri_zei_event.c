/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <stdio.h>
#include <assert.h>
#include <level_zero/ze_api.h>
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri_zei.h"
#include <stdlib.h>

/* update event usage bracket [ev_lb, ev_ub] */
#define update_event_usage(device_state, event_idx)                     \
        if (event_idx == device_state->ev_ub)                           \
            device_state->ev_lb = device_state->ev_ub = -1;             \
        else {                                                          \
            device_state->ev_lb = event_idx + 1;                        \
            if (device_state->ev_lb == yaksuri_zei_global.ev_pool_cap)  \
                device_state->ev_lb = 0;                                \
        }

int create_ze_event(int dev_id, ze_event_handle_t * ze_event, int *idx)
{
    ze_result_t zerr;
    int rc = YAKSA_SUCCESS;
    int ev_idx;

    yaksuri_zei_device_state_s *device_state = yaksuri_zei_global.device_states + dev_id;
    pthread_mutex_lock(&device_state->mutex);
    /* sanity check - abort when all events in the event pool are used up */
    assert(device_state->ev_pool_idx != device_state->ev_lb);
    ev_idx = device_state->ev_pool_idx;
    if (idx)
        *idx = ev_idx;
    /* reset the event before using it again */
    if (device_state->events[ev_idx]) {
        zeEventHostReset(device_state->events[ev_idx]);
        *ze_event = device_state->events[ev_idx];
    } else {
        ze_event_desc_t event_desc = {
            .stype = ZE_STRUCTURE_TYPE_EVENT_DESC,
            .pNext = NULL,
            .index = ev_idx,
            .signal = 0,
            .wait = ZE_EVENT_SCOPE_FLAG_HOST
        };
        zerr = zeEventCreate(device_state->ep, &event_desc, ze_event);
        YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);
        device_state->events[ev_idx] = *ze_event;
    }
    device_state->ev_pool_idx++;
    if (device_state->ev_pool_idx == yaksuri_zei_global.ev_pool_cap)
        device_state->ev_pool_idx = 0;
    /* update event bracket [ev_lb,ev_ub] */
    device_state->ev_ub = ev_idx;
    if (device_state->ev_lb == -1)
        device_state->ev_lb = device_state->ev_ub;
    pthread_mutex_unlock(&device_state->mutex);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static void free_to_marker(yaksuri_zei_device_state_s * device_state, ze_command_list_handle_t cl)
{
    int i;
    /* find cl marker */
    for (i = 0; i < device_state->num_cl; i++) {
        if (device_state->cl[i] == cl)
            break;
    }
    int marker = i;
    if (marker == device_state->num_cl)
        return;
    /* don't free the last one */
    for (i = 0; i < marker; i++) {
        if (device_state->cl[i] == NULL)
            continue;
        recycle_command_list(device_state->cl[i], device_state->dev_id);
        device_state->cl[i] = NULL;
    }
    if (marker == device_state->num_cl - 1) {
        device_state->num_cl = 1;
        device_state->cl[0] = cl;
    }
}

int yaksuri_zei_event_record(int device, void **event_)
{
    int rc = YAKSA_SUCCESS;
    ze_result_t zerr;
    ze_event_handle_t ze_event;
    yaksuri_zei_event_s *event;
    int idx;

    event = (yaksuri_zei_event_s *) malloc(sizeof(yaksuri_zei_event_s));

    rc = create_ze_event(device, &ze_event, &idx);
    YAKSU_ERR_CHECK(rc, fn_fail);

    event->dev_id = device;
    event->ze_event = ze_event;
    event->cl = NULL;
    event->idx = idx;

    yaksuri_zei_device_state_s *device_state = yaksuri_zei_global.device_states + device;
    if (device_state->num_cl) {
        pthread_mutex_lock(&device_state->mutex);
        ze_command_list_handle_t cl = device_state->cl[device_state->num_cl - 1];
        assert(cl);
        zerr = zeCommandListAppendSignalEvent(cl, event->ze_event);
        YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);
        event->cl = cl;
        pthread_mutex_unlock(&device_state->mutex);
    }
    assert(event->cl);

    *event_ = event;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_zei_event_query(void *event_, int *completed)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_zei_event_s *event = (yaksuri_zei_event_s *) event_;
    *completed = 1;
    ze_result_t zerr = zeEventQueryStatus(event->ze_event);
    if (zerr == ZE_RESULT_NOT_READY) {
        *completed = 0;
    } else {
        YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);
        yaksuri_zei_device_state_s *device_state = yaksuri_zei_global.device_states + event->dev_id;
        pthread_mutex_lock(&device_state->mutex);
        assert(event->cl);
        if (event->idx == device_state->last_event_idx) {
            device_state->last_event_idx = -1;
        }
        /* update event bracket [ev_lb, ev_ub] */
        /* freeing event->idx means all events below this index are done */
        update_event_usage(device_state, event->idx);
        free_to_marker(device_state, event->cl);
        pthread_mutex_unlock(&device_state->mutex);
        free(event);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_zei_add_dependency(int device1, int device2)
{
    ze_result_t zerr;
    int rc = YAKSA_SUCCESS;
    ze_event_handle_t last_event;
    ze_command_list_handle_t cl;

    /* wait for the last event on device1 to finish */
    yaksuri_zei_device_state_s *device_state = yaksuri_zei_global.device_states + device1;
    pthread_mutex_lock(&device_state->mutex);
    assert(device_state->last_event_idx >= 0);
    last_event = device_state->events[device_state->last_event_idx];
    assert(device_state->num_cl > 0);
    cl = device_state->cl[device_state->num_cl - 1];
    pthread_mutex_unlock(&device_state->mutex);
    if (last_event) {
        int completed = 0;
        while (!completed) {
            zerr = zeEventQueryStatus(last_event);
            if (zerr == ZE_RESULT_SUCCESS)
                completed = 1;
            else if (zerr != ZE_RESULT_NOT_READY) {
                YAKSURI_ZEI_ZE_ERR_CHKANDJUMP(zerr, rc, fn_fail);
            }
        }
        pthread_mutex_lock(&device_state->mutex);
        /* update event bracket [ev_lb, ev_ub] */
        update_event_usage(device_state, device_state->last_event_idx);
        free_to_marker(device_state, cl);
        pthread_mutex_unlock(&device_state->mutex);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
