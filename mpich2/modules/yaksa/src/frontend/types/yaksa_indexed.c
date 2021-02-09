/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <stdlib.h>
#include <assert.h>

int yaksi_type_create_hindexed(int count, const int *array_of_blocklengths,
                               const intptr_t * array_of_displs, yaksi_type_s * intype,
                               yaksi_type_s ** newtype)
{
    int rc = YAKSA_SUCCESS;

    /* shortcut for hindexed_block types */
    bool is_hindexed_block = true;
    for (int i = 1; i < count; i++) {
        if (array_of_blocklengths[i] != array_of_blocklengths[i - 1])
            is_hindexed_block = false;
    }
    if (is_hindexed_block) {
        rc = yaksi_type_create_hindexed_block(count, array_of_blocklengths[0], array_of_displs,
                                              intype, newtype);
        YAKSU_ERR_CHECK(rc, fn_fail);
        goto fn_exit;
    }

    /* regular hindexed type */
    yaksi_type_s *outtype;
    outtype = (yaksi_type_s *) malloc(sizeof(yaksi_type_s));
    YAKSU_ERR_CHKANDJUMP(!outtype, rc, YAKSA_ERR__OUT_OF_MEM, fn_fail);
    yaksu_atomic_store(&outtype->refcount, 1);

    yaksu_atomic_incr(&intype->refcount);

    outtype->kind = YAKSI_TYPE_KIND__HINDEXED;
    outtype->tree_depth = intype->tree_depth + 1;

    outtype->size = 0;
    for (int i = 0; i < count; i++)
        outtype->size += intype->size * array_of_blocklengths[i];
    outtype->alignment = intype->alignment;

    int is_set;
    is_set = 0;
    for (int idx = 0; idx < count; idx++) {
        if (array_of_blocklengths[idx] == 0)
            continue;

        intptr_t lb, ub;
        if (intype->extent > 0) {
            lb = array_of_displs[idx] + intype->lb;
            ub = array_of_displs[idx] + intype->ub +
                intype->extent * (array_of_blocklengths[idx] - 1);
        } else {
            lb = array_of_displs[idx] + intype->lb +
                intype->extent * (array_of_blocklengths[idx] - 1);
            ub = array_of_displs[idx] + intype->ub;
        }

        intptr_t true_lb = lb - intype->lb + intype->true_lb;
        intptr_t true_ub = ub - intype->ub + intype->true_ub;

        if (is_set) {
            outtype->lb = YAKSU_MIN(lb, outtype->lb);
            outtype->true_lb = YAKSU_MIN(true_lb, outtype->true_lb);
            outtype->ub = YAKSU_MAX(ub, outtype->ub);
            outtype->true_ub = YAKSU_MAX(true_ub, outtype->true_ub);
        } else {
            outtype->lb = lb;
            outtype->true_lb = true_lb;
            outtype->ub = ub;
            outtype->true_ub = true_ub;
        }

        is_set = 1;
    }

    outtype->extent = outtype->ub - outtype->lb;

    outtype->u.hindexed.count = count;
    outtype->u.hindexed.array_of_blocklengths = (int *) malloc(count * sizeof(intptr_t));
    outtype->u.hindexed.array_of_displs = (intptr_t *) malloc(count * sizeof(intptr_t));
    for (int i = 0; i < count; i++) {
        outtype->u.hindexed.array_of_blocklengths[i] = array_of_blocklengths[i];
        outtype->u.hindexed.array_of_displs[i] = array_of_displs[i];
    }
    outtype->u.hindexed.child = intype;

    /* detect if the outtype is contiguous */
    if (intype->is_contig && ((outtype->ub - outtype->lb) == outtype->size)) {
        outtype->is_contig = true;

        int left = 0;
        while (array_of_blocklengths[left] == 0)
            left++;
        int right = left + 1;
        while (right < count && array_of_blocklengths[right] == 0)
            right++;
        while (right < count) {
            if (array_of_displs[right] <= array_of_displs[left]) {
                outtype->is_contig = false;
                break;
            } else {
                left = right;
                right++;
                while (right < count && array_of_blocklengths[right] == 0)
                    right++;
            }
        }
    } else {
        outtype->is_contig = false;
    }

    if (outtype->is_contig) {
        outtype->num_contig = 1;
    } else if (intype->is_contig) {
        uintptr_t tmp = 0;
        for (int i = 0; i < count; i++)
            if (array_of_blocklengths[i])
                tmp++;
        outtype->num_contig = intype->num_contig * tmp;
    } else {
        uintptr_t tmp = 0;
        for (int i = 0; i < count; i++)
            tmp += array_of_blocklengths[i];
        outtype->num_contig = intype->num_contig * tmp;
    }

    yaksur_type_create_hook(outtype);
    *newtype = outtype;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksa_type_create_hindexed(int count, const int *array_of_blocklengths,
                               const intptr_t * array_of_displs, yaksa_type_t oldtype,
                               yaksa_info_t info, yaksa_type_t * newtype)
{
    int rc = YAKSA_SUCCESS;

    assert(yaksu_atomic_load(&yaksi_is_initialized));

    yaksi_type_s *intype;
    rc = yaksi_type_get(oldtype, &intype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    uintptr_t total_size;
    total_size = 0;
    for (int i = 0; i < count; i++) {
        total_size += intype->size * array_of_blocklengths[i];
    }
    if (total_size == 0) {
        *newtype = YAKSA_TYPE__NULL;
        goto fn_exit;
    }

    yaksi_type_s *outtype;
    rc = yaksi_type_create_hindexed(count, array_of_blocklengths, array_of_displs, intype,
                                    &outtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksi_type_handle_alloc(outtype, newtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksa_type_create_indexed(int count, const int *array_of_blocklengths,
                              const int *array_of_displs, yaksa_type_t oldtype,
                              yaksa_info_t info, yaksa_type_t * newtype)
{
    int rc = YAKSA_SUCCESS;
    intptr_t *real_array_of_displs = (intptr_t *) malloc(count * sizeof(intptr_t));

    assert(yaksu_atomic_load(&yaksi_is_initialized));

    yaksi_type_s *intype;
    rc = yaksi_type_get(oldtype, &intype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    uintptr_t total_size;
    total_size = 0;
    for (int i = 0; i < count; i++) {
        total_size += intype->size * array_of_blocklengths[i];
    }
    if (total_size == 0) {
        *newtype = YAKSA_TYPE__NULL;
        goto fn_exit;
    }

    for (int i = 0; i < count; i++)
        real_array_of_displs[i] = array_of_displs[i] * intype->extent;

    yaksi_type_s *outtype;
    rc = yaksi_type_create_hindexed(count, array_of_blocklengths, real_array_of_displs, intype,
                                    &outtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksi_type_handle_alloc(outtype, newtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    free(real_array_of_displs);
    return rc;
  fn_fail:
    goto fn_exit;
}
