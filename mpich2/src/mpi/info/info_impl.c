/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

static int info_find_key(MPIR_Info * info_ptr, const char *key)
{
    for (int i = 0; i < info_ptr->size; i++) {
        if (strncmp(info_ptr->entries[i].key, key, MPI_MAX_INFO_KEY) == 0) {
            return i;
        }
    }
    return -1;
}

static int info_find_key_array(MPIR_Info * info_ptr, const char *key)
{
    for (int i = 0; i < info_ptr->array_size; i++) {
        if (strncmp(info_ptr->array_entries[i].key, key, MPI_MAX_INFO_KEY) == 0) {
            return i;
        }
    }
    return -1;
}

static int info_delete_array(MPIR_Info * info_ptr, const char *key)
{
    int found_index = info_find_key_array(info_ptr, key);

    /* Only delete if found (no error if not). */
    if (found_index >= 0) {

        MPL_direct_free(info_ptr->array_entries[found_index].key);
        for (int i = 0; i < info_ptr->array_entries[found_index].num_values; i++) {
            MPL_direct_free(info_ptr->array_entries[found_index].values[i]);
        }
        MPL_direct_free(info_ptr->array_entries[found_index].values);

        /* move up the later entries */
        for (int i = found_index + 1; i < info_ptr->array_size; i++) {
            info_ptr->array_entries[i - 1] = info_ptr->array_entries[i];
        }
        info_ptr->array_size--;
    }

    return found_index;
}

const char *MPIR_Info_lookup(const MPIR_Info * info_ptr, const char *key)
{
    if (!info_ptr) {
        return NULL;
    }

    for (int i = 0; i < info_ptr->size; i++) {
        if (strncmp(info_ptr->entries[i].key, key, MPI_MAX_INFO_KEY) == 0) {
            return info_ptr->entries[i].value;
        }
    }
    return NULL;
}

const char *MPIR_Info_lookup_array(MPIR_Info * info_ptr, const char *key, int index,
                                   int *num_values)
{
    if (!info_ptr) {
        return NULL;
    }

    for (int i = 0; i < info_ptr->array_size; i++) {
        if (strncmp(info_ptr->array_entries[i].key, key, MPI_MAX_INFO_KEY) == 0) {
            MPIR_Assertp(index < info_ptr->array_entries[i].num_values);
            if (num_values) {
                *num_values = info_ptr->array_entries[i].num_values;
            }
            return info_ptr->array_entries[i].values[index];
        }
    }
    return NULL;
}

/* All the MPIR_Info routines may be called before initialization or after finalization of MPI. */
int MPIR_Info_delete_impl(MPIR_Info * info_ptr, const char *key)
{
    int mpi_errno = MPI_SUCCESS;

    int found_index = info_find_key(info_ptr, key);
    MPIR_ERR_CHKANDJUMP1((found_index < 0), mpi_errno, MPI_ERR_INFO_NOKEY, "**infonokey",
                         "**infonokey %s", key);

    /* MPI_Info objects are allocated by MPL_direct_malloc(), so they need to be
     * freed by MPL_direct_free(), not MPL_free(). */
    MPL_direct_free(info_ptr->entries[found_index].key);
    MPL_direct_free(info_ptr->entries[found_index].value);

    /* move up the later entries */
    for (int i = found_index + 1; i < info_ptr->size; i++) {
        info_ptr->entries[i - 1] = info_ptr->entries[i];
    }
    info_ptr->size--;

    /* delete a possibly existing array type entry as well */
    info_delete_array(info_ptr, key);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_Info_dup_impl(MPIR_Info * info_ptr, MPIR_Info ** new_info_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    *new_info_ptr = NULL;
    if (!info_ptr)
        goto fn_exit;

    MPIR_Info *info_new;
    mpi_errno = MPIR_Info_alloc(&info_new);
    MPIR_ERR_CHECK(mpi_errno);

    for (int i = 0; i < info_ptr->size; i++) {
        MPIR_Info_push(info_new, info_ptr->entries[i].key, info_ptr->entries[i].value);
        MPIR_ERR_CHECK(mpi_errno);
    }
    for (int i = 0; i < info_ptr->array_size; i++) {
        int num_values = info_ptr->array_entries[i].num_values;
        /* Create a new array entry for the given key that in turn has `num_values` value enties,
         * and directly also set the first value with index = 0 to `array_entries[i].values[0]`. */
        mpi_errno = MPIR_Info_push_array(info_new, 0, num_values, info_ptr->array_entries[i].key,
                                         info_ptr->array_entries[i].values[0]);
        MPIR_ERR_CHECK(mpi_errno);
        /* Now also copy/set the other (num_values-1) value entries from index = 1 on. */
        for (int j = 1; j < num_values; j++) {
            mpi_errno = MPIR_Info_set_array(info_new, j, info_ptr->array_entries[i].key,
                                            info_ptr->array_entries[i].values[j]);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

    *new_info_ptr = info_new;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_Info_merge_from_array_impl(int count, MPIR_Info * array_of_info_ptrs[],
                                    MPIR_Info ** new_info_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    *new_info_ptr = NULL;

    MPIR_Info *info_new;
    mpi_errno = MPIR_Info_alloc(&info_new);
    MPIR_ERR_CHECK(mpi_errno);

    int found = 0;

    /* Loop over all entries of the `array_of_info_ptrs` array given as a parameter. */
    for (int i = 0; i < count; i++) {
        if (!array_of_info_ptrs[i])
            continue;

        /* Loop over all key entries of the current (`i`) info object in the given array. */
        for (int j = 0; j < array_of_info_ptrs[i]->size; j++) {

            char *key = array_of_info_ptrs[i]->entries[j].key;
            /* Current key already an array type and/or already handled and stored as such? */
            if (!strcmp(array_of_info_ptrs[i]->entries[j].value, MPIR_INFO_INFOKEY_ARRAY_TYPE) ||
                MPIR_Info_lookup_array(info_new, key, 0, NULL))
                continue;

            /* Loop over all remaining (not yet checked) info objects in the given array. */
            for (int k = i + 1; k < count; k++) {
                if (!array_of_info_ptrs[k])
                    continue;

                /* Use the current key to check if it is also used in one of the other (`k`)
                 * info objects in the given array. */
                const char *value = MPIR_Info_lookup(array_of_info_ptrs[k], key);
                if (value && strcmp(value, MPIR_INFO_INFOKEY_ARRAY_TYPE)) {
                    /* If a value is found and it's not the wildcard for an "array type"
                     * (because the merge function is not intended to work recursively),
                     * then this value has to be added to an array type entry for key. */
                    if (!found) {
                        /* If it's the first value for this array type entry, then we first
                         * need to create/allocate the data structure for this entry by calling
                         * `MPIR_Info_push_array()`, which will also add the first value at
                         * index `k` for the respective key. */
                        mpi_errno = MPIR_Info_push_array(info_new, k, count, key, value);
                        MPIR_ERR_CHECK(mpi_errno);
                        found = 1;
                    } else {
                        /* If the respective entry has already been created (`found == 1`),
                         * then we can use `MPIR_Info_set_array()` to put this value into the
                         * right position (`k`) within this existing array type entry. */
                        mpi_errno = MPIR_Info_set_array(info_new, k, key, value);
                        MPIR_ERR_CHECK(mpi_errno);
                    }
                }
            }

            if (found) {
                /* If a duplicate key was found (`found == 1), the value for the key used for the
                 * current search must also be set to the correct position (`i`) within the already
                 * created array type entry. */
                mpi_errno = MPIR_Info_set_array(info_new, i, array_of_info_ptrs[i]->entries[j].key,
                                                array_of_info_ptrs[i]->entries[j].value);
                MPIR_ERR_CHECK(mpi_errno);
                found = 0;
            }
        }
    }

    /* Finally, we must also copy all other key/value entries for which no duplicates were found.
     * For this, we loop again over all info objects in the given array and check for each key/value
     * pair if the key has also been added as an array entry type to the newly created info object.
     * If this is not the case, we just copy the key/value pair.
     * If it is the case, we also add this key as a non-array entry but with the "array type" wildcard
     * as the value so that it can be identified via the common MPI API as being in fact an array entry. */
    for (int i = 0; i < count; i++) {
        if (!array_of_info_ptrs[i])
            continue;

        for (int j = 0; j < array_of_info_ptrs[i]->size; j++) {
            char *key = array_of_info_ptrs[i]->entries[j].key;
            char *value = array_of_info_ptrs[i]->entries[j].value;

            if (strcmp(value, MPIR_INFO_INFOKEY_ARRAY_TYPE)) {
                if (!MPIR_Info_lookup_array(info_new, key, 0, NULL)) {
                    mpi_errno = MPIR_Info_push(info_new, key, value);
                    MPIR_ERR_CHECK(mpi_errno);
                } else if (!MPIR_Info_lookup(info_new, key)) {
                    mpi_errno = MPIR_Info_push(info_new, key, MPIR_INFO_INFOKEY_ARRAY_TYPE);
                    MPIR_ERR_CHECK(mpi_errno);
                }
            }
        }
    }

    *new_info_ptr = info_new;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_Info_split_into_array_impl(int *count, MPIR_Info ** array_of_info_ptrs,
                                    MPIR_Info * info_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    int new_count = 0;
    int max_j = *count;

    /* Loop over all keys and check for array type entries. */
    for (int i = 0; i < info_ptr->size; i++) {
        char *key = info_ptr->entries[i].key;
        char *value = info_ptr->entries[i].value;

        if (!strcmp(value, MPIR_INFO_INFOKEY_ARRAY_TYPE)) {
            /* This *is* an array type entry: Check for the number of values and the first value. */
            int num_values;
            const char *array_value = MPIR_Info_lookup_array(info_ptr, key, 0, &num_values);

            /* We assume that there is at least one value stored for each existing array type entry. */
            MPIR_Assertp(array_value && num_values);

            /* `newcount` must always be the maximum number we encounter. */
            if (new_count < num_values) {
                new_count = num_values;
            }

            /* Adjust `max_j` to the upper limit as given via `count` parameter */
            max_j = num_values;
            if (max_j > *count) {
                max_j = *count;
            }

            /* Loop over the stored values of this array type entry, but at max up to the number given in `*count`. */
            for (int j = 0; j < max_j; j++) {

                /* Skip the first value as this has already been fetched above. */
                if (j) {
                    array_value = MPIR_Info_lookup_array(info_ptr, key, j, NULL);
                    MPIR_Assertp(array_value);
                }

                /* Finally put the current value with its key into the respective info object in the given array. */
                mpi_errno = MPIR_Info_set_impl(array_of_info_ptrs[j], key, array_value);
                MPIR_ERR_CHECK(mpi_errno);
            }
        } else {
            /* This is not an array type entry: Just copy it into all given info objects in the array. */
            for (int j = 0; j < *count; j++) {
                mpi_errno = MPIR_Info_set_impl(array_of_info_ptrs[j], key, value);
                MPIR_ERR_CHECK(mpi_errno);
            }
        }
    }

    *count = new_count;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_Info_get_impl(MPIR_Info * info_ptr, const char *key, int valuelen, char *value, int *flag)
{
    int mpi_errno = MPI_SUCCESS;

    const char *v = MPIR_Info_lookup(info_ptr, key);
    if (!v) {
        *flag = 0;
    } else {
        *flag = 1;
        /* +1 because the MPI Standard says "In C, valuelen
         * (passed to MPI_Info_get) should be one less than the
         * amount of allocated space to allow for the null
         * terminator*/
        int err = MPL_strncpy(value, v, valuelen + 1);
        if (err != 0) {
            mpi_errno = MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,
                                             MPI_ERR_INFO_VALUE, "**infovallong", NULL);
        }
    }

    return mpi_errno;
}

int MPIR_Info_get_nkeys_impl(MPIR_Info * info_ptr, int *nkeys)
{
    *nkeys = info_ptr->size;

    return MPI_SUCCESS;
}

int MPIR_Info_get_nthkey_impl(MPIR_Info * info_ptr, int n, char *key)
{
    int mpi_errno = MPI_SUCCESS;

    /* verify that n is valid */
    MPIR_ERR_CHKANDJUMP2((n >= info_ptr->size), mpi_errno, MPI_ERR_ARG, "**infonkey",
                         "**infonkey %d %d", n, info_ptr->size);

    /* if key is MPI_MAX_INFO_KEY long, MPL_strncpy will null-terminate it for
     * us */
    MPL_strncpy(key, info_ptr->entries[n].key, MPI_MAX_INFO_KEY);
    /* Eventually, we could remember the location of this key in
     * the head using the key/value locations (and a union datatype?) */

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_Info_get_valuelen_impl(MPIR_Info * info_ptr, const char *key, int *valuelen, int *flag)
{
    const char *v = MPIR_Info_lookup(info_ptr, key);

    if (!v) {
        *flag = 0;
    } else {
        *valuelen = (int) strlen(v);
        *flag = 1;
    }

    return MPI_SUCCESS;
}

int MPIR_Info_set_impl(MPIR_Info * info_ptr, const char *key, const char *value)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    int found_index = info_find_key(info_ptr, key);
    if (found_index < 0) {
        /* Key not present, insert value */
        mpi_errno = MPIR_Info_push(info_ptr, key, value);
        MPIR_ERR_CHECK(mpi_errno);
    } else {
        /* Key already present; replace value */
        MPL_direct_free(info_ptr->entries[found_index].value);
        info_ptr->entries[found_index].value = MPL_direct_strdup(value);
        MPIR_ERR_CHKANDJUMP(!info_ptr->entries[found_index].value, mpi_errno, MPI_ERR_OTHER,
                            "**nomem");

        /* ...and delete a possibly existing array type entry */
        info_delete_array(info_ptr, key);
    }

  fn_exit:
    MPIR_FUNC_EXIT;
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

int MPIR_Info_get_string_impl(MPIR_Info * info_ptr, const char *key, int *buflen, char *value,
                              int *flag)
{
    const char *v = MPIR_Info_lookup(info_ptr, key);
    if (!v) {
        *flag = 0;
    } else {
        *flag = 1;

        int old_buflen = *buflen;
        /* It needs to include a terminator. */
        int new_buflen = (int) (strlen(v) + 1);
        if (old_buflen > 0) {
            /* Copy the value. */
            MPL_strncpy(value, v, old_buflen);
            /* No matter whether MPL_strncpy() returns an error or not
             * (i.e., whether value fits or not), it is not an error. */
        }
        *buflen = new_buflen;
    }

    return MPI_SUCCESS;
}

int MPIR_Info_create_env_impl(int argc, char **argv, MPIR_Info ** new_info_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    /* Allocate an empty info object. */
    MPIR_Info *info_ptr = NULL;
    mpi_errno = MPIR_Info_alloc(&info_ptr);
    MPIR_ERR_CHECK(mpi_errno);
    /* Set up the info value. */
    MPIR_Info_setup_env(info_ptr, argc, argv);

    *new_info_ptr = info_ptr;

  fn_exit:
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

int MPIR_Info_set_hex_impl(MPIR_Info * info_ptr, const char *key, const void *value, int value_size)
{
    int mpi_errno = MPI_SUCCESS;

    char value_buf[1024];

    int rc;
    int len_out;
    rc = MPL_hex_encode(value, value_size, value_buf, 1024, &len_out);
    MPIR_Assertp(rc == MPL_SUCCESS);

    mpi_errno = MPIR_Info_set_impl(info_ptr, key, value_buf);

    return mpi_errno;
}

int MPIR_Info_decode_hex(const char *str, void *buf, int len)
{
    int mpi_errno = MPI_SUCCESS;

    int len_out;
    int rc = MPL_hex_decode(str, buf, len, &len_out);
    MPIR_ERR_CHKANDJUMP(rc || len_out != len, mpi_errno, MPI_ERR_OTHER, "**infohexinvalid");

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
