/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2023 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#ifndef _MPID_PSP_DATATYPE_H_
#define _MPID_PSP_DATATYPE_H_

#include "mpidimpl.h"

typedef struct MPIDI_PSP_Datatype_packed {
    MPI_Aint datatype;
    int datatype_sz;
    void *datatype_encoded;
} MPIDI_PSP_Datatype_packed_t;

#define MPIDI_PSP_Datatype_get_size_dt_ptr(count_, datatype_,   \
                                       data_sz_out_, dt_ptr_)   \
    do {                                                        \
        if (HANDLE_IS_BUILTIN(datatype_)) {                     \
            (dt_ptr_)        = NULL;                            \
            (data_sz_out_)   = (size_t)(count_) *               \
                MPIR_Datatype_get_basic_size(datatype_);        \
        } else {                                                \
            MPIR_Datatype_get_ptr((datatype_), (dt_ptr_));      \
            (data_sz_out_)   = (dt_ptr_) ? (size_t)(count_) *   \
                (dt_ptr_)->size : 0;                            \
        }                                                       \
    } while (0)

#define MPIDI_PSP_Datatype_check_size(datatype_,count_,data_sz_out_)    \
    do {                                                                \
        if (HANDLE_IS_BUILTIN(datatype_)) {                             \
            (data_sz_out_)   = (size_t)(count_) *                       \
                MPIR_Datatype_get_basic_size(datatype_);                \
        } else {                                                        \
            MPIR_Datatype *dt_ptr_;                                     \
            MPIR_Datatype_get_ptr((datatype_), (dt_ptr_));              \
            (data_sz_out_)   = (dt_ptr_) ? (size_t)(count_) *           \
                (dt_ptr_)->size : 0;                                    \
        }                                                               \
    } while (0)


static inline
    void MPIDI_PSP_Datatype_map_to_basic_type(MPI_Datatype in_datatype,
                                              int in_count,
                                              MPI_Datatype * out_datatype, int *out_count)
{
    MPIR_Datatype *dt_ptr;
    uint64_t data_sz;
    size_t basic_type_size;

    MPIDI_PSP_Datatype_get_size_dt_ptr(in_count, in_datatype, data_sz, dt_ptr);

    if (HANDLE_IS_BUILTIN(in_datatype)) {
        *out_datatype = in_datatype;
        *out_count = in_count;
    } else {
        *out_datatype = (dt_ptr) ? dt_ptr->basic_type : MPI_DATATYPE_NULL;
        MPIR_Datatype_get_size_macro(*out_datatype, basic_type_size);
        *out_count = (basic_type_size > 0) ? data_sz / basic_type_size : 0;
    }
}

/*
 * get the size required to encode the datatype described by info.
 * Use like:
 * encode = malloc(MPID_PSP_Datatype_get_size(info));
 * MPID_PSP_Datatype_encode(info, encode);
 */
static inline unsigned int MPID_PSP_Datatype_get_size(MPI_Datatype datatype)
{
    MPIR_Datatype *dtp;
    int flattened_size = 0;

    if (!MPIR_DATATYPE_IS_PREDEFINED(datatype)) {
        MPIR_Datatype_get_ptr(datatype, dtp);
        MPIR_Typerep_flatten_size(dtp, &flattened_size);
    }

    return flattened_size + sizeof(MPIDI_PSP_Datatype_packed_t);
}

/*
 * Encode the (MPI_Datatype)datatype to *encode. Caller has to allocate at least
 * info->encode_size bytes at encode.
 */
void MPID_PSP_Datatype_encode(MPI_Datatype datatype, void *encode);


/*
 * Create a new (MPI_Datatype)new_datatype with refcnt 1. Caller has to call
 * MPID_PSP_Datatype_release(new_datatype) after usage.
 */
MPI_Datatype MPID_PSP_Datatype_decode(void *encode);


void MPID_PSP_Datatype_release(MPI_Datatype datatype);
void MPID_PSP_Datatype_add_ref(MPI_Datatype datatype);

#endif /* _MPID_PSP_DATATYPE_H_ */
