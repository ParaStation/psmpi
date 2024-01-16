/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2024 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpid_psp_datatype.h"

/*
 * Encode the (MPI_Datatype)datatype to *encode. Caller has to allocate at least
 * MPIR_Typerep_flatten_size() bytes at encode.
 */
void MPID_PSP_Datatype_encode(MPI_Datatype datatype, void *encode)
{
    int ret;
    MPIDI_PSP_Datatype_packed_t *packed_datatype = (MPIDI_PSP_Datatype_packed_t *) encode;

    /* do not encode predefined datatypes */
    if (MPIR_DATATYPE_IS_PREDEFINED(datatype)) {
        packed_datatype->datatype = datatype;
        packed_datatype->datatype_sz = 0;
    } else {
        MPIR_Datatype *dtp;

        MPIR_Datatype_get_ptr(datatype, dtp);
        ret = MPIR_Typerep_flatten(dtp, &packed_datatype->datatype_encoded);
        MPIR_Assert(ret == MPI_SUCCESS);
        ret = MPIR_Typerep_flatten_size(dtp, &packed_datatype->datatype_sz);
        MPIR_Assert(ret == MPI_SUCCESS);
    }

    return;
}

/*
 * Create a new (MPI_Datatype)new_datatype with refcnt 1. Caller has to call
 * MPIR_Datatype_ptr_release(new_datatype) after usage.
 */
MPI_Datatype MPID_PSP_Datatype_decode(void *encode)
{
    int ret;
    MPI_Datatype datatype;

    MPIDI_PSP_Datatype_packed_t *packed_datatype = (MPIDI_PSP_Datatype_packed_t *) encode;
    MPIR_Datatype *new_dtp;

    /* do not decode predefined datatypes */
    if (!packed_datatype->datatype_sz) {
        datatype = packed_datatype->datatype;
    } else {

        new_dtp = (MPIR_Datatype *) MPIR_Handle_obj_alloc(&MPIR_Datatype_mem);
        if (!new_dtp) {
            goto err_alloc_dtp;
        }

        MPIR_Object_set_ref(new_dtp, 1);
        ret = MPIR_Typerep_unflatten(new_dtp, &packed_datatype->datatype_encoded);
        MPIR_Assert(ret == MPI_SUCCESS);

        datatype = new_dtp->handle;
    }

    return datatype;

    /* --- */
  err_alloc_dtp:
    {
        static int warn = 0;
        if (!warn) {
            fprintf(stderr, "Warning: unhandled error in " __FILE__ ":%d", __LINE__);
            warn = 1;
        }
    }
    return 0;
}

void MPID_PSP_Datatype_release(MPI_Datatype datatype)
{
    if (!MPIR_DATATYPE_IS_PREDEFINED(datatype)) {
        MPIR_Datatype *dtp;
        MPIR_Datatype_get_ptr(datatype, dtp);
        MPIR_Datatype_ptr_release(dtp);
    }
}

void MPID_PSP_Datatype_add_ref(MPI_Datatype datatype)
{
    if (!MPIR_DATATYPE_IS_PREDEFINED(datatype)) {
        MPIR_Datatype *dtp;
        MPIR_Datatype_get_ptr(datatype, dtp);
        // TODO: check if MPID_PSP_Datatype_add_ref() is required at all.
        MPIR_Datatype_ptr_add_ref(dtp);
    }
}
