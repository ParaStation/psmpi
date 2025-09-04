/*
 * ParaStation
 *
 * Copyright (C) 2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#ifndef _MPID_UCC_DTYPES_H_
#define _MPID_UCC_DTYPES_H_

#define MPIDI_COMMON_UCC_DTYPE_NULL ((ucc_datatype_t)-1)
#define MPIDI_COMMON_UCC_DTYPE_UNSUPPORTED ((ucc_datatype_t)-2)

static inline ucc_datatype_t mpidi_mpi_dtype_to_ucc_dtype(MPI_Datatype datatype)
{
    switch (datatype) {
        case MPI_CHAR:
        case MPI_SIGNED_CHAR:
            return UCC_DT_INT8;
        case MPI_SHORT:
            return UCC_DT_INT16;
        case MPI_INT:
            return UCC_DT_INT32;
        case MPI_LONG:
        case MPI_LONG_LONG:
            return UCC_DT_INT64;
            /* return UCC_DT_INT128; */
        case MPI_BYTE:
        case MPI_UNSIGNED_CHAR:
            return UCC_DT_UINT8;
        case MPI_UNSIGNED_SHORT:
            return UCC_DT_UINT16;
        case MPI_UNSIGNED:
            return UCC_DT_UINT32;
        case MPI_UNSIGNED_LONG:
        case MPI_UNSIGNED_LONG_LONG:
            return UCC_DT_UINT64;
            /* return UCC_DT_UINT128; */
        case MPI_FLOAT:
            return UCC_DT_FLOAT32;
        case MPI_DOUBLE:
            return UCC_DT_FLOAT64;
        case MPI_LONG_DOUBLE:
            return UCC_DT_FLOAT128;
        default:
            return MPIDI_COMMON_UCC_DTYPE_UNSUPPORTED;
    }
}

static inline const char *mpidi_ucc_dtype_to_str(ucc_datatype_t datatype)
{
    switch (datatype) {
        case UCC_DT_INT8:
            return "UCC_DT_INT8";
        case UCC_DT_INT16:
            return "UCC_DT_INT16";
        case UCC_DT_INT32:
            return "UCC_DT_INT32";
        case UCC_DT_INT64:
            return "UCC_DT_INT64";
        case UCC_DT_INT128:
            return "UCC_DT_INT128";
        case UCC_DT_UINT8:
            return "UCC_DT_UINT8";
        case UCC_DT_UINT16:
            return "UCC_DT_UINT16";
        case UCC_DT_UINT32:
            return "UCC_DT_UINT32";
        case UCC_DT_UINT64:
            return "UCC_DT_UINT64";
        case UCC_DT_UINT128:
            return "UCC_DT_UINT128";
        case UCC_DT_FLOAT32:
            return "UCC_DT_FLOAT32";
        case UCC_DT_FLOAT64:
            return "UCC_DT_FLOAT64";
        case UCC_DT_FLOAT128:
            return "UCC_DT_FLOAT128";
        default:
            return "unknown";
    }
}


#define MPIDI_COMMON_UCC_REDUCTION_OP_NULL ((ucc_reduction_op_t)-1)
#define MPIDI_COMMON_UCC_REDUCTION_OP_UNSUPPORTED ((ucc_reduction_op_t)-2)

static inline ucc_reduction_op_t mpidi_mpi_op_to_ucc_reduction_op(MPI_Op operation)
{
    switch (operation) {
        case MPI_MAX:
            return UCC_OP_MAX;
        case MPI_MIN:
            return UCC_OP_MIN;
        case MPI_SUM:
            return UCC_OP_SUM;
        case MPI_PROD:
            return UCC_OP_PROD;
        case MPI_LAND:
            return UCC_OP_LAND;
        case MPI_BAND:
            return UCC_OP_BAND;
        case MPI_LOR:
            return UCC_OP_LOR;
        case MPI_BOR:
            return UCC_OP_BOR;
        case MPI_LXOR:
            return UCC_OP_LXOR;
        case MPI_BXOR:
            return UCC_OP_BXOR;
        default:
            return MPIDI_COMMON_UCC_REDUCTION_OP_UNSUPPORTED;
    }
}

static inline const char *mpidi_ucc_reduction_op_to_str(ucc_reduction_op_t operation)
{
    switch (operation) {
        case UCC_OP_MAX:
            return "UCC_OP_MAX";
        case UCC_OP_MIN:
            return "UCC_OP_MIN";
        case UCC_OP_SUM:
            return "UCC_OP_SUM";
        case UCC_OP_PROD:
            return "UCC_OP_PROD";
        case UCC_OP_LAND:
            return "UCC_OP_LAND";
        case UCC_OP_BAND:
            return "UCC_OP_BAND";
        case UCC_OP_LOR:
            return "UCC_OP_LOR";
        case UCC_OP_BOR:
            return "UCC_OP_BOR";
        case UCC_OP_LXOR:
            return "UCC_OP_LXOR";
        case UCC_OP_BXOR:
            return "UCC_OP_BXOR";
        default:
            return "unknown";
    }
}

#endif /*_MPID_UCC_DTYPES_H_*/
