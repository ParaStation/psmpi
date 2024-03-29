/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"
#include "mpir_op_util.h"
#ifdef HAVE_FORTRAN_BINDING
#include "mpii_fortlogical.h"
#endif

/*
 * In MPI-2.1, this operation is valid only for C integer and Logical
 * types (5.9.2 Predefined reduce operations)
 */
#ifndef MPIR_LLXOR
#define MPIR_LLXOR(a,b) (((a)&&(!b))||((!a)&&(b)))
#endif

void MPIR_LXOR(void *invec, void *inoutvec, MPI_Aint * Len, MPI_Datatype * type)
{
    MPI_Aint i, len = *Len;

    switch (*type) {
#undef MPIR_OP_TYPE_MACRO
#define MPIR_OP_TYPE_MACRO(mpi_type_, c_type_, type_name_) MPIR_OP_TYPE_REDUCE_CASE(mpi_type_, c_type_, MPIR_LLXOR)
            /* no semicolons by necessity */
            MPIR_OP_TYPE_GROUP(C_INTEGER)

                /* MPI_LOGICAL requires special handling (MPIR_{TO,FROM}_FLOG) */
#if defined(HAVE_FORTRAN_BINDING)
#undef MPIR_OP_TYPE_MACRO_HAVE_FORTRAN
#define MPIR_OP_TYPE_MACRO_HAVE_FORTRAN(mpi_type_, c_type_, type_name_)  \
        case (mpi_type_): {                                                \
                c_type_ * restrict a = (c_type_ *)inoutvec;                \
                c_type_ * restrict b = (c_type_ *)invec;                   \
                for (i=0; i<len; i++)                                      \
                    a[i] = MPII_TO_FLOG(MPIR_LLXOR(MPII_FROM_FLOG(a[i]),   \
                                                   MPII_FROM_FLOG(b[i]))); \
                break;                                                     \
        }
                /* expand logicals (which may include MPI_C_BOOL, a non-Fortran type) */
                MPIR_OP_TYPE_GROUP(LOGICAL)
                MPIR_OP_TYPE_GROUP(LOGICAL_EXTRA)
                /* now revert _HAVE_FORTRAN macro to default */
#undef MPIR_OP_TYPE_MACRO_HAVE_FORTRAN
#define MPIR_OP_TYPE_MACRO_HAVE_FORTRAN(mpi_type_, c_type_, type_name_) MPIR_OP_TYPE_MACRO(mpi_type_, c_type_, type_name_)
#else
                /* if we don't have Fortran support then we don't have to jump through
                 * any hoops, simply expand the group */
                MPIR_OP_TYPE_GROUP(LOGICAL)
                MPIR_OP_TYPE_GROUP(LOGICAL_EXTRA)
#endif
                /* extra types that are not required to be supported by the MPI Standard */
                MPIR_OP_TYPE_GROUP(C_INTEGER_EXTRA)
                MPIR_OP_TYPE_GROUP(FORTRAN_INTEGER)
                MPIR_OP_TYPE_GROUP(FORTRAN_INTEGER_EXTRA)
#undef MPIR_OP_TYPE_MACRO
        default:
            MPIR_Assert(0);
            break;
    }
}


int MPIR_LXOR_check_dtype(MPI_Datatype type)
{
    switch (type) {
#undef MPIR_OP_TYPE_MACRO
#define MPIR_OP_TYPE_MACRO(mpi_type_, c_type_, type_name_) case (mpi_type_):
            MPIR_OP_TYPE_GROUP(C_INTEGER)
                MPIR_OP_TYPE_GROUP(LOGICAL)     /* no special handling needed in check_dtype code */
                MPIR_OP_TYPE_GROUP(LOGICAL_EXTRA)

                /* extra types that are not required to be supported by the MPI Standard */
                MPIR_OP_TYPE_GROUP(C_INTEGER_EXTRA)
                MPIR_OP_TYPE_GROUP(FORTRAN_INTEGER)
                MPIR_OP_TYPE_GROUP(FORTRAN_INTEGER_EXTRA)
#undef MPIR_OP_TYPE_MACRO
                return MPI_SUCCESS;
            /* --BEGIN ERROR HANDLING-- */
        default:
            return MPIR_Err_create_code(MPI_SUCCESS, MPIR_ERR_RECOVERABLE, __func__, __LINE__,
                                        MPI_ERR_OP, "**opundefined", "**opundefined %s",
                                        "MPI_LXOR");
            /* --END ERROR HANDLING-- */
    }
}
