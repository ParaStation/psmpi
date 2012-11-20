/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*  
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 * This file is automatically generated by buildiface 
 * DO NOT EDIT
 */
#include "mpi_fortimpl.h"


/* Begin MPI profiling block */
#if defined(USE_WEAK_SYMBOLS) && !defined(USE_ONLY_MPI_NAMES) 
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_CART_GET( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_get__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_get( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_get_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_CART_GET = PMPI_CART_GET
#pragma weak mpi_cart_get__ = PMPI_CART_GET
#pragma weak mpi_cart_get_ = PMPI_CART_GET
#pragma weak mpi_cart_get = PMPI_CART_GET
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_CART_GET = pmpi_cart_get__
#pragma weak mpi_cart_get__ = pmpi_cart_get__
#pragma weak mpi_cart_get_ = pmpi_cart_get__
#pragma weak mpi_cart_get = pmpi_cart_get__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_CART_GET = pmpi_cart_get_
#pragma weak mpi_cart_get__ = pmpi_cart_get_
#pragma weak mpi_cart_get_ = pmpi_cart_get_
#pragma weak mpi_cart_get = pmpi_cart_get_
#else
#pragma weak MPI_CART_GET = pmpi_cart_get
#pragma weak mpi_cart_get__ = pmpi_cart_get
#pragma weak mpi_cart_get_ = pmpi_cart_get
#pragma weak mpi_cart_get = pmpi_cart_get
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_CART_GET( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak MPI_CART_GET = PMPI_CART_GET
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_get__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_cart_get__ = pmpi_cart_get__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_get( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_cart_get = pmpi_cart_get
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_get_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_cart_get_ = pmpi_cart_get_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_CART_GET  MPI_CART_GET
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_cart_get__  mpi_cart_get__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_cart_get  mpi_cart_get
#else
#pragma _HP_SECONDARY_DEF pmpi_cart_get_  mpi_cart_get_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_CART_GET as PMPI_CART_GET
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_cart_get__ as pmpi_cart_get__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_cart_get as pmpi_cart_get
#else
#pragma _CRI duplicate mpi_cart_get_ as pmpi_cart_get_
#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK) && \
    defined(USE_ONLY_MPI_NAMES)
extern FORT_DLL_SPEC void FORT_CALL MPI_CART_GET( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_get__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_get( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_get_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_cart_get__ = MPI_CART_GET
#pragma weak mpi_cart_get_ = MPI_CART_GET
#pragma weak mpi_cart_get = MPI_CART_GET
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_CART_GET = mpi_cart_get__
#pragma weak mpi_cart_get_ = mpi_cart_get__
#pragma weak mpi_cart_get = mpi_cart_get__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_CART_GET = mpi_cart_get_
#pragma weak mpi_cart_get__ = mpi_cart_get_
#pragma weak mpi_cart_get = mpi_cart_get_
#else
#pragma weak MPI_CART_GET = mpi_cart_get
#pragma weak mpi_cart_get__ = mpi_cart_get
#pragma weak mpi_cart_get_ = mpi_cart_get
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_CART_GET( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_get__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_get_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_get( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_cart_get__ = PMPI_CART_GET
#pragma weak pmpi_cart_get_ = PMPI_CART_GET
#pragma weak pmpi_cart_get = PMPI_CART_GET
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_CART_GET = pmpi_cart_get__
#pragma weak pmpi_cart_get_ = pmpi_cart_get__
#pragma weak pmpi_cart_get = pmpi_cart_get__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_CART_GET = pmpi_cart_get_
#pragma weak pmpi_cart_get__ = pmpi_cart_get_
#pragma weak pmpi_cart_get = pmpi_cart_get_
#else
#pragma weak PMPI_CART_GET = pmpi_cart_get
#pragma weak pmpi_cart_get__ = pmpi_cart_get
#pragma weak pmpi_cart_get_ = pmpi_cart_get
#endif /* Test on name mapping */
#endif /* Use multiple pragma weak */

#ifdef F77_NAME_UPPER
#define mpi_cart_get_ PMPI_CART_GET
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_cart_get_ pmpi_cart_get__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_cart_get_ pmpi_cart_get
#else
#define mpi_cart_get_ pmpi_cart_get_
#endif /* Test on name mapping */

/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Cart_get
#define MPI_Cart_get PMPI_Cart_get 

#else

#ifdef F77_NAME_UPPER
#define mpi_cart_get_ MPI_CART_GET
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_cart_get_ mpi_cart_get__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_cart_get_ mpi_cart_get
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_cart_get_ ( MPI_Fint *v1, MPI_Fint *v2, MPI_Fint *v3, MPI_Fint *v4, MPI_Fint *v5, MPI_Fint *ierr ){
    *ierr = MPI_Cart_get( (MPI_Comm)(*v1), *v2, v3, v4, v5 );

    if (*ierr == MPI_SUCCESS) {int li;
     for (li=0; li<*v2; li++) {
        v4[li] = MPIR_TO_FLOG(v4[li]);
     }
    }
}
