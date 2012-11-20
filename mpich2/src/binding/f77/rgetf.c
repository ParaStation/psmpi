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
extern FORT_DLL_SPEC void FORT_CALL MPIX_RGET( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_rget__( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_rget( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_rget_( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPIX_RGET = PMPIX_RGET
#pragma weak mpix_rget__ = PMPIX_RGET
#pragma weak mpix_rget_ = PMPIX_RGET
#pragma weak mpix_rget = PMPIX_RGET
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPIX_RGET = pmpix_rget__
#pragma weak mpix_rget__ = pmpix_rget__
#pragma weak mpix_rget_ = pmpix_rget__
#pragma weak mpix_rget = pmpix_rget__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPIX_RGET = pmpix_rget_
#pragma weak mpix_rget__ = pmpix_rget_
#pragma weak mpix_rget_ = pmpix_rget_
#pragma weak mpix_rget = pmpix_rget_
#else
#pragma weak MPIX_RGET = pmpix_rget
#pragma weak mpix_rget__ = pmpix_rget
#pragma weak mpix_rget_ = pmpix_rget
#pragma weak mpix_rget = pmpix_rget
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPIX_RGET( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak MPIX_RGET = PMPIX_RGET
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpix_rget__( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_rget__ = pmpix_rget__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpix_rget( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_rget = pmpix_rget
#else
extern FORT_DLL_SPEC void FORT_CALL mpix_rget_( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_rget_ = pmpix_rget_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPIX_RGET  MPIX_RGET
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpix_rget__  mpix_rget__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpix_rget  mpix_rget
#else
#pragma _HP_SECONDARY_DEF pmpix_rget_  mpix_rget_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPIX_RGET as PMPIX_RGET
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpix_rget__ as pmpix_rget__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpix_rget as pmpix_rget
#else
#pragma _CRI duplicate mpix_rget_ as pmpix_rget_
#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK) && \
    defined(USE_ONLY_MPI_NAMES)
extern FORT_DLL_SPEC void FORT_CALL MPIX_RGET( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_rget__( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_rget( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_rget_( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpix_rget__ = MPIX_RGET
#pragma weak mpix_rget_ = MPIX_RGET
#pragma weak mpix_rget = MPIX_RGET
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPIX_RGET = mpix_rget__
#pragma weak mpix_rget_ = mpix_rget__
#pragma weak mpix_rget = mpix_rget__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPIX_RGET = mpix_rget_
#pragma weak mpix_rget__ = mpix_rget_
#pragma weak mpix_rget = mpix_rget_
#else
#pragma weak MPIX_RGET = mpix_rget
#pragma weak mpix_rget__ = mpix_rget
#pragma weak mpix_rget_ = mpix_rget
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPIX_RGET( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpix_rget__( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpix_rget_( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpix_rget( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpix_rget__ = PMPIX_RGET
#pragma weak pmpix_rget_ = PMPIX_RGET
#pragma weak pmpix_rget = PMPIX_RGET
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPIX_RGET = pmpix_rget__
#pragma weak pmpix_rget_ = pmpix_rget__
#pragma weak pmpix_rget = pmpix_rget__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPIX_RGET = pmpix_rget_
#pragma weak pmpix_rget__ = pmpix_rget_
#pragma weak pmpix_rget = pmpix_rget_
#else
#pragma weak PMPIX_RGET = pmpix_rget
#pragma weak pmpix_rget__ = pmpix_rget
#pragma weak pmpix_rget_ = pmpix_rget
#endif /* Test on name mapping */
#endif /* Use multiple pragma weak */

#ifdef F77_NAME_UPPER
#define mpix_rget_ PMPIX_RGET
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpix_rget_ pmpix_rget__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpix_rget_ pmpix_rget
#else
#define mpix_rget_ pmpix_rget_
#endif /* Test on name mapping */

/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPIX_Rget
#define MPIX_Rget PMPIX_Rget 

#else

#ifdef F77_NAME_UPPER
#define mpix_rget_ MPIX_RGET
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpix_rget_ mpix_rget__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpix_rget_ mpix_rget
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpix_rget_ ( void*v1, MPI_Fint *v2, MPI_Fint *v3, MPI_Fint *v4, MPI_Fint *v5, MPI_Fint *v6, MPI_Fint *v7, MPI_Fint *v8, MPI_Fint *v9, MPI_Fint *ierr ){
    *ierr = MPIX_Rget( v1, *v2, (MPI_Datatype)(*v3), *v4, *v5, *v6, (MPI_Datatype)(*v7), *v8, (MPI_Request *)(v9) );
}
