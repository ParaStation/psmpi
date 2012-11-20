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
extern FORT_DLL_SPEC void FORT_CALL MPIX_IREDUCE( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_ireduce__( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_ireduce( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_ireduce_( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPIX_IREDUCE = PMPIX_IREDUCE
#pragma weak mpix_ireduce__ = PMPIX_IREDUCE
#pragma weak mpix_ireduce_ = PMPIX_IREDUCE
#pragma weak mpix_ireduce = PMPIX_IREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPIX_IREDUCE = pmpix_ireduce__
#pragma weak mpix_ireduce__ = pmpix_ireduce__
#pragma weak mpix_ireduce_ = pmpix_ireduce__
#pragma weak mpix_ireduce = pmpix_ireduce__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPIX_IREDUCE = pmpix_ireduce_
#pragma weak mpix_ireduce__ = pmpix_ireduce_
#pragma weak mpix_ireduce_ = pmpix_ireduce_
#pragma weak mpix_ireduce = pmpix_ireduce_
#else
#pragma weak MPIX_IREDUCE = pmpix_ireduce
#pragma weak mpix_ireduce__ = pmpix_ireduce
#pragma weak mpix_ireduce_ = pmpix_ireduce
#pragma weak mpix_ireduce = pmpix_ireduce
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPIX_IREDUCE( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak MPIX_IREDUCE = PMPIX_IREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpix_ireduce__( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_ireduce__ = pmpix_ireduce__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpix_ireduce( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_ireduce = pmpix_ireduce
#else
extern FORT_DLL_SPEC void FORT_CALL mpix_ireduce_( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_ireduce_ = pmpix_ireduce_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPIX_IREDUCE  MPIX_IREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpix_ireduce__  mpix_ireduce__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpix_ireduce  mpix_ireduce
#else
#pragma _HP_SECONDARY_DEF pmpix_ireduce_  mpix_ireduce_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPIX_IREDUCE as PMPIX_IREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpix_ireduce__ as pmpix_ireduce__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpix_ireduce as pmpix_ireduce
#else
#pragma _CRI duplicate mpix_ireduce_ as pmpix_ireduce_
#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK) && \
    defined(USE_ONLY_MPI_NAMES)
extern FORT_DLL_SPEC void FORT_CALL MPIX_IREDUCE( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_ireduce__( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_ireduce( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_ireduce_( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpix_ireduce__ = MPIX_IREDUCE
#pragma weak mpix_ireduce_ = MPIX_IREDUCE
#pragma weak mpix_ireduce = MPIX_IREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPIX_IREDUCE = mpix_ireduce__
#pragma weak mpix_ireduce_ = mpix_ireduce__
#pragma weak mpix_ireduce = mpix_ireduce__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPIX_IREDUCE = mpix_ireduce_
#pragma weak mpix_ireduce__ = mpix_ireduce_
#pragma weak mpix_ireduce = mpix_ireduce_
#else
#pragma weak MPIX_IREDUCE = mpix_ireduce
#pragma weak mpix_ireduce__ = mpix_ireduce
#pragma weak mpix_ireduce_ = mpix_ireduce
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPIX_IREDUCE( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpix_ireduce__( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpix_ireduce_( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpix_ireduce( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpix_ireduce__ = PMPIX_IREDUCE
#pragma weak pmpix_ireduce_ = PMPIX_IREDUCE
#pragma weak pmpix_ireduce = PMPIX_IREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPIX_IREDUCE = pmpix_ireduce__
#pragma weak pmpix_ireduce_ = pmpix_ireduce__
#pragma weak pmpix_ireduce = pmpix_ireduce__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPIX_IREDUCE = pmpix_ireduce_
#pragma weak pmpix_ireduce__ = pmpix_ireduce_
#pragma weak pmpix_ireduce = pmpix_ireduce_
#else
#pragma weak PMPIX_IREDUCE = pmpix_ireduce
#pragma weak pmpix_ireduce__ = pmpix_ireduce
#pragma weak pmpix_ireduce_ = pmpix_ireduce
#endif /* Test on name mapping */
#endif /* Use multiple pragma weak */

#ifdef F77_NAME_UPPER
#define mpix_ireduce_ PMPIX_IREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpix_ireduce_ pmpix_ireduce__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpix_ireduce_ pmpix_ireduce
#else
#define mpix_ireduce_ pmpix_ireduce_
#endif /* Test on name mapping */

/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPIX_Ireduce
#define MPIX_Ireduce PMPIX_Ireduce 

#else

#ifdef F77_NAME_UPPER
#define mpix_ireduce_ MPIX_IREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpix_ireduce_ mpix_ireduce__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpix_ireduce_ mpix_ireduce
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpix_ireduce_ ( void*v1, void*v2, MPI_Fint *v3, MPI_Fint *v4, MPI_Fint *v5, MPI_Fint *v6, MPI_Fint *v7, MPI_Fint *v8, MPI_Fint *ierr ){

#ifndef HAVE_MPI_F_INIT_WORKS_WITH_C
    if (MPIR_F_NeedInit){ mpirinitf_(); MPIR_F_NeedInit = 0; }
#endif
    if (v1 == MPIR_F_MPI_IN_PLACE) v1 = MPI_IN_PLACE;
    *ierr = MPIX_Ireduce( v1, v2, *v3, (MPI_Datatype)(*v4), *v5, *v6, (MPI_Comm)(*v7), (MPI_Request *)(v8) );
}
