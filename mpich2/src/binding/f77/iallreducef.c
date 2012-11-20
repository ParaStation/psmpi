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
extern FORT_DLL_SPEC void FORT_CALL MPIX_IALLREDUCE( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_iallreduce__( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_iallreduce( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_iallreduce_( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPIX_IALLREDUCE = PMPIX_IALLREDUCE
#pragma weak mpix_iallreduce__ = PMPIX_IALLREDUCE
#pragma weak mpix_iallreduce_ = PMPIX_IALLREDUCE
#pragma weak mpix_iallreduce = PMPIX_IALLREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPIX_IALLREDUCE = pmpix_iallreduce__
#pragma weak mpix_iallreduce__ = pmpix_iallreduce__
#pragma weak mpix_iallreduce_ = pmpix_iallreduce__
#pragma weak mpix_iallreduce = pmpix_iallreduce__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPIX_IALLREDUCE = pmpix_iallreduce_
#pragma weak mpix_iallreduce__ = pmpix_iallreduce_
#pragma weak mpix_iallreduce_ = pmpix_iallreduce_
#pragma weak mpix_iallreduce = pmpix_iallreduce_
#else
#pragma weak MPIX_IALLREDUCE = pmpix_iallreduce
#pragma weak mpix_iallreduce__ = pmpix_iallreduce
#pragma weak mpix_iallreduce_ = pmpix_iallreduce
#pragma weak mpix_iallreduce = pmpix_iallreduce
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPIX_IALLREDUCE( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak MPIX_IALLREDUCE = PMPIX_IALLREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpix_iallreduce__( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_iallreduce__ = pmpix_iallreduce__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpix_iallreduce( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_iallreduce = pmpix_iallreduce
#else
extern FORT_DLL_SPEC void FORT_CALL mpix_iallreduce_( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_iallreduce_ = pmpix_iallreduce_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPIX_IALLREDUCE  MPIX_IALLREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpix_iallreduce__  mpix_iallreduce__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpix_iallreduce  mpix_iallreduce
#else
#pragma _HP_SECONDARY_DEF pmpix_iallreduce_  mpix_iallreduce_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPIX_IALLREDUCE as PMPIX_IALLREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpix_iallreduce__ as pmpix_iallreduce__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpix_iallreduce as pmpix_iallreduce
#else
#pragma _CRI duplicate mpix_iallreduce_ as pmpix_iallreduce_
#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK) && \
    defined(USE_ONLY_MPI_NAMES)
extern FORT_DLL_SPEC void FORT_CALL MPIX_IALLREDUCE( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_iallreduce__( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_iallreduce( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_iallreduce_( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpix_iallreduce__ = MPIX_IALLREDUCE
#pragma weak mpix_iallreduce_ = MPIX_IALLREDUCE
#pragma weak mpix_iallreduce = MPIX_IALLREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPIX_IALLREDUCE = mpix_iallreduce__
#pragma weak mpix_iallreduce_ = mpix_iallreduce__
#pragma weak mpix_iallreduce = mpix_iallreduce__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPIX_IALLREDUCE = mpix_iallreduce_
#pragma weak mpix_iallreduce__ = mpix_iallreduce_
#pragma weak mpix_iallreduce = mpix_iallreduce_
#else
#pragma weak MPIX_IALLREDUCE = mpix_iallreduce
#pragma weak mpix_iallreduce__ = mpix_iallreduce
#pragma weak mpix_iallreduce_ = mpix_iallreduce
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPIX_IALLREDUCE( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpix_iallreduce__( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpix_iallreduce_( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpix_iallreduce( void*, void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpix_iallreduce__ = PMPIX_IALLREDUCE
#pragma weak pmpix_iallreduce_ = PMPIX_IALLREDUCE
#pragma weak pmpix_iallreduce = PMPIX_IALLREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPIX_IALLREDUCE = pmpix_iallreduce__
#pragma weak pmpix_iallreduce_ = pmpix_iallreduce__
#pragma weak pmpix_iallreduce = pmpix_iallreduce__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPIX_IALLREDUCE = pmpix_iallreduce_
#pragma weak pmpix_iallreduce__ = pmpix_iallreduce_
#pragma weak pmpix_iallreduce = pmpix_iallreduce_
#else
#pragma weak PMPIX_IALLREDUCE = pmpix_iallreduce
#pragma weak pmpix_iallreduce__ = pmpix_iallreduce
#pragma weak pmpix_iallreduce_ = pmpix_iallreduce
#endif /* Test on name mapping */
#endif /* Use multiple pragma weak */

#ifdef F77_NAME_UPPER
#define mpix_iallreduce_ PMPIX_IALLREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpix_iallreduce_ pmpix_iallreduce__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpix_iallreduce_ pmpix_iallreduce
#else
#define mpix_iallreduce_ pmpix_iallreduce_
#endif /* Test on name mapping */

/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPIX_Iallreduce
#define MPIX_Iallreduce PMPIX_Iallreduce 

#else

#ifdef F77_NAME_UPPER
#define mpix_iallreduce_ MPIX_IALLREDUCE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpix_iallreduce_ mpix_iallreduce__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpix_iallreduce_ mpix_iallreduce
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpix_iallreduce_ ( void*v1, void*v2, MPI_Fint *v3, MPI_Fint *v4, MPI_Fint *v5, MPI_Fint *v6, MPI_Fint *v7, MPI_Fint *ierr ){

#ifndef HAVE_MPI_F_INIT_WORKS_WITH_C
    if (MPIR_F_NeedInit){ mpirinitf_(); MPIR_F_NeedInit = 0; }
#endif
    if (v1 == MPIR_F_MPI_IN_PLACE) v1 = MPI_IN_PLACE;
    *ierr = MPIX_Iallreduce( v1, v2, *v3, (MPI_Datatype)(*v4), *v5, (MPI_Comm)(*v6), (MPI_Request *)(v7) );
}
