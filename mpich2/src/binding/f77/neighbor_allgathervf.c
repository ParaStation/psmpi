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
extern FORT_DLL_SPEC void FORT_CALL MPIX_NEIGHBOR_ALLGATHERV( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_neighbor_allgatherv__( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_neighbor_allgatherv( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_neighbor_allgatherv_( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPIX_NEIGHBOR_ALLGATHERV = PMPIX_NEIGHBOR_ALLGATHERV
#pragma weak mpix_neighbor_allgatherv__ = PMPIX_NEIGHBOR_ALLGATHERV
#pragma weak mpix_neighbor_allgatherv_ = PMPIX_NEIGHBOR_ALLGATHERV
#pragma weak mpix_neighbor_allgatherv = PMPIX_NEIGHBOR_ALLGATHERV
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPIX_NEIGHBOR_ALLGATHERV = pmpix_neighbor_allgatherv__
#pragma weak mpix_neighbor_allgatherv__ = pmpix_neighbor_allgatherv__
#pragma weak mpix_neighbor_allgatherv_ = pmpix_neighbor_allgatherv__
#pragma weak mpix_neighbor_allgatherv = pmpix_neighbor_allgatherv__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPIX_NEIGHBOR_ALLGATHERV = pmpix_neighbor_allgatherv_
#pragma weak mpix_neighbor_allgatherv__ = pmpix_neighbor_allgatherv_
#pragma weak mpix_neighbor_allgatherv_ = pmpix_neighbor_allgatherv_
#pragma weak mpix_neighbor_allgatherv = pmpix_neighbor_allgatherv_
#else
#pragma weak MPIX_NEIGHBOR_ALLGATHERV = pmpix_neighbor_allgatherv
#pragma weak mpix_neighbor_allgatherv__ = pmpix_neighbor_allgatherv
#pragma weak mpix_neighbor_allgatherv_ = pmpix_neighbor_allgatherv
#pragma weak mpix_neighbor_allgatherv = pmpix_neighbor_allgatherv
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPIX_NEIGHBOR_ALLGATHERV( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak MPIX_NEIGHBOR_ALLGATHERV = PMPIX_NEIGHBOR_ALLGATHERV
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpix_neighbor_allgatherv__( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_neighbor_allgatherv__ = pmpix_neighbor_allgatherv__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpix_neighbor_allgatherv( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_neighbor_allgatherv = pmpix_neighbor_allgatherv
#else
extern FORT_DLL_SPEC void FORT_CALL mpix_neighbor_allgatherv_( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_neighbor_allgatherv_ = pmpix_neighbor_allgatherv_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPIX_NEIGHBOR_ALLGATHERV  MPIX_NEIGHBOR_ALLGATHERV
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpix_neighbor_allgatherv__  mpix_neighbor_allgatherv__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpix_neighbor_allgatherv  mpix_neighbor_allgatherv
#else
#pragma _HP_SECONDARY_DEF pmpix_neighbor_allgatherv_  mpix_neighbor_allgatherv_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPIX_NEIGHBOR_ALLGATHERV as PMPIX_NEIGHBOR_ALLGATHERV
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpix_neighbor_allgatherv__ as pmpix_neighbor_allgatherv__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpix_neighbor_allgatherv as pmpix_neighbor_allgatherv
#else
#pragma _CRI duplicate mpix_neighbor_allgatherv_ as pmpix_neighbor_allgatherv_
#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK) && \
    defined(USE_ONLY_MPI_NAMES)
extern FORT_DLL_SPEC void FORT_CALL MPIX_NEIGHBOR_ALLGATHERV( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_neighbor_allgatherv__( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_neighbor_allgatherv( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_neighbor_allgatherv_( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpix_neighbor_allgatherv__ = MPIX_NEIGHBOR_ALLGATHERV
#pragma weak mpix_neighbor_allgatherv_ = MPIX_NEIGHBOR_ALLGATHERV
#pragma weak mpix_neighbor_allgatherv = MPIX_NEIGHBOR_ALLGATHERV
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPIX_NEIGHBOR_ALLGATHERV = mpix_neighbor_allgatherv__
#pragma weak mpix_neighbor_allgatherv_ = mpix_neighbor_allgatherv__
#pragma weak mpix_neighbor_allgatherv = mpix_neighbor_allgatherv__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPIX_NEIGHBOR_ALLGATHERV = mpix_neighbor_allgatherv_
#pragma weak mpix_neighbor_allgatherv__ = mpix_neighbor_allgatherv_
#pragma weak mpix_neighbor_allgatherv = mpix_neighbor_allgatherv_
#else
#pragma weak MPIX_NEIGHBOR_ALLGATHERV = mpix_neighbor_allgatherv
#pragma weak mpix_neighbor_allgatherv__ = mpix_neighbor_allgatherv
#pragma weak mpix_neighbor_allgatherv_ = mpix_neighbor_allgatherv
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPIX_NEIGHBOR_ALLGATHERV( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpix_neighbor_allgatherv__( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpix_neighbor_allgatherv_( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpix_neighbor_allgatherv( void*, MPI_Fint *, MPI_Fint *, void*, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpix_neighbor_allgatherv__ = PMPIX_NEIGHBOR_ALLGATHERV
#pragma weak pmpix_neighbor_allgatherv_ = PMPIX_NEIGHBOR_ALLGATHERV
#pragma weak pmpix_neighbor_allgatherv = PMPIX_NEIGHBOR_ALLGATHERV
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPIX_NEIGHBOR_ALLGATHERV = pmpix_neighbor_allgatherv__
#pragma weak pmpix_neighbor_allgatherv_ = pmpix_neighbor_allgatherv__
#pragma weak pmpix_neighbor_allgatherv = pmpix_neighbor_allgatherv__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPIX_NEIGHBOR_ALLGATHERV = pmpix_neighbor_allgatherv_
#pragma weak pmpix_neighbor_allgatherv__ = pmpix_neighbor_allgatherv_
#pragma weak pmpix_neighbor_allgatherv = pmpix_neighbor_allgatherv_
#else
#pragma weak PMPIX_NEIGHBOR_ALLGATHERV = pmpix_neighbor_allgatherv
#pragma weak pmpix_neighbor_allgatherv__ = pmpix_neighbor_allgatherv
#pragma weak pmpix_neighbor_allgatherv_ = pmpix_neighbor_allgatherv
#endif /* Test on name mapping */
#endif /* Use multiple pragma weak */

#ifdef F77_NAME_UPPER
#define mpix_neighbor_allgatherv_ PMPIX_NEIGHBOR_ALLGATHERV
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpix_neighbor_allgatherv_ pmpix_neighbor_allgatherv__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpix_neighbor_allgatherv_ pmpix_neighbor_allgatherv
#else
#define mpix_neighbor_allgatherv_ pmpix_neighbor_allgatherv_
#endif /* Test on name mapping */

/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPIX_Neighbor_allgatherv
#define MPIX_Neighbor_allgatherv PMPIX_Neighbor_allgatherv 

#else

#ifdef F77_NAME_UPPER
#define mpix_neighbor_allgatherv_ MPIX_NEIGHBOR_ALLGATHERV
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpix_neighbor_allgatherv_ mpix_neighbor_allgatherv__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpix_neighbor_allgatherv_ mpix_neighbor_allgatherv
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpix_neighbor_allgatherv_ ( void*v1, MPI_Fint *v2, MPI_Fint *v3, void*v4, MPI_Fint v5[], MPI_Fint v6[], MPI_Fint *v7, MPI_Fint *v8, MPI_Fint *ierr ){
    *ierr = MPIX_Neighbor_allgatherv( v1, *v2, (MPI_Datatype)(*v3), v4, v5, v6, (MPI_Datatype)(*v7), (MPI_Comm)(*v8) );
}
