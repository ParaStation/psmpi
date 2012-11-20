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
extern FORT_DLL_SPEC void FORT_CALL MPIX_MUTEX_LOCK( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_mutex_lock__( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_mutex_lock( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_mutex_lock_( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPIX_MUTEX_LOCK = PMPIX_MUTEX_LOCK
#pragma weak mpix_mutex_lock__ = PMPIX_MUTEX_LOCK
#pragma weak mpix_mutex_lock_ = PMPIX_MUTEX_LOCK
#pragma weak mpix_mutex_lock = PMPIX_MUTEX_LOCK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPIX_MUTEX_LOCK = pmpix_mutex_lock__
#pragma weak mpix_mutex_lock__ = pmpix_mutex_lock__
#pragma weak mpix_mutex_lock_ = pmpix_mutex_lock__
#pragma weak mpix_mutex_lock = pmpix_mutex_lock__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPIX_MUTEX_LOCK = pmpix_mutex_lock_
#pragma weak mpix_mutex_lock__ = pmpix_mutex_lock_
#pragma weak mpix_mutex_lock_ = pmpix_mutex_lock_
#pragma weak mpix_mutex_lock = pmpix_mutex_lock_
#else
#pragma weak MPIX_MUTEX_LOCK = pmpix_mutex_lock
#pragma weak mpix_mutex_lock__ = pmpix_mutex_lock
#pragma weak mpix_mutex_lock_ = pmpix_mutex_lock
#pragma weak mpix_mutex_lock = pmpix_mutex_lock
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPIX_MUTEX_LOCK( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak MPIX_MUTEX_LOCK = PMPIX_MUTEX_LOCK
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpix_mutex_lock__( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_mutex_lock__ = pmpix_mutex_lock__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpix_mutex_lock( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_mutex_lock = pmpix_mutex_lock
#else
extern FORT_DLL_SPEC void FORT_CALL mpix_mutex_lock_( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpix_mutex_lock_ = pmpix_mutex_lock_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPIX_MUTEX_LOCK  MPIX_MUTEX_LOCK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpix_mutex_lock__  mpix_mutex_lock__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpix_mutex_lock  mpix_mutex_lock
#else
#pragma _HP_SECONDARY_DEF pmpix_mutex_lock_  mpix_mutex_lock_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPIX_MUTEX_LOCK as PMPIX_MUTEX_LOCK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpix_mutex_lock__ as pmpix_mutex_lock__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpix_mutex_lock as pmpix_mutex_lock
#else
#pragma _CRI duplicate mpix_mutex_lock_ as pmpix_mutex_lock_
#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK) && \
    defined(USE_ONLY_MPI_NAMES)
extern FORT_DLL_SPEC void FORT_CALL MPIX_MUTEX_LOCK( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_mutex_lock__( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_mutex_lock( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpix_mutex_lock_( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpix_mutex_lock__ = MPIX_MUTEX_LOCK
#pragma weak mpix_mutex_lock_ = MPIX_MUTEX_LOCK
#pragma weak mpix_mutex_lock = MPIX_MUTEX_LOCK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPIX_MUTEX_LOCK = mpix_mutex_lock__
#pragma weak mpix_mutex_lock_ = mpix_mutex_lock__
#pragma weak mpix_mutex_lock = mpix_mutex_lock__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPIX_MUTEX_LOCK = mpix_mutex_lock_
#pragma weak mpix_mutex_lock__ = mpix_mutex_lock_
#pragma weak mpix_mutex_lock = mpix_mutex_lock_
#else
#pragma weak MPIX_MUTEX_LOCK = mpix_mutex_lock
#pragma weak mpix_mutex_lock__ = mpix_mutex_lock
#pragma weak mpix_mutex_lock_ = mpix_mutex_lock
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPIX_MUTEX_LOCK( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpix_mutex_lock__( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpix_mutex_lock_( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpix_mutex_lock( MPIX_Mutex, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpix_mutex_lock__ = PMPIX_MUTEX_LOCK
#pragma weak pmpix_mutex_lock_ = PMPIX_MUTEX_LOCK
#pragma weak pmpix_mutex_lock = PMPIX_MUTEX_LOCK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPIX_MUTEX_LOCK = pmpix_mutex_lock__
#pragma weak pmpix_mutex_lock_ = pmpix_mutex_lock__
#pragma weak pmpix_mutex_lock = pmpix_mutex_lock__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPIX_MUTEX_LOCK = pmpix_mutex_lock_
#pragma weak pmpix_mutex_lock__ = pmpix_mutex_lock_
#pragma weak pmpix_mutex_lock = pmpix_mutex_lock_
#else
#pragma weak PMPIX_MUTEX_LOCK = pmpix_mutex_lock
#pragma weak pmpix_mutex_lock__ = pmpix_mutex_lock
#pragma weak pmpix_mutex_lock_ = pmpix_mutex_lock
#endif /* Test on name mapping */
#endif /* Use multiple pragma weak */

#ifdef F77_NAME_UPPER
#define mpix_mutex_lock_ PMPIX_MUTEX_LOCK
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpix_mutex_lock_ pmpix_mutex_lock__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpix_mutex_lock_ pmpix_mutex_lock
#else
#define mpix_mutex_lock_ pmpix_mutex_lock_
#endif /* Test on name mapping */

/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPIX_Mutex_lock
#define MPIX_Mutex_lock PMPIX_Mutex_lock 

#else

#ifdef F77_NAME_UPPER
#define mpix_mutex_lock_ MPIX_MUTEX_LOCK
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpix_mutex_lock_ mpix_mutex_lock__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpix_mutex_lock_ mpix_mutex_lock
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpix_mutex_lock_ ( MPIX_Mutex v1, MPI_Fint *v2, MPI_Fint *v3, MPI_Fint *ierr ){
    *ierr = MPIX_Mutex_lock( v1, *v2, *v3 );
}
