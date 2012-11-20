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
extern FORT_DLL_SPEC void FORT_CALL MPI_BSEND_INIT( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_bsend_init__( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_bsend_init( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_bsend_init_( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_BSEND_INIT = PMPI_BSEND_INIT
#pragma weak mpi_bsend_init__ = PMPI_BSEND_INIT
#pragma weak mpi_bsend_init_ = PMPI_BSEND_INIT
#pragma weak mpi_bsend_init = PMPI_BSEND_INIT
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_BSEND_INIT = pmpi_bsend_init__
#pragma weak mpi_bsend_init__ = pmpi_bsend_init__
#pragma weak mpi_bsend_init_ = pmpi_bsend_init__
#pragma weak mpi_bsend_init = pmpi_bsend_init__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_BSEND_INIT = pmpi_bsend_init_
#pragma weak mpi_bsend_init__ = pmpi_bsend_init_
#pragma weak mpi_bsend_init_ = pmpi_bsend_init_
#pragma weak mpi_bsend_init = pmpi_bsend_init_
#else
#pragma weak MPI_BSEND_INIT = pmpi_bsend_init
#pragma weak mpi_bsend_init__ = pmpi_bsend_init
#pragma weak mpi_bsend_init_ = pmpi_bsend_init
#pragma weak mpi_bsend_init = pmpi_bsend_init
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_BSEND_INIT( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak MPI_BSEND_INIT = PMPI_BSEND_INIT
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_bsend_init__( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_bsend_init__ = pmpi_bsend_init__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_bsend_init( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_bsend_init = pmpi_bsend_init
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_bsend_init_( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_bsend_init_ = pmpi_bsend_init_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_BSEND_INIT  MPI_BSEND_INIT
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_bsend_init__  mpi_bsend_init__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_bsend_init  mpi_bsend_init
#else
#pragma _HP_SECONDARY_DEF pmpi_bsend_init_  mpi_bsend_init_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_BSEND_INIT as PMPI_BSEND_INIT
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_bsend_init__ as pmpi_bsend_init__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_bsend_init as pmpi_bsend_init
#else
#pragma _CRI duplicate mpi_bsend_init_ as pmpi_bsend_init_
#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK) && \
    defined(USE_ONLY_MPI_NAMES)
extern FORT_DLL_SPEC void FORT_CALL MPI_BSEND_INIT( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_bsend_init__( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_bsend_init( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_bsend_init_( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_bsend_init__ = MPI_BSEND_INIT
#pragma weak mpi_bsend_init_ = MPI_BSEND_INIT
#pragma weak mpi_bsend_init = MPI_BSEND_INIT
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_BSEND_INIT = mpi_bsend_init__
#pragma weak mpi_bsend_init_ = mpi_bsend_init__
#pragma weak mpi_bsend_init = mpi_bsend_init__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_BSEND_INIT = mpi_bsend_init_
#pragma weak mpi_bsend_init__ = mpi_bsend_init_
#pragma weak mpi_bsend_init = mpi_bsend_init_
#else
#pragma weak MPI_BSEND_INIT = mpi_bsend_init
#pragma weak mpi_bsend_init__ = mpi_bsend_init
#pragma weak mpi_bsend_init_ = mpi_bsend_init
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_BSEND_INIT( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_bsend_init__( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_bsend_init_( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_bsend_init( void*, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_bsend_init__ = PMPI_BSEND_INIT
#pragma weak pmpi_bsend_init_ = PMPI_BSEND_INIT
#pragma weak pmpi_bsend_init = PMPI_BSEND_INIT
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_BSEND_INIT = pmpi_bsend_init__
#pragma weak pmpi_bsend_init_ = pmpi_bsend_init__
#pragma weak pmpi_bsend_init = pmpi_bsend_init__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_BSEND_INIT = pmpi_bsend_init_
#pragma weak pmpi_bsend_init__ = pmpi_bsend_init_
#pragma weak pmpi_bsend_init = pmpi_bsend_init_
#else
#pragma weak PMPI_BSEND_INIT = pmpi_bsend_init
#pragma weak pmpi_bsend_init__ = pmpi_bsend_init
#pragma weak pmpi_bsend_init_ = pmpi_bsend_init
#endif /* Test on name mapping */
#endif /* Use multiple pragma weak */

#ifdef F77_NAME_UPPER
#define mpi_bsend_init_ PMPI_BSEND_INIT
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_bsend_init_ pmpi_bsend_init__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_bsend_init_ pmpi_bsend_init
#else
#define mpi_bsend_init_ pmpi_bsend_init_
#endif /* Test on name mapping */

/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Bsend_init
#define MPI_Bsend_init PMPI_Bsend_init 

#else

#ifdef F77_NAME_UPPER
#define mpi_bsend_init_ MPI_BSEND_INIT
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_bsend_init_ mpi_bsend_init__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_bsend_init_ mpi_bsend_init
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_bsend_init_ ( void*v1, MPI_Fint *v2, MPI_Fint *v3, MPI_Fint *v4, MPI_Fint *v5, MPI_Fint *v6, MPI_Fint *v7, MPI_Fint *ierr ){
    *ierr = MPI_Bsend_init( v1, *v2, (MPI_Datatype)(*v3), *v4, *v5, (MPI_Comm)(*v6), (MPI_Request *)(v7) );
}
