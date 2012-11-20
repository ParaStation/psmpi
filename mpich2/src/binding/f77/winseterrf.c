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
extern FORT_DLL_SPEC void FORT_CALL MPI_WIN_SET_ERRHANDLER( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_set_errhandler__( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_set_errhandler( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_set_errhandler_( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_WIN_SET_ERRHANDLER = PMPI_WIN_SET_ERRHANDLER
#pragma weak mpi_win_set_errhandler__ = PMPI_WIN_SET_ERRHANDLER
#pragma weak mpi_win_set_errhandler_ = PMPI_WIN_SET_ERRHANDLER
#pragma weak mpi_win_set_errhandler = PMPI_WIN_SET_ERRHANDLER
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_WIN_SET_ERRHANDLER = pmpi_win_set_errhandler__
#pragma weak mpi_win_set_errhandler__ = pmpi_win_set_errhandler__
#pragma weak mpi_win_set_errhandler_ = pmpi_win_set_errhandler__
#pragma weak mpi_win_set_errhandler = pmpi_win_set_errhandler__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_WIN_SET_ERRHANDLER = pmpi_win_set_errhandler_
#pragma weak mpi_win_set_errhandler__ = pmpi_win_set_errhandler_
#pragma weak mpi_win_set_errhandler_ = pmpi_win_set_errhandler_
#pragma weak mpi_win_set_errhandler = pmpi_win_set_errhandler_
#else
#pragma weak MPI_WIN_SET_ERRHANDLER = pmpi_win_set_errhandler
#pragma weak mpi_win_set_errhandler__ = pmpi_win_set_errhandler
#pragma weak mpi_win_set_errhandler_ = pmpi_win_set_errhandler
#pragma weak mpi_win_set_errhandler = pmpi_win_set_errhandler
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_WIN_SET_ERRHANDLER( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak MPI_WIN_SET_ERRHANDLER = PMPI_WIN_SET_ERRHANDLER
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_win_set_errhandler__( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_win_set_errhandler__ = pmpi_win_set_errhandler__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_win_set_errhandler( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_win_set_errhandler = pmpi_win_set_errhandler
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_win_set_errhandler_( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_win_set_errhandler_ = pmpi_win_set_errhandler_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_WIN_SET_ERRHANDLER  MPI_WIN_SET_ERRHANDLER
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_win_set_errhandler__  mpi_win_set_errhandler__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_win_set_errhandler  mpi_win_set_errhandler
#else
#pragma _HP_SECONDARY_DEF pmpi_win_set_errhandler_  mpi_win_set_errhandler_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_WIN_SET_ERRHANDLER as PMPI_WIN_SET_ERRHANDLER
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_win_set_errhandler__ as pmpi_win_set_errhandler__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_win_set_errhandler as pmpi_win_set_errhandler
#else
#pragma _CRI duplicate mpi_win_set_errhandler_ as pmpi_win_set_errhandler_
#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK) && \
    defined(USE_ONLY_MPI_NAMES)
extern FORT_DLL_SPEC void FORT_CALL MPI_WIN_SET_ERRHANDLER( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_set_errhandler__( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_set_errhandler( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_set_errhandler_( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_win_set_errhandler__ = MPI_WIN_SET_ERRHANDLER
#pragma weak mpi_win_set_errhandler_ = MPI_WIN_SET_ERRHANDLER
#pragma weak mpi_win_set_errhandler = MPI_WIN_SET_ERRHANDLER
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_WIN_SET_ERRHANDLER = mpi_win_set_errhandler__
#pragma weak mpi_win_set_errhandler_ = mpi_win_set_errhandler__
#pragma weak mpi_win_set_errhandler = mpi_win_set_errhandler__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_WIN_SET_ERRHANDLER = mpi_win_set_errhandler_
#pragma weak mpi_win_set_errhandler__ = mpi_win_set_errhandler_
#pragma weak mpi_win_set_errhandler = mpi_win_set_errhandler_
#else
#pragma weak MPI_WIN_SET_ERRHANDLER = mpi_win_set_errhandler
#pragma weak mpi_win_set_errhandler__ = mpi_win_set_errhandler
#pragma weak mpi_win_set_errhandler_ = mpi_win_set_errhandler
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_WIN_SET_ERRHANDLER( MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_set_errhandler__( MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_set_errhandler_( MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_set_errhandler( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_win_set_errhandler__ = PMPI_WIN_SET_ERRHANDLER
#pragma weak pmpi_win_set_errhandler_ = PMPI_WIN_SET_ERRHANDLER
#pragma weak pmpi_win_set_errhandler = PMPI_WIN_SET_ERRHANDLER
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_WIN_SET_ERRHANDLER = pmpi_win_set_errhandler__
#pragma weak pmpi_win_set_errhandler_ = pmpi_win_set_errhandler__
#pragma weak pmpi_win_set_errhandler = pmpi_win_set_errhandler__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_WIN_SET_ERRHANDLER = pmpi_win_set_errhandler_
#pragma weak pmpi_win_set_errhandler__ = pmpi_win_set_errhandler_
#pragma weak pmpi_win_set_errhandler = pmpi_win_set_errhandler_
#else
#pragma weak PMPI_WIN_SET_ERRHANDLER = pmpi_win_set_errhandler
#pragma weak pmpi_win_set_errhandler__ = pmpi_win_set_errhandler
#pragma weak pmpi_win_set_errhandler_ = pmpi_win_set_errhandler
#endif /* Test on name mapping */
#endif /* Use multiple pragma weak */

#ifdef F77_NAME_UPPER
#define mpi_win_set_errhandler_ PMPI_WIN_SET_ERRHANDLER
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_win_set_errhandler_ pmpi_win_set_errhandler__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_win_set_errhandler_ pmpi_win_set_errhandler
#else
#define mpi_win_set_errhandler_ pmpi_win_set_errhandler_
#endif /* Test on name mapping */

/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Win_set_errhandler
#define MPI_Win_set_errhandler PMPI_Win_set_errhandler 

#else

#ifdef F77_NAME_UPPER
#define mpi_win_set_errhandler_ MPI_WIN_SET_ERRHANDLER
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_win_set_errhandler_ mpi_win_set_errhandler__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_win_set_errhandler_ mpi_win_set_errhandler
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_win_set_errhandler_ ( MPI_Fint *v1, MPI_Fint *v2, MPI_Fint *ierr ){
    *ierr = MPI_Win_set_errhandler( *v1, *v2 );
}
