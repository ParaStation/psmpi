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
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_SEEK( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_seek__( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_seek( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_seek_( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_FILE_SEEK = PMPI_FILE_SEEK
#pragma weak mpi_file_seek__ = PMPI_FILE_SEEK
#pragma weak mpi_file_seek_ = PMPI_FILE_SEEK
#pragma weak mpi_file_seek = PMPI_FILE_SEEK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_FILE_SEEK = pmpi_file_seek__
#pragma weak mpi_file_seek__ = pmpi_file_seek__
#pragma weak mpi_file_seek_ = pmpi_file_seek__
#pragma weak mpi_file_seek = pmpi_file_seek__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_FILE_SEEK = pmpi_file_seek_
#pragma weak mpi_file_seek__ = pmpi_file_seek_
#pragma weak mpi_file_seek_ = pmpi_file_seek_
#pragma weak mpi_file_seek = pmpi_file_seek_
#else
#pragma weak MPI_FILE_SEEK = pmpi_file_seek
#pragma weak mpi_file_seek__ = pmpi_file_seek
#pragma weak mpi_file_seek_ = pmpi_file_seek
#pragma weak mpi_file_seek = pmpi_file_seek
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_SEEK( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );

#pragma weak MPI_FILE_SEEK = PMPI_FILE_SEEK
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_file_seek__( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_file_seek__ = pmpi_file_seek__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_file_seek( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_file_seek = pmpi_file_seek
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_file_seek_( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_file_seek_ = pmpi_file_seek_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_FILE_SEEK  MPI_FILE_SEEK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_file_seek__  mpi_file_seek__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_file_seek  mpi_file_seek
#else
#pragma _HP_SECONDARY_DEF pmpi_file_seek_  mpi_file_seek_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_FILE_SEEK as PMPI_FILE_SEEK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_file_seek__ as pmpi_file_seek__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_file_seek as pmpi_file_seek
#else
#pragma _CRI duplicate mpi_file_seek_ as pmpi_file_seek_
#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK) && \
    defined(USE_ONLY_MPI_NAMES)
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_SEEK( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_seek__( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_seek( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_seek_( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_file_seek__ = MPI_FILE_SEEK
#pragma weak mpi_file_seek_ = MPI_FILE_SEEK
#pragma weak mpi_file_seek = MPI_FILE_SEEK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_FILE_SEEK = mpi_file_seek__
#pragma weak mpi_file_seek_ = mpi_file_seek__
#pragma weak mpi_file_seek = mpi_file_seek__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_FILE_SEEK = mpi_file_seek_
#pragma weak mpi_file_seek__ = mpi_file_seek_
#pragma weak mpi_file_seek = mpi_file_seek_
#else
#pragma weak MPI_FILE_SEEK = mpi_file_seek
#pragma weak mpi_file_seek__ = mpi_file_seek
#pragma weak mpi_file_seek_ = mpi_file_seek
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_FILE_SEEK( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_seek__( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_seek_( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_seek( MPI_Fint *, MPI_Offset *, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_file_seek__ = PMPI_FILE_SEEK
#pragma weak pmpi_file_seek_ = PMPI_FILE_SEEK
#pragma weak pmpi_file_seek = PMPI_FILE_SEEK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_FILE_SEEK = pmpi_file_seek__
#pragma weak pmpi_file_seek_ = pmpi_file_seek__
#pragma weak pmpi_file_seek = pmpi_file_seek__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_FILE_SEEK = pmpi_file_seek_
#pragma weak pmpi_file_seek__ = pmpi_file_seek_
#pragma weak pmpi_file_seek = pmpi_file_seek_
#else
#pragma weak PMPI_FILE_SEEK = pmpi_file_seek
#pragma weak pmpi_file_seek__ = pmpi_file_seek
#pragma weak pmpi_file_seek_ = pmpi_file_seek
#endif /* Test on name mapping */
#endif /* Use multiple pragma weak */

#ifdef F77_NAME_UPPER
#define mpi_file_seek_ PMPI_FILE_SEEK
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_file_seek_ pmpi_file_seek__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_file_seek_ pmpi_file_seek
#else
#define mpi_file_seek_ pmpi_file_seek_
#endif /* Test on name mapping */

/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_File_seek
#define MPI_File_seek PMPI_File_seek 

#else

#ifdef F77_NAME_UPPER
#define mpi_file_seek_ MPI_FILE_SEEK
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_file_seek_ mpi_file_seek__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_file_seek_ mpi_file_seek
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_file_seek_ ( MPI_Fint *v1, MPI_Offset *v2, MPI_Fint *v3, MPI_Fint *ierr ){
#ifdef MPI_MODE_RDONLY
    *ierr = MPI_File_seek( MPI_File_f2c(*v1), *v2, *v3 );
#else
*ierr = MPI_ERR_INTERN;
#endif
}
