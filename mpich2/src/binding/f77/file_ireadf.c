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
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_IREAD( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_iread__( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_iread( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_iread_( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_FILE_IREAD = PMPI_FILE_IREAD
#pragma weak mpi_file_iread__ = PMPI_FILE_IREAD
#pragma weak mpi_file_iread_ = PMPI_FILE_IREAD
#pragma weak mpi_file_iread = PMPI_FILE_IREAD
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_FILE_IREAD = pmpi_file_iread__
#pragma weak mpi_file_iread__ = pmpi_file_iread__
#pragma weak mpi_file_iread_ = pmpi_file_iread__
#pragma weak mpi_file_iread = pmpi_file_iread__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_FILE_IREAD = pmpi_file_iread_
#pragma weak mpi_file_iread__ = pmpi_file_iread_
#pragma weak mpi_file_iread_ = pmpi_file_iread_
#pragma weak mpi_file_iread = pmpi_file_iread_
#else
#pragma weak MPI_FILE_IREAD = pmpi_file_iread
#pragma weak mpi_file_iread__ = pmpi_file_iread
#pragma weak mpi_file_iread_ = pmpi_file_iread
#pragma weak mpi_file_iread = pmpi_file_iread
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_IREAD( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );

#pragma weak MPI_FILE_IREAD = PMPI_FILE_IREAD
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_file_iread__( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );

#pragma weak mpi_file_iread__ = pmpi_file_iread__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_file_iread( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );

#pragma weak mpi_file_iread = pmpi_file_iread
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_file_iread_( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );

#pragma weak mpi_file_iread_ = pmpi_file_iread_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_FILE_IREAD  MPI_FILE_IREAD
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_file_iread__  mpi_file_iread__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_file_iread  mpi_file_iread
#else
#pragma _HP_SECONDARY_DEF pmpi_file_iread_  mpi_file_iread_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_FILE_IREAD as PMPI_FILE_IREAD
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_file_iread__ as pmpi_file_iread__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_file_iread as pmpi_file_iread
#else
#pragma _CRI duplicate mpi_file_iread_ as pmpi_file_iread_
#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK) && \
    defined(USE_ONLY_MPI_NAMES)
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_IREAD( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_iread__( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_iread( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_iread_( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_file_iread__ = MPI_FILE_IREAD
#pragma weak mpi_file_iread_ = MPI_FILE_IREAD
#pragma weak mpi_file_iread = MPI_FILE_IREAD
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_FILE_IREAD = mpi_file_iread__
#pragma weak mpi_file_iread_ = mpi_file_iread__
#pragma weak mpi_file_iread = mpi_file_iread__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_FILE_IREAD = mpi_file_iread_
#pragma weak mpi_file_iread__ = mpi_file_iread_
#pragma weak mpi_file_iread = mpi_file_iread_
#else
#pragma weak MPI_FILE_IREAD = mpi_file_iread
#pragma weak mpi_file_iread__ = mpi_file_iread
#pragma weak mpi_file_iread_ = mpi_file_iread
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_FILE_IREAD( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_iread__( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_iread_( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_iread( MPI_Fint *, void*, MPI_Fint *, MPI_Fint *, MPIO_Request*, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_file_iread__ = PMPI_FILE_IREAD
#pragma weak pmpi_file_iread_ = PMPI_FILE_IREAD
#pragma weak pmpi_file_iread = PMPI_FILE_IREAD
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_FILE_IREAD = pmpi_file_iread__
#pragma weak pmpi_file_iread_ = pmpi_file_iread__
#pragma weak pmpi_file_iread = pmpi_file_iread__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_FILE_IREAD = pmpi_file_iread_
#pragma weak pmpi_file_iread__ = pmpi_file_iread_
#pragma weak pmpi_file_iread = pmpi_file_iread_
#else
#pragma weak PMPI_FILE_IREAD = pmpi_file_iread
#pragma weak pmpi_file_iread__ = pmpi_file_iread
#pragma weak pmpi_file_iread_ = pmpi_file_iread
#endif /* Test on name mapping */
#endif /* Use multiple pragma weak */

#ifdef F77_NAME_UPPER
#define mpi_file_iread_ PMPI_FILE_IREAD
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_file_iread_ pmpi_file_iread__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_file_iread_ pmpi_file_iread
#else
#define mpi_file_iread_ pmpi_file_iread_
#endif /* Test on name mapping */

/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_File_iread
#define MPI_File_iread PMPI_File_iread 

#else

#ifdef F77_NAME_UPPER
#define mpi_file_iread_ MPI_FILE_IREAD
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_file_iread_ mpi_file_iread__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_file_iread_ mpi_file_iread
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_file_iread_ ( MPI_Fint *v1, void*v2, MPI_Fint *v3, MPI_Fint *v4, MPIO_Request*v5, MPI_Fint *ierr ){
#ifdef MPI_MODE_RDONLY
    *ierr = MPI_File_iread( MPI_File_f2c(*v1), v2, *v3, (MPI_Datatype)(*v4), v5 );
#else
*ierr = MPI_ERR_INTERN;
#endif
}
