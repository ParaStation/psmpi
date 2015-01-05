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
extern FORT_DLL_SPEC void FORT_CALL MPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_PACK_SIZE = PMPI_PACK_SIZE
#pragma weak mpi_pack_size__ = PMPI_PACK_SIZE
#pragma weak mpi_pack_size_ = PMPI_PACK_SIZE
#pragma weak mpi_pack_size = PMPI_PACK_SIZE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_PACK_SIZE = pmpi_pack_size__
#pragma weak mpi_pack_size__ = pmpi_pack_size__
#pragma weak mpi_pack_size_ = pmpi_pack_size__
#pragma weak mpi_pack_size = pmpi_pack_size__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_PACK_SIZE = pmpi_pack_size_
#pragma weak mpi_pack_size__ = pmpi_pack_size_
#pragma weak mpi_pack_size_ = pmpi_pack_size_
#pragma weak mpi_pack_size = pmpi_pack_size_
#else
#pragma weak MPI_PACK_SIZE = pmpi_pack_size
#pragma weak mpi_pack_size__ = pmpi_pack_size
#pragma weak mpi_pack_size_ = pmpi_pack_size
#pragma weak mpi_pack_size = pmpi_pack_size
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak MPI_PACK_SIZE = PMPI_PACK_SIZE
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_pack_size__ = pmpi_pack_size__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_pack_size = pmpi_pack_size
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_pack_size_ = pmpi_pack_size_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_PACK_SIZE  MPI_PACK_SIZE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_pack_size__  mpi_pack_size__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_pack_size  mpi_pack_size
#else
#pragma _HP_SECONDARY_DEF pmpi_pack_size_  mpi_pack_size_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_PACK_SIZE as PMPI_PACK_SIZE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_pack_size__ as pmpi_pack_size__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_pack_size as pmpi_pack_size
#else
#pragma _CRI duplicate mpi_pack_size_ as pmpi_pack_size_
#endif

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_PACK_SIZE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_PACK_SIZE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_PACK_SIZE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_PACK_SIZE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size")));

#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(USE_ONLY_MPI_NAMES)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_pack_size__ = MPI_PACK_SIZE
#pragma weak mpi_pack_size_ = MPI_PACK_SIZE
#pragma weak mpi_pack_size = MPI_PACK_SIZE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_PACK_SIZE = mpi_pack_size__
#pragma weak mpi_pack_size_ = mpi_pack_size__
#pragma weak mpi_pack_size = mpi_pack_size__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_PACK_SIZE = mpi_pack_size_
#pragma weak mpi_pack_size__ = mpi_pack_size_
#pragma weak mpi_pack_size = mpi_pack_size_
#else
#pragma weak MPI_PACK_SIZE = mpi_pack_size
#pragma weak mpi_pack_size__ = mpi_pack_size
#pragma weak mpi_pack_size_ = mpi_pack_size
#endif
#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_PACK_SIZE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_PACK_SIZE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_PACK_SIZE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_pack_size__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_pack_size__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_pack_size__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_pack_size_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_pack_size_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_pack_size_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_pack_size")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_pack_size")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_pack_size")));
extern FORT_DLL_SPEC void FORT_CALL mpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_pack_size__ = PMPI_PACK_SIZE
#pragma weak pmpi_pack_size_ = PMPI_PACK_SIZE
#pragma weak pmpi_pack_size = PMPI_PACK_SIZE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_PACK_SIZE = pmpi_pack_size__
#pragma weak pmpi_pack_size_ = pmpi_pack_size__
#pragma weak pmpi_pack_size = pmpi_pack_size__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_PACK_SIZE = pmpi_pack_size_
#pragma weak pmpi_pack_size__ = pmpi_pack_size_
#pragma weak pmpi_pack_size = pmpi_pack_size_
#else
#pragma weak PMPI_PACK_SIZE = pmpi_pack_size
#pragma weak pmpi_pack_size__ = pmpi_pack_size
#pragma weak pmpi_pack_size_ = pmpi_pack_size
#endif /* Test on name mapping */

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL pmpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_PACK_SIZE")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_PACK_SIZE")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_PACK_SIZE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_pack_size( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size_")));

#else
extern FORT_DLL_SPEC void FORT_CALL PMPI_PACK_SIZE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_pack_size__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_pack_size_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_pack_size")));

#endif /* Test on name mapping */
#endif /* HAVE_MULTIPLE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */

#ifdef F77_NAME_UPPER
#define mpi_pack_size_ PMPI_PACK_SIZE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_pack_size_ pmpi_pack_size__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_pack_size_ pmpi_pack_size
#else
#define mpi_pack_size_ pmpi_pack_size_
#endif /* Test on name mapping */

/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Pack_size
#define MPI_Pack_size PMPI_Pack_size 

#else

#ifdef F77_NAME_UPPER
#define mpi_pack_size_ MPI_PACK_SIZE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_pack_size_ mpi_pack_size__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_pack_size_ mpi_pack_size
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_pack_size_ ( MPI_Fint *v1, MPI_Fint *v2, MPI_Fint *v3, MPI_Fint *v4, MPI_Fint *ierr ){
    *ierr = MPI_Pack_size( (int)*v1, (MPI_Datatype)(*v2), (MPI_Comm)(*v3), v4 );
}