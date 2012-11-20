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
extern FORT_DLL_SPEC void FORT_CALL MPI_COMM_SET_NAME( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_comm_set_name__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_comm_set_name( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_comm_set_name_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_COMM_SET_NAME = PMPI_COMM_SET_NAME
#pragma weak mpi_comm_set_name__ = PMPI_COMM_SET_NAME
#pragma weak mpi_comm_set_name_ = PMPI_COMM_SET_NAME
#pragma weak mpi_comm_set_name = PMPI_COMM_SET_NAME
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_COMM_SET_NAME = pmpi_comm_set_name__
#pragma weak mpi_comm_set_name__ = pmpi_comm_set_name__
#pragma weak mpi_comm_set_name_ = pmpi_comm_set_name__
#pragma weak mpi_comm_set_name = pmpi_comm_set_name__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_COMM_SET_NAME = pmpi_comm_set_name_
#pragma weak mpi_comm_set_name__ = pmpi_comm_set_name_
#pragma weak mpi_comm_set_name_ = pmpi_comm_set_name_
#pragma weak mpi_comm_set_name = pmpi_comm_set_name_
#else
#pragma weak MPI_COMM_SET_NAME = pmpi_comm_set_name
#pragma weak mpi_comm_set_name__ = pmpi_comm_set_name
#pragma weak mpi_comm_set_name_ = pmpi_comm_set_name
#pragma weak mpi_comm_set_name = pmpi_comm_set_name
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_COMM_SET_NAME( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#pragma weak MPI_COMM_SET_NAME = PMPI_COMM_SET_NAME
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_comm_set_name__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#pragma weak mpi_comm_set_name__ = pmpi_comm_set_name__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_comm_set_name( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#pragma weak mpi_comm_set_name = pmpi_comm_set_name
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_comm_set_name_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#pragma weak mpi_comm_set_name_ = pmpi_comm_set_name_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_COMM_SET_NAME  MPI_COMM_SET_NAME
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_comm_set_name__  mpi_comm_set_name__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_comm_set_name  mpi_comm_set_name
#else
#pragma _HP_SECONDARY_DEF pmpi_comm_set_name_  mpi_comm_set_name_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_COMM_SET_NAME as PMPI_COMM_SET_NAME
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_comm_set_name__ as pmpi_comm_set_name__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_comm_set_name as pmpi_comm_set_name
#else
#pragma _CRI duplicate mpi_comm_set_name_ as pmpi_comm_set_name_
#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK) && \
    defined(USE_ONLY_MPI_NAMES)
extern FORT_DLL_SPEC void FORT_CALL MPI_COMM_SET_NAME( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_comm_set_name__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_comm_set_name( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_comm_set_name_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_comm_set_name__ = MPI_COMM_SET_NAME
#pragma weak mpi_comm_set_name_ = MPI_COMM_SET_NAME
#pragma weak mpi_comm_set_name = MPI_COMM_SET_NAME
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_COMM_SET_NAME = mpi_comm_set_name__
#pragma weak mpi_comm_set_name_ = mpi_comm_set_name__
#pragma weak mpi_comm_set_name = mpi_comm_set_name__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_COMM_SET_NAME = mpi_comm_set_name_
#pragma weak mpi_comm_set_name__ = mpi_comm_set_name_
#pragma weak mpi_comm_set_name = mpi_comm_set_name_
#else
#pragma weak MPI_COMM_SET_NAME = mpi_comm_set_name
#pragma weak mpi_comm_set_name__ = mpi_comm_set_name
#pragma weak mpi_comm_set_name_ = mpi_comm_set_name
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_COMM_SET_NAME( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_comm_set_name__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_comm_set_name_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_comm_set_name( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_comm_set_name__ = PMPI_COMM_SET_NAME
#pragma weak pmpi_comm_set_name_ = PMPI_COMM_SET_NAME
#pragma weak pmpi_comm_set_name = PMPI_COMM_SET_NAME
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_COMM_SET_NAME = pmpi_comm_set_name__
#pragma weak pmpi_comm_set_name_ = pmpi_comm_set_name__
#pragma weak pmpi_comm_set_name = pmpi_comm_set_name__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_COMM_SET_NAME = pmpi_comm_set_name_
#pragma weak pmpi_comm_set_name__ = pmpi_comm_set_name_
#pragma weak pmpi_comm_set_name = pmpi_comm_set_name_
#else
#pragma weak PMPI_COMM_SET_NAME = pmpi_comm_set_name
#pragma weak pmpi_comm_set_name__ = pmpi_comm_set_name
#pragma weak pmpi_comm_set_name_ = pmpi_comm_set_name
#endif /* Test on name mapping */
#endif /* Use multiple pragma weak */

#ifdef F77_NAME_UPPER
#define mpi_comm_set_name_ PMPI_COMM_SET_NAME
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_comm_set_name_ pmpi_comm_set_name__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_comm_set_name_ pmpi_comm_set_name
#else
#define mpi_comm_set_name_ pmpi_comm_set_name_
#endif /* Test on name mapping */

/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Comm_set_name
#define MPI_Comm_set_name PMPI_Comm_set_name 

#else

#ifdef F77_NAME_UPPER
#define mpi_comm_set_name_ MPI_COMM_SET_NAME
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_comm_set_name_ mpi_comm_set_name__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_comm_set_name_ mpi_comm_set_name
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_comm_set_name_ ( MPI_Fint *v1, char *v2 FORT_MIXED_LEN(d2), MPI_Fint *ierr FORT_END_LEN(d2) ){
    char *p2;

    {char *p = v2 + d2 - 1;
     int  li;
        while (*p == ' ' && p > v2) p--;
        p++;
        p2 = (char *)MPIU_Malloc( p-v2 + 1 );
        for (li=0; li<(p-v2); li++) { p2[li] = v2[li]; }
        p2[li] = 0; 
    }
    *ierr = MPI_Comm_set_name( (MPI_Comm)(*v1), p2 );
    MPIU_Free( p2 );
}
