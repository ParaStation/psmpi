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
extern FORT_DLL_SPEC void FORT_CALL MPI_DIST_GRAPH_NEIGHBORS( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_dist_graph_neighbors__( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_dist_graph_neighbors( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_dist_graph_neighbors_( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_DIST_GRAPH_NEIGHBORS = PMPI_DIST_GRAPH_NEIGHBORS
#pragma weak mpi_dist_graph_neighbors__ = PMPI_DIST_GRAPH_NEIGHBORS
#pragma weak mpi_dist_graph_neighbors_ = PMPI_DIST_GRAPH_NEIGHBORS
#pragma weak mpi_dist_graph_neighbors = PMPI_DIST_GRAPH_NEIGHBORS
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_DIST_GRAPH_NEIGHBORS = pmpi_dist_graph_neighbors__
#pragma weak mpi_dist_graph_neighbors__ = pmpi_dist_graph_neighbors__
#pragma weak mpi_dist_graph_neighbors_ = pmpi_dist_graph_neighbors__
#pragma weak mpi_dist_graph_neighbors = pmpi_dist_graph_neighbors__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_DIST_GRAPH_NEIGHBORS = pmpi_dist_graph_neighbors_
#pragma weak mpi_dist_graph_neighbors__ = pmpi_dist_graph_neighbors_
#pragma weak mpi_dist_graph_neighbors_ = pmpi_dist_graph_neighbors_
#pragma weak mpi_dist_graph_neighbors = pmpi_dist_graph_neighbors_
#else
#pragma weak MPI_DIST_GRAPH_NEIGHBORS = pmpi_dist_graph_neighbors
#pragma weak mpi_dist_graph_neighbors__ = pmpi_dist_graph_neighbors
#pragma weak mpi_dist_graph_neighbors_ = pmpi_dist_graph_neighbors
#pragma weak mpi_dist_graph_neighbors = pmpi_dist_graph_neighbors
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_DIST_GRAPH_NEIGHBORS( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );

#pragma weak MPI_DIST_GRAPH_NEIGHBORS = PMPI_DIST_GRAPH_NEIGHBORS
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_dist_graph_neighbors__( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );

#pragma weak mpi_dist_graph_neighbors__ = pmpi_dist_graph_neighbors__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_dist_graph_neighbors( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );

#pragma weak mpi_dist_graph_neighbors = pmpi_dist_graph_neighbors
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_dist_graph_neighbors_( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );

#pragma weak mpi_dist_graph_neighbors_ = pmpi_dist_graph_neighbors_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_DIST_GRAPH_NEIGHBORS  MPI_DIST_GRAPH_NEIGHBORS
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_dist_graph_neighbors__  mpi_dist_graph_neighbors__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_dist_graph_neighbors  mpi_dist_graph_neighbors
#else
#pragma _HP_SECONDARY_DEF pmpi_dist_graph_neighbors_  mpi_dist_graph_neighbors_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_DIST_GRAPH_NEIGHBORS as PMPI_DIST_GRAPH_NEIGHBORS
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_dist_graph_neighbors__ as pmpi_dist_graph_neighbors__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_dist_graph_neighbors as pmpi_dist_graph_neighbors
#else
#pragma _CRI duplicate mpi_dist_graph_neighbors_ as pmpi_dist_graph_neighbors_
#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK) && \
    defined(USE_ONLY_MPI_NAMES)
extern FORT_DLL_SPEC void FORT_CALL MPI_DIST_GRAPH_NEIGHBORS( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_dist_graph_neighbors__( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_dist_graph_neighbors( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_dist_graph_neighbors_( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_dist_graph_neighbors__ = MPI_DIST_GRAPH_NEIGHBORS
#pragma weak mpi_dist_graph_neighbors_ = MPI_DIST_GRAPH_NEIGHBORS
#pragma weak mpi_dist_graph_neighbors = MPI_DIST_GRAPH_NEIGHBORS
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_DIST_GRAPH_NEIGHBORS = mpi_dist_graph_neighbors__
#pragma weak mpi_dist_graph_neighbors_ = mpi_dist_graph_neighbors__
#pragma weak mpi_dist_graph_neighbors = mpi_dist_graph_neighbors__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_DIST_GRAPH_NEIGHBORS = mpi_dist_graph_neighbors_
#pragma weak mpi_dist_graph_neighbors__ = mpi_dist_graph_neighbors_
#pragma weak mpi_dist_graph_neighbors = mpi_dist_graph_neighbors_
#else
#pragma weak MPI_DIST_GRAPH_NEIGHBORS = mpi_dist_graph_neighbors
#pragma weak mpi_dist_graph_neighbors__ = mpi_dist_graph_neighbors
#pragma weak mpi_dist_graph_neighbors_ = mpi_dist_graph_neighbors
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS) && defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_DIST_GRAPH_NEIGHBORS( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_dist_graph_neighbors__( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_dist_graph_neighbors_( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_dist_graph_neighbors( MPI_Fint *, MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_dist_graph_neighbors__ = PMPI_DIST_GRAPH_NEIGHBORS
#pragma weak pmpi_dist_graph_neighbors_ = PMPI_DIST_GRAPH_NEIGHBORS
#pragma weak pmpi_dist_graph_neighbors = PMPI_DIST_GRAPH_NEIGHBORS
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_DIST_GRAPH_NEIGHBORS = pmpi_dist_graph_neighbors__
#pragma weak pmpi_dist_graph_neighbors_ = pmpi_dist_graph_neighbors__
#pragma weak pmpi_dist_graph_neighbors = pmpi_dist_graph_neighbors__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_DIST_GRAPH_NEIGHBORS = pmpi_dist_graph_neighbors_
#pragma weak pmpi_dist_graph_neighbors__ = pmpi_dist_graph_neighbors_
#pragma weak pmpi_dist_graph_neighbors = pmpi_dist_graph_neighbors_
#else
#pragma weak PMPI_DIST_GRAPH_NEIGHBORS = pmpi_dist_graph_neighbors
#pragma weak pmpi_dist_graph_neighbors__ = pmpi_dist_graph_neighbors
#pragma weak pmpi_dist_graph_neighbors_ = pmpi_dist_graph_neighbors
#endif /* Test on name mapping */
#endif /* Use multiple pragma weak */

#ifdef F77_NAME_UPPER
#define mpi_dist_graph_neighbors_ PMPI_DIST_GRAPH_NEIGHBORS
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_dist_graph_neighbors_ pmpi_dist_graph_neighbors__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_dist_graph_neighbors_ pmpi_dist_graph_neighbors
#else
#define mpi_dist_graph_neighbors_ pmpi_dist_graph_neighbors_
#endif /* Test on name mapping */

/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Dist_graph_neighbors
#define MPI_Dist_graph_neighbors PMPI_Dist_graph_neighbors 

#else

#ifdef F77_NAME_UPPER
#define mpi_dist_graph_neighbors_ MPI_DIST_GRAPH_NEIGHBORS
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_dist_graph_neighbors_ mpi_dist_graph_neighbors__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_dist_graph_neighbors_ mpi_dist_graph_neighbors
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_dist_graph_neighbors_ ( MPI_Fint *v1, MPI_Fint *v2, MPI_Fint v3[], MPI_Fint v4[], MPI_Fint *v5, MPI_Fint v6[], MPI_Fint v7[], MPI_Fint *ierr ){

#ifndef HAVE_MPI_F_INIT_WORKS_WITH_C
    if (MPIR_F_NeedInit){ mpirinitf_(); MPIR_F_NeedInit = 0; }
#endif
    *ierr = MPI_Dist_graph_neighbors( (MPI_Comm)(*v1), *v2, v3, v4, *v5, v6, v7 );
}
