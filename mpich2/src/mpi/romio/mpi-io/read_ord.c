/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpioimpl.h"
#include <limits.h>
#include <assert.h>

#ifdef HAVE_WEAK_SYMBOLS

#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_File_read_ordered = PMPI_File_read_ordered
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_File_read_ordered MPI_File_read_ordered
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_File_read_ordered as PMPI_File_read_ordered
/* end of weak pragmas */
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_File_read_ordered(MPI_File fh, void *buf, int count, MPI_Datatype datatype,
                          MPI_Status * status)
    __attribute__ ((weak, alias("PMPI_File_read_ordered")));
#endif

#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_File_read_ordered_c = PMPI_File_read_ordered_c
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_File_read_ordered_c MPI_File_read_ordered_c
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_File_read_ordered_c as PMPI_File_read_ordered_c
/* end of weak pragmas */
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_File_read_ordered_c(MPI_File fh, void *buf, MPI_Count count, MPI_Datatype datatype,
                            MPI_Status * status)
    __attribute__ ((weak, alias("PMPI_File_read_ordered_c")));
#endif

/* Include mapping from MPI->PMPI */
#define MPIO_BUILD_PROFILING
#include "mpioprof.h"
#endif

/* status object not filled currently */

/*@
    MPI_File_read_ordered - Collective read using shared file pointer

Input Parameters:
. fh - file handle (handle)
. count - number of elements in buffer (nonnegative integer)
. datatype - datatype of each buffer element (handle)

Output Parameters:
. buf - initial address of buffer (choice)
. status - status object (Status)

.N fortran
@*/
int MPI_File_read_ordered(MPI_File fh, void *buf, int count,
                          MPI_Datatype datatype, MPI_Status * status)
{
    return MPIOI_File_read_ordered(fh, buf, count, datatype, status);
}

/* large count function */



/*@
    MPI_File_read_ordered_c - Collective read using shared file pointer

Input Parameters:
. fh - file handle (handle)
. count - number of elements in buffer (nonnegative integer)
. datatype - datatype of each buffer element (handle)

Output Parameters:
. buf - initial address of buffer (choice)
. status - status object (Status)

.N fortran
@*/
int MPI_File_read_ordered_c(MPI_File fh, void *buf, MPI_Count count,
                            MPI_Datatype datatype, MPI_Status * status)
{
    return MPIOI_File_read_ordered(fh, buf, count, datatype, status);
}

#ifdef MPIO_BUILD_PROFILING
int MPIOI_File_read_ordered(MPI_File fh, void *buf, MPI_Aint count,
                            MPI_Datatype datatype, MPI_Status * status)
{
    int error_code, nprocs, myrank;
    ADIO_Offset incr;
    MPI_Count datatype_size;
    int source, dest;
    static char myname[] = "MPI_FILE_READ_ORDERED";
    ADIO_Offset shared_fp = 0;
    ADIO_File adio_fh;

    ROMIO_THREAD_CS_ENTER();

    adio_fh = MPIO_File_resolve(fh);

    /* --BEGIN ERROR HANDLING-- */
    MPIO_CHECK_FILE_HANDLE(adio_fh, myname, error_code);
    MPIO_CHECK_COUNT(adio_fh, count, myname, error_code);
    MPIO_CHECK_DATATYPE(adio_fh, datatype, myname, error_code);
    /* --END ERROR HANDLING-- */

    MPI_Type_size_x(datatype, &datatype_size);

    /* --BEGIN ERROR HANDLING-- */
    MPIO_CHECK_INTEGRAL_ETYPE(adio_fh, count, datatype_size, myname, error_code);
    MPIO_CHECK_FS_SUPPORTS_SHARED(adio_fh, myname, error_code);
    MPIO_CHECK_COUNT_SIZE(adio_fh, count, datatype_size, myname, error_code);
    /* --END ERROR HANDLING-- */

    ADIOI_TEST_DEFERRED(adio_fh, "MPI_File_read_ordered", &error_code);

    MPI_Comm_size(adio_fh->comm, &nprocs);
    MPI_Comm_rank(adio_fh->comm, &myrank);

    incr = (count * datatype_size) / adio_fh->etype_size;

    /* Use a message as a 'token' to order the operations */
    source = myrank - 1;
    dest = myrank + 1;
    if (source < 0)
        source = MPI_PROC_NULL;
    if (dest >= nprocs)
        dest = MPI_PROC_NULL;
    MPI_Recv(NULL, 0, MPI_BYTE, source, 0, adio_fh->comm, MPI_STATUS_IGNORE);

    ADIO_Get_shared_fp(adio_fh, incr, &shared_fp, &error_code);
    /* --BEGIN ERROR HANDLING-- */
    if (error_code != MPI_SUCCESS) {
        error_code = MPIO_Err_return_file(adio_fh, error_code);
        goto fn_exit;
    }
    /* --END ERROR HANDLING-- */

    MPI_Send(NULL, 0, MPI_BYTE, dest, 0, adio_fh->comm);

    ADIO_ReadStridedColl(adio_fh, buf, count, datatype, ADIO_EXPLICIT_OFFSET,
                         shared_fp, status, &error_code);

    /* --BEGIN ERROR HANDLING-- */
    if (error_code != MPI_SUCCESS)
        error_code = MPIO_Err_return_file(adio_fh, error_code);
    /* --END ERROR HANDLING-- */

  fn_exit:
    ROMIO_THREAD_CS_EXIT();

    /* FIXME: Check for error code from ReadStridedColl? */
    return error_code;
}
#endif
