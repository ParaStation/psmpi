! This file created from test/mpi/f77/coll/vw_inplacef.f with f77tof90
! -*- Mode: Fortran; -*- 
!
! (C) 2012 by Argonne National Laboratory.
!     See COPYRIGHT in top-level directory.
!
! A simple test for Fortran support of the MPI_IN_PLACE value in Alltoall[vw].
!
       program main
       use mpi
       integer ierr, errs
       integer comm, root
       integer rank, size
       integer i
       integer MAX_SIZE
       parameter (MAX_SIZE=1024)
       integer rbuf(MAX_SIZE)
       integer rdispls(MAX_SIZE), rcounts(MAX_SIZE), rtypes(MAX_SIZE)

       errs = 0
       call mtest_init( ierr )

       comm = MPI_COMM_WORLD
       call mpi_comm_rank( comm, rank, ierr )
       call mpi_comm_size( comm, size, ierr )

       do i=1,MAX_SIZE
           rbuf(i) = rank
       enddo

! Alltoallv and Alltoallw with inplace
! The test does not even check if receive buffer is processed correctly,
! because it merely aims to make sure MPI_IN_PLACE can be handled by
! Fortran MPI_Alltoall[vw]. The C version of these tests should have checked
! the buffer.
       do i=1,size
           rcounts(i) = i-1 + rank
           rdispls(i) = (i-1) * (2*size)
           rtypes(i)  = MPI_INTEGER
       enddo
       call mpi_alltoallv( MPI_IN_PLACE, 0, 0, MPI_DATATYPE_NULL, &
      &                     rbuf, rcounts, rdispls, MPI_INTEGER, &
      &                     comm, ierr )

       call mpi_alltoallw( MPI_IN_PLACE, 0, 0, MPI_DATATYPE_NULL, &
      &                     rbuf, rcounts, rdispls, rtypes, &
      &                     comm, ierr )


       call mtest_finalize( errs )
       call mpi_finalize( ierr )

       end
