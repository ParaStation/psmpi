! This file created from test/mpi/f77/datatype/allctypesf.f with f77tof90
      program main
      use mpi
      integer atype, ierr
!
      call mpi_init(ierr)
!
!     Check that all Ctypes are available in Fortran (MPI 2.1, p 483, line 46)
!
       atype = MPI_CHAR           
       atype = MPI_SIGNED_CHAR    
       atype = MPI_UNSIGNED_CHAR  
       atype = MPI_BYTE           
       atype = MPI_WCHAR          
       atype = MPI_SHORT          
       atype = MPI_UNSIGNED_SHORT 
       atype = MPI_INT            
       atype = MPI_UNSIGNED       
       atype = MPI_LONG           
       atype = MPI_UNSIGNED_LONG  
       atype = MPI_FLOAT          
       atype = MPI_DOUBLE         
       atype = MPI_LONG_DOUBLE    
       atype = MPI_LONG_LONG_INT  
       atype = MPI_UNSIGNED_LONG_LONG 
       atype = MPI_LONG_LONG      
       atype = MPI_PACKED         
       atype = MPI_LB             
       atype = MPI_UB             
       atype = MPI_FLOAT_INT         
       atype = MPI_DOUBLE_INT        
       atype = MPI_LONG_INT          
       atype = MPI_SHORT_INT         
       atype = MPI_2INT              
       atype = MPI_LONG_DOUBLE_INT   
!
! This is a compilation test - If we can compile it, we pass.
       call MPI_Finalize( ierr )
       print *, " No Errors"
       stop 
       end
