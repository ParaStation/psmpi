!
! Copyright (C) by Argonne National Laboratory
!     See COPYRIGHT in top-level directory
!

function PMPIR_Wtick_f08() result(res)
    use :: mpi_c_interface_nobuf, only : MPIR_Wtick_c
    double precision :: res
    res = MPIR_Wtick_c()
end function PMPIR_Wtick_f08
