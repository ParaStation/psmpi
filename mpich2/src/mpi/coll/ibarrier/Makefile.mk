##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

# mpi_sources includes only the routines that are MPI function entry points
# The code for the MPI operations (e.g., MPI_SUM) is not included in
# mpi_sources
mpi_sources +=                                  \
    src/mpi/coll/ibarrier/ibarrier.c

mpi_core_sources +=                                      \
    src/mpi/coll/ibarrier/ibarrier_intra_sched_recursive_doubling.c  \
    src/mpi/coll/ibarrier/ibarrier_inter_sched_bcast.c  \
    src/mpi/coll/ibarrier/ibarrier_intra_gentran_recexch.c
