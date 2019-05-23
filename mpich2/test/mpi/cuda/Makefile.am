# -*- Mode: Makefile; -*-
# vim: set ft=automake :
#
# (C) 2011 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.
#

NVCC = nvcc
NVCFLAGS = -m64
MPICC_CUDA_CONF = -config=cuda

AM_CFLAGS = $(MPICC_CUDA_CONF)

include $(top_srcdir)/Makefile.mtest

EXTRA_DIST = testlist

cuda_simple_kernel_SOURCES = cuda_simple_kernel.cu
cuda_laplace_mpi_SOURCES = cuda_laplace_mpi.cu

.cu.o:
	$(CC) -cc=$(NVCC) $(MPICC_CUDA_CONF) $(NVCFLAGS) -o $@ -c $<

## for all programs that are just built from the single corresponding source
## file, we don't need per-target _SOURCES rules, automake will infer them
## correctly
noinst_PROGRAMS =  \
	cuda_pingpong_mixed \
	cuda_pingpong_dtype \
	cuda_coll_suite \
	cuda_reduce_family \
	cuda_reduce_prefix \
	cuda_reduce_local \
	cuda_reduce_vec \
	cuda_allreduce_vec \
	cuda_simple_kernel \
	cuda_laplace_mpi \
	cuda_allgather cuda_accfence \
	cuda_cas \
	cuda_fop \
	cuda_sync_memops_ptr_attr \
	cuda_sync_memops_explicitly \
	cuda_sync_memops_memcpy


cuda_sync_memops_ptr_attr_SOURCES   = cuda_sync_memops.c
cuda_sync_memops_explicitly_SOURCES = cuda_sync_memops.c
cuda_sync_memops_memcpy_SOURCES     = cuda_sync_memops.c

cuda_sync_memops_ptr_attr_CPPFLAGS   = -DSYNC_VIA_PTR_ATTR $(AM_CPPFLAGS)
cuda_sync_memops_explicitly_CPPFLAGS = -DSYNC_VIA_MEMCPY $(AM_CPPFLAGS)
cuda_sync_memops_memcpy_CPPFLAGS     = -DSYNC_EXPLICITLY $(AM_CPPFLAGS)
