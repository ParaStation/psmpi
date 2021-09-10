##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

pack_testlists = $(top_srcdir)/test/pack/testlist.gen $(top_srcdir)/test/pack/testlist.threads.gen
EXTRA_DIST += $(top_srcdir)/test/pack/testlist.gen

EXTRA_PROGRAMS += \
	test/pack/pack

test_pack_pack_CPPFLAGS = $(test_cppflags)

test_pack_pack_SOURCES = test/pack/pack.c        \
                         test/pack/pack-common.c \
                         test/pack/pack-cuda.c   \
                         test/pack/pack-ze.c

if BUILD_CUDA_TESTS
include $(top_srcdir)/test/pack/Makefile.cuda.mk
endif BUILD_CUDA_TESTS

if BUILD_ZE_TESTS
include $(top_srcdir)/test/pack/Makefile.ze.mk
endif BUILD_ZE_TESTS

testlists += $(pack_testlists)

test-pack:
	@$(top_srcdir)/test/runtests.py --summary=$(top_builddir)/test/pack/summary.junit.xml \
                $(pack_testlists)
