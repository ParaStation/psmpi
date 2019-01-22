# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

TESTS = \
	thread_scale_tkt \
	thread_scale_mcs \
	thread_scale_hmcs \
	thread_ws_scale_tkt \
	thread_ws_scale_mcs \
	thread_ws_scale_hmcs \
	thread_scale_tlp \
	thread_scale_mcsp

check_PROGRAMS = $(TESTS)
noinst_PROGRAMS = $(TESTS)

include $(top_srcdir)/test/Makefile.mk

thread_scale_tkt_SOURCES = thread_scale.c
thread_scale_mcs_SOURCES = thread_scale.c
thread_scale_hmcs_SOURCES = thread_scale.c
thread_ws_scale_tkt_SOURCES = thread_ws_scale.c
thread_ws_scale_mcs_SOURCES = thread_ws_scale.c
thread_ws_scale_hmcs_SOURCES = thread_ws_scale.c
thread_scale_tlp_SOURCES = thread_scale_tlp.c
thread_scale_mcsp_SOURCES = thread_scale_tlp.c

thread_scale_tkt_CFLAGS = -DZMTEST_USE_TICKET -fopenmp
thread_scale_mcs_CFLAGS = -DZMTEST_USE_MCS -fopenmp
thread_scale_hmcs_CFLAGS = -DZMTEST_USE_HMCS -fopenmp
thread_ws_scale_tkt_CFLAGS = -DZMTEST_USE_TICKET -fopenmp
thread_ws_scale_mcs_CFLAGS = -DZMTEST_USE_MCS -fopenmp
thread_ws_scale_hmcs_CFLAGS = -DZMTEST_USE_HMCS -fopenmp
thread_scale_tlp_CFLAGS = -DZMTEST_USE_TLP -fopenmp
thread_scale_mcsp_CFLAGS = -DZMTEST_USE_MCSP -fopenmp

thread_scale_tkt_LDFLAGS = -fopenmp
thread_scale_mcs_LDFLAGS = -fopenmp
thread_scale_hmcs_LDFLAGS = -fopenmp -lstdc++
thread_ws_scale_tkt_LDFLAGS = -fopenmp
thread_ws_scale_mcs_LDFLAGS = -fopenmp
thread_ws_scale_hmcs_LDFLAGS = -fopenmp
thread_scale_tlp_LDFLAGS = -fopenmp -lstdc++
thread_scale_mcsp_LDFLAGS = -fopenmp
