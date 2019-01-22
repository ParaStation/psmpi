# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

TESTS = \
	rr_sched \
	prio_chain \
	wskip_test

XFAIL_TESTS =

check_PROGRAMS = $(TESTS)
noinst_PROGRAMS = $(TESTS)

include $(top_srcdir)/test/Makefile.mk

rr_sched_SOURCES       = rr_sched.c
prio_chain_SOURCES     = prio_chain.c
wskip_test_SOURCES     = wskip_test.c

rr_sched_CFLAGS        =

rr_sched_LDFLAGS       = -pthread
prio_chain_LDFLAGS     = -pthread
wskip_test_LDFLAGS     = -pthread
