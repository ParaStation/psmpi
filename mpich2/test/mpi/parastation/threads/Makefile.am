#
# ParaStation
#
# Copyright (C) 2024-2025 ParTec AG, Munich
#
# This file may be distributed under the terms of the Q Public License
# as defined in the file LICENSE.QPL included in the packaging of this
# file.
 
include $(top_srcdir)/Makefile_threads.mtest

EXTRA_DIST = testlist

mprobe_threaded_LDADD = $(LDADD) -lpthread

noinst_PROGRAMS = mprobe_threaded
