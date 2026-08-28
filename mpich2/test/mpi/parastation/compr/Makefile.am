#
# ParaStation
#
# Copyright (C) 2025-2026 ParTec AG, Munich
#
# This file may be distributed under the terms of the Q Public License
# as defined in the file LICENSE.QPL included in the packaging of this
# file.

include $(top_srcdir)/Makefile_single.mtest

EXTRA_DIST = testlist

noinst_libdir = $(abs_top_srcdir)/parastation/compr/
noinst_lib_LTLIBRARIES = libmpix_compressor_nop.la libmpix_compressor_d2i.la libmpix_compressor_c2i.la libmpix_compressor_c2d2i.la

libmpix_compressor_nop_la_CPPFLAGS = -DWITH_COMPRESSOR_PLUGIN
libmpix_compressor_nop_la_SOURCES = mpix_compressor_nop.c

libmpix_compressor_d2i_la_CPPFLAGS = -DWITH_COMPRESSOR_PLUGIN
libmpix_compressor_d2i_la_SOURCES = mpix_compressor_d2i.c

libmpix_compressor_c2i_la_CPPFLAGS = -DWITH_COMPRESSOR_PLUGIN
libmpix_compressor_c2i_la_SOURCES = mpix_compressor_c2i.c

libmpix_compressor_c2d2i_la_CPPFLAGS = -DWITH_COMPRESSOR_PLUGIN
libmpix_compressor_c2d2i_la_SOURCES = mpix_compressor_c2d2i.c

mpix_compressor_plg_nop_SOURCES = mpix_compressor_plg.c
mpix_compressor_plg_nop_LDADD = libmpix_compressor_nop.la

mpix_compressor_plg_d2i_SOURCES = mpix_compressor_plg.c
mpix_compressor_plg_d2i_LDADD = libmpix_compressor_d2i.la

mpix_compressor_plg_c2i_SOURCES = mpix_compressor_plg.c
mpix_compressor_plg_c2i_LDADD = libmpix_compressor_c2i.la

mpix_compressor_plg_c2d2i_SOURCES = mpix_compressor_plg.c
mpix_compressor_plg_c2d2i_LDADD = libmpix_compressor_c2d2i.la

noinst_PROGRAMS = \
    mpix_compressor_nop \
    mpix_compressor_d2i \
    mpix_compressor_c2i \
    mpix_compressor_c2d2i \
    mpix_compressor_plg_nop \
    mpix_compressor_plg_d2i \
    mpix_compressor_plg_c2i \
    mpix_compressor_plg_c2d2i \
    mpix_compressor_info
