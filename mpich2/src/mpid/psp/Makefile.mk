## -*- Mode: Makefile -*-

if BUILD_PSP

errnames_txt_files += src/mpid/psp/errnames.txt

nodist_include_HEADERS += src/mpid/psp/include/mpi-ext.h

AM_CPPFLAGS += -I$(top_srcdir)/src/mpid/psp/include	\
               -I$(top_builddir)/src/mpid/psp/include

noinst_HEADERS += src/mpid/psp/include/list.h		\
                  src/mpid/psp/include/mpidimpl.h	\
                  src/mpid/psp/include/mpidpost.h	\
				  src/mpid/psp/include/mpid_sched.h \
				  src/mpid/psp/include/mpid_coll.h  \
                  src/mpid/psp/include/mpidpre.h

include $(top_srcdir)/src/mpid/psp/src/Makefile.mk

endif BUILD_PSP
