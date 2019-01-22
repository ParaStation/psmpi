# This Makefile.am generated automatically by f77tof90
# from f77/pt2pt/Makefile.am.  DO NOT EDIT
# -*- Mode: Makefile; -*-
# vim: set ft=automake :
#
# (C) 2011 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.
#

include $(top_srcdir)/Makefile_f90.mtest

EXTRA_DIST = testlist

# avoid having to write many "foo_SOURCES = foo.f90" lines
AM_DEFAULT_SOURCE_EXT = .f90

noinst_PROGRAMS = sendf90 \
                  ssendf90 \
                  rsendf90 \
                  isendf90 \
                  irsendf90 \
                  issendf90 \
                  psendf90 \
                  prsendf90 \
                  pssendf90 \
                  sendrecvf90 \
                  sendrecvreplf90 \
                  statusesf90 greqf90 \
                  mprobef90

sendf90_SOURCES = sendf90.f90 utilsf90.f90
ssendf90_SOURCES = ssendf90.f90 utilsf90.f90
rsendf90_SOURCES = rsendf90.f90 utilsf90.f90
isendf90_SOURCES = isendf90.f90 utilsf90.f90
irsendf90_SOURCES = irsendf90.f90 utilsf90.f90
issendf90_SOURCES = issendf90.f90 utilsf90.f90
psendf90_SOURCES = psendf90.f90 utilsf90.f90
prsendf90_SOURCES = prsendf90.f90 utilsf90.f90
pssendf90_SOURCES = pssendf90.f90 utilsf90.f90
sendrecvf90_SOURCES = sendrecvf90.f90 utilsf90.f90
sendrecvreplf90_SOURCES = sendrecvreplf90.f90 utilsf90.f90
greqf90_SOURCES = greqf90.f90 dummyf90.f90

## attr1aints.h will be distributed because it's listed in AC_CONFIG_FILES/AC_OUTPUT

# ensure that dependent tests will be rebuilt when typeaints.h is updated

if MAINTAINER_MODE

# f77tof90 only replaces output files if the contents differ, so we must use a
# separate timestamp file in order avoid constantly trying to remake the f90
# copy of the Makefile when the f77 copy is updated in a way that does not
# change its contents
$(srcdir)/Makefile.am: $(srcdir)/Makefile.am-stamp

$(srcdir)/Makefile.am-stamp: $(top_srcdir)/f77/pt2pt/Makefile.am $(top_srcdir)/maint/f77tof90
	( cd $(top_srcdir) && ./maint/f77tof90 f77/pt2pt f90/pt2pt Makefile.am Makefile.ap )

MAINTAINERCLEANFILES = Makefile.am-stamp
EXTRA_DIST += Makefile.am-stamp  

endif MAINTAINER_MODE

