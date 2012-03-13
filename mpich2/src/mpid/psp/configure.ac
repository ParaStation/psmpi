#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
AC_INIT(configure.ac)
# AC_INIT(["mpich2-psp"],[""],["hauke@par-tec.com"])
AC_CONFIG_SRCDIR([configure.ac])



dnl
dnl Set the directory that contains support scripts such as install-sh and
dnl config.guess
dnl
AC_CONFIG_AUX_DIR(../../../confdb)

echo "RUNNING mpid/psp/configure FOR THE ParaStation DEVICE"



AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([src/Makefile])

AC_CONFIG_FILES([localdefs])


#
# bring in the default implementation of the MPID_Thread package
#
MPID_THREAD_SRCDIR="../common/thread"
builtin(include,../common/thread/mpid_thread.m4)

PAC_PROG_MAKE

AC_SUBST(AR)
AC_SUBST(RANLIB)
AC_SUBST(MPILIBNAME)
AC_SUBST(CC)
AC_SUBST(DEFS)
AC_SUBST(CFLAGS)
AC_SUBST(CPPFLAGS)
PAC_CC_SUBDIR_SHLIBS
AC_SUBST(master_top_srcdir)
# Master_top_builddir is imported from the top; use it for the "top"
AC_SUBST(master_top_builddir)
export master_top_builddir
dnl Dependency handling
AC_SUBST(MAKE_DEPEND_C)
dnl Etags
AC_CHECK_PROGS(ETAGS,etags,true)
AC_SUBST(ETAGS)
AC_SUBST(ETAGSADD)
export ETAGS
export ETAGSADD

AC_SUBST(EXTRA_LIBS)

PAC_SUBDIR_CACHE
dnl AC_CONFIG_SUBDIRS(channels/${channel_name})

AC_OUTPUT($MPID_THREAD_OUTPUT_FILES)

PAC_SUBDIR_CACHE_CLEANUP



#
# Local Variables:
#  compile-command: "autoconf -I ../../../confdb"
# End:
#
#
