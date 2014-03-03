#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
AC_INIT([psmpi2],[],[hauke@par-tec.com])
AC_CONFIG_SRCDIR([configure.ac])

AC_CONFIG_AUX_DIR([scripts])
AC_CANONICAL_TARGET([])

# AC_PROG_INSTALL
# AC_PROG_RANLIB

path_cleanup()
{
	case $1 in
	[[\\/]]* | ?:[[\\/]]* ) # Absolute path.
		echo $1 ;;
	*) # Relative path.
		echo `pwd`/$1 ;;
	esac
}

TOP_SRCDIR=`cd \`dirname $0\`;pwd`
TOP_BUILDDIR=`pwd`
AC_SUBST([TOP_SRCDIR])
AC_SUBST([TOP_BUILDDIR])

CONFIGARGS="$ac_configure_args"
AC_SUBST([CONFIGARGS])

# Try to find the version and release date
. ${TOP_SRCDIR}/mpich2/maint/Version

if test -z "$MPICH2_VERSION"; then
   AC_MSG_ERROR([MPICH2_VERSION not found. Configuration aborted.])
fi
AC_SUBST([MPICH2_VERSION])
AC_SUBST([MPICH2_RELEASE_DATE])
# configure mpich2

AC_ARG_WITH([mpd], [
  --with-mpd            Install also default mpd.
],[
	ENABLE_MPD="true"
],[
	ENABLE_MPD="false"
])
AC_SUBST([ENABLE_MPD])


AC_ARG_WITH([confset], [
  --with-confset[=set]  Use predefined configuration set ["none"]
	   none       : Do not configure mpich.
			Prepare only for tar,rpm and srpm build
	   default    : like gcc

	   gcc        : Use Gnu compiler (gcc)
	   intel      : Use intel compiler (icc)
	   pgi	      : Portland group compiler (pgcc)
	   psc	      : Pathscale compiler (pathcc)
	   cellgcc    : ppu-gcc
	   cellxlc    :	ppuxlc

	   devel      : With errorchecking and debug info (gcc)
	   user       : No predefined options
	   ch3        : orginal MPIch2 ch3 device (no parastation)
	   "--xyz --abc" : Use this to configure mpich without a set
],[
	CONFSET="$withval"
],[
	CONFSET="none"
])


AC_ARG_WITH([mpichconf], [
  --with-mpichconf[=opts]  Additional mpich configuration options
],[
	MPICHCONF="$withval"
])



#Some influential environment variables:
#  CC          C compiler command
#  CFLAGS      C compiler flags
#  LDFLAGS     linker flags, e.g. -L<lib dir> if you have libraries in a
#              nonstandard directory <lib dir>
#  CPPFLAGS    C/C++ preprocessor flags, e.g. -I<include dir> if you have
#              headers in a nonstandard directory <include dir>
#  F77         Fortran 77 compiler command
#  FFLAGS      Fortran 77 compiler flags
#  FC          Fortran 90 compiler command
#  FCFLAGS     Fortran 90 compiler flags
#  CXX         C++ compiler command
#  CXXFLAGS    C++ compiler flags
#  CPP         C preprocessor
#

EXTRA_VERSION_INFO=""
# CONF_OPS_COMMON="$CONF_OPS_COMMON --enable-error-messages=generic"
CONF_OPS_COMMON="$CONF_OPS_COMMON --enable-fast=defopt,notiming"
CONF_OPS_COMMON="$CONF_OPS_COMMON --enable-cache"
CONF_OPS_COMMON="$CONF_OPS_COMMON --enable-lib-depend"
CONF_OPS_COMMON="$CONF_OPS_COMMON --disable-rpath"

case $CONFSET in
"devel")
	echo "Using confset=devel"
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS --enable-cache"
	CONF_OPS="$CONF_OPS --enable-g=all"
	CONF_OPS="$CONF_OPS --enable-dependencies"
	CONF_OPS="$CONF_OPS --disable-f77"
	CONF_OPS="$CONF_OPS --disable-fc"
	CONF_OPS="$CONF_OPS --disable-cxx"
	CONF_OPS="$CONF_OPS --disable-romio"
	CONF_OPS="$CONF_OPS --disable-mpe"
	CONF_OPS="$CONF_OPS --enable-shared"
	CONF_OPS="$CONF_OPS --enable-error-checking=all"

	CONF_OPS="$CONF_OPS CFLAGS=-O3"
	;;
"gcc"|\
"default")
	echo "Using confset=default"
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_COMMON"
	CONF_OPS="$CONF_OPS --enable-cache"
	CONF_OPS="$CONF_OPS --enable-cxx"
	CONF_OPS="$CONF_OPS --enable-romio"
	CONF_OPS="$CONF_OPS --enable-mpe"	# need also romio!
	CONF_OPS="$CONF_OPS --enable-shared"

	CONF_OPS="$CONF_OPS CC=gcc CXX=g++ CFLAGS=-O3"
	if which gfortran >/dev/null 2>&1 ;then
	   # found gfortran
	   CONF_OPS="$CONF_OPS --enable-f77"
	   CONF_OPS="$CONF_OPS --enable-fc"
	   CONF_OPS="$CONF_OPS FC=gfortran"
	fi
	EXTRA_VERSION_INFO=" compiler: $(gcc --version|head -n 1)"
	;;
"intel")
	echo "Using confset=intel"
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_COMMON"
	CONF_OPS="$CONF_OPS --enable-f77"
	CONF_OPS="$CONF_OPS --enable-fc"
	CONF_OPS="$CONF_OPS --enable-cxx"
	CONF_OPS="$CONF_OPS --enable-romio"
	CONF_OPS="$CONF_OPS --enable-mpe"	# need also romio!
	CONF_OPS="$CONF_OPS --enable-shared"

	CONF_OPS="$CONF_OPS CC=icc CXX=icpc FC=ifort"
	CONF_OPS="$CONF_OPS CFLAGS=-O3"
	EXTRA_VERSION_INFO=" compiler: $(icc -V 2>&1 |head -n 1)"
	;;
"pgi")
	echo "Using confset=pgi"
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_COMMON"
	CONF_OPS="$CONF_OPS --enable-f77"
	CONF_OPS="$CONF_OPS --enable-fc"
	CONF_OPS="$CONF_OPS --enable-cxx"
	CONF_OPS="$CONF_OPS --enable-romio"
	CONF_OPS="$CONF_OPS --enable-mpe"	# need also romio!
	CONF_OPS="$CONF_OPS --enable-shared"

	CONF_OPS="$CONF_OPS CC=pgcc CXX=pgc++ FC=pgf90"
	CONF_OPS="$CONF_OPS CFLAGS=-O3"
	;;
"pgi_pgCC")
	echo "Using confset=pgi"
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_COMMON"
	CONF_OPS="$CONF_OPS --enable-f77"
	CONF_OPS="$CONF_OPS --enable-fc"
	CONF_OPS="$CONF_OPS --enable-cxx"
	CONF_OPS="$CONF_OPS --enable-romio"
	CONF_OPS="$CONF_OPS --enable-mpe"	# need also romio!
	CONF_OPS="$CONF_OPS --enable-shared"

	CONF_OPS="$CONF_OPS CC=pgcc CXX=pgCC FC=pgf90"
	CONF_OPS="$CONF_OPS CFLAGS=-O3"
	;;
"psc")
	echo "Using confset=psc"
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_COMMON"
	CONF_OPS="$CONF_OPS --enable-f77"
	CONF_OPS="$CONF_OPS --enable-fc"
	CONF_OPS="$CONF_OPS --enable-cxx"
	CONF_OPS="$CONF_OPS --enable-romio"
	CONF_OPS="$CONF_OPS --enable-mpe"	# need also romio!
	CONF_OPS="$CONF_OPS --enable-shared"

	CONF_OPS="$CONF_OPS CC=pathcc CXX=pathCC FC=pathf90"
	CONF_OPS="$CONF_OPS CFLAGS=-O3"
	;;
"cellgcc")
	echo "Using confset=cellgcc"
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_COMMON"
	CONF_OPS="$CONF_OPS --enable-f77"
	CONF_OPS="$CONF_OPS --enable-fc"
	CONF_OPS="$CONF_OPS --enable-cxx"
	CONF_OPS="$CONF_OPS --enable-romio"
	CONF_OPS="$CONF_OPS --enable-mpe"	# need also romio!
	CONF_OPS="$CONF_OPS --enable-shared"

	CONF_OPS="$CONF_OPS CC=ppu-gcc CXX=ppu-c++ FC=ppu-gfortran"
	CONF_OPS="$CONF_OPS CFLAGS=-O3"
	;;
"cellxlc")
	echo "Using confset=cellxlc"
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_COMMON"
	CONF_OPS="$CONF_OPS --enable-f77"
	CONF_OPS="$CONF_OPS --enable-fc"
	CONF_OPS="$CONF_OPS --enable-cxx"
	CONF_OPS="$CONF_OPS --enable-romio"
	CONF_OPS="$CONF_OPS --enable-mpe"	# need also romio!
	CONF_OPS="$CONF_OPS --enable-shared"

	CONF_OPS="$CONF_OPS CC='ppuxlc -q64' CXX='ppuxlC -q64' FC='ppuxlf -q64'"
	CONF_OPS="$CONF_OPS CFLAGS=-O3"
	;;
"none")
	# No build
	;;
"user")
	CONF_OPS=
	;;
"ch3")
	echo "Using confset=ch3"
	CONF_OPS="$CONF_OPS $CONF_OPS_COMMON"
	CONF_OPS="$CONF_OPS --enable-f77"
	CONF_OPS="$CONF_OPS --enable-fc"
	CONF_OPS="$CONF_OPS --enable-cxx"
	CONF_OPS="$CONF_OPS --enable-romio"
	CONF_OPS="$CONF_OPS --enable-mpe"	# need also romio!
	CONF_OPS="$CONF_OPS --enable-shared"

	CONF_OPS="$CONF_OPS CC=gcc CXX=g++ FC=gfortran CFLAGS=-O3"

	ENABLE_MPD="true"
	;;
*)
	echo "Sorry. Unknown confset ${CONFSET}"
	exit 1
	;;
esac


AC_ARG_WITH([threading], [
  --with-threading      Enable threading 'multiple' in device psp
],[
	export PSP_THREADING=multiple
])


AC_SUBST([CONFSET])
AC_SUBST([EXTRA_VERSION_INFO])

AC_CONFIG_FILES([Makefile])

AC_OUTPUT

if test "${CONFSET}" != "none" ; then
   # Call mpich2 configure:
   mkdir -p "${TOP_BUILDDIR}/mpich2"
   cd "${TOP_BUILDDIR}/mpich2"
   echo "############### configure mpich #############"
   echo "wd=$(pwd)"
   #   echo ${TOP_SRCDIR}/mpich2/configure ${CONF_OPS} "$@"

   # call original mpich configure
   # --disable-maintainer-mode prevent from refreshing autoconf generated files in the source folder.
   eval ${TOP_SRCDIR}/mpich2/configure ${CONF_OPS} ${MPICHCONF} ${CONFIGARGS} --disable-maintainer-mode
fi

#
# Local Variables:
#  compile-command: "autoconf"
# End:
#
#
