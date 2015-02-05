#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
AC_INIT([psmpi],[],[hauke@par-tec.com])
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

if test -z "$MPICH_VERSION"; then
   AC_MSG_ERROR([MPICH_VERSION not found. Configuration aborted.])
fi
AC_SUBST([MPICH_VERSION])
AC_SUBST([MPICH_RELEASE_DATE])
# configure mpich2

AC_ARG_WITH([hydra], [
  --with-hydra            Install also Hydra (mpiexec).
],[
	ENABLE_HYDRA="true"
],[
	ENABLE_HYDRA="false"
])
AC_SUBST([ENABLE_HYDRA])


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
CONF_OPS_COMMON="$CONF_OPS_COMMON --enable-fast=all"
CONF_OPS_COMMON="$CONF_OPS_COMMON --enable-cache"
CONF_OPS_COMMON="$CONF_OPS_COMMON --enable-lib-depend"
CONF_OPS_COMMON="$CONF_OPS_COMMON --disable-rpath"
CONF_OPS_COMMON="$CONF_OPS_COMMON --enable-shared"
CONF_OPS_COMMON="$CONF_OPS_COMMON --disable-static"
MODULEFILE_CONFSET="$CONFSET"

case $CONFSET in
"devel")
	echo "Using confset=devel"
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS --enable-cache"
	CONF_OPS="$CONF_OPS --enable-g=all"
	CONF_OPS="$CONF_OPS --enable-dependencies"
	# CONF_OPS="$CONF_OPS --disable-f77"
	# CONF_OPS="$CONF_OPS --disable-fc"
	# CONF_OPS="$CONF_OPS --disable-cxx"
	CONF_OPS="$CONF_OPS --enable-romio"
	# CONF_OPS="$CONF_OPS --disable-mpe"
	CONF_OPS="$CONF_OPS --enable-shared"
	CONF_OPS="$CONF_OPS --disable-static"
	CONF_OPS="$CONF_OPS --enable-error-checking=all"
	CONF_OPS="$CONF_OPS --enable-strict" # Compile with -Wall and other checks.

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

	CONF_OPS="$CONF_OPS CC=gcc CXX=g++ CFLAGS=-O3"
	if which gfortran >/dev/null 2>&1 ;then
	   # found gfortran
	   CONF_OPS="$CONF_OPS --enable-f77"
	   CONF_OPS="$CONF_OPS --enable-fc"
	   CONF_OPS="$CONF_OPS FC=gfortran"
	fi
	MODULEFILE_CONFSET=""
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

	CONF_OPS="$CONF_OPS CC=pathcc CXX=pathCC FC=pathf90"
	CONF_OPS="$CONF_OPS CFLAGS=-O3"
	;;
"micgcc")
	echo "Using confset=micgcc"
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_COMMON"
	CONF_OPS="$CONF_OPS --enable-cxx"
	CONF_OPS="$CONF_OPS --enable-romio"
	CONF_OPS="$CONF_OPS --enable-mpe"	# need also romio!

	CONF_OPS="$CONF_OPS --host=x86_64-k1om-linux"

	CONF_OPS="$CONF_OPS 'CC=x86_64-k1om-linux-gcc'"
	CONF_OPS="$CONF_OPS 'CXX=x86_64-k1om-linux-g++'"
	CONF_OPS="$CONF_OPS 'F77=x86_64-k1om-linux-gfortran'"
	CONF_OPS="$CONF_OPS 'FC=x86_64-k1om-linux-gfortran'"
	CONF_OPS="$CONF_OPS 'AR=x86_64-k1om-linux-ar'"
	CONF_OPS="$CONF_OPS 'RANLIB=x86_64-k1om-linux-ranlib'"

	# FIXME: check whether all of these are valid in mpich 3.0.4,
	# they were gathered with 1.5a2
	CONF_OPS="$CONF_OPS 'CROSS_F77_FALSE_VALUE=0'"
	CONF_OPS="$CONF_OPS 'CROSS_F77_SIZEOF_DOUBLE_PRECISION=8'"
	CONF_OPS="$CONF_OPS 'CROSS_F77_SIZEOF_INTEGER=4'"
	CONF_OPS="$CONF_OPS 'CROSS_F77_SIZEOF_REAL=4'"
	CONF_OPS="$CONF_OPS 'CROSS_F77_TRUE_VALUE=1'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_ADDRESS_KIND=8'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_ALL_INTEGER_MODELS=2,1,4,2,9,4,18,8,'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_DOUBLE_MODEL=15,307'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_INTEGER_KIND=4'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_INTEGER_MODEL=9'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_INTEGER_MODEL_MAP={2,1,1},{4,2,2},{9,4,4},{18,8,8},'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_OFFSET_KIND=8'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_REAL_MODEL=6,37'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_SIZEOF_DOUBLE_PRECISION=8'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_SIZEOF_INTEGER=4'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_SIZEOF_REAL=4'"
	CONF_OPS="$CONF_OPS 'CROSS_MPI_STATUS_SIZE=5'"

	CONF_OPS="$CONF_OPS CFLAGS=-O3"
	EXTRA_VERSION_INFO=" compiler: $(x86_64-k1om-linux-gcc --version|head -n 1)"
	MODULEFILE_CONFSET="mic"
	;;
"micintel")
	echo "Using confset=micintel"
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_COMMON"
	CONF_OPS="$CONF_OPS --enable-cxx"
	CONF_OPS="$CONF_OPS --enable-romio"
	CONF_OPS="$CONF_OPS --enable-mpe"	# need also romio!

	# Get Performance
	CONF_OPS="$CONF_OPS --enable-g=none"
	CONF_OPS="$CONF_OPS --enable-error-checking=no"

	CONF_OPS="$CONF_OPS --host=x86_64-k1om-linux"
	CONF_OPS="$CONF_OPS 'CC=icc -mmic'"
	CONF_OPS="$CONF_OPS 'CXX=icpc -mmic'"
	CONF_OPS="$CONF_OPS 'F77=ifort -mmic'"
	CONF_OPS="$CONF_OPS 'FC=ifort -mmic'"

	# FIXME: check whether all of these are valid in mpich 3.0.4,
	# they were gathered with 1.5a2
	CONF_OPS="$CONF_OPS 'CROSS_F77_FALSE_VALUE=0'"
	CONF_OPS="$CONF_OPS 'CROSS_F77_SIZEOF_DOUBLE_PRECISION=8'"
	CONF_OPS="$CONF_OPS 'CROSS_F77_SIZEOF_INTEGER=4'"
	CONF_OPS="$CONF_OPS 'CROSS_F77_SIZEOF_REAL=4'"
	CONF_OPS="$CONF_OPS 'CROSS_F77_TRUE_VALUE=1'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_ADDRESS_KIND=8'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_ALL_INTEGER_MODELS=2,1,4,2,9,4,18,8,'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_DOUBLE_MODEL=15,307'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_INTEGER_KIND=4'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_INTEGER_MODEL=9'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_INTEGER_MODEL_MAP={2,1,1},{4,2,2},{9,4,4},{18,8,8},'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_OFFSET_KIND=8'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_REAL_MODEL=6,37'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_SIZEOF_DOUBLE_PRECISION=8'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_SIZEOF_INTEGER=4'"
	CONF_OPS="$CONF_OPS 'CROSS_F90_SIZEOF_REAL=4'"
	CONF_OPS="$CONF_OPS 'CROSS_MPI_STATUS_SIZE=5'"

	CONF_OPS="$CONF_OPS CFLAGS=-O3"
	EXTRA_VERSION_INFO=" compiler: $(icc -V 2>&1 |head -n 1)"
	MODULEFILE_CONFSET="intel-mic"
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

	CONF_OPS="$CONF_OPS CC=gcc CXX=g++ FC=gfortran CFLAGS=-O3"

	ENABLE_HYDRA="true"
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
	MODULEFILE_CONFSET="$MODULEFILE_CONFSET${MODULEFILE_CONFSET:+-}mt"
])

AC_ARG_WITH([modulefile], [
  --with-modulefile=filename
			destinaton of the generated modulefile
],[
	MODULEFILE="$withval"
],[
	# MODULEFILE="${sysconfdir}/modulefiles/mpi/psmpi-${host_cpu}$MODULEFILE_CONFSET"
	MODULEFILE="${sysconfdir}/modulefiles/mpi/psmpi"
	if test "${MODULEFILE_CONFSET}"x != "x" ; then
		MODULEFILE="$MODULEFILE-$MODULEFILE_CONFSET"
	fi
	MODULEFILE="$MODULEFILE-\$(VER)"
])


AC_SUBST([CONFSET])
AC_SUBST([MODULEFILE])
AC_SUBST([EXTRA_VERSION_INFO])

# ToDo: directly  use PSCOM_LDFLAGS from mpich2/src/mpid/psp/subconfigure.m4
PSCOM_LDFLAGS="${PSCOM_LDFLAGS-"-L/opt/parastation/lib64"}"
AC_SUBST([PSCOM_LDFLAGS])

AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([scripts/generate_modulefile.sh], [chmod +x scripts/generate_modulefile.sh])

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
