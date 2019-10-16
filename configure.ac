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
	   icc        : Like 'intel'
	   pgi	      : Portland group compiler (pgcc)
	   psc	      : Pathscale compiler (pathcc)
	   cellgcc    : ppu-gcc
	   cellxlc    :	ppuxlc

	   devel      : With error checking and debug info (gcc)
	   user       : No predefined options
	   ch3        : original mpich ch3 device (no parastation)
	   "--xyz --abc" : Use this to configure mpich without a set
],[
	CONFSET="$withval"
],[
	CONFSET="none"
])


AC_ARG_WITH([mpichconf], [
  --with-mpichconf[=opts]   Additional mpich configuration options
],[
	MPICHCONF="$withval"
])


AC_ARG_WITH([profdir], [
  --with-profdir[=path]     Set path to profile directory
],[
	PROFILE_DIR="$withval"
],[
	PROFILE_DIR="${TOP_SRCDIR}/profile"
])

AC_ARG_WITH([profile], [
  --with-profile[=use|gen]  Use profile for guided optimization
],[
	ENABLE_PROFILE="$withval"
	if test "$ENABLE_PROFILE" = "yes" ; then
		ENABLE_PROFILE="use"
	fi
	if test "$ENABLE_PROFILE" = "gen" ; then
		if ! test -d "${PROFILE_DIR}" ; then
			mkdir -p "${PROFILE_DIR}" 2>/dev/null
		fi
	fi
])

AC_ARG_WITH([pscom-builtin], [
  --with-pscom-builtin[=list]
                          Define list of pscom plugins that shall be built-in
                          Default: openib,psm2 / Needs pscom source code fusion
],[
	PSCOM_BUILTIN_PLUGINS="$withval"
])

AC_ARG_WITH([pscom-allin], [
  --with-pscom-allin[=path]
                          Enable pscom source code fusion and set path to pscom
],[
	export PSCOM_ALLIN="true"
	PSCOM_ALLIN_DIR="$withval"
	AC_CHECK_FILE(["${PSCOM_ALLIN_DIR}/lib/all/pscom_all.c"], [
		PSCOM_ALLIN_DIR="${PSCOM_ALLIN_DIR}/lib/all/"
	])
	export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DPSCOM_ALLIN -I${PSCOM_ALLIN_DIR}/ -I${PSCOM_ALLIN_DIR}/../../include/ -I${PSCOM_ALLIN_DIR}/../../lib/pscom/"

	if test -x "${PSCOM_ALLIN_DIR}/../../scripts/vcversion" ; then
		if ! test -d "${TOP_BUILDDIR}/pscom" ; then
                        mkdir -p "${TOP_BUILDDIR}/pscom" 2>/dev/null
                fi
		PSCOM_VC_VERSION=`${PSCOM_ALLIN_DIR}/../../scripts/vcversion -r ${PSCOM_ALLIN_DIR}/../.. -n`
		echo "Including pscom version ${PSCOM_VC_VERSION}"
		echo "#define VC_VERSION \"${PSCOM_VC_VERSION}\"" > "${TOP_BUILDDIR}/pscom/vc_version.h"
		export MPICHLIB_CFLAGS="-I${TOP_BUILDDIR}/pscom/ ${MPICHLIB_CFLAGS}"
	fi

	AC_CHECK_HEADER([infiniband/verbs.h], [have_openib=yes], [have_openib=no])
	AC_CHECK_HEADER([psm2.h], [have_psm2=yes], [have_psm2=no])

	if test -n "$PSCOM_BUILTIN_PLUGINS" ; then
		IFS=","
		for plugin in $PSCOM_BUILTIN_PLUGINS ; do
			case "$plugin" in
				"openib") openib_enabled="yes" ;;
				"psm2")   psm2_enabled="yes"   ;;
			esac
		done
		unset IFS
	else
		openib_enabled="$have_openib"
		psm2_enabled="$have_psm2"
	fi

	if test "$openib_enabled" = "yes" ; then
		if test "$have_openib" = "yes" ; then
			export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DPSCOM_ALLIN_OPENIB"
			export PSCOM_ALLIN_LIBS="${PSCOM_ALLIN_LIBS} -libverbs"
			echo "Including openib as a built-in plugin for pscom"
		else
			AC_MSG_ERROR([openib support requested but not found])
		fi
	fi
	if test "$psm2_enabled" = "yes" ; then
		if test "$have_psm2" = "yes" ; then
			export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DPSCOM_ALLIN_PSM2"
			export PSCOM_ALLIN_LIBS="${PSCOM_ALLIN_LIBS} -lpsm2"
			echo "Including psm2 as a built-in plugin for pscom"
		else
			AC_MSG_ERROR([psm2 support requested but not found])
		fi
	fi

	case $CONFSET in
		"default"|"gcc"|"intel"|"icc") ;;
		*) confset_and_allin_does_not_match="yes" ;;
	esac
	if test "$confset_and_allin_does_not_match" = "yes" ; then
		AC_MSG_ERROR([confset=$CONFSET does not allow for --with-pscom-allin])
	fi
],[
	PSCOM_ALLIN="false"
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
CONF_OPS_COMMON="$CONF_OPS_COMMON --enable-fast=O3,ndebug"
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

	CONF_OPS="$CONF_OPS"
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

	CONF_OPS="$CONF_OPS CC=gcc CXX=g++"
	if which gfortran >/dev/null 2>&1 ;then
	   # found gfortran
	   CONF_OPS="$CONF_OPS --enable-f77"
	   CONF_OPS="$CONF_OPS --enable-fc"
	   CONF_OPS="$CONF_OPS FC=gfortran"
	fi
	MODULEFILE_CONFSET=""
	EXTRA_VERSION_INFO=" compiler: $(gcc --version|head -n 1)"

	if test "$ENABLE_PROFILE" = "use" ; then
		export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -fprofile-use -fprofile-dir=$PROFILE_DIR -Wno-error=coverage-mismatch"
	else
		if test "$ENABLE_PROFILE" = "gen" ; then
			export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -fprofile-generate -fprofile-dir=$PROFILE_DIR"
			# Fix me! Disable Fortran for avoiding issues when compiling with FDO enabled:
			CONF_OPS=`echo "$CONF_OPS" | sed -e s/enable-f77/disable-fortran/g`
		fi
	fi
	;;
"icc"|\
"intel")
	echo "Using confset=intel"
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_COMMON"
	CONF_OPS="$CONF_OPS --enable-f77"
	CONF_OPS="$CONF_OPS --enable-fc"
	CONF_OPS="$CONF_OPS --enable-cxx"
	CONF_OPS="$CONF_OPS --enable-romio"
	CONF_OPS="$CONF_OPS --enable-mpe"	# need also romio!

	CONF_OPS="$CONF_OPS CC=icc CXX=icpc"
	# Append "-reentrancy threaded" to FC.
	# FFLAGS, FCFLAGS, MPICHLIB_FCFLAGS, MPICHLIB_LDFLAGS and friends
	# are not forwarded to the final linking with libtool which uses $(FC).
	# "libtool -XCClinker" should be set somehow. Up to than we have
	# '-reentrancy threaded' also inside mpif77, mpif90 and mpifort.
	CONF_OPS="$CONF_OPS FC='ifort -reentrancy threaded'"
	CONF_OPS="$CONF_OPS"
	EXTRA_VERSION_INFO=" compiler: $(icc -V 2>&1 |head -n 1)"

	if test "$ENABLE_PROFILE" = "use" ; then
		export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -prof-use -prof-dir=$PROFILE_DIR"
        else
                if test "$ENABLE_PROFILE" = "gen" ; then
			export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -prof-gen -prof-dir=$PROFILE_DIR"
                fi
        fi
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

	CONF_OPS="$CONF_OPS CC=pgcc CXX=pgc++ FC=pgf90"
	CONF_OPS="$CONF_OPS"
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
	CONF_OPS="$CONF_OPS"
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

	CONF_OPS="$CONF_OPS"
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

	CONF_OPS="$CONF_OPS"
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
	CONF_OPS="$CONF_OPS"
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
	CONF_OPS="$CONF_OPS"
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

	CONF_OPS="$CONF_OPS CC=gcc CXX=g++ FC=gfortran"

	ENABLE_HYDRA="true"
	;;
*)
	echo "Sorry. Unknown confset ${CONFSET}"
	exit 1
	;;
esac

AC_ARG_WITH([threading], [
  --with-threading        Enable threading 'multiple' in device psp
],[
	export PSP_THREADING=multiple
	MODULEFILE_CONFSET="$MODULEFILE_CONFSET${MODULEFILE_CONFSET:+-}mt"
])

AC_ARG_WITH([modulefile], [
  --with-modulefile=filename
			  Destinaton of the generated modulefile
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
AC_ARG_WITH([topology-awareness], [
  --with-topology-awareness
			  Enable topology/hierarchy awareness for collectives
],[
	export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DMPID_PSP_TOPOLOGY_AWARENESS"
	ENABLE_TOPOLOGY_AWARENESS="true"
],[
	export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DMPIR_CONTEXT_SUBCOMM_WIDTH=0"
	ENABLE_TOPOLOGY_AWARENESS="false"
])
AC_SUBST([ENABLE_TOPOLOGY_AWARENESS])
export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DMPIR_CONTEXT_DYNAMIC_PROC_WIDTH=0"

AC_ARG_WITH([cuda], [
  --with-cuda
			  Enable CUDA awareness within ParaStationMPI
],[
	CONF_OPS="$CONF_OPS --enable-psp-cuda-awareness"

	CUDA_HOME="${CUDA_HOME-"/usr/local/cuda/"}"
	if ! test -d "$CUDA_HOME" ; then
		unset CUDA_HOME
	fi
	CONF_OPS="$CONF_OPS PSP_CPPFLAGS='-DMPID_PSP_WITH_CUDA_AWARENESS'"
	CONF_OPS="$CONF_OPS"
])


if test "$PSP_THREADING" = "multiple" ; then
   echo "Preparing psmpi/mpich with MPI_THREAD_MULTIPLE support."
fi
if test "$ENABLE_HYDRA" = "true" ; then
   echo "Preparing psmpi/mpich with Hydra's mpiexec support."
fi
if test "$ENABLE_TOPOLOGY_AWARENESS" = "true" ; then
   echo "Preparing psmpi/mpich with topology-awareness."
fi
if test "$PSCOM_ALLIN" = "true" ; then
   echo "Preparing psmpi/mpich with pscom included."
fi
sleep 1

AC_SUBST([CONFSET])
AC_SUBST([MODULEFILE])
AC_SUBST([EXTRA_VERSION_INFO])
AC_SUBST([CUDA_HOME])

# ToDo: directly  use PSCOM_LDFLAGS from mpich2/src/mpid/psp/subconfigure.m4
PSCOM_LDFLAGS="${PSCOM_LDFLAGS-"-L/opt/parastation/lib64"}"
AC_SUBST([PSCOM_LDFLAGS])

PSCOM_LIBRARY="${PSCOM_LIBRARY-"-lpscom"}"
AC_SUBST([PSCOM_LIBRARY])

AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([scripts/generate_modulefile.sh], [chmod +x scripts/generate_modulefile.sh])
AC_CONFIG_FILES([scripts/generate_mpicc-conf.sh], [chmod +x scripts/generate_mpicc-conf.sh])

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
#  compile-command: "module load automake-1.15 && autoreconf -vif"
# End:
