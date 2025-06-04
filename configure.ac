#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

# Retrieve version information from MPICH
m4_include([mpich2/maint/version.m4])

AC_INIT([psmpi],[],[support@par-tec.com])
AC_CONFIG_SRCDIR([configure.ac])

AC_CONFIG_AUX_DIR([scripts])
AC_CONFIG_MACRO_DIR([mpich2/confdb])
AC_CANONICAL_TARGET([])
AC_PROG_CC

CONF_ADI3_DEVICE="psp"
CONF_ECHO_PREFIX="psmpi: "

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

# Get the version and release date from MPICH
MPICH_VERSION=MPICH_VERSION_m4
AC_SUBST([MPICH_VERSION])
MPICH_RELEASE_DATE="MPICH_RELEASE_DATE_m4"
AC_SUBST([MPICH_RELEASE_DATE])

# configure mpich2

DISABLE_HWLOC="true" # For the time being, disable HWLOC by default!
AC_ARG_WITH([hwloc], [
  --with-hwloc[[=path]]     Use the HWLOC support in MPICH (and specify path to HWLOC directory).
],[
	if test "x$with_hwloc" = "xno" ; then
	   DISABLE_HWLOC="true"
	else
	   if test "x$with_hwloc" = "xyes" ; then
	      WITH_HWLOC_DIR=""
	   else
	      # Path to be passed to MPICH's configure for HWLOC:
	      WITH_HWLOC_DIR="${with_hwloc}"
	   fi
	   DISABLE_HWLOC="false"
	fi
])
AC_SUBST([DISABLE_HWLOC])

DISABLE_HCOLL="true" # For the time being, disable HCOLL by default!
AC_ARG_WITH([hcoll], [
  --with-hcoll[[=path]]     Use HCOLL support (and specify path to HCOLL directory).
],[
	if test "x$with_hcoll" = "xno" ; then
	   DISABLE_HCOLL="true"
	else
	   DISABLE_HCOLL="false"
	   # Path to be passed to src/mpid/common/hcoll/subconfigure.m4:
	   WITH_HCOLL_DIR="${with_hcoll}"
	fi
])
AC_SUBST([DISABLE_HCOLL])

AC_ARG_WITH([confset], [
  --with-confset[=set]  Use predefined configuration set ["default"]
	   default    : Like 'gcc'
	   none       : Do not configure mpich.
			Prepare only for tar, rpm and srpm build

	   gcc        : Use Gnu compiler (gcc)
	   intel      : Use Intel compiler (icx)
	   icx        : Like 'intel'
	   pgi	      : Portland Group compiler (pgcc)
	   nvhpc      : Nvidia HPC compiler (nvc)
	   llvm       : LLVM-based toolchains (e.g., AOCC)

	   devel      : With error checking and debug info (gcc)
	   debug      : Like 'devel' but without any optimizations

	   user       : No predefined options
	   ch3|ch4    : original mpich ch3|ch4 device (no ParaStation device)
	   "--xyz --abc" : Use this to configure mpich without a set
],[
	CONFSET="$withval"
],[
	CONFSET="default"
])

AC_ARG_ENABLE(confset-overwrite,
              AS_HELP_STRING([--enable-confset-overwrite],
                             [Enable overwriting of compilers selected by confset via environment variables CC, CXX and FC]),
                             enable_confset_overwrite=yes, enable_confset_overwrite=no
)

AC_ARG_WITH([pscom-allin], [
  --with-pscom-allin[=path]
			  Use pscom source code fusion and set path to pscom
],,[with_pscom_allin=no])

AC_ARG_WITH([pscom], [
  --with-pscom[=path]
			  Use pscom as communication transport (used by default, specify path to the pscom installation)
],[],[with_pscom=yes])

AC_ARG_WITH([mpichconf], [
  --with-mpichconf[=opts]   Additional mpich configuration options
],[
	MPICHCONF="$withval"
],[
	unset MPICHCONF
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

CONF_OPS_COMMON="$CONF_OPS_COMMON --enable-cache"
CONF_OPS_COMMON="$CONF_OPS_COMMON --disable-static"
CONF_OPS_COMMON="$CONF_OPS_COMMON --enable-shared"
CONF_OPS_COMMON="$CONF_OPS_COMMON --enable-romio"

CONF_OPS_DEVEL="$CONF_OPS_DEVEL $CONF_OPS_COMMON"
CONF_OPS_DEVEL="$CONF_OPS_DEVEL --enable-g=all"
CONF_OPS_DEVEL="$CONF_OPS_DEVEL --enable-dependencies"
CONF_OPS_DEVEL="$CONF_OPS_DEVEL --enable-error-checking=all"
CONF_OPS_DEVEL="$CONF_OPS_DEVEL --enable-strict" # Compile with -Wall and other checks.

CONF_OPS_DEPLOY="$CONF_OPS_DEPLOY $CONF_OPS_COMMON"
CONF_OPS_DEPLOY="$CONF_OPS_DEPLOY --enable-fast=O3,ndebug"
CONF_OPS_DEPLOY="$CONF_OPS_DEPLOY --enable-lib-depend"
CONF_OPS_DEPLOY="$CONF_OPS_DEPLOY --disable-rpath"
CONF_OPS_DEPLOY="$CONF_OPS_DEPLOY --enable-cxx"
CONF_OPS_DEPLOY="$CONF_OPS_DEPLOY --enable-fortran=all"

MODULEFILE_CONFSET="$CONFSET"

case $CONFSET in
"devel")
	CONF_OPS="$CONF_OPS $CONF_OPS_DEVEL"
	;;
"debug")
	CONF_OPS="$CONF_OPS $CONF_OPS_DEVEL"
	CONF_OPS="$CONF_OPS --disable-fast" # Complile with -O0
	;;
"gcc"|\
"default")
	CONF_OPS="$CONF_OPS $CONF_OPS_DEPLOY"

	if [[ -n "$CC" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting C compiler with $CC])
	else
		CC=gcc
	fi
	if [[ -n "$CXX" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting C++ compiler with $CXX])
	else
		CXX=g++
	fi
	if [[ -n "$FC" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting Fortan compiler with $FC])
	else
		FC=gfortran
	fi

	MODULEFILE_CONFSET=""
	EXTRA_VERSION_INFO=" compiler: $($CC --version|head -n 1)"

	if test "$ENABLE_PROFILE" = "use" ; then
		export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -fprofile-use -fprofile-dir=$PROFILE_DIR -Wno-error=coverage-mismatch"
	else
		if test "$ENABLE_PROFILE" = "gen" ; then
			export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -fprofile-generate -fprofile-dir=$PROFILE_DIR"
			# Fix me! Disable Fortran for avoiding issues when compiling with FDO enabled:
			CONF_OPS=`echo "$CONF_OPS" | sed -e s/enable-fortran=all/enable-fortran=no/g`
		fi
	fi
	;;
"icx"|\
"intel")
	CONF_OPS="$CONF_OPS $CONF_OPS_DEPLOY"

	if [[ -n "$CC" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting C compiler with $CC])
	else
		CC=icx
	fi
	if [[ -n "$CXX" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting C++ compiler with $CXX])
	else
		CXX=icpx
	fi

	# Append "-reentrancy threaded" to FC.
	# FFLAGS, FCFLAGS, MPICHLIB_FCFLAGS, MPICHLIB_LDFLAGS and friends
	# are not forwarded to the final linking with libtool which uses $(FC).
	# "libtool -XCClinker" should be set somehow. Up to than we have
	# '-reentrancy threaded' also inside mpif77, mpif90 and mpifort.
	if [[ -n "$FC" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting Fortan compiler with $FC, using with -reentrancy threaded])
		FC="$FC -reentrancy threaded"
	else
		FC='ifx -reentrancy threaded'
	fi

	EXTRA_VERSION_INFO=" compiler: $($CC -V 2>&1 |head -n 1)"

	if test "$ENABLE_PROFILE" = "use" ; then
		export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -fprofile-use=$PROFILE_DIR"
	else
		if test "$ENABLE_PROFILE" = "gen" ; then
			export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -fprofile-generate=$PROFILE_DIR"
			# Fix me! Disable Fortran for avoiding issues when compiling with PGO enabled:
			CONF_OPS=`echo "$CONF_OPS" | sed -e s/enable-fortran=all/enable-fortran=no/g`
		fi
	fi
	;;
"pgi")
	CONF_OPS="$CONF_OPS $CONF_OPS_DEPLOY"

	if [[ -n "$CC" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting C compiler with $CC])
	else
		CC=pgcc
	fi
	if [[ -n "$CXX" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting C++ compiler with $CXX])
	else
		CXX=pgc++
	fi
	if [[ -n "$FC" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting Fortan compiler with $FC])
	else
		FC=pgf90
	fi
	;;

"nvhpc")
	CONF_OPS="$CONF_OPS $CONF_OPS_DEPLOY"

	if [[ -n "$CC" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting C compiler with $CC])
	else
		CC=nvc
	fi
	if [[ -n "$CXX" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting C++ compiler with $CXX])
	else
		CXX=nvc++
	fi
	if [[ -n "$FC" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting Fortan compiler with $FC])
	else
		FC=nvfortran
	fi
	;;
"llvm")
	CONF_OPS="$CONF_OPS $CONF_OPS_DEPLOY"

	if [[ -n "$CC" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting C compiler with $CC])
	else
		CC=clang
	fi
	if [[ -n "$CXX" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting C++ compiler with $CXX])
	else
		CXX=clang++
	fi
	if [[ -n "$FC" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting Fortan compiler with $FC])
	else
		FC=flang
	fi
	;;
"none")
	# No build
	;;
"user")
	CONF_OPS=
	;;
"ch"[[34]]*)
	CONF_ADI3_DEVICE="${CONFSET}"
	# Explicitly set the UCX netwomod
	if test "x${CONF_ADI3_DEVICE}" = "xch4"; then
		CONF_ADI3_DEVICE="ch4:ucx"
	fi

	MODULEFILE_CONFSET="mpich"

	CONF_OPS="$CONF_OPS $CONF_OPS_DEPLOY"

	if [[ -n "$CC" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting C compiler with $CC])
	else
		CC=gcc
	fi
	if [[ -n "$CXX" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting C++ compiler with $CXX])
	else
		CXX=g++
	fi
	if [[ -n "$FC" -a "$enable_confset_overwrite" = "yes" ]]; then
		AC_MSG_WARN([confset $CONFSET: Overwriting Fortan compiler with $FC])
	else
		FC=gfortran
	fi

	CHANNEL_DEVICE=`echo ${CONFSET} | cut -c 1-2`
	;;
*)
	echo "${CONF_ECHO_PREFIX}Sorry. Unknown confset '${CONFSET}'."
	exit 1
	;;
esac

# Set the ADI3 device to be used with all the options parsed and set so far.
CONF_OPS="--with-device=${CONF_ADI3_DEVICE} ${CONF_OPS}"

# Additional psp-specific options
if test "${CONF_ADI3_DEVICE}" = "psp" ; then
	CONF_OPS="$CONF_OPS --with-psp-confset=$CONFSET"  # Make confset available to subconfigure.m4
	CONF_OPS="$CONF_OPS --with-datatype-engine=yaksa" # 'yaksa' is the default datatype enginge
	CONF_OPS="$CONF_OPS --with-psp-pscom=${with_pscom}"
fi

AC_ARG_WITH([pscom-builtin], [
  --with-pscom-builtin[=list]
			  Define list of pscom plugins that shall be built-in
			  Default: psm2,ucp / Requires '--with-pscom-allin'
],[
	PSCOM_BUILTIN_PLUGINS="$withval"
])

CONF_ECHO_ALLIN=""

AS_IF([test "x$with_pscom_allin" != "xno"],[
	# Ensure somethin else than "yes" was given
	AS_IF([test "x$with_pscom_allin" == "xyes"],[
		AC_MSG_ERROR(["You have to provide a path to the pscom sources. Abort!"])
	])

	export PSCOM_ALLIN="true"
	PSCOM_ALLIN_DIR="$with_pscom_allin"
	AC_CHECK_FILE(["${PSCOM_ALLIN_DIR}/lib/all/pscom_all.c"], [
		PSCOM_ALLIN_DIR="${PSCOM_ALLIN_DIR}/lib/all/"
	])
	export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DPSCOM_ALLIN -I${PSCOM_ALLIN_DIR}/ -I${PSCOM_ALLIN_DIR}/../../include/ -I${PSCOM_ALLIN_DIR}/../../lib/pscom/"

	if test -x "${PSCOM_ALLIN_DIR}/../../scripts/vcversion" ; then
		if ! test -d "${TOP_BUILDDIR}/pscom" ; then
			mkdir -p "${TOP_BUILDDIR}/pscom" 2>/dev/null
		fi
		PSCOM_VC_VERSION=`${PSCOM_ALLIN_DIR}/../../scripts/vcversion -r ${PSCOM_ALLIN_DIR}/../.. -n`
		CONF_ECHO_ALLIN="${CONF_ECHO_ALLIN:+${CONF_ECHO_ALLIN}\n}"
		CONF_ECHO_ALLIN="${CONF_ECHO_ALLIN}Including pscom version '${PSCOM_VC_VERSION}'."
		echo "#define VC_VERSION \"${PSCOM_VC_VERSION}\"" > "${TOP_BUILDDIR}/pscom/vc_version.h"
		export MPICHLIB_CFLAGS="-I${TOP_BUILDDIR}/pscom/ ${MPICHLIB_CFLAGS}"
	fi

	AC_CHECK_HEADER([infiniband/verbs.h], [have_openib=yes], [have_openib=no])
	AC_CHECK_HEADER([psm2.h], [have_psm2=yes], [have_psm2=no])
	AC_CHECK_HEADER([ucp/api/ucp.h], [have_ucp_h=yes], [have_ucp_h=no])
	AC_CHECK_HEADER([ucp/api/ucp_def.h], [have_ucp_def_h=yes], [have_ucp_def_h=no])
	AS_IF([test "x${have_ucp_h}" = "xyes" -a "x${have_ucp_def_h}" = "xyes"], [have_ucp=yes], [have_ucp=no])

	if test -n "$PSCOM_BUILTIN_PLUGINS" ; then
		IFS=","
		for plugin in $PSCOM_BUILTIN_PLUGINS ; do
			case "$plugin" in
				"openib") openib_enabled="yes" ;;
				"psm2")   psm2_enabled="yes"   ;;
				"ucp")    ucp_enabled="yes"    ;;
			esac
		done
		unset IFS
	else
		psm2_enabled="$have_psm2"
		ucp_enabled="$have_ucp"
	fi

	if test "$openib_enabled" = "yes" ; then
		if test "$have_openib" = "yes" ; then
			export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DPSCOM_ALLIN_OPENIB"
			export PSCOM_ALLIN_LIBS="${PSCOM_ALLIN_LIBS} -libverbs"
			CONF_ECHO_ALLIN="${CONF_ECHO_ALLIN:+${CONF_ECHO_ALLIN}\n}"
			CONF_ECHO_ALLIN="${CONF_ECHO_ALLIN}Including openib as a built-in plugin for pscom."
		else
			AC_MSG_ERROR([openib support requested but not found])
		fi
	fi
	if test "$psm2_enabled" = "yes" ; then
		if test "$have_psm2" = "yes" ; then
			export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DPSCOM_ALLIN_PSM2"
			export PSCOM_ALLIN_LIBS="${PSCOM_ALLIN_LIBS} -lpsm2"
			CONF_ECHO_ALLIN="${CONF_ECHO_ALLIN:+${CONF_ECHO_ALLIN}\n}"
			CONF_ECHO_ALLIN="${CONF_ECHO_ALLIN}Including psm2 as a built-in plugin for pscom."
		else
			AC_MSG_ERROR([psm2 support requested but not found])
		fi
	fi
	if test "$ucp_enabled" = "yes" ; then
		if test "$have_ucp" = "yes" ; then
			export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DPSCOM_ALLIN_UCP"
			export PSCOM_ALLIN_LIBS="${PSCOM_ALLIN_LIBS} -lucp -lucs"
			CONF_ECHO_ALLIN="${CONF_ECHO_ALLIN:+${CONF_ECHO_ALLIN}\n}"
			CONF_ECHO_ALLIN="${CONF_ECHO_ALLIN}Including ucp as a built-in plugin for pscom."

		else
			AC_MSG_ERROR([ucp support requested but not found])
		fi
	fi
	case $CONFSET in
		"default"|"gcc"|"intel"|"nvhpc"|"icx"|"llvm"|"devel"|"debug") ;;
		*) confset_and_allin_does_not_match="yes" ;;
	esac
	if test "$confset_and_allin_does_not_match" = "yes" ; then
		AC_MSG_ERROR([confset=$CONFSET does not allow for --with-pscom-allin])
	fi

	# -- visibility ("protected") support --

	# Does __attribute__ ((visibility ("protected"))) work with functions?
	AC_CACHE_CHECK(linker support for protected functions,
		pscom_cv_protected_func,
		[cat > conftest.c <<EOF
		__attribute__ ((visibility("protected")))
		int bar(void) { return 1; }
EOF
		pscom_cv_protected_func=no
		if AC_TRY_COMMAND(${CC-cc} -nostdlib -nostartfiles $no_ssp -fPIC -shared conftest.c -o conftest.so); then
			cat > conftest.c <<EOF
			extern int bar(void);
			int main (void) { return bar(); }
EOF
			if AC_TRY_COMMAND(${CC-cc} -nostdlib -nostartfiles $no_ssp conftest.c -o conftest conftest.so); then
				pscom_cv_protected_func=yes
			fi
		fi
		rm -f conftest.*
	])
	AC_SUBST(pscom_cv_protected_func)
	AS_IF([test "$pscom_cv_protected_func" = "no"],
		# AC_DEFINE([NO_PROTECTED_FUNC], [1], [Define to 1 if visibility ("protected") does not work on functions])
		export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DNO_PROTECTED_FUNC=1 -DENABLE_PLUGIN_LOADING=0"
	)

],[
	PSCOM_ALLIN="false"
])


AC_ARG_ENABLE(threading,
              AS_HELP_STRING([--enable-threading],
                             [Enable threading 'multiple' in device psp]),,
                             enable_threading=no
)
if test "$enable_threading" = "yes" ; then
    export PSP_THREADING=multiple
    MODULEFILE_CONFSET="$MODULEFILE_CONFSET${MODULEFILE_CONFSET:+-}mt"
fi

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

AC_ARG_ENABLE(hydra,
              AS_HELP_STRING([--enable-hydra],
                             ["Build and install MPICH's process manager hydra"]),,
                             enable_hydra=no
)
if test "x$enable_hydra" = "xyes" ; then
    ENABLE_HYDRA="true"
    CONF_OPS="$CONF_OPS --with-pm=hydra"
else
    ENABLE_HYDRA="false"
    CONF_OPS="$CONF_OPS --with-pm=none"
fi

AC_ARG_ENABLE(statistics,
              AS_HELP_STRING([--enable-statistics],
                             [Enable the collection of statistical information]),,
                             enable_statistics=no
)
if test "$enable_statistics" = "yes" ; then
    ENABLE_STATISTICS="true"
    CONF_OPS="$CONF_OPS --enable-psp-statistics"
fi

AC_ARG_ENABLE(msa-awareness,
              AS_HELP_STRING([--enable-msa-awareness],
                             [Enable MSA awareness like hierarchical collectives]),,
                             enable_msa_awareness=no
)
if test "$enable_msa_awareness" = "yes" ; then
    ENABLE_MSA_AWARENESS="true"
	CONF_OPS="$CONF_OPS --enable-psp-msa-awareness"
else
	ENABLE_MSA_AWARENESS="false"
fi

AC_SUBST([ENABLE_MSA_AWARENESS])
if test -z "${CHANNEL_DEVICE}" ; then
   export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DMPIR_CONTEXT_DYNAMIC_PROC_WIDTH=0"
fi

# Check for OpenPMIx library
openpmix_min_version=4.2.0
openpmix_min_version_numeric="0x00040200"

PAC_CHECK_HEADER_LIB_OPTIONAL(pmix, pmix.h, pmix, PMIx_Init)

if test "$pac_have_pmix" = "yes"; then
    AC_MSG_CHECKING([if OpenPMIx version >= $openpmix_min_version])
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <pmix_version.h>]],
                      [[
#if PMIX_NUMERIC_VERSION < $openpmix_min_version_numeric
#error "OpenPMIx version is less than $openpmix_min_version"
#endif
                      ]])],
                      [AC_MSG_RESULT([yes])
                      ENABLE_PMIX="true"
                      ],
                      [AC_MSG_RESULT([no])
                      if test -n "${with_pmix}" ; then
                          AC_MSG_ERROR([OpenPMIx version < $openpmix_min_version but PMIx support was requested!])
                      else
                          AC_MSG_WARN([OpenPMIx version < $openpmix_min_version. Compiling psmpi without PMIx support!])
                          ENABLE_PMIX="false"
                      fi
                      ])
else
    ENABLE_PMIX="false"
fi

if test "x$ENABLE_PMIX" == "xtrue"; then
    # Always enable MPICH's built-in PMI1/2 support along with OpenPMIx
    CONF_OPS="$CONF_OPS --with-pmix${with_pmix:+=$with_pmix} --with-pmilib=mpich"
else
    # OpenPMIx library not available or cannot be used due to minimum version requirement
    # Enable only MPICH's embedded PMI1 support
    CONF_OPS="$CONF_OPS --without-pmix --with-pmi=pmi1"
fi

AC_ARG_WITH([cuda], [
  --with-cuda[[=path]]
			  Use CUDA awareness within ParaStation MPI (and specify path to CUDA directory)
],,[with_cuda=no])

if test "x${with_cuda}" != "xno"; then
	# Determine CUDA_HOME directory
	AS_IF(	[test "x${with_cuda}" = "xyes"],
		[CUDA_HOME="${CUDA_HOME-"/usr/local/cuda"}"],
		[CUDA_HOME="${with_cuda}"]
	)

	# Check if CUDA_HOME is a valid directory
	AC_MSG_CHECKING([if ${CUDA_HOME} is a valid directory])
	AS_IF(	[test -d "${CUDA_HOME}"],
		[
			AC_MSG_RESULT(yes)
			have_cuda=yes
		],
		[
			AC_MSG_RESULT(no)
			have_cuda=no
		]
	)

	# Use user-provided $NVCC_FLAGS or set ccbin to $CC
	AS_IF([test -n "$NVCC_FLAGS"],[
		AC_MSG_WARN([nvcc flags: Overwriting NVCC_FLAGS with '${NVCC_FLAGS}'])
		nvcc_flags_overwrite=yes
	],[
		NVCC_FLAGS="-ccbin ${CC}"
		# nvc doesn't work, use nvc++ instead
		NVCC_FLAGS=$(echo ${NVCC_FLAGS} | sed -e 's/nvc/nvc++/g' -e 's/pgcc/pgc++/g')
		nvcc_flags_overwrite=no
	])

	# save language settings, customize ac_ext and ac_compile to support CUDA
	AC_LANG_PUSH([C])
	ac_ext=cu
	ac_compile="${CUDA_HOME}/bin/nvcc ${NVCC_FLAGS} -c conftest.$ac_ext >&5"

	AC_MSG_CHECKING([whether nvcc works with NVCC_FLAGS='${NVCC_FLAGS}'])
	AC_COMPILE_IFELSE([AC_LANG_PROGRAM([__global__ void foo(int x) {}],[])],[
		AC_MSG_RESULT([yes])
		have_nvcc=yes
	],[
		AC_MSG_RESULT([no])

		AS_IF([test "x$nvcc_flags_overwrite" == "xyes"],[
			AC_MSG_ERROR(["nvcc is not functional with NVCC_FLAGS='${NVCC_FLAGS}'. Abort!"])
			have_nvcc=no
		],[
			# Try fallback to gcc
			AC_MSG_WARN([Falling back to 'gcc' as host compiler])

			NVCC_FLAGS='-ccbin gcc'
			ac_compile="${CUDA_HOME}/bin/nvcc ${NVCC_FLAGS} -c conftest.$ac_ext >&5"

			AC_MSG_CHECKING([whether nvcc works with 'gcc' as host compiler])
			AC_COMPILE_IFELSE([AC_LANG_PROGRAM([__global__ void foo(int x) {}],[])],[
				AC_MSG_RESULT([yes])
				have_nvcc=yes
			],[
				AC_MSG_RESULT([no])
				have_nvcc=no
			])
		])
	])

	# done with CUDA, back to C
	AC_LANG_POP([C])

	NVCC="${CUDA_HOME}/bin/nvcc ${NVCC_FLAGS}"

	# CUDA could be found and enabled
	AS_IF(	[test "x${have_cuda}" = "xyes" -a "x${have_nvcc}" = "xyes"],
		[
			CUDA_SUPPORT="true"
			CONF_OPS="$CONF_OPS --enable-psp-cuda-awareness --with-cuda=${CUDA_HOME}"
			MPICCCONFDIR="${sysconfdir}"
		],
		[test "x${have_cuda}" = "xyes" -a "x{have_nvcc}" != "xyes"],
		[AC_MSG_ERROR([CUDA found, but the 'nvcc' compiler is not working.])],
		[
			AC_MSG_ERROR([CUDA support is requested but CUDA home directory cannot be found])
		]
	)
else
	CUDA_SUPPORT="no"
	MPICCCONFDIR=""
	CONF_OPS="$CONF_OPS --without-cuda"
fi

test -n "$CC"      && CONF_OPS="$CONF_OPS 'CC=$CC'"
test -n "$CXX"     && CONF_OPS="$CONF_OPS 'CXX=$CXX'"
test -n "$FC"      && CONF_OPS="$CONF_OPS 'FC=$FC'"
test -n "$F77"     && CONF_OPS="$CONF_OPS 'F77=$F77'"
test -n "$AR"      && CONF_OPS="$CONF_OPS 'AR=$AR'"
test -n "$NVCC"    && CONF_OPS="$CONF_OPS 'NVCC=$NVCC'"
test -n "$RANLIB"  && CONF_OPS="$CONF_OPS 'RANLIB=$RANLIB'"
test -n "$FFLAGS"  && CONF_OPS="$CONF_OPS 'FFLAGS=$FFLAGS'"
test -n "$FCFLAGS" && CONF_OPS="$CONF_OPS 'FCFLAGS=$FCFLAGS'"

AC_SUBST([CONFSET])
AC_SUBST([CUDA_SUPPORT])
AC_SUBST([MODULEFILE])
AC_SUBST([EXTRA_VERSION_INFO])
AC_SUBST([CUDA_HOME])
AC_SUBST([MPICCCONFDIR])

# ToDo: directly  use PSCOM_LDFLAGS from mpich2/src/mpid/psp/subconfigure.m4
PSCOM_LDFLAGS="${PSCOM_LDFLAGS-"-L/opt/parastation/lib64"}"
AS_IF([test "x$with_pscom" != "xno" -a "x$with_pscom" != "xyes"],[
	PSCOM_LDFLAGS="-L${with_pscom}/lib64"
	PSCOM_LD_LIBRARY_PATH="${with_pscom}/lib64"
])
AC_SUBST([PSCOM_LDFLAGS])
AC_SUBST([PSCOM_LD_LIBRARY_PATH])

PSCOM_LIBRARY="${PSCOM_LIBRARY-"-lpscom"}"
AC_SUBST([PSCOM_LIBRARY])

AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([scripts/generate_modulefile.sh], [chmod +x scripts/generate_modulefile.sh])
AC_CONFIG_FILES([scripts/generate_mpicc-conf.sh], [chmod +x scripts/generate_mpicc-conf.sh])

AC_OUTPUT

if test "x$DISABLE_HWLOC" = "xtrue" ; then
   CONF_OPS="$CONF_OPS --without-hwloc"
   CONF_OPS="$CONF_OPS --with-hydra-topolib='no'"
else
   if test -n "${WITH_HWLOC_DIR}" ; then
      CONF_OPS="$CONF_OPS --with-hwloc=${WITH_HWLOC_DIR}"
   else
       # Mute --with-hwloc from the top-level configure and let MPICH check for external or embedded HWLOC.
       CONFIGARGS=`echo "$CONFIGARGS" | sed "s/'--with-hwloc'//"`
   fi
fi

if test "x$DISABLE_HCOLL" = "xtrue" ; then
   CONF_OPS="$CONF_OPS --without-hcoll"
elif test -n "$WITH_HCOLL_DIR" ; then
   CONF_OPS="$CONF_OPS --with-hcoll=${WITH_HCOLL_DIR}"
fi

if test "${CONFSET}" != "none" ; then
   # Call mpich2 configure:
   mkdir -p "${TOP_BUILDDIR}/mpich2"
   cd "${TOP_BUILDDIR}/mpich2"
   echo "############### configure mpich #############"
   echo "wd=$(pwd)"
   #   echo ${TOP_SRCDIR}/mpich2/configure ${CONF_OPS} "$@"

   # call original mpich configure
   # --disable-maintainer-mode prevent from refreshing autoconf generated files in the source folder.
   eval ${TOP_SRCDIR}/mpich2/configure ${CONFIGARGS} ${CONF_OPS} ${MPICHCONF} --disable-maintainer-mode
fi

echo "####### SUMMARY of psmpi configuration ########"
echo "${CONF_ECHO_PREFIX}Using confset='${CONFSET}'."
CONF_ECHO_SUMMARY_PREFIX="${CONF_ECHO_PREFIX}Configured mpich/${CONF_ADI3_DEVICE} with "
conf_echo() {
   echo "${CONF_ECHO_SUMMARY_PREFIX}$1"
   CONF_ECHO_SUMMARY_PRINTED=yes
}
if test "$PSP_THREADING" = "multiple" ; then
   conf_echo "MPI_THREAD_MULTIPLE support."
fi
if test "$ENABLE_HYDRA" = "true" ; then
   conf_echo "Hydra's mpiexec support."
fi
if test "$ENABLE_PMIX" = "true" ; then
   conf_echo "PMIx support."
fi
if test "$ENABLE_STATISTICS" = "true" ; then
   conf_echo "support for statistics."
fi
if test "$ENABLE_MSA_AWARENESS" = "true" ; then
   conf_echo "MSA awareness."
fi
if test "$PSCOM_ALLIN" = "true" ; then
   conf_echo "pscom included."
   if test -n "${CONF_ECHO_ALLIN}" ; then
      CONF_ECHO_ALLIN=`echo -en ${CONF_ECHO_ALLIN} | sed -z "s/\n/\n${CONF_ECHO_PREFIX}/g;s/,$/\n/"`
      echo -e "${CONF_ECHO_PREFIX}${CONF_ECHO_ALLIN}"
   fi
fi
if test "$CUDA_SUPPORT" = "true" ; then
   conf_echo "CUDA support."
fi
if test -n "$WITH_HCOLL_DIR" ; then
   conf_echo "HCOLL support."
fi
if test "${CONF_ADI3_DEVICE}" != "psp" -a "${CONF_ECHO_SUMMARY_PRINTED}" = "yes" ; then
   echo "${CONF_ECHO_PREFIX}WARNING Top-level configure arguments for devices other than psp are propagated, but their effectiveness is not checked."
   echo "${CONF_ECHO_PREFIX}WARNING The canonical way for passing configure arguments on to the ch3/4 devices is the use of --with-mpichconf=[[args]]."
fi

#
# Local Variables:
#  compile-command: "module load automake-1.15 && autoreconf -vif"
# End:
