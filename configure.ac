#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

# Retrieve version information from MPICH
m4_include([mpich2/maint/version.m4])

AC_INIT([psmpi],[],[support@par-tec.com])
AC_CONFIG_SRCDIR([configure.ac])

AC_CONFIG_AUX_DIR([scripts])
AC_CONFIG_MACRO_DIR([mpich2/confdb])
AC_CANONICAL_TARGET([])

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

AC_ARG_WITH([hydra], [
  --with-hydra            Install also Hydra (mpiexec).
],[
	if test "x$with_hydra" = "xno" ; then
	   ENABLE_HYDRA="false"
	else
           ENABLE_HYDRA="true"
	fi
],[
	ENABLE_HYDRA="false"
])
AC_SUBST([ENABLE_HYDRA])

DISABLE_HWLOC="true" # For the time being, disable HWLOC by default!
AC_ARG_WITH([hwloc], [
  --with-hwloc[[=PATH]]     Enable the HWLOC support in MPICH.
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
  --with-hcoll[[=PATH]]     Enable HCOLL support (and specify path to HCOLL directory).
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
	   intel      : Use Intel compiler (icc)
	   icc        : Like 'intel'
	   pgi	      : Portland Group compiler (pgcc)
	   nvhpc      : Nvidia HPC compiler (nvc)

	   devel      : With error checking and debug info (gcc)
	   debug      : Like 'devel' but without any optimizations

	   user       : No predefined options
	   ch3|ch4    : original mpich ch3|ch4 device (no parastation)
	   "--xyz --abc" : Use this to configure mpich without a set
],[
	CONFSET="$withval"
],[
	CONFSET="default"
])


AC_ARG_WITH([mpichconf], [
  --with-mpichconf[=opts]   Additional mpich configuration options
],[
	MPICHCONF="$withval"
],[
	unset MPICHCONF
])

# Only test fortran-related behavior, if `--enable-fortran=[no|none]` was *not*
# given. Otherwise, fortran might be disabled completely making this check
# obsolete.
if ! echo ${MPICHCONF} | grep -E 'enable-fortran=no'; then
    PAC_PROG_F77_MISMATCHED_ARGS(addarg,yes)
    if test "X$addarg" != "X" ; then
	AC_MSG_WARN([The Fortran compiler $F77 does not accept programs that call the same routine with arguments of different types without the option $addarg. Running configure with FFLAGS=$addarg.])
	FFLAGS="${FFLAGS} ${addarg}"
    fi
fi



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
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_DEVEL"
	;;
"debug")
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_DEVEL"
	CONF_OPS="$CONF_OPS --disable-fast" # Complile with -O0
	;;
"gcc"|\
"default")
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_DEPLOY"

	CC=gcc
	CXX=g++
	FC=gfortran

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
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_DEPLOY"

	CC=icc
	CXX=icpc

	# Append "-reentrancy threaded" to FC.
	# FFLAGS, FCFLAGS, MPICHLIB_FCFLAGS, MPICHLIB_LDFLAGS and friends
	# are not forwarded to the final linking with libtool which uses $(FC).
	# "libtool -XCClinker" should be set somehow. Up to than we have
	# '-reentrancy threaded' also inside mpif77, mpif90 and mpifort.
	FC='ifort -reentrancy threaded'

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
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_DEPLOY"

	CC=pgcc
	CXX=pgc++
	FC=pgf90
	;;

"nvhpc")
	CONF_OPS="--with-device=psp"
	CONF_OPS="$CONF_OPS $CONF_OPS_DEPLOY"

	CC=nvc
	CXX=nvc++
	FC=nvfortran
	;;
"none")
	# No build
	;;
"user")
	CONF_OPS=
	;;
"ch"[[34]]*)
	# Explicitly set the UCX netwomod
	if test "x${CONFSET}" = "xch4"; then
		CONFSET="ch4:ucx"
	fi

	CONF_OPS="--with-device=${CONFSET}"
	MODULEFILE_CONFSET="mpich"

	CONF_OPS="$CONF_OPS $CONF_OPS_DEPLOY"

	CC=gcc
	CXX=g++
	FC=gfortran

	ENABLE_HYDRA="true"
	CHANNEL_DEVICE=`echo ${CONFSET} | cut -c 1-2`
	;;
*)
	echo "${CONF_ECHO_PREFIX}Sorry. Unknown confset '${CONFSET}'."
	exit 1
	;;
esac

# Additional psp-specific options
if ! echo ${CONFSET} | grep -Eq "ch[34]"; then
	CONF_OPS="$CONF_OPS --with-psp-confset=$CONFSET"  # Make confset available to subconfigure.m4
	CONF_OPS="$CONF_OPS --with-datatype-engine=yaksa" # 'yaksa' is the default datatype enginge
fi

test -n "$CC"  && CONF_OPS="$CONF_OPS 'CC=$CC'"
test -n "$CXX" && CONF_OPS="$CONF_OPS 'CXX=$CXX'"
test -n "$FC"  && CONF_OPS="$CONF_OPS 'FC=$FC'"
test -n "$F77" && CONF_OPS="$CONF_OPS 'F77=$F77'"
test -n "$AR"  && CONF_OPS="$CONF_OPS 'AR=$AR'"
test -n "$RANLIB" && CONF_OPS="$CONF_OPS 'RANLIB=$RANLIB'"
test -n "$FFLAGS" && CONF_OPS="$CONF_OPS 'FFLAGS=$FFLAGS'"

AC_ARG_WITH([pscom-builtin], [
  --with-pscom-builtin[=list]
			  Define list of pscom plugins that shall be built-in
			  Default: openib,psm2 / Needs pscom source code fusion
],[
	PSCOM_BUILTIN_PLUGINS="$withval"
])

CONF_ECHO_ALLIN=""

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
		CONF_ECHO_ALLIN="${CONF_ECHO_ALLIN:+${CONF_ECHO_ALLIN}\n}"
		CONF_ECHO_ALLIN="${CONF_ECHO_ALLIN}Including pscom version '${PSCOM_VC_VERSION}'."
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

	case $CONFSET in
		"default"|"gcc"|"intel"|"icc") ;;
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


AC_ARG_WITH([threading], [
  --with-threading        Enable threading 'multiple' in device psp
],[
	if test "x${withval}" = "xyes"; then
		export PSP_THREADING=multiple
		MODULEFILE_CONFSET="$MODULEFILE_CONFSET${MODULEFILE_CONFSET:+-}mt"
	fi
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

AC_ARG_WITH([session-statistics], [
  --with-session-statistics
			  Enable the collection of statistical information
],[
	if test "x${withval}" != "xno"; then
		ENABLE_SESSION_STATISTICS="true"
		CONF_OPS="$CONF_OPS --enable-psp-session-statistics"
	fi
])

AC_ARG_WITH([topology-awareness], [
  --with-topology-awareness
			  Enable topology/hierarchy awareness for collectives
],[
	if test "x${withval}" != "xno"; then
		ENABLE_TOPOLOGY_AWARENESS="true"
		CONF_OPS="$CONF_OPS --enable-psp-topology-awareness"
	else
		export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DMPIR_CONTEXT_SUBCOMM_WIDTH=0"
		ENABLE_TOPOLOGY_AWARENESS="false"
	fi
],[
	if test -n "$WITH_HCOLL_DIR" ; then
		ENABLE_TOPOLOGY_AWARENESS="hcoll"
		CONF_OPS="$CONF_OPS --enable-psp-topology-awareness"
	else
		if test -z "${CHANNEL_DEVICE}" ; then
			export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DMPIR_CONTEXT_SUBCOMM_WIDTH=0"
		fi
	fi
])

AC_SUBST([ENABLE_TOPOLOGY_AWARENESS])
if test -z "${CHANNEL_DEVICE}" ; then
   export MPICHLIB_CFLAGS="${MPICHLIB_CFLAGS} -DMPIR_CONTEXT_DYNAMIC_PROC_WIDTH=0"
fi

AC_ARG_WITH([pmix], [
  --with-pmix
			  Use PMIx to talk to the process manager
],,[with_pmix=no])

if test "x${with_pmix}" != "xno"; then
	# Determine PMIx home
	AS_IF(	[test "x${with_pmix}" = "xyes"],
		[PMIX_HOME="${PMIX_HOME-"/usr"}"],
		[PMIX_HOME="${with_pmix}"]
	)

	# Check if PMIX_HOME is a valid directory
	AC_MSG_CHECKING([if ${PMIX_HOME} is a valid directory])
	AS_IF(	[test -d "${PMIX_HOME}"],
		[
			AC_MSG_RESULT(yes)
			have_pmix=yes
		],
		[
			AC_MSG_RESULT(no)
			have_pmix=no
		]
	)

	# PMIX_HOME could be found
	AS_IF(
		[test "x${have_pmix}" = "xyes" -a "x${ENABLE_HYDRA}" = "xtrue"],
		[AC_MSG_ERROR([PMIx support and Hydra are mutually exclusive.])],
		[test "x${have_pmix}" = "xyes"],
		[
			ENABLE_PMIX="true"
			CONF_OPS="$CONF_OPS --with-pmix=${PMIX_HOME}"
		],
		[
			AC_MSG_ERROR([PMIx support is requested but PMIx home directory cannot be found])
		]
	)
else
	ENABLE_PMIX="false"
	CONF_OPS="$CONF_OPS --without-pmix"
fi


AC_ARG_WITH([cuda], [
  --with-cuda
			  Enable CUDA awareness within ParaStationMPI
],,[with_cuda=no])

if test "x${with_cuda}" != "xno"; then
	# Determine CUDA_HOME directory
	AS_IF(	[test "x${with_cuda}" = "xyes"],
		[CUDA_HOME="${CUDA_HOME-"/usr/local/cuda/"}"],
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

	# Check whether nvcc works (required for yaksa)
	AC_MSG_CHECKING([if nvcc works])
	cat>conftest.cu<<EOF
    __global__ void foo(int x) {}
EOF
	${CUDA_HOME}/bin/nvcc -c conftest.cu 2> /dev/null
	AS_IF(	[test "x$?" == "x0"],
		[
			AC_MSG_RESULT(yes)
			have_nvcc=yes
		],
		[
			AC_MSG_RESULT(no)
			have_nvcc=no
		]
	)

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

echo "${CONF_ECHO_PREFIX}Using confset='${CONFSET}'."
CONF_ECHO_PREP_PHRASE="${CONF_ECHO_PREFIX}Preparing mpich/psp with "
if test "$PSP_THREADING" = "multiple" ; then
   echo "${CONF_ECHO_PREP_PHRASE}MPI_THREAD_MULTIPLE support."
fi
if test "$ENABLE_HYDRA" = "true" ; then
   echo "${CONF_ECHO_PREP_PHRASE}Hydra's mpiexec support."
fi
if test "$ENABLE_PMIX" = "true" ; then
   echo "${CONF_ECHO_PREP_PHRASE}PMIx support."
else
   echo "${CONF_ECHO_PREP_PHRASE}PMI support."
fi
if test "$ENABLE_SESSION_STATISTICS" = "true" ; then
   echo "${CONF_ECHO_PREP_PHRASE}support for statistics."
fi
if test "$ENABLE_TOPOLOGY_AWARENESS" = "true" ; then
   echo "${CONF_ECHO_PREP_PHRASE}topology awareness."
fi
if test "$PSCOM_ALLIN" = "true" ; then
   echo "${CONF_ECHO_PREP_PHRASE}pscom included."
   if test -n "${CONF_ECHO_ALLIN}" ; then
      CONF_ECHO_ALLIN=`echo -en ${CONF_ECHO_ALLIN} | sed -z "s/\n/\n${CONF_ECHO_PREFIX}/g;s/,$/\n/"`
      echo -e "${CONF_ECHO_PREFIX}${CONF_ECHO_ALLIN}"
   fi
fi
if test "$CUDA_SUPPORT" = "true" ; then
   echo "${CONF_ECHO_PREP_PHRASE}CUDA support."
fi
if test -n "$WITH_HCOLL_DIR" ; then
   if test "$ENABLE_TOPOLOGY_AWARENESS" = "hcoll" ; then
	echo "${CONF_ECHO_PREP_PHRASE}topology awareness by HCOLL support."
   else
	echo "${CONF_ECHO_PREP_PHRASE}HCOLL support."
   fi
fi
sleep 1

AC_SUBST([CONFSET])
AC_SUBST([CUDA_SUPPORT])
AC_SUBST([MODULEFILE])
AC_SUBST([EXTRA_VERSION_INFO])
AC_SUBST([CUDA_HOME])
AC_SUBST([MPICCCONFDIR])

# ToDo: directly  use PSCOM_LDFLAGS from mpich2/src/mpid/psp/subconfigure.m4
PSCOM_LDFLAGS="${PSCOM_LDFLAGS-"-L/opt/parastation/lib64"}"
AC_SUBST([PSCOM_LDFLAGS])

PSCOM_LIBRARY="${PSCOM_LIBRARY-"-lpscom"}"
AC_SUBST([PSCOM_LIBRARY])

AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([scripts/generate_modulefile.sh], [chmod +x scripts/generate_modulefile.sh])
AC_CONFIG_FILES([scripts/generate_mpicc-conf.sh], [chmod +x scripts/generate_mpicc-conf.sh])

AC_OUTPUT

if test "x$DISABLE_HWLOC" = "xtrue" ; then
   CONF_OPS="$CONF_OPS --with-hwloc-prefix='no'"
   CONF_OPS="$CONF_OPS --with-hydra-topolib='no'"
else
   if test -n "${WITH_HWLOC_DIR}" ; then
      CONF_OPS="$CONF_OPS --with-hwloc-prefix=${WITH_HWLOC_DIR}"
      export CPATH=$CPATH:${WITH_HWLOC_DIR}/include
      export LIBRARY_PATH=$LIBRARY_PATH:${WITH_HWLOC_DIR}/lib:${WITH_HWLOC_DIR}/lib64
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${WITH_HWLOC_DIR}/lib:${WITH_HWLOC_DIR}/lib64
      export MPICHLIB_CFLAGS="$MPICHLIB_CFLAGS -I${WITH_HWLOC_DIR}/include"
      export MPICHLIB_CFLAGS="$MPICHLIB_CFLAGS -L${WITH_HWLOC_DIR}/lib"
      export MPICHLIB_CFLAGS="$MPICHLIB_CFLAGS -L${WITH_HWLOC_DIR}/lib64"
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

#
# Local Variables:
#  compile-command: "module load automake-1.15 && autoreconf -vif"
# End:
