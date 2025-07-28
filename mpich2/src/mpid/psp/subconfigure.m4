[#] start of __file__

[#] dependencies
[#] ======================================================================
# todo: check whether we need all of them
dnl MPICH_SUBCFG_BEFORE=src/mpid/common/sched
dnl MPICH_SUBCFG_BEFORE=src/mpid/common/datatype
dnl MPICH_SUBCFG_BEFORE=src/mpid/common/thread

[#] prereq (formerly mpich2prereq, setup_device, etc.)
[#] ======================================================================
AC_DEFUN([PAC_SUBCFG_PREREQ_]PAC_SUBCFG_AUTO_SUFFIX,[

AM_CONDITIONAL([BUILD_PSP],[test "$device_name" = "psp"])
AM_COND_IF([BUILD_PSP],[
AC_MSG_NOTICE([RUNNING PREREQ FOR PSP DEVICE])

AC_ARG_WITH(psp-confset,
	AC_HELP_STRING(
		[--with-psp-confset=confset],
		[Define the configuration set to be used for the PSP device]
		),,with_psp_confset="default")
AC_DEFINE_UNQUOTED([MPIDI_PSP_CONFSET], ["$with_psp_confset"], [The configuration set of the PSP layer])

AC_ARG_WITH(psp-pscom,
	AC_HELP_STRING(
		[--with-psp-pscom=path],
		[Define the path to the pscom installation to be used]
		),,with_psp_pscom="yes")

# maximum length of a processor name, as used by
# MPI_GET_PROCESSOR_NAME
MPID_MAX_PROCESSOR_NAME=1024

# maximum length of an error string
MPID_MAX_ERROR_STRING=1024

#
# MPID_NO_SPAWN=yes
if test "$PSP_THREADING" != "multiple" ; then
    MPID_MAX_THREAD_LEVEL=MPI_THREAD_SERIALIZED
    echo "        mpid/psp : set MPID_MAX_THREAD_LEVEL=MPI_THREAD_SERIALIZED"
else
    MPID_MAX_THREAD_LEVEL=MPI_THREAD_MULTIPLE
    echo "        mpid/psp : set MPID_MAX_THREAD_LEVEL=MPI_THREAD_MULTIPLE"
fi

# Evaluate argument of '--with-pscom' option:
AS_IF([test "x$with_psp_pscom" != "xno"],[
	AS_IF([test "x$with_psp_pscom" != "xyes"],[
		# build with pscom lib in specified path
		PSCOM_CPPFLAGS="-I${with_psp_pscom}/include"
		PSCOM_LDFLAGS="-L${with_psp_pscom}/lib64"
		PSCOM_LIBRARY="${PSCOM_LIBRARY-"-lpscom"}"
		PSCOM_ALLIN_LIBS=""
	], [
		AS_IF([test "x$PSCOM_ALLIN" != "xtrue"],[
			# build with pscom lib in default path
			PSCOM_CPPFLAGS="${PSCOM_CPPFLAGS-"-I/opt/parastation/include"}"
			PSCOM_LDFLAGS="${PSCOM_LDFLAGS-"-L/opt/parastation/lib64"}"
			PSCOM_LIBRARY="${PSCOM_LIBRARY-"-lpscom"}"
			PSCOM_ALLIN_LIBS=""
		], [
			# build pscom all-in
			PSCOM_LDFLAGS=""
			for lib in ${PSCOM_ALLIN_LIBS} ; do
				if [[ "${lib}" == "-lrrcomm" ]] ; then
					PSCOM_LDFLAGS="-L/opt/parastation/lib -L/opt/parastation/lib64" # rrcomm static library path
					echo "        mpid/psp : set PSCOM_LDFLAGS to ${PSCOM_LDFLAGS} for rrcomm library"
					break
				fi
			done
			PSCOM_CPPFLAGS=""
			PSCOM_LIBRARY=""
			PSCOM_ALLIN_LIBS="${PSCOM_ALLIN_LIBS} -lpthread -ldl"
			echo "        mpid/psp : set PSCOM_ALLIN_LIBS to ${PSCOM_ALLIN_LIBS}"
		])
	])
],[
	AS_IF([test "x$PSCOM_ALLIN" != "xtrue"],[
		AC_MSG_ERROR(["A pscom installation is required for building ParaStation MPI using the psp device. Abort!"])
	])
])

AC_ARG_VAR([PSCOM_CPPFLAGS], [C preprocessor flags for PSCOM headers
			    (default: "-I/opt/parastation/include")])
AC_ARG_VAR([PSCOM_LDFLAGS], [linker flags for PSCOM libraries
			    (default: "-L/opt/parastation/lib64")])
AC_ARG_VAR([PSCOM_LIBRARY], [file name for PSCOM library
			    (default: "-lpscom")])
AC_ARG_VAR([PSCOM_RUNPATH], [RUNPATH to be added to libmpi.so
			    (default: "-Wl,-rpath=/opt/parastation/lib64")])

# Add 'RUNPATH' for pscom to 'libpsmpi.so' for non-allin build:
if test "x$PSCOM_ALLIN" != "xtrue" ; then
	PSCOM_RUNPATH="${PSCOM_RUNPATH-"${PSCOM_LDFLAGS//-L/-Wl,-rpath=}"}"
	case "$PSCOM_RUNPATH" in '/'*) PSCOM_RUNPATH="-Wl,-rpath=${PSCOM_RUNPATH}" ;; esac
	PAC_APPEND_FLAG([${PSCOM_RUNPATH}],[PSP_LDFLAGS])
fi

AC_SUBST([PSCOM_CPPFLAGS])
AC_SUBST([PSCOM_LDFLAGS])
AC_SUBST([PSCOM_LIBRARY])
AC_SUBST([PSCOM_ALLIN_LIBS])

AC_ARG_VAR([PSP_CPPFLAGS], [C preprocessor flags for PSP macros])
AC_ARG_VAR([PSP_LDFLAGS], [additional linker flags for the PSP device])
AC_ARG_VAR([PSP_LIBS], [additional libraries for the PSP device])

AC_SUBST([PSP_CPPFLAGS])
AC_SUBST([PSP_LDFLAGS])
AC_SUBST([PSP_LIBS])

# Statistics
AC_ARG_ENABLE(psp-statistics,
    AC_HELP_STRING(
        [--enable-psp-statistics],
        [Enable statistics collection for the PSP device
    ]),,enable_psp_statistics=no)
if test "$enable_psp_statistics" = "yes" ; then
   AC_DEFINE([MPIDI_PSP_WITH_STATISTICS], [], [Define to enable statistics collection by PSP device])
fi

# Topology awareness
AC_ARG_ENABLE(psp-msa-awareness,
    AC_HELP_STRING(
        [--enable-psp-msa-awareness],
        [Enable topology awareness for the PSP device
    ]),,enable_psp_msa_awareness=no)
PSP_MSA_AWARENESS=0
if test "$enable_psp_msa_awareness" = "yes" ; then
   PSP_MSA_AWARENESS=1
   AC_DEFINE([MPIDI_PSP_WITH_MSA_AWARENESS], [], [Define to enable topology awareness in PSP device])
fi
AC_SUBST([PSP_MSA_AWARENESS])

# CUDA support
AC_ARG_ENABLE(psp-cuda-awareness,
    AC_HELP_STRING(
        [--enable-psp-cuda-awareness],
        [Enable CUDA awareness for the PSP device
    ]),,enable_psp_cuda_awareness=no)
PSP_CUDA_AWARE_SUPPORT=0
if test "$enable_psp_cuda_awareness" = "yes" ; then

	# add PSCOM_CPPFLAGS to CPPFLAGS and backup
	save_CPPFLAGS="$CPPFLAGS"
	CPPFLAGS="$CPPFLAGS $PSCOM_CPPFLAGS"

	# check if we try to build against non-CUDA-aware pscom
	AC_MSG_CHECKING(if pscom is CUDA-aware)
	AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
#define PSCOM_CUDA_AWARENESS
#include <pscom.h>
#ifndef PSCOM_CUDA_AWARENESS_SUPPORT
# error macro not defined
#endif]])],
		[AC_MSG_RESULT(yes)
		pscom_is_cuda_aware=yes
	],
		[AC_MSG_RESULT(no)
		pscom_is_cuda_aware=no
	])

	# restore CPPFLAGS
	CPPFLAGS="$save_CPPFLAGS"

	# Check whether pscom is present and provides pscom_memcpy
	AS_IF([test "x$PSCOM_ALLIN"  != "xtrue"],[
		# add '--allow-shlib-undefined' to the LDFLAGS since we only want to
		# check whether 'pscom_memcpy()' is present
		save_LDFLAGS="$LDFLAGS"
		LDFLAGS="$LDFLAGS -Wl,--allow-shlib-undefined"

		AX_CHECK_LIB([PSCOM],
			[pscom.h],
			[pscom],
			[pscom_memcpy],
			have_pscom_memcpy=yes)

		# restore LDFLAGS
		LDFLAGS="$save_LDFLAGS"
	])

	AS_IF([test "$pscom_is_cuda_aware" = "no" ],[
		AC_MSG_ERROR([The pscom library is missing CUDA awareness. Abort!])
	],[
		AS_IF([test "x$PSCOM_ALLIN"  != "xtrue" -a "$have_pscom_memcpy" != "yes" ],[
			AC_MSG_ERROR([The pscom library is lacking the pscom_memcpy() symbol. Please re-compile the pscom with CUDA support. Abort!])
		],[
			PSP_CUDA_AWARE_SUPPORT=1
			AC_DEFINE([MPIDI_PSP_WITH_CUDA_AWARENESS], [], [Define to enable GPU memory awareness in PSP device if CUDA is found])
			PAC_APPEND_FLAG([-DMPIR_USE_DEVICE_MEMCPY], [CPPFLAGS])
		])
	])
else
	# Check whether pscom is present for non-ALLIN builds
	AS_IF([test "x$PSCOM_ALLIN"  != "xtrue"],[
		AX_CHECK_LIB([PSCOM],
	                     [pscom.h],
	                     [pscom],
	                     ,
	                     have_pscom=yes,
	                     have_pscom=no)

		AS_IF([test "$have_pscom" != "yes" ],[
			AC_MSG_ERROR([Could not find the pscom library. Abort!])
		])
	])
fi
AC_SUBST([PSP_CUDA_AWARE_SUPPORT])

# Determine PS-MPI version string
PSP_VC_VERSION=$(${main_top_srcdir}/../scripts/vcversion -r ${main_top_srcdir}/.. -n)
AC_DEFINE_UNQUOTED([MPIDI_PSP_VC_VERSION], ["$PSP_VC_VERSION"], [Version string for debugging purpose])

# Determine PSCOM version string
AC_MSG_CHECKING([pscom version])
PSP_PSCOM_VERSION="(unknown)"
AS_IF([test "x$PSCOM_ALLIN" = "xtrue"],[
	PSP_PSCOM_VERSION="$PSCOM_VC_VERSION (allin)"
],[
	AS_IF([test "x$with_psp_pscom" != "xno" -a "x$with_psp_pscom" != "xyes"],
		[PSCOM_INFO_BIN="${with_psp_pscom}/bin/pscom_info"],
		[PSCOM_INFO_BIN="$(which pscom_info)"])
	AS_IF([test -n ${PSCOM_INFO_BIN}],
	        [[PSP_PSCOM_VERSION=$(${PSCOM_INFO_BIN} -v |& sed -En 's/.*(([0-9]+)\.([0-9]+)\.([0-9]+)-.*)\)>/\1/p')]])
])
AC_MSG_RESULT([$PSP_PSCOM_VERSION])
AC_DEFINE_UNQUOTED([MPIDI_PSP_PSCOM_VERSION],["$PSP_PSCOM_VERSION"],[Version string of pscom])

AC_CONFIG_FILES([
src/mpid/psp/include/mpi-ext.h
src/mpid/psp/include/mpid_cuda_aware.h
])


# todo: check whether we need all of them
build_mpid_common_sched=yes
build_mpid_common_datatype=yes
build_mpid_common_thread=yes

])dnl end AM_COND_IF([BUILD_PSP],...)
])dnl end SUBCFG_PREREQ

[#] main part
[#] ======================================================================
AC_DEFUN([PAC_SUBCFG_BODY_]PAC_SUBCFG_AUTO_SUFFIX,[
AM_COND_IF([BUILD_PSP],[
AC_MSG_NOTICE([RUNNING CONFIGURE FOR PSP DEVICE])

])dnl end AM_COND_IF([BUILD_PSP],...)
])dnl end SUBCFG_BODY

[#] end of __file__
