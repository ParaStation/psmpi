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

PSCOM_CPPFLAGS="${PSCOM_CPPFLAGS-"-I/opt/parastation/include"}"
PSCOM_LDFLAGS="${PSCOM_LDFLAGS-"-L/opt/parastation/lib64"}"
PSCOM_LIBRARY="${PSCOM_LIBRARY-"-lpscom"}"
PSCOM_RPATHLINK="${PSCOM_RPATHLINK-"${PSCOM_LDFLAGS//-L/-Wl,-rpath-link=}"}"
case "$PSCOM_RPATHLINK" in '/'*) PSCOM_RPATHLINK="-Wl,-rpath-link=${PSCOM_RPATHLINK}" ;; esac
AC_ARG_VAR([PSCOM_CPPFLAGS], [C preprocessor flags for PSCOM headers
			    (default: "-I/opt/parastation/include")])
AC_ARG_VAR([PSCOM_LDFLAGS], [linker flags for PSCOM libraries
			    (default: "-L/opt/parastation/lib64")])
AC_ARG_VAR([PSCOM_LIBRARY], [file name for PSCOM library
			    (default: "-lpscom")])
AC_ARG_VAR([PSCOM_RPATHLINK], [mpicc wrapper option for -Wl,-rpath-link
			    (default: "-Wl,-rpath-link=/opt/parastation/lib64")])

PAC_APPEND_FLAG([${PSCOM_RPATHLINK}],[WRAPPER_LDFLAGS])

if test "$PSCOM_ALLIN" = "true" ; then
    PSCOM_LDFLAGS=""
    PSCOM_CPPFLAGS=""
    PSCOM_LIBRARY=""
    PSCOM_ALLIN_LIBS="${PSCOM_ALLIN_LIBS} -lpthread -ldl"
    echo "        mpid/psp : set PSCOM_ALLIN_LIBS to ${PSCOM_ALLIN_LIBS}"
else
    PSCOM_ALLIN_LIBS=""
fi

AC_SUBST([PSCOM_CPPFLAGS])
AC_SUBST([PSCOM_LDFLAGS])
AC_SUBST([PSCOM_LIBRARY])
AC_SUBST([PSCOM_RPATHLINK])
AC_SUBST([PSCOM_ALLIN_LIBS])

AC_ARG_VAR([PSP_CPPFLAGS], [C preprocessor flags for PSP macros])
AC_ARG_VAR([PSP_LDFLAGS], [additional linker flags for the PSP device])
AC_ARG_VAR([PSP_LIBS], [additional libraries for the PSP device])

AC_SUBST([PSP_CPPFLAGS])
AC_SUBST([PSP_LDFLAGS])
AC_SUBST([PSP_LIBS])

# Session statistics
AC_ARG_ENABLE(psp-session-statistics,
    AC_HELP_STRING(
        [--enable-psp-session-statistics],
        [Enable session statistics for the PSP device
    ]),,enable_psp_session_statistics=no)
if test "$enable_psp_session_statistics" = "yes" ; then
   AC_DEFINE([MPIDI_PSP_WITH_SESSION_STATISTICS], [], [Define to enable session statistics by PSP device])
fi

# Topology awareness
AC_ARG_ENABLE(psp-topology-awareness,
    AC_HELP_STRING(
        [--enable-psp-topology-awareness],
        [Enable topology awareness for the PSP device
    ]),,enable_psp_topology_awareness=no)
PSP_TOPOLOGY_AWARENESS=0
if test "$enable_psp_topology_awareness" = "yes" ; then
   PSP_TOPOLOGY_AWARENESS=1
   AC_DEFINE([MPIDI_PSP_WITH_TOPOLOGY_AWARENESS], [], [Define to enable topology awareness in PSP device])
fi
AC_SUBST([PSP_TOPOLOGY_AWARENESS])

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
			AC_MSG_ERROR([The pscom library is lacking the pscom_memcpy() symbol. Please re-compile the pscom with "--enable-cuda". Abort!])
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
PSP_VC_VERSION=$(${master_top_srcdir}/../scripts/vcversion -r ${master_top_srcdir}/.. -n)
AC_DEFINE_UNQUOTED([MPIDI_PSP_VC_VERSION], ["$PSP_VC_VERSION"], [Version string for debugging purpose])

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
