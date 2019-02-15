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
# AX_CHECK_LIBRARY([PSCOM], [pscom.h], [pscom])
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

AC_SUBST([PSP_CPPFLAGS])

AC_ARG_ENABLE(psp-cuda-awareness,
    AC_HELP_STRING(
        [--enable-psp-cuda-awareness],
        [Enable CUDA-awareness for the PSP device
    ]),,enable_psp_cuda_awareness=no)

# CUDA support
pscom_is_cuda_aware=no
if test "$enable_psp_cuda_awareness" = "yes" ; then
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
	])

    AS_IF([test "$pscom_is_cuda_aware" = "no" ],[
		AC_MSG_WARN([!!!!!!! BUILDING AGAINST NON-CUDA-AWARE PSCOM !!!!!!!])
	],[
		PAC_APPEND_FLAG([-DMPIR_USE_DEVICE_MEMCPY], [CPPFLAGS])
	])
fi


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
