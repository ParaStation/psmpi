[#] start of __file__

[#] dependencies
[#] ======================================================================
# todo: check whether we need all of them
dnl MPICH2_SUBCFG_BEFORE=src/mpid/common/sched
dnl MPICH2_SUBCFG_BEFORE=src/mpid/common/datatype
dnl MPICH2_SUBCFG_BEFORE=src/mpid/common/thread

[#] prereq (formerly mpich2prereq, setup_device, etc.)
[#] ======================================================================
AC_DEFUN([PAC_SUBCFG_PREREQ_]PAC_SUBCFG_AUTO_SUFFIX,[

AM_CONDITIONAL([BUILD_PSP],[test "$device_name" = "psp"])
AM_COND_IF([BUILD_PSP],[

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
# AX_CHECK_LIBRARY([PSCOM], [pscom.h], [pscom])
AC_ARG_VAR([PSCOM_CPPFLAGS], [C preprocessor flags for PSCOM headers])
AC_ARG_VAR([PSCOM_LDFLAGS], [linker flags for PSCOM libraries])

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
