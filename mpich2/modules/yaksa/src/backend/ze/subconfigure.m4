##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##


##########################################################################
##### capture user arguments
##########################################################################

# --with=ze
PAC_SET_HEADER_LIB_PATH([ze])
if test x"${with_ze}" != x ; then
    PAC_CHECK_HEADER_LIB([level_zero/ze_api.h],[ze_loader],[zeCommandQueueCreate],[have_ze=yes],[have_ze=no])
    AC_MSG_CHECKING([whether ocloc is installed])
    if ! command -v ocloc &> /dev/null; then
        AC_MSG_ERROR([ocloc not found; either install it or disable ze support])
    else
        AC_MSG_RESULT([yes])
    fi
fi
if test "${have_ze}" = "yes" ; then
    AC_DEFINE([HAVE_ZE],[1],[Define is ZE is available])
fi
AM_CONDITIONAL([BUILD_ZE_BACKEND], [test x${have_ze} = xyes])
AM_CONDITIONAL([BUILD_ZE_TESTS], [test x${have_ze} = xyes])


# --with-ze-p2p
AC_ARG_ENABLE([ze-p2p],AS_HELP_STRING([--enable-ze-p2p={yes|no|cliques}],[controls ZE P2P capability]),,
              [enable_ze_p2p=yes])
if test "${have_ze}" = "yes" ; then
    if test "${enable_ze_p2p}" = "yes" ; then
        AC_DEFINE([ZE_P2P],[ZE_P2P_ENABLED],[Define if ZE P2P is enabled])
    elif test "${enable_ze_p2p}" = "cliques" ; then
        AC_DEFINE([ZE_P2P],[ZE_P2P_CLIQUES],[Define if ZE P2P is enabled in clique mode])
    else
        AC_DEFINE([ZE_P2P],[ZE_P2P_DISABLED],[Define if ZE P2P is disabled])
    fi
fi

# --ze-native=[skl|dg1|ats]
AC_ARG_ENABLE([ze-native],AS_HELP_STRING([--enable-ze-native=device],[compile GPU kernel to native binary]),,
              [enable_ze_native=])
if test "${have_ze}" = "yes" -a x"${enable_ze_native}" != x; then
    AC_MSG_CHECKING([whether ocloc works])
    cat>conftest.cl<<EOF
    __kernel void foo(int x) {}
EOF
    ocloc compile -file conftest.cl -device ${enable_ze_native} -options "-cl-std=CL2.0" > /dev/null 2>&1
    if test "$?" = "0" ; then
        AC_MSG_RESULT([yes])
    else
        AC_MSG_RESULT([no])
        enable_ze_native=
        AC_MSG_ERROR([ocloc compiler is not compatible with ze_native])
    fi
    rm -f conftest.*
fi
if test x"${enable_ze_native}" != x; then
    AC_DEFINE(ZE_NATIVE, 1, [Compile kernels to binary])
else
    AC_DEFINE(ZE_NATIVE, 0, [No native format])
fi
AC_SUBST(enable_ze_native)
AM_CONDITIONAL([BUILD_ZE_NATIVE],[test x"$enable_ze_native" != x])

##########################################################################
##### analyze the user arguments and setup internal infrastructure
##########################################################################

if test "${have_ze}" = "yes" ; then
    supported_backends="${supported_backends},ze"
fi
