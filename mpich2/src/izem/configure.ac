# -*- Mode: Autoconf; -*-
#
# See COPYRIGHT in top-level directory.
#

AC_PREREQ(2.63)

m4_include([maint/version.m4])
AC_INIT([izem],ZM_VERSION_m4)

ZM_VERSION=ZM_VERSION_m4
AC_SUBST([ZM_VERSION])

libzm_so_version="libzm_so_version_m4"
AC_SUBST([libzm_so_version])

# Produce a numeric version assuming the following format:
# Version: [MAJ].[MIN].[REV][EXT][EXT_NUMBER]
# Example: 1.0.7rc1 has
#          MAJ = 1
#          MIN = 0
#          REV = 7
#          EXT = rc
#          EXT_NUMBER = 1
#
# Converting to numeric version will convert EXT to a format number:
#          ALPHA (a) = 0
#          BETA (b)  = 1
#          RC (rc)   = 2
#          PATCH (p) = 3
# Regular releases are treated as patch 0
#
# Numeric version will have 1 digit for MAJ, 2 digits for MIN,
# 2 digits for REV, 1 digit for EXT and 2 digits for EXT_NUMBER.
changequote(<<,>>)
V1=`expr $ZM_VERSION : '\([0-9]*\)\.[0-9]*\.*[0-9]*[a-zA-Z]*[0-9]*'`
V2=`expr $ZM_VERSION : '[0-9]*\.\([0-9]*\)\.*[0-9]*[a-zA-Z]*[0-9]*'`
V3=`expr $ZM_VERSION : '[0-9]*\.[0-9]*\.*\([0-9]*\)[a-zA-Z]*[0-9]*'`
V4=`expr $ZM_VERSION : '[0-9]*\.[0-9]*\.*[0-9]*\([a-zA-Z]*\)[0-9]*'`
V5=`expr $ZM_VERSION : '[0-9]*\.[0-9]*\.*[0-9]*[a-zA-Z]*\([0-9]*\)'`
changequote([,])

if test "$V2" -le 9 ; then V2="0$V2" ; fi
if test "$V3" = "" ; then V3="0"; fi
if test "$V3" -le 9 ; then V3="0$V3" ; fi
if test "$V4" = "a" ; then
    V4=0
elif test "$V4" = "b" ; then
    V4=1
elif test "$V4" = "rc" ; then
    V4=2
elif test "$V4" = "" ; then
    V4=3
    V5=0
elif test "$V4" = "p" ; then
    V4=3
fi
if test "$V5" -le 9 ; then V5="0$V5" ; fi

ZM_NUMVERSION=`expr $V1$V2$V3$V4$V5 + 0`
AC_SUBST(ZM_NUMVERSION)

AC_CONFIG_AUX_DIR(m4)
AC_CONFIG_MACRO_DIR(m4)
AC_PROG_CC
AC_PROG_CXX
AC_HEADER_STDC
AC_PROG_CC_C99

if test "x$ac_cv_prog_cc_c99" = "xno"; then
    AC_MSG_ERROR([C99 or an earlier C standard required])
fi

dnl ----------------------------------------------------------------------------
dnl check the environment and the function availability
dnl ----------------------------------------------------------------------------
# compute canonical system types
AC_CANONICAL_HOST

dnl ----------------------------------------------------------------------------
dnl user option/feature selection
dnl ----------------------------------------------------------------------------
# embedding option
AC_ARG_ENABLE(embedded,
    AC_HELP_STRING([--enable-embedded], [Build Izem in embedded mode (default is no)]),
    [embedded=yes],
    [embedded=no])
AM_CONDITIONAL([ZM_EMBEDDED_MODE],[test "x${embedded}" = "xyes"])

# debug options
AC_ARG_ENABLE([debug],
[  --enable-debug@<:@=OPTS@:>@   control the level of debugging. "OPTS" is a list of
                          comma separated names below. Default is "none".
                           none - no debugging
                           yes  - add compiler flags, -g -O0 -Wall
                           log  - enable debug event logging
                           all  - all of the above choices],,
[enable_debug=none])

# strip off multiple options, separated by commas
save_IFS="$IFS"
IFS=","
for option in $enable_debug ; do
    case "$option" in
        no|none)
        ;;
        yes)
            debug_flags=yes
        ;;
        log)
            debug_log=yes
        ;;
        all)
            debug_flags=yes
            debug_log=yes
        ;;
        *)
            IFS=$save_IFS
            AC_MSG_WARN([Unknown value $option for enable-debug])
            IFS=","
        ;;
    esac
done
IFS="$save_IFS"

AS_IF([test "x$debug_flags" = "xyes"], [CFLAGS="$CFLAGS -g -O0 -Wall"])
AS_IF([test "x$debug_log" = "xyes"],
    [AC_DEFINE(ZM_CONFIG_USE_DEBUG_LOG, 1, [Define to enable debug logging])])

# Testing for atomic

AC_MSG_CHECKING([for gcc __atomic builtins (memory model aware)])
have_gcc_atomic_mem_aware_builtins=no
AC_TRY_LINK(
  [], [char l; unsigned long v; __atomic_test_and_set(&l, __ATOMIC_RELAXED); __atomic_store_n(&v, 1, __ATOMIC_RELEASE); __atomic_load_n(&v, __ATOMIC_ACQUIRE);],
  [have_gcc_atomic_mem_aware_builtins=yes],
)
if test "x$have_gcc_atomic_mem_aware_builtins" = "xyes"; then
  AC_DEFINE([HAVE_GCC_MEM_AWARE_ATOMIC_BUILTINS], [1], [Define if the gcc __atomic builtins (memory model aware) are available])
fi
AC_MSG_RESULT([$have_gcc_atomic_mem_aware_builtins])

AC_MSG_CHECKING([for C11 atomics])
have_c11_atomics=no
AC_TRY_LINK(
  [], [#include <stdatomic.h>
       atomic_char l;
       atomic_long v;
       atomic_intptr_t p;
       atomic_flag_test_and_set_explicit(&l, memory_order_relaxed);
       atomic_store_explicit(&v, 1, memory_order_release);
       atomic_load_explicit(&v, memory_order_acquire);
       atomic_store_explicit(&p, NULL, memory_order_release);
       intptr_t q = NULL;
       atomic_compare_exchange_strong_explicit(&p, &q, &l, memory_order_acq_rel, memory_order_acquire);
       atomic_compare_exchange_weak_explicit(&p, &q, &l, memory_order_acq_rel, memory_order_acquire);
],
  [have_c11_atomics=yes],
)
if test "x$have_c11_atomics" = "xyes"; then
  AC_DEFINE([HAVE_C11_ATOMICS], [1], [Define if the C11 atomics are available])
fi
AC_MSG_RESULT([$have_c11_atomics])

AC_MSG_CHECKING([for gcc __sync atomics])
have_sync_atomics=no
AC_TRY_LINK(
  [], [char l = 'm'; unsigned long v; __sync_bool_compare_and_swap(&l, 'm', 'l'); __sync_synchronize ();  __sync_fetch_and_add(&v, 1);],
  [have_sync_atomics=yes],
)
if test "x$have_sync_atomics" = "xyes"; then
  AC_DEFINE([HAVE_SYNC_ATOMICS], [1], [Define if the sync atomics are available])
fi
AC_MSG_RESULT([$have_sync_atomics])


if test "x$have_c11_atomics" = "xyes"; then
  memory_model="C11"
elif test "x$have_gcc_atomic_mem_aware_builtins" = "xyes"; then
    memory_model="GCC_ATOM"
elif test "x$have_sync_atomics" = "xyes"; then
     memory_model="GCC_SYNC"
else
    AC_MSG_ERROR([No atomic/memory model found with this compiler])
fi

ZM_MEMORY_MODEL=ZM_MEMORY_MODEL_$memory_model
AC_SUBST(ZM_MEMORY_MODEL)

# Default lock interface

AC_ARG_WITH([lock_if],
[  --with-lock-if@<:@=LOCK_IF@:>@   define the default lock interface being
                          exposed by Izem (i.e., zm_lock_t and associated
                          routines. LOCK_IF represents the name of the
                          underlying lock to be used. Possible values are:
                          tkt  - Ticket
                          mcs  - MCS
                          mmcs - Memorizing MCS. It memorizes the local
                                 context of the lock holder. It allows
                                 reacquiring and releasing the lock without
                                 bringing a local context. It is useful for
                                 lock passing in nested scopes (e.g., MPICH)
                          hmcs - Multilevel Hierarchical MCS (HMCS)
                          tlp  - Generic Two-Level Priority
                          mcsp - Two-Level MCS lock
],,
[with_lock_if=tkt])

case "$with_lock_if" in
    tkt)
        ZM_LOCK_IF=ZM_TICKET_IF
    ;;
    mcs)
        ZM_LOCK_IF=ZM_MCS_IF
    ;;
    mmcs)
        ZM_LOCK_IF=ZM_MMCS_IF
    ;;
    hmcs)
        ZM_LOCK_IF=ZM_HMCS_IF
    ;;
    tlp)
        ZM_LOCK_IF=ZM_TLP_IF
    ;;
    mcsp)
        ZM_LOCK_IF=ZM_MCSP_IF
    ;;
    *)
        AC_MSG_WARN([Unknown value $with_lock_if for with-lock-if])
    ;;
esac

AC_SUBST(ZM_LOCK_IF)

# TLP lock components
AC_ARG_WITH([tlp_locks],
[  --with-tlp-locks@<:@=HIGH-LOW@:>@   define the high and low priority locks
                          in the TLP lock. HIGH and LOW represent the locks
                          used by the high priority and low priority threads,
                          repspectively,
                          tkt-tkt  - both are ticket locks (default)
                          mcs-mcs  - both are mcs locks
                          tkt-mcs  - HIGH is ticket LOW is MCS
                          mcs-tkt  - HIGH is MCS LOW is ticket
                          hmcs-hmcs  - both are HMCS locks
                          hmcs-mcs   - HIGH is HMCS and LOW is MCS
],,
[with_tlp_locks=tkt-tkt])

case "$with_tlp_locks" in
    tkt-tkt)
        ZM_TLP_HIGH_P=ZM_TICKET
        ZM_TLP_LOW_P=ZM_TICKET
    ;;
    mcs-mcs)
        ZM_TLP_HIGH_P=ZM_MCS
        ZM_TLP_LOW_P=ZM_MCS
    ;;
    tkt-mcs)
        ZM_TLP_HIGH_P=ZM_TICKET
        ZM_TLP_LOW_P=ZM_MCS
    ;;
    mcs-tkt)
        ZM_TLP_HIGH_P=ZM_MCS
        ZM_TLP_LOW_P=ZM_TICKET
    ;;
    hmcs-hmcs)
        ZM_TLP_HIGH_P=ZM_HMCS
        ZM_TLP_LOW_P=ZM_HMCS
    ;;
    hmcs-mcs)
        ZM_TLP_HIGH_P=ZM_HMCS
        ZM_TLP_LOW_P=ZM_MCS

    ;;
    *)
        AC_MSG_WARN([Unknown value $with_tlp_locks for with-tlp-locks])
    ;;
esac

AC_SUBST(ZM_TLP_HIGH_P)
AC_SUBST(ZM_TLP_LOW_P)

# Default cond var interface

AC_ARG_WITH([cond_if],
[  --with-cond-if@<:@=COND_IF@:>@   define the default condition variable
                          interface being exposed by Izem (i.e., zm_cond_t
                          and associated routines. COND_IF represents the
                          name of the underlying condition variable to be
                          used. Possible values are:
                          ccond  - CAS-based spinning
],,
[with_cond_if=ccond])

case "$with_cond_if" in
    ccond)
        ZM_COND_IF=ZM_CCOND_IF
    ;;
    *)
        AC_MSG_WARN([Unknown value $with_cond_if for with-cond-if])
    ;;
esac

AC_SUBST(ZM_COND_IF)

# Default queue interface

AC_ARG_WITH([queue_if],
[  --with-queue-if@<:@=QUEUE_IF@:>@   define the default queue interface being
                          exposed by Izem (i.e., zm_queue_t and associated
                          routines. QUEUE_IF represents the name of the
                          underlying queue to be used. Possible values are:
                          gl   - Global lock (both enq and deq compete for the same lock)
                          ms   - Michael and Scott''s nonblocking queue (PODC''96)
                          swp  - SWAP-based MPSC queue inspired by the MCS lock
                          fa   - FAA-based MPSC queue derived from WF (Yang, PPoPP''16)
                          runtime - Runtime selection through ZM_QUEUE_IF environment variable
],,
[with_queue_if=swp])

case "$with_queue_if" in
    gl)
        ZM_QUEUE_CONF=ZM_GLQUEUE_IF
    ;;
    ms)
        ZM_QUEUE_CONF=ZM_MSQUEUE_IF
    ;;
    swp)
        ZM_QUEUE_CONF=ZM_SWPQUEUE_IF
    ;;
    fa)
        ZM_QUEUE_CONF=ZM_FAQUEUE_IF
    ;;
    # mbpqueue has a different function signature, so won't be supported here
    runtime)
        ZM_QUEUE_CONF=ZM_RUNTIMEQUEUE_IF
    ;;
    *)
        AC_MSG_ERROR([Unknown value $with_queue_if for with-queue-if])
    ;;
esac

AC_SUBST(ZM_QUEUE_CONF)

AC_ARG_WITH([hwloc],
            [AS_HELP_STRING([--with-hwloc], [Set path to hwloc. Default: auto detect.])],
            [LIBS="$LIBS -L$with_hwloc/lib"
             CFLAGS="$CFLAGS -I$with_hwloc/include"],
            [])

AC_CHECK_LIB(hwloc,
             hwloc_topology_init,
             [],
             AC_MSG_WARN([Could not find hwloc. Set the path to hwloc using --with-hwloc]),
             [])

AM_CONDITIONAL([ZM_HAVE_HWLOC],[test "${with_hwloc+set}" = set])

AM_INIT_AUTOMAKE([-Wall -Wno-portability-recursive -Werror foreign 1.11.3 subdir-objects])

AM_SILENT_RULES([yes])
AM_PROG_AS
AM_PROG_AR

LT_INIT

AC_PROG_INSTALL

dnl ----------------------------------------------------------------------------
dnl config headers
dnl ----------------------------------------------------------------------------
AC_CONFIG_HEADERS([src/include/zm_config.h])
AH_TOP([
/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ZM_CONFIG_H_INCLUDED
#define ZM_CONFIG_H_INCLUDED
])
AH_BOTTOM([
#endif /* ZM_CONFIG_H_INCLUDED */
])
dnl ----------------------------------------------------------------------------

AC_CONFIG_FILES([Makefile
                 Doxyfile
                 maint/izem.pc
                 src/Makefile
                 src/include/common/zm_common.h
                 src/include/queue/zm_queue.h
                 test/Makefile
                 test/regres/Makefile
                 test/regres/list/Makefile
                 test/regres/queue/Makefile
                 test/perf/Makefile
                 test/perf/queue/Makefile
                 ])
if test "${with_hwloc+set}" = set; then
AC_CONFIG_FILES([src/include/lock/zm_lock.h
                 src/include/lock/zm_lock_types.h
                 src/include/cond/zm_cond.h
                 test/regres/lock/Makefile
                 test/regres/cond/Makefile
                 test/perf/lock/Makefile
                 ])
fi
AC_OUTPUT
