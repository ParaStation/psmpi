# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

prefixdir = ${prefix}

zm_headers = \
	include/common/zm_common.h \
	include/queue/zm_queue_types.h \
	include/queue/zm_glqueue.h \
	include/queue/zm_swpqueue.h \
	include/queue/zm_faqueue.h \
	include/queue/zm_mpbqueue.h \
	include/queue/zm_msqueue.h


if ZM_HAVE_HWLOC
zm_headers += \
	include/lock/zm_lock.h \
	include/lock/zm_lock_types.h \
	include/lock/zm_ticket.h \
	include/lock/zm_mcs.h \
	include/lock/zm_mmcs.h \
	include/lock/zm_tlp.h \
	include/lock/zm_mcsp.h \
	include/lock/zm_hmcs.h \
	include/lock/zm_hmpr.h \
	include/cond/zm_cond.h \
	include/cond/zm_cond_types.h \
	include/cond/zm_ccond.h \
	include/cond/zm_scount.h \
	include/cond/zm_wskip.h
endif

noinst_HEADERS = \
	include/zm_config.h \
	include/mem/zm_hzdptr.h \
	include/list/zm_sdlist.h

if ZM_EMBEDDED_MODE
noinst_HEADERS += ${zm_headers}
else
nobase_prefix_HEADERS = ${zm_headers}
endif
