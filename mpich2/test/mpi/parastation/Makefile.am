# -*- Mode: Makefile; -*-
# vim: set ft=automake :
#
# (C) 2011 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.
#

include $(top_srcdir)/Makefile.mtest

EXTRA_DIST = testlist

mprobe_LDADD = -lpthread

## for all programs that are just built from the single corresponding source
## file, we don't need per-target _SOURCES rules, automake will infer them
## correctly
noinst_PROGRAMS =  \
    mprobe locknull \
    win_sync_err_flush win_sync_err_flushall win_sync_err_lockall_pt win_sync_err_lockall_at win_sync_err_unlockall \
    win_flavor_err_shm win_flavor_err_dyn \
    mpi_spawn_isend \
    numcontexts2k numcontexts4k numcontexts8k numcontexts16k \
    get-acc-loc
