# -*- Mode: Makefile; -*-
# vim: set ft=automake :
#
# (C) 2011 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.
#

include $(top_srcdir)/Makefile_single.mtest

EXTRA_DIST = testlist

bin_PROGRAMS = mprobe

mprobe_SOURCES = mprobe.c
mprobe_LDADD = -lpthread

## for all programs that are just built from the single corresponding source
## file, we don't need per-target _SOURCES rules, automake will infer them
## correctly
noinst_PROGRAMS =               \
	locknull                \
	get_acc_loc             \
	win_sync_err_flush      \
	win_sync_err_flushall   \
	win_sync_err_lockall_pt \
	win_sync_err_lockall_at \
	win_sync_err_unlockall  \
	win_flavor_err_shm      \
	win_flavor_err_dyn      \
	win_shm_dup_comm        \
	win_shm_recursive       \
	win_shm_multiple        \
	win_shm_large           \
	win_test_ordering       \
	split_type_integrity    \
	spawn_isend             \
	spawn_anysrc            \
	spawn_ssend             \
	spawn_rma               \
	spawn_univ_split        \
	spawn_univ_comm         \
	spawn_univ_anysrc       \
	spawn_univ_anysrc_order \
	spawn_univ_anysrc_order_mixed_comms \
	spawn_univ_anysrc_cancel\
	spawn_univ_discon       \
	num_contexts_2k         \
	num_contexts_4k         \
	num_contexts_8k         \
	num_contexts_16k        \
	large_msg_typed         \
	large_msg_builtin       \
	large_msg_noncontig     \
	large_msg_coll_bcast    \
	large_msg_coll_ibcast   \
	large_msg_coll_allred   \
	large_msg_coll_iallred  \
	large_msg_coll_reduce   \
	large_msg_coll_ireduce  \
	large_msg_coll_scatter  \
	large_msg_coll_iscatter \
	large_msg_coll_gather   \
	large_msg_coll_igather  \
	large_msg_coll_allgather  \
	large_msg_coll_iallgather \
	large_msg_coll_alltoall   \
	large_msg_coll_ialltoall  \
	large_msg_coll_alltoall_inplace  \
	large_msg_coll_ialltoall_inplace \
	intercomm_anysrc        \
	intercomm_stress        \
	large_acc_flush_local   \
	acc_pairtype            \
	acc_vecsum              \
	get_acc_pairtype        \
	get_acc_shm_pairtype    \
	putfence1               \
	getfence1               \
	msa_split_type          \
	msa_info_env
