#
# ParaStation
#
# Copyright (C) 2024-2026 ParTec AG, Munich
#
# This file may be distributed under the terms of the Q Public License
# as defined in the file LICENSE.QPL included in the packaging of this
# file.
 
include $(top_srcdir)/Makefile_single.mtest

EXTRA_DIST = testlist

noinst_PROGRAMS = spawn_anysrc \
    spawn_intercomm_stress \
    spawn_isend \
    spawn_psmgmt_info \
    spawn_rma \
    spawn_ssend \
    spawn_univ_anysrc_cancel \
    spawn_univ_anysrc_order_mixed_comms \
    spawn_univ_anysrc_order \
    spawn_univ_anysrc \
    spawn_univ_comm \
    spawn_univ_discon \
    spawn_univ_split \
    spawn_ex_world \
    spawn_ex_world_nb \
    spawn_ex_session \
    spawn_ex_session_nb

spawn_ex_world_SOURCES = spawn_ex.c
spawn_ex_world_CPPFLAGS = $(AM_CPPFLAGS)
spawn_ex_world_nb_SOURCES = spawn_ex.c
spawn_ex_world_nb_CPPFLAGS = -DNONBLOCKING $(AM_CPPFLAGS)
spawn_ex_session_SOURCES = spawn_ex.c
spawn_ex_session_CPPFLAGS = -DSESSION_INIT $(AM_CPPFLAGS)
spawn_ex_session_nb_SOURCES = spawn_ex.c
spawn_ex_session_nb_CPPFLAGS = -DSESSION_INIT -DNONBLOCKING $(AM_CPPFLAGS)
