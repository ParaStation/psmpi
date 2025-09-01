## -*- Mode: Makefile -*-

mpi_core_sources +=               src/mpid/psp/src/mpid_abort.c			\
                                  src/mpid/psp/src/mpid_cancel.c		\
                                  src/mpid/psp/src/mpid_coll.c    		\
                                  src/mpid/psp/src/mpid_comm.c    		\
                                  src/mpid/psp/src/mpid_datatype.c		\
                                  src/mpid/psp/src/mpid_debug.c			\
                                  src/mpid/psp/src/mpid_finalize.c		\
                                  src/mpid/psp/src/mpid_get_processor_name.c	\
                                  src/mpid/psp/src/mpid_init.c			\
                                  src/mpid/psp/src/mpid_irecv.c			\
                                  src/mpid/psp/src/mpid_isend.c			\
                                  src/mpid/psp/src/mpid_op.c			\
                                  src/mpid/psp/src/mpid_part.c			\
                                  src/mpid/psp/src/mpid_persistent.c		\
                                  src/mpid/psp/src/mpid_progress.c		\
                                  src/mpid/psp/src/mpid_spawn.c			\
                                  src/mpid/psp/src/mpid_pg.c			\
                                  src/mpid/psp/src/mpid_psp_connect.c   \
                                  src/mpid/psp/src/mpid_psp_ctrl_msgs.c		\
                                  src/mpid/psp/src/mpid_psp_datatype.c		\
                                  src/mpid/psp/src/mpid_psp_packed_msg_acc.c	\
                                  src/mpid/psp/src/mpid_psp_request.c		\
                                  src/mpid/psp/src/mpid_recv.c			\
                                  src/mpid/psp/src/mpid_rma_accumulate.c	\
                                  src/mpid/psp/src/mpid_rma.c			\
                                  src/mpid/psp/src/mpid_rma_get.c		\
                                  src/mpid/psp/src/mpid_rma_put.c		\
                                  src/mpid/psp/src/mpid_rma_sync.c		\
                                  src/mpid/psp/src/mpid_rma_callbacks.c		\
                                  src/mpid/psp/src/mpid_send.c			\
                                  src/mpid/psp/src/mpid_start.c			\
                                  src/mpid/psp/src/mpid_stream_enqueue.c	\
                                  src/mpid/psp/src/mpid_unresolved.c		\
                                  src/mpid/psp/src/mpid_vc.c                    \
                                  src/mpid/psp/src/mpid_win_info.c

external_libs += @PSCOM_LIBRARY@ @PSCOM_ALLIN_LIBS@ @PSP_LIBS@
external_ldflags += @PSCOM_LDFLAGS@ @PSP_LDFLAGS@
AM_CPPFLAGS += @PSCOM_CPPFLAGS@ @PSP_CPPFLAGS@
