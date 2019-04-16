## -*- Mode: Makefile -*-

mpi_core_sources +=               src/mpid/psp/src/mpid_abort.c			\
                                  src/mpid/psp/src/mpid_cancel.c		\
                                  src/mpid/psp/src/mpid_coll.c    		\
                                  src/mpid/psp/src/mpid_debug.c			\
                                  src/mpid/psp/src/mpid_finalize.c		\
                                  src/mpid/psp/src/mpid_get_processor_name.c	\
                                  src/mpid/psp/src/mpid_init.c			\
                                  src/mpid/psp/src/mpid_irecv.c			\
                                  src/mpid/psp/src/mpid_isend.c			\
                                  src/mpid/psp/src/mpid_persistent.c		\
                                  src/mpid/psp/src/mpid_progress.c		\
                                  src/mpid/psp/src/mpid_port.c			\
                                  src/mpid/psp/src/mpid_pg.c			\
                                  src/mpid/psp/src/mpid_psp_datatype.c		\
                                  src/mpid/psp/src/mpid_psp_packed_msg_acc.c	\
                                  src/mpid/psp/src/mpid_psp_request.c		\
                                  src/mpid/psp/src/mpid_recv.c			\
                                  src/mpid/psp/src/mpid_rma_accumulate.c	\
                                  src/mpid/psp/src/mpid_rma.c			\
                                  src/mpid/psp/src/mpid_rma_get.c		\
                                  src/mpid/psp/src/mpid_rma_put.c		\
                                  src/mpid/psp/src/mpid_rma_sync.c		\
                                  src/mpid/psp/src/mpid_send.c			\
                                  src/mpid/psp/src/mpid_unresolved.c		\
                                  src/mpid/psp/src/mpid_vc.c

external_libs += @PSCOM_LIBRARY@ @PSCOM_ALLIN_LIBS@
external_ldflags += @PSCOM_LDFLAGS@
AM_CPPFLAGS += @PSCOM_CPPFLAGS@ @PSP_CPPFLAGS@
