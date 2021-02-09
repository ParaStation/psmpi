##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

AM_CPPFLAGS += -I$(top_srcdir)/src/backend/ze/pup

libyaksa_la_SOURCES += \
	src/backend/ze/pup/yaksuri_zei_event.c \
	src/backend/ze/pup/yaksuri_zei_get_ptr_attr.c

include src/backend/ze/pup/Makefile.pup.mk
include src/backend/ze/pup/Makefile.populate_pupfns.mk

ze_native_TARGET = @enable_ze_native@

if BUILD_ZE_NATIVE

.cl.c:
	@echo "  GEN (native)  $@" ; \
	echo "ocloc compile -file $< -device $(ze_native_TARGET) -out_dir `dirname $@` -output_no_suffix -options \"-I $(top_srcdir)/src/backend/ze/include -cl-std=CL2.0\""; \
	ocloc compile -file $< -device $(ze_native_TARGET) -out_dir `dirname $@` -output_no_suffix -options "-I $(top_srcdir)/src/backend/ze/include -cl-std=CL2.0" && \
	mv $(@:.c=) $(@:.c=.bin) && /bin/rm -f $(@:.c=.gen); \
	$(top_srcdir)/src/backend/ze/pup/inline.py $(@:.c=.bin) $@ $(top_srcdir) 1

else

.cl.c:
	@echo " GEN (spirv)  $@"; \
	echo "ocloc compile -file $< -device skl -spv_only -out_dir `dirname $@` -output_no_suffix -options \"-I $(top_srcdir)/src/backend/ze/include -cl-std=CL2.0\""; \
	ocloc compile -file $< -device skl -spv_only -out_dir `dirname $@` -output_no_suffix -options "-I $(top_srcdir)/src/backend/ze/include -cl-std=CL2.0" && \
	/bin/rm -f $(@:.c=.gen); \
	$(top_srcdir)/src/backend/ze/pup/inline.py $< $@ $(top_srcdir) 0

endif
