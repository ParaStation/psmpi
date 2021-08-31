##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

cuda_pack_testlists = $(top_srcdir)/test/pack/testlist.cuda.d-d-d.gen \
	$(top_srcdir)/test/pack/testlist.cuda.d-rh-d.gen \
	$(top_srcdir)/test/pack/testlist.cuda.d-urh-d.gen \
	$(top_srcdir)/test/pack/testlist.cuda.d-m-d.gen \
	$(top_srcdir)/test/pack/testlist.cuda.rh-d-rh.gen \
	$(top_srcdir)/test/pack/testlist.cuda.urh-d-urh.gen \
	$(top_srcdir)/test/pack/testlist.cuda.md.d-d-d.gen \
	$(top_srcdir)/test/pack/testlist.cuda.md.urh-d-urh.gen \
	$(top_srcdir)/test/pack/testlist.cuda.md-stride.d-d-d.gen

pack_testlists += $(cuda_pack_testlists)
EXTRA_DIST += $(cuda_pack_testlists)
