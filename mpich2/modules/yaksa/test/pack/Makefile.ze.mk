##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

ze_pack_testlists = $(top_srcdir)/test/pack/testlist.ze.d-d-d.gen \
	$(top_srcdir)/test/pack/testlist.ze.d-rh-d.gen \
	$(top_srcdir)/test/pack/testlist.ze.d-urh-d.gen \
	$(top_srcdir)/test/pack/testlist.ze.rh-d-rh.gen \
	$(top_srcdir)/test/pack/testlist.ze.urh-d-urh.gen \
	$(top_srcdir)/test/pack/testlist.ze.md.d-d-d.gen \
	$(top_srcdir)/test/pack/testlist.ze.md.urh-d-urh.gen \
	$(top_srcdir)/test/pack/testlist.ze.md-stride.d-d-d.gen

pack_testlists += $(ze_pack_testlists)
EXTRA_DIST += $(ze_pack_testlists)
