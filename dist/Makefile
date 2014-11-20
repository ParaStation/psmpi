#                                                       -*- Makefile -*-
# ParaStation
#
# Copyright (C) 2003 ParTec AG, Karlsruhe
# Copyright (C) 2005-2010 ParTec Cluster Competence Center GmbH, Munich
#
# This file may be distributed under the terms of the Q Public License
# as defined in the file LICENSE.QPL included in the packaging of this
# file.
#
# Authors:      Jens Hauke <hauke@par-tec.com>
#               Thomas Moschny <moschny@par-tec.com>
#

TOP_SRCDIR?= ..
CONFIGARGS?=

# Makefile.sources have to define $(PKG_NAME) and $(SOURCE_FILES)
-include $(TOP_SRCDIR)/dist/Makefile.sources
######################################################

all: tar srpm rpm

VERSION:=$(shell cd $(TOP_SRCDIR) && scripts/vcversion --fmt version)
RELEASE:=$(shell cd $(TOP_SRCDIR) && scripts/vcversion --fmt release)
DISTNAME=$(PKG_NAME)-$(VERSION)-$(RELEASE)
RPMBUILDDIR=/var/tmp/$(DISTNAME)-$(USER)
RPMBUILDLOG=$(PKG_NAME)$(shell echo -n $(CONFIGARGS)|tr -c "[0-9A-Za-z]" "_")-$(VERSION)-$(RELEASE).rpmbuild.log
CHANGELOG=$(TOP_SRCDIR)/ChangeLog


$(DISTNAME)/%: $(TOP_SRCDIR)/%
	@mkdir -p $(dir $@)
	@cp -a $< $@

$(DISTNAME)/$(DISTNAME).spec: $(TOP_SRCDIR)/dist/$(PKG_NAME).spec.templ
	mkdir -p $(DISTNAME)
	sed ":t \
		s,@VERSION\(_$(PKG_NAME)\)\?@,$(VERSION),;t t \
		s,@RELEASE\(_$(PKG_NAME)\)\?@,$(RELEASE),;t t \
		/\%changelog/r $(CHANGELOG)" $< > $@

$(DISTNAME)/VERSION:
	echo $(VERSION)-$(RELEASE) > $@

$(DISTNAME).tar.gz: $(patsubst %,$(DISTNAME)/%,$(SOURCE_FILES)) \
			$(DISTNAME)/$(DISTNAME).spec \
			$(DISTNAME)/VERSION
	@echo "cp $(words $?) from $(words $^) sourcefiles to \"$(DISTNAME)\"."
	tar --owner=0 --group=0 -czf $@ $(DISTNAME)

$(DISTNAME).src.rpm: $(DISTNAME)/$(DISTNAME).spec $(DISTNAME).tar.gz
	rpmbuild -bs --define="_sourcedir $$PWD" --define="_srcrpmdir $$PWD" $<

# $(RPMBUILDLOG) as the target because we know the filename.(in opposite to the rpm filename)
$(RPMBUILDLOG): $(DISTNAME)/$(DISTNAME).spec $(DISTNAME).tar.gz
	mkdir -p $(RPMBUILDDIR)
	rpmbuild --clean $(TARGETARGS) \
		-bb --define="_sourcedir $$PWD" --define "_rpmdir $$PWD" \
		--define="_rpmfilename %%{NAME}-%%{VERSION}-%%{RELEASE}.%%{ARCH}.rpm" \
		--define="_builddir $(RPMBUILDDIR)" $(CONFIGARGS) \
		$< > $(RPMBUILDLOG)_ 2>&1 \
		|| { cat $(RPMBUILDLOG)_; exit 1; }
	mv $(RPMBUILDLOG)_ $(RPMBUILDLOG)
	@grep -B4 -E "warn|error|Wrote:" $(RPMBUILDLOG) || :
	-rmdir $(RPMBUILDDIR)
	@echo "RPM build log: $(RPMBUILDLOG)"

.PHONY: tar
tar: $(DISTNAME).tar.gz

.PHONY: srpm
srpm: $(DISTNAME).src.rpm

.PHONY: rpm
rpm: $(RPMBUILDLOG)

.PHONY: deb
deb: rpm
# Hack: generate debian packages with alien. Find rpms by scanning the buildlog.
	grep -E ^Wrote: $(RPMBUILDLOG) | while read x pkg; do			\
		if $(TOP_SRCDIR)/scripts/vcversion -r $(TOP_SRCDIR) --isrelease; then	\
			fakeroot alien -k "$$pkg";				\
		else								\
			fakeroot alien "$$pkg";					\
		fi								\
	done

.PHONY: Makefile.sources
Makefile.sources:
#	Delete old SOURCE_FILES=... from Makefile.sources:
	sed -i '1h;1!H;$${g;s/\n[ ]*SOURCE_FILES=\([^\n]\|\\\n\)*//;p};d' $@
#	Append new SOURCE_FILES= from git ls-files removing
#	filenams matching one of the patterns in Makefile.sources.exclude.
	(cd $(TOP_SRCDIR) && git ls-files) | if test -f "$@.exclude"; then \
	 grep -vf "$@.exclude"; else cat;fi | LC_ALL=C sort | \
	 gawk ' !f{printf "SOURCE_FILES=";f=1} f{gsub("([:%])","\\\\&");printf " \\\n" $$0} END{printf "\n"}' >> $@

######################################################
.PHONY: version
version:
	@echo VERSION="$(VERSION)"
	@echo RELEASE="$(RELEASE)"

.PHONY: tag
tag:
	@echo 'Use "git tag -a <tagname>" to tag a version!'
	@exit 1

######################################################
# get the latest update from the git@git.ccc:psbuild project
.PHONY: update
update: $(TOP_SRCDIR)/.git
	git pull git@git.ccc:psbuild master


# check for git repository
$(TOP_SRCDIR)/.git:
	@echo "Missing $(TOP_SRCDIR)/.git!"
	@echo "Is this a git project?"
	@echo "Get the git with:"
	@echo "> git clone git@git.ccc:$(PKG_NAME).git"
	@exit 1

######################################################
DISTNAMEPATTERN=$(PKG_NAME)-*[0-9].[0-9]*.[0-9]*-*
DISTCLEAN+= $(DISTNAMEPATTERN).tar.gz
DISTCLEAN+= $(DISTNAMEPATTERN).rpm
DISTCLEAN+= $(DISTNAMEPATTERN).spec
DISTCLEAN+= $(DISTNAMEPATTERN).rpmbuild.log

######################################################
.PHONY: clean distclean
clean:
	$(RM) -r $(DISTNAME)

distclean: clean
	$(RM) $(DISTCLEAN)
	$(RM) -r $(DISTNAMEPATTERN)
