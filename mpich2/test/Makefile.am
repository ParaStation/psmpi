##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

## FIXME: commented out temporarily, really just want "mpi" to be dealt with at
## distclean time for now
##SUBDIRS = mpi util basic commands .
SUBDIRS = commands .

# Test both the MPI routines and the MPICH command scripts
testing: mpi/Makefile
	(NOXMLCLOSE=YES && export NOXMLCLOSE && cd mpi && $(MAKE) testing)
	(XMLFILE=../mpi/summary.xml && XMLCONTINUE=YES && \
	export XMLFILE && export XMLCONTINUE && \
	cd commands && $(MAKE) testing)

mpi/Makefile:
	cd mpi && ./autogen.sh && ./configure --with-mpi=${prefix}
