#! /bin/sh
#
# ParaStation
#
# Copyright (C) 2021 ParTec AG, Munich
#
# This file may be distributed under the terms of the Q Public License
# as defined in the file LICENSE.QPL included in the packaging of this
# file.




########################################################################
## Utility functions
########################################################################

error() {
    echo "===> ERROR:   $@"
}

echo_n() {
    # "echo -n" isn't portable, must portably implement with printf
    printf "%s" "$*"
}

########################################################################
echo
echo "####################################"
echo "## Checking user environment"
echo "####################################"
echo

########################################################################
## Checks to make sure we are running from the correct location
########################################################################

echo_n "Verifying the location of autogen.sh... "
if [ ! -d dist -o ! -s dist/psmpi.spec.templ ] ; then
    error "must execute at top level directory for now"
    exit 1
fi
# Set the SRCROOTDIR to be used later and avoid "cd ../../"-like usage.
SRCROOTDIR=$PWD
echo "done"

########################################################################
## Call autogen.sh from MPICH source tree
########################################################################
( cd ${SRCROOTDIR}/mpich2 && ./autogen.sh --with-doc && cd ${SRCROOTDIR} )


########################################################################
## Run autotools in the source root
########################################################################
autoreconf -vif
