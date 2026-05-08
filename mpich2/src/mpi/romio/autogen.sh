#! /bin/sh
##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

if test -z "$AUTORECONF" ; then
    AUTORECONF="autoreconf"
fi

$AUTORECONF -vif || exit 1

if test -d mpl ; then
    echo "=== running autogen.sh in 'mpl' ==="
    (cd mpl && ./autogen.sh) || exit 1
fi
