#!/bin/bash
if [ $# -gt 0 ] ; then
	if [ `basename $1` = "configure" ] ; then
		cd `dirname $1`
	else
		cd $1
	fi
fi
cp configure configure.orig
cat configure.orig | sed "s#pgcc\*#pgcc* | nvc*#g" | sed "s#pgc++\*#pgc++* | nvc++*#g" | sed "s#pgfortran\*#pgfortran* | nvfortran*#g" > configure
if [ $? -eq 0 ] ; then
	rm configure.orig
fi
