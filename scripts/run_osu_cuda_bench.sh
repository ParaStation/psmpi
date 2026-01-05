#!/bin/bash
#
# ParaStation
#
# Copyright (C) 2016-2021 ParTec Cluster Competence Center GmbH, Munich
# Copyright (C) 2021-2026 ParTec AG, Munich
#
# This file may be distributed under the terms of the Q Public License
# as defined in the file LICENSE.QPL included in the packaging of this
# file.
#
# Run all CUDA-related tests of the OMB (OSU Micro-Benchmarks) suite.
# (see http://mvapich.cse.ohio-state.edu/benchmarks/)

benchprefix="lib/osu-micro-benchmarks/mpi"

if [ "$#" -ge "3" ] ; then
    exec=`basename $3`
    if [ "$exec" == "mpiexec" ] ; then
	MPIRUN="$3"
    else
	MPIRUN="$3/mpiexec"
    fi
else
    MPIRUN=`which mpiexec`
fi

if [ "$#" -ge "2" ] ; then
    exec=`basename $2`
    if [ "$exec" == "mpiccc" ] ; then
	MPICC="$2"
    else
	MPICC="$2/mpicc"
    fi
else
    MPICC=`which mpicc`
fi

if [ "$#" -ge "1" ] ; then
    cd $1
    if test -f README ; then
	OSU=`cat README | grep "OMB (OSU Micro-Benchmarks)"`
	if [ -n "$OSU" ] ; then
	    rm -rf build
	    mkdir build
	    cd build
	    echo "../configure CC=$MPICC --prefix=$PWD --enable-cuda=basic --with-cuda=/usr/local/cuda"
	    ../configure CC=$MPICC --prefix=$PWD --enable-cuda=basic --with-cuda=/usr/local/cuda
	    make
	    make install

	    cat > "/tmp/osu-cuda-testlist" << EOF
pt2pt/osu_bibw 2
pt2pt/osu_bw 2
pt2pt/osu_latency 2
collective/osu_allgather 4
collective/osu_allgatherv 4
collective/osu_allreduce 4
collective/osu_alltoall 4
collective/osu_alltoallv 4
collective/osu_bcast 4
collective/osu_gather 4
collective/osu_gatherv 4
collective/osu_reduce 4
collective/osu_reduce_scatter 4
collective/osu_scatter 4
collective/osu_scatterv 4
EOF
	    testlist=`cat /tmp/osu-cuda-testlist`
	    testname=""
	    for test in $testlist ; do
		testnum=$test
		if [ "$testnum" -eq "$testnum" ] 2>/dev/null ; then

		    echo "mpiexec -np $testnum $PWD/$benchprefix/$testname"
		    mpiexec -np "$testnum" "$PWD/$benchprefix/$testname" -d cuda D D

		else
		    testname="$testnum"
		fi
	    done

	    exit 1
	else
	    echo "!ERROR! OMB (OSU Micro-Benchmarks) not found."
	fi
    fi
fi

echo "!ERROR! Usage: ./osu_cuda_runs.sh <path/to/osu-micro-benchmarks> [path/to/mpicc] [path/to/mpiexec]"
