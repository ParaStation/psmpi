#!/usr/bin/env bash

export OMP_NUM_THREADS=88 && export OMP_PLACES=threads && export OMP_PROC_BIND=close

LOCKS="mtx tkt mcs hmcs"
NITER=10

echo "lock,nthreads,thruput" >  thread_scale_${OMP_NUM_THREADS}.csv
for lock in `echo $LOCKS`
do
    for (( i=0; i <= $NITER; i++ ));
    do
        ./thread_scale_${lock} > foo.csv
        sed "1d; s/^/$lock,/" foo.csv >> thread_scale_${OMP_NUM_THREADS}.csv
        rm foo.csv
    done
done
