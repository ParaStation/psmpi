/*
 *  (C) 2006 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 *  Portions of this code were written by Intel Corporation.
 *  Copyright (C) 2011-2012 Intel Corporation.  Intel provides this material
 *  to Argonne National Laboratory subject to Software Grant and Corporate
 *  Contributor License Agreement dated February 8, 2012.
 *
 *  Portions of this code were written/modified by
 *  ParTec Cluster Competence Center GmbH, Munich
 *  (C) 2016 ParTec Cluster Competence Center GmbH
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mpitest.h"

typedef struct {
    int val;
    int loc;
} twoint_t;

static int errors = 0;

int main(int argc, char **argv) {
    int me, nproc;
    twoint_t *data = NULL;
    twoint_t mine;
    twoint_t res;
    MPI_Win win;

    MTest_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (me == 0) {
        MPI_Alloc_mem(sizeof(twoint_t), MPI_INFO_NULL, &data);
    }

    MPI_Win_create(data, me == 0 ? sizeof(twoint_t) : 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, win);

    /* All processes perform MAXLOC and MINLOC operations on a 2INT on rank 0.
     * The loc is the origin process' rank, and the value is (nproc-me).  In
     * the case of MAXLOC, rank 0 should win and in the case of MINLOC, rank
     * nproc-1 should win.
     */

    /** Test MAXLOC **/

    if (me == 0) {
        data->val = 0;
        data->loc = -1;
    }
    MPI_Win_fence(0, win);

    mine.loc = me;
    mine.val = nproc - me;
    MPI_Get_accumulate(&mine, 1, MPI_2INT, &res, 1, MPI_2INT, 0, 0, 1, MPI_2INT, MPI_MAXLOC, win);

    MPI_Win_fence(0, win);

    if (me == 0 && (data->loc != 0 || data->val != nproc)) {
        errors++;
        printf("Error in test MAXLOC + Fence -> Expected: { loc = %d, val = %d }  Actual: { loc = %d, val = %d }\n",
               0, nproc, data->loc, data->val);
    }

    if ( (res.loc != -1) && (res.val != nproc-res.loc) ) {
	 errors++;
	 printf("Error in test MAXLOC + Fence: res.val (=%d) != nproc (=%d) - res.loc (=%d)\n",
		res.val, nproc, res.loc);
    }

    /** Test MINLOC **/

    if (me == 0) {
        data->val = nproc;
        data->loc = -1;
    }
    MPI_Win_fence(0, win);

    mine.loc = me;
    mine.val = nproc - me;
    MPI_Get_accumulate(&mine, 1, MPI_2INT, &res, 1, MPI_2INT, 0, 0, 1, MPI_2INT, MPI_MINLOC, win);
    MPI_Win_fence(0, win);

    if (me == 0 && (data->loc != nproc-1 || data->val != 1)) {
        errors++;
        printf("Error in test MINLOC + Fence -> Expected: { loc = %d, val = %d }  Actual: { loc = %d, val = %d }\n",
               nproc-1, 1, data->loc, data->val);
    }

    if ( (res.loc != -1) && (res.val != nproc-res.loc) ) {
	 errors++;
	 printf("Error in test MINLOC + Fence -> res.val (=%d) != nproc (=%d) - res.loc (=%d)\n",
		res.val, nproc, res.loc);
    }


    /* All processes perform MAXLOC and MINLOC operations on a 2INT on rank 0.
     * The loc is the origin process' rank, and the value is 1.  In both cases,
     * rank 0 should win because the values are equal and it has the lowest
     * loc.
     */

    /** Test MAXLOC **/

    if (me == 0) {
        data->val = 0;
        data->loc = -1;
    }
    MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

    mine.loc = me;
    mine.val = 1;

    MPI_Win_lock(MPI_LOCK_SHARED, 0, MPI_MODE_NOCHECK, win);
    MPI_Get_accumulate(&mine, 1, MPI_2INT, &res, 1, MPI_2INT, 0, 0, 1, MPI_2INT, MPI_MAXLOC, win);
    MPI_Win_unlock(0, win);

    MPI_Barrier(MPI_COMM_WORLD);

    if (me == 0 && (data->loc != 0 || data->val != 1)) {
        errors++;
        printf("Error in test MAXLOC + Lock -> Expected: { loc = %d, val = %d }  Actual: { loc = %d, val = %d }\n",
               0, 1, data->loc, data->val);

    }

    MPI_Win_fence(MPI_MODE_NOPRECEDE, win);

    /** Test MINLOC **/

    if (me == 0) {
        data->val = nproc;
        data->loc = -1;
    }
    MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

    mine.loc = me;
    mine.val = 1;

    MPI_Win_lock(MPI_LOCK_SHARED, 0, MPI_MODE_NOCHECK, win);
    MPI_Get_accumulate(&mine, 1, MPI_2INT, &res, 1, MPI_2INT, 0, 0, 1, MPI_2INT, MPI_MINLOC, win);
    MPI_Win_unlock(0, win);

    MPI_Barrier(MPI_COMM_WORLD);

    if (me == 0 && (data->loc != 0 || data->val != 1)) {
        errors++;
        printf("Error in test MINLOC + Lock -> Expected: { loc = %d, val = %d }  Actual: { loc = %d, val = %d }\n",
               0, 1, data->loc, data->val);
    }

    MPI_Win_free(&win);

    MTest_Finalize(errors);

    return MTestReturnValue(errors);
}
