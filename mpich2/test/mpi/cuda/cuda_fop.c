/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2003 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 *  Portions of this code were written/modified by ParTec AG
 *  Copyright (C) 2019-2021 ParTec Cluster Competence Center GmbH, Munich
 *  Copyright (C) 2021-2022 ParTec AG, Munich
 */

#include "mpi.h"
#include "mpitest.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define ITER 100

#define TYPE_C   double
#define TYPE_MPI MPI_DOUBLE
#define TYPE_FMT "%f"

#define CMP(x, y) ((x - ((TYPE_C) (y))) > 1.0e-9)

#define MALLOC(x)          malloc(x)
#define FREE(x)            free(x)

#define CUDA_MALLOC(x,y)   cudaMalloc(x,y)
#define CUDA_FREE(x)       cudaFree(x)
#define CUDA_CHECK(call)                                                          \
	do {                                                                          \
		if((call) != cudaSuccess) {                                               \
			cudaError_t err = cudaGetLastError();                                 \
			fprintf(stderr, "CUDA error calling \""#call"\", code is %d\n", err); \
			MPI_Abort(MPI_COMM_WORLD, err);                                       \
		}                                                                         \
	} while(0);

static const int SQ_LIMIT = 10;
static int SQ_COUNT = 0;
static int SQ_VERBOSE = 0;

#define SQUELCH(X)                              \
  do {                                          \
    if (SQ_COUNT < SQ_LIMIT || SQ_VERBOSE) {    \
      SQ_COUNT++;                               \
      X                                         \
    }                                           \
  } while (0)




void reset_vars(TYPE_C * val_ptr, TYPE_C * res_ptr, TYPE_C * cval_ptr, TYPE_C * cres_ptr, MPI_Win win)
{
    int i, rank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
    for (i = 0; i < nproc; i++) {
        val_ptr[i] = 0;
        res_ptr[i] = -1;
    }
	CUDA_CHECK(cudaMemcpy(cval_ptr, val_ptr, sizeof(TYPE_C) * nproc, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cres_ptr, res_ptr, sizeof(TYPE_C) * nproc, cudaMemcpyHostToDevice));
    MPI_Win_unlock(rank, win);


    MPI_Barrier(MPI_COMM_WORLD);
}

/* ------ Based on rma/fetch_and_op.c  ------ */
int main(int argc, char **argv)
{
    int i, rank, nproc, mpi_type_size;
    int errors = 0, all_errors = 0;
    TYPE_C *val_ptr, *res_ptr, *cval_ptr, *cres_ptr;
    MPI_Win win;
    MPI_Info info = MPI_INFO_NULL;

    MTest_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    MPI_Type_size(TYPE_MPI, &mpi_type_size);
    assert(mpi_type_size == sizeof(TYPE_C));

    val_ptr = malloc(sizeof(TYPE_C) * nproc);
    res_ptr = malloc(sizeof(TYPE_C) * nproc);
	CUDA_CHECK(CUDA_MALLOC((void**)&cval_ptr, sizeof(TYPE_C) * nproc));
	CUDA_CHECK(CUDA_MALLOC((void**)&cres_ptr, sizeof(TYPE_C) * nproc));

    MTEST_VG_MEM_INIT(val_ptr, sizeof(TYPE_C) * nproc);
    MTEST_VG_MEM_INIT(res_ptr, sizeof(TYPE_C) * nproc);
	CUDA_CHECK(cudaMemcpy(cval_ptr, val_ptr, sizeof(TYPE_C) * nproc, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cres_ptr, res_ptr, sizeof(TYPE_C) * nproc, cudaMemcpyHostToDevice));

#ifdef TEST_HWACC_INFO
    MPI_Info_create(&info);
    MPI_Info_set(info, "disable_shm_accumulate", "true");
#endif

#ifdef TEST_ACCOPS_INFO
    if (info == MPI_INFO_NULL)
        MPI_Info_create(&info);
    MPI_Info_set(info, "which_accumulate_ops", "sum,no_op");
#endif

    MPI_Win_create(cval_ptr, sizeof(TYPE_C) * nproc, sizeof(TYPE_C), info, MPI_COMM_WORLD, &win);

#if defined(TEST_HWACC_INFO) || defined(TEST_ACCOPS_INFO)
    MPI_Info_free(&info);
#endif

    /* Test self communication */

    reset_vars(val_ptr, res_ptr, cval_ptr, cres_ptr, win);

    for (i = 0; i < ITER; i++) {
        TYPE_C one = 1, result = -1;
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
        MPI_Fetch_and_op(&one, &result, TYPE_MPI, rank, 0, MPI_SUM, win);
        MPI_Win_unlock(rank, win);
    }

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
	CUDA_CHECK(cudaMemcpy(val_ptr, cval_ptr, sizeof(TYPE_C) * nproc, cudaMemcpyDeviceToHost));
    if (CMP(val_ptr[0], ITER)) {
        SQUELCH(printf
                ("%d->%d -- SELF: expected " TYPE_FMT ", got " TYPE_FMT "\n", rank, rank,
                 (TYPE_C) ITER, val_ptr[0]););
        errors++;
    }
    MPI_Win_unlock(rank, win);

    /* Test neighbor communication */

    reset_vars(val_ptr, res_ptr, cval_ptr, cres_ptr, win);

    for (i = 0; i < ITER; i++) {
		TYPE_C *one, *result, *cone, *cresult;
		one = (TYPE_C *)malloc(sizeof(TYPE_C));
		result = (TYPE_C *)malloc(sizeof(TYPE_C));
		CUDA_CHECK(CUDA_MALLOC((void**)&cone, sizeof(TYPE_C)));
		CUDA_CHECK(CUDA_MALLOC((void**)&cresult, sizeof(TYPE_C)));
        *one = 1;
		*result = -1;
		CUDA_CHECK(cudaMemcpy(cresult, result, sizeof(TYPE_C), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(cone, one, sizeof(TYPE_C), cudaMemcpyHostToDevice));


        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, (rank + 1) % nproc, 0, win);
        MPI_Fetch_and_op(cone, cresult, TYPE_MPI, (rank + 1) % nproc, 0, MPI_SUM, win);
        MPI_Win_unlock((rank + 1) % nproc, win);

		CUDA_CHECK(cudaMemcpy(result, cresult, sizeof(TYPE_C), cudaMemcpyDeviceToHost));
        if (CMP(*result, i)) {
            SQUELCH(printf
                    ("%d->%d -- NEIGHBOR[%d]: expected result " TYPE_FMT ", got " TYPE_FMT "\n",
                     (rank + 1) % nproc, rank, i, (TYPE_C) i, result););
            errors++;
        }

		CUDA_CHECK(CUDA_FREE(cone));
		CUDA_CHECK(CUDA_FREE(cresult));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
    if (CMP(val_ptr[0], ITER)) {
        SQUELCH(printf
                ("%d->%d -- NEIGHBOR: expected " TYPE_FMT ", got " TYPE_FMT "\n",
                 (rank + 1) % nproc, rank, (TYPE_C) ITER, val_ptr[0]););
        errors++;
    }
    MPI_Win_unlock(rank, win);

    /* Test contention */

    reset_vars(val_ptr, res_ptr, cval_ptr, cres_ptr, win);

    if (rank != 0) {
        for (i = 0; i < ITER; i++) {
            TYPE_C one = 1, result;
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
            MPI_Fetch_and_op(&one, &result, TYPE_MPI, 0, 0, MPI_SUM, win);
            MPI_Win_unlock(0, win);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
	CUDA_CHECK(cudaMemcpy(val_ptr, cval_ptr, sizeof(TYPE_C) * nproc, cudaMemcpyDeviceToHost));
    if (rank == 0 && nproc > 1) {
        if (CMP(val_ptr[0], ITER * (nproc - 1))) {
            SQUELCH(printf
                    ("*->%d - CONTENTION: expected=" TYPE_FMT " val=" TYPE_FMT "\n", rank,
                     (TYPE_C) ITER * (nproc - 1), val_ptr[0]););
            errors++;
        }
    }
    MPI_Win_unlock(rank, win);

    /* Test all-to-all communication (fence) */

    reset_vars(val_ptr, res_ptr, cval_ptr, cres_ptr, win);

    for (i = 0; i < ITER; i++) {
        int j;

        MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
        for (j = 0; j < nproc; j++) {
            TYPE_C rank_cnv = (TYPE_C) rank;
            MPI_Fetch_and_op(&rank_cnv, &res_ptr[j], TYPE_MPI, j, rank, MPI_SUM, win);
        }
        MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
        MPI_Barrier(MPI_COMM_WORLD);

        for (j = 0; j < nproc; j++) {
            if (CMP(res_ptr[j], i * rank)) {
                SQUELCH(printf
                        ("%d->%d -- ALL-TO-ALL (FENCE) [%d]: expected result " TYPE_FMT ", got "
                         TYPE_FMT "\n", rank, j, i, (TYPE_C) i * rank, res_ptr[j]););
                errors++;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
    for (i = 0; i < nproc; i++) {
        if (CMP(val_ptr[i], ITER * i)) {
            SQUELCH(printf
                    ("%d->%d -- ALL-TO-ALL (FENCE): expected " TYPE_FMT ", got " TYPE_FMT "\n", i,
                     rank, (TYPE_C) ITER * i, val_ptr[i]););
            errors++;
        }
    }
    MPI_Win_unlock(rank, win);

    /* Test all-to-all communication (lock-all) */

    reset_vars(val_ptr, res_ptr, cval_ptr, cres_ptr, win);

    for (i = 0; i < ITER; i++) {
        int j;

        MPI_Win_lock_all(0, win);
        for (j = 0; j < nproc; j++) {
            TYPE_C rank_cnv = (TYPE_C) rank;
            MPI_Fetch_and_op(&rank_cnv, &res_ptr[j], TYPE_MPI, j, rank, MPI_SUM, win);
        }
        MPI_Win_unlock_all(win);
        MPI_Barrier(MPI_COMM_WORLD);

        for (j = 0; j < nproc; j++) {
            if (CMP(res_ptr[j], i * rank)) {
                SQUELCH(printf
                        ("%d->%d -- ALL-TO-ALL (LOCK-ALL) [%d]: expected result " TYPE_FMT ", got "
                         TYPE_FMT "\n", rank, j, i, (TYPE_C) i * rank, res_ptr[j]););
                errors++;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
    for (i = 0; i < nproc; i++) {
        if (CMP(val_ptr[i], ITER * i)) {
            SQUELCH(printf
                    ("%d->%d -- ALL-TO-ALL (LOCK-ALL): expected " TYPE_FMT ", got " TYPE_FMT "\n",
                     i, rank, (TYPE_C) ITER * i, val_ptr[i]););
            errors++;
        }
    }
    MPI_Win_unlock(rank, win);

    /* Test all-to-all communication (lock-all+flush) */

    reset_vars(val_ptr, res_ptr, cval_ptr, cres_ptr, win);

    for (i = 0; i < ITER; i++) {
        int j;

        MPI_Win_lock_all(0, win);
        for (j = 0; j < nproc; j++) {
            TYPE_C rank_cnv = (TYPE_C) rank;
            MPI_Fetch_and_op(&rank_cnv, &res_ptr[j], TYPE_MPI, j, rank, MPI_SUM, win);
            MPI_Win_flush(j, win);
        }
        MPI_Win_unlock_all(win);
        MPI_Barrier(MPI_COMM_WORLD);

        for (j = 0; j < nproc; j++) {
            if (CMP(res_ptr[j], i * rank)) {
                SQUELCH(printf
                        ("%d->%d -- ALL-TO-ALL (LOCK-ALL+FLUSH) [%d]: expected result " TYPE_FMT
                         ", got " TYPE_FMT "\n", rank, j, i, (TYPE_C) i * rank, res_ptr[j]););
                errors++;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
    for (i = 0; i < nproc; i++) {
        if (CMP(val_ptr[i], ITER * i)) {
            SQUELCH(printf
                    ("%d->%d -- ALL-TO-ALL (LOCK-ALL+FLUSH): expected " TYPE_FMT ", got " TYPE_FMT
                     "\n", i, rank, (TYPE_C) ITER * i, val_ptr[i]););
            errors++;
        }
    }
    MPI_Win_unlock(rank, win);

    /* Test NO_OP (neighbor communication) */

    MPI_Barrier(MPI_COMM_WORLD);
    reset_vars(val_ptr, res_ptr, cval_ptr, cres_ptr, win);

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
    for (i = 0; i < nproc; i++)
        val_ptr[i] = (TYPE_C) rank;
	CUDA_CHECK(cudaMemcpy(cval_ptr, val_ptr, sizeof(TYPE_C) * nproc, cudaMemcpyHostToDevice));
    MPI_Win_unlock(rank, win);
    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < ITER; i++) {
        int target = (rank + 1) % nproc;

        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, 0, win);
        MPI_Fetch_and_op(NULL, res_ptr, TYPE_MPI, target, 0, MPI_NO_OP, win);
        MPI_Win_unlock(target, win);

        if (res_ptr[0] != (TYPE_C) target) {
            SQUELCH(printf("%d->%d -- NOP[%d]: expected " TYPE_FMT ", got " TYPE_FMT "\n",
                           target, rank, i, (TYPE_C) target, res_ptr[0]););
            errors++;
        }
    }

    /* Test NO_OP (self communication) */

    MPI_Barrier(MPI_COMM_WORLD);
    reset_vars(val_ptr, res_ptr, cval_ptr, cres_ptr, win);

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
    for (i = 0; i < nproc; i++)
        val_ptr[i] = (TYPE_C) rank;
	CUDA_CHECK(cudaMemcpy(cval_ptr, val_ptr, sizeof(TYPE_C) * nproc, cudaMemcpyHostToDevice));
    MPI_Win_unlock(rank, win);
    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < ITER; i++) {
        int target = rank;

        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, 0, win);
        MPI_Fetch_and_op(NULL, cres_ptr, TYPE_MPI, target, 0, MPI_NO_OP, win);
        MPI_Win_unlock(target, win);

		CUDA_CHECK(cudaMemcpy(res_ptr, cres_ptr, sizeof(TYPE_C) * nproc, cudaMemcpyDeviceToHost));
        if (res_ptr[0] != (TYPE_C) target) {
            SQUELCH(printf("%d->%d -- NOP_SELF[%d]: expected " TYPE_FMT ", got " TYPE_FMT "\n",
                           target, rank, i, (TYPE_C) target, res_ptr[0]););
            errors++;
        }
    }

    MPI_Win_free(&win);

    free(val_ptr);
    free(res_ptr);
	CUDA_CHECK(CUDA_FREE(cval_ptr));
	CUDA_CHECK(CUDA_FREE(cres_ptr));
    MTest_Finalize(errors);

    return MTestReturnValue(all_errors);
}
