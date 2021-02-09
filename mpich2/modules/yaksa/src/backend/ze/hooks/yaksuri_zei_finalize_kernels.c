/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * DO NOT EDIT: AUTOMATICALLY GENERATED FILE !!
 */

#include "stdlib.h"
#include "yaksi.h"
#include "yaksuri_zei.h"
#include "level_zero/ze_api.h"

ze_result_t yaksuri_ze_finalize_module_kernel() {
    ze_result_t zerr = ZE_RESULT_SUCCESS;
    int i, k; 

    for (k=0; k<3100; k++) {
        if (yaksuri_ze_kernels[k] == NULL) continue;
        for (i=0; i<yaksuri_zei_global.ndevices; i++) {
            if (yaksuri_ze_kernels[k][i] == NULL) continue;
            zerr = zeKernelDestroy(yaksuri_ze_kernels[k][i]);
            if (zerr != ZE_RESULT_SUCCESS) { 
                fprintf(stderr, "ZE Error (%s:%s,%d): %d \n", __func__, __FILE__, __LINE__, zerr); 
                goto fn_fail; 
            }
            yaksuri_ze_kernels[k][i] = NULL;
        }
    }

    for (k=0; k<300; k++) {
        if (yaksuri_ze_modules[k] == NULL) continue;
        for (i=0; i<yaksuri_zei_global.ndevices; i++) {
            if (yaksuri_ze_modules[k][i] == NULL) continue;
            zerr = zeModuleDestroy(yaksuri_ze_modules[k][i]);
            if (zerr != ZE_RESULT_SUCCESS) { 
                fprintf(stderr, "ZE Error (%s:%s,%d): %d \n", __func__, __FILE__, __LINE__, zerr); 
                goto fn_fail; 
            }
            yaksuri_ze_modules[k][i] = NULL;
        }
    }

fn_exit:
    return zerr; 
fn_fail:
    goto fn_exit; 
}
