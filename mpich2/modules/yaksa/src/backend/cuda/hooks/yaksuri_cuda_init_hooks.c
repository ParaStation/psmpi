/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksuri_cudai.h"
#include <assert.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

static void *cuda_host_malloc(uintptr_t size)
{
    void *ptr = NULL;

    cudaError_t cerr = cudaMallocHost(&ptr, size);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);

    return ptr;
}

static void *cuda_gpu_malloc(uintptr_t size, int device)
{
    void *ptr = NULL;
    cudaError_t cerr;

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);

    if (cur_device != device) {
        cerr = cudaSetDevice(device);
        YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
    }

    cerr = cudaMalloc(&ptr, size);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);

    if (cur_device != device) {
        cerr = cudaSetDevice(cur_device);
        YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
    }

    return ptr;
}

static void cuda_host_free(void *ptr)
{
    cudaError_t cerr = cudaFreeHost(ptr);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

static void cuda_gpu_free(void *ptr)
{
    cudaError_t cerr = cudaFree(ptr);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

yaksuri_cudai_global_s yaksuri_cudai_global;

static int finalize_hook(void)
{
    int rc = YAKSA_SUCCESS;
    cudaError_t cerr;

    for (int i = 0; i < yaksuri_cudai_global.ndevices; i++) {
        cerr = cudaSetDevice(i);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

        cerr = cudaStreamDestroy(yaksuri_cudai_global.stream[i]);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

        free(yaksuri_cudai_global.p2p[i]);
    }
    free(yaksuri_cudai_global.stream);
    free(yaksuri_cudai_global.p2p);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int get_num_devices(int *ndevices)
{
    *ndevices = yaksuri_cudai_global.ndevices;

    return YAKSA_SUCCESS;
}

static int check_p2p_comm(int sdev, int ddev, bool * is_enabled)
{
#if CUDA_P2P == CUDA_P2P_ENABLED
    *is_enabled = yaksuri_cudai_global.p2p[sdev][ddev];
#elif CUDA_P2P == CUDA_P2P_CLIQUES
    if ((sdev + ddev) % 2)
        *is_enabled = 0;
    else
        *is_enabled = yaksuri_cudai_global.p2p[sdev][ddev];
#else
    *is_enabled = 0;
#endif

    return YAKSA_SUCCESS;
}

int yaksuri_cuda_init_hook(yaksur_gpudriver_hooks_s ** hooks)
{
    int rc = YAKSA_SUCCESS;
    cudaError_t cerr;

    cerr = cudaGetDeviceCount(&yaksuri_cudai_global.ndevices);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    if (getenv("CUDA_VISIBLE_DEVICES") == NULL) {
        /* user did not do any filtering for us; if any of the devices
         * is in exclusive mode, disable GPU support to avoid
         * incorrect device sharing */
        bool excl = false;
        for (int i = 0; i < yaksuri_cudai_global.ndevices; i++) {
            struct cudaDeviceProp prop;

            cerr = cudaGetDeviceProperties(&prop, i);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

            if (prop.computeMode != cudaComputeModeDefault) {
                excl = true;
                break;
            }
        }

        if (excl == true) {
            fprintf(stderr, "[yaksa] ====> Disabling CUDA support <====\n");
            fprintf(stderr,
                    "[yaksa] CUDA is setup in exclusive compute mode, but CUDA_VISIBLE_DEVICES is not set\n");
            fprintf(stderr,
                    "[yaksa] You can silence this warning by setting CUDA_VISIBLE_DEVICES\n");
            fflush(stderr);
            *hooks = NULL;
            goto fn_exit;
        }
    }

    yaksuri_cudai_global.stream = (cudaStream_t *)
        malloc(yaksuri_cudai_global.ndevices * sizeof(cudaStream_t));

    yaksuri_cudai_global.p2p = (bool **) malloc(yaksuri_cudai_global.ndevices * sizeof(bool *));
    for (int i = 0; i < yaksuri_cudai_global.ndevices; i++) {
        yaksuri_cudai_global.p2p[i] = (bool *) malloc(yaksuri_cudai_global.ndevices * sizeof(bool));
    }

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    for (int i = 0; i < yaksuri_cudai_global.ndevices; i++) {
        cerr = cudaSetDevice(i);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

        cerr = cudaStreamCreateWithFlags(&yaksuri_cudai_global.stream[i], cudaStreamNonBlocking);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

        for (int j = 0; j < yaksuri_cudai_global.ndevices; j++) {
            if (i == j) {
                yaksuri_cudai_global.p2p[i][j] = 1;
            } else {
                int val;
                cerr = cudaDeviceCanAccessPeer(&val, i, j);
                YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

                if (val) {
                    cerr = cudaDeviceEnablePeerAccess(j, 0);
                    if (cerr != cudaErrorPeerAccessAlreadyEnabled) {
                        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
                    }
                    yaksuri_cudai_global.p2p[i][j] = 1;
                } else {
                    yaksuri_cudai_global.p2p[i][j] = 0;
                }
            }
        }
    }

    cerr = cudaSetDevice(cur_device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    *hooks = (yaksur_gpudriver_hooks_s *) malloc(sizeof(yaksur_gpudriver_hooks_s));
    (*hooks)->get_num_devices = get_num_devices;
    (*hooks)->check_p2p_comm = check_p2p_comm;
    (*hooks)->finalize = finalize_hook;
    (*hooks)->get_iov_pack_threshold = yaksuri_cudai_get_iov_pack_threshold;
    (*hooks)->get_iov_unpack_threshold = yaksuri_cudai_get_iov_unpack_threshold;
    (*hooks)->ipack = yaksuri_cudai_ipack;
    (*hooks)->iunpack = yaksuri_cudai_iunpack;
    (*hooks)->pup_is_supported = yaksuri_cudai_pup_is_supported;
    (*hooks)->host_malloc = cuda_host_malloc;
    (*hooks)->host_free = cuda_host_free;
    (*hooks)->gpu_malloc = cuda_gpu_malloc;
    (*hooks)->gpu_free = cuda_gpu_free;
    (*hooks)->get_ptr_attr = yaksuri_cudai_get_ptr_attr;
    (*hooks)->event_record = yaksuri_cudai_event_record;
    (*hooks)->event_query = yaksuri_cudai_event_query;
    (*hooks)->add_dependency = yaksuri_cudai_add_dependency;
    (*hooks)->type_create = yaksuri_cudai_type_create_hook;
    (*hooks)->type_free = yaksuri_cudai_type_free_hook;
    (*hooks)->info_create = yaksuri_cudai_info_create_hook;
    (*hooks)->info_free = yaksuri_cudai_info_free_hook;
    (*hooks)->info_keyval_append = yaksuri_cudai_info_keyval_append;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
