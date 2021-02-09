##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##


##########################################################################
##### capture user arguments
##########################################################################

# --with-cuda-sm
AC_ARG_WITH([cuda-sm],
            [
  --with-cuda-sm=<options> (https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
          Comma-separated list of below options:
                all - build compatibility for all GPUs supported by the CUDA version (can increase compilation time)

                # Kepler architecture
                kepler - build compatibility for all Kepler GPUs
                30     - generic kepler architecture (generic - Tesla K40/K80, GeForce 700, GT-730)
                35     - specific Tesla K40 (adds support for dynamic parallelism)
                37     - specific Tesla K80 (adds more registers)

                # Maxwell architecture
                maxwell - build compatibility for all Maxwell GPUs
                50      - Tesla/Quadro M series
                52      - Quadro M6000, GeForce 900, GTX-970, GTX-980, GTX Titan X
                53      - Tegra (Jetson) TX1 / Tegra X1, Drive CX, Drive PX, Jetson Nano

                # Pascal architecture
                pascal - build compatibility for all Pascal GPUs
                60     - Quadro GP100, Tesla P100, DGX-1 (Generic Pascal)
                61     - GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp,
                         Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2
                62     - Integrated GPU on the NVIDIA Drive PX2, Tegra (Jetson) TX2

                # Volta architecture
                volta - build compatibility for all Volta GPUs
                70    - DGX-1 with Volta, Tesla V100, GTX 1180 (GV104), Titan V, Quadro GV100
                72    - Jetson AGX Xavier, Drive AGX Pegasus, Xavier NX

                # Turing architecture
                turing - build compatibility for all Turing GPUs
                75     - GTX/RTX Turing - GTX 1660 Ti, RTX 2060, RTX 2070, RTX 2080, Titan RTX,
                         Quadro RTX 4000, Quadro RTX 5000, Quadro RTX 6000, Quadro RTX 8000,
                         Quadro T1000/T2000, Tesla T4

                # Ampere architecture
                ampere - build compatibility for all Ampere GPUs
                80     - RTX Ampere - RTX 3080

                # Other
                <numeric> - specific SM numeric to use
            ],,
            [with_cuda_sm=all])


# --with=cuda
PAC_SET_HEADER_LIB_PATH([cuda])
PAC_CHECK_HEADER_LIB([cuda_runtime_api.h],[cudart],[cudaStreamSynchronize],[have_cuda=yes],[have_cuda=no])
if test "${have_cuda}" = "yes" ; then
    AC_MSG_CHECKING([whether nvcc works])
    cat>conftest.cu<<EOF
    __global__ void foo(int x) {}
EOF
    ${with_cuda}/bin/nvcc -c conftest.cu 2> /dev/null
    if test "$?" = "0" ; then
        AC_DEFINE([HAVE_CUDA],[1],[Define is CUDA is available])
        AS_IF([test -n "${with_cuda}"],[NVCC=${with_cuda}/bin/nvcc],[NVCC=nvcc])
        AC_SUBST(NVCC)
        # nvcc compiled applications need libstdc++ to be able to link
        # with a C compiler
        PAC_PUSH_FLAG([LIBS])
        PAC_APPEND_FLAG([-lstdc++],[LIBS])
        AC_LINK_IFELSE(
            [AC_LANG_PROGRAM([int x = 5;],[x++;])],
            [libstdcpp_works=yes],
            [libstdcpp_works=no])
        PAC_POP_FLAG([LIBS])
        if test "${libstdcpp_works}" = "yes" ; then
            PAC_APPEND_FLAG([-lstdc++],[LIBS])
            AC_MSG_RESULT([yes])
        else
            have_cuda=no
            AC_MSG_RESULT([no])
        fi
    else
        have_cuda=no
        AC_MSG_RESULT([no])
    fi
    rm -f conftest.*
fi
AM_CONDITIONAL([BUILD_CUDA_BACKEND], [test x${have_cuda} = xyes])
AM_CONDITIONAL([BUILD_CUDA_TESTS], [test x${have_cuda} = xyes])


# --with-cuda-p2p
AC_ARG_ENABLE([cuda-p2p],AS_HELP_STRING([--enable-cuda-p2p={yes|no|cliques}],[controls CUDA P2P capability]),,
              [enable_cuda_p2p=yes])
if test "${enable_cuda_p2p}" = "yes" ; then
    AC_DEFINE([CUDA_P2P],[CUDA_P2P_ENABLED],[Define if CUDA P2P is enabled])
elif test "${enable_cuda_p2p}" = "cliques" ; then
    AC_DEFINE([CUDA_P2P],[CUDA_P2P_CLIQUES],[Define if CUDA P2P is enabled in clique mode])
else
    AC_DEFINE([CUDA_P2P],[CUDA_P2P_DISABLED],[Define if CUDA P2P is disabled])
fi


##########################################################################
##### analyze the user arguments and setup internal infrastructure
##########################################################################

if test "${have_cuda}" = "yes" ; then
    for maj_version in 11 10 9 8 7 6 5 ; do
        version=$((maj_version * 1000))
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([
                              #include <cuda.h>
                              int x[[CUDA_VERSION - $version]];
                          ],)],[cuda_version=${maj_version}],[])
        if test ! -z ${cuda_version} ; then break ; fi
    done
    PAC_PUSH_FLAG([IFS])
    IFS=","
    CUDA_SM=
    for sm in ${with_cuda_sm} ; do
        case "$sm" in
            all)
                if test ${cuda_version} -ge 11 ; then
                    # maxwell (52) to ampere (80)
                    supported_cuda_sms="52 53 60 61 62 70 72 75 80"
                elif test ${cuda_version} -ge 10 ; then
                    # kepler (30) to turing (75)
                    supported_cuda_sms="30 35 37 50 52 53 60 61 62 70 72 75"
                elif test ${cuda_version} -ge 9 ; then
                    # kepler (30) to volta (72)
                    supported_cuda_sms="30 35 37 50 52 53 60 61 62 70 72"
                elif test ${cuda_version} -ge 8 ; then
                    # kepler (30) to pascal (62)
                    supported_cuda_sms="30 35 37 50 52 53 60 61 62"
                elif test ${cuda_version} -ge 6 ; then
                    # kepler (30) to maxwell (53)
                    supported_cuda_sms="30 35 37 50 52 53"
                elif test ${cuda_version} -ge 5 ; then
                    # kepler (30) to kepler (37)
                    supported_cuda_sms="30 35 37"
                fi

                for supported_cuda_sm in $supported_cuda_sms ; do
                    PAC_APPEND_FLAG([$supported_cuda_sm],[CUDA_SM])
                done
                ;;

            kepler)
                PAC_APPEND_FLAG([30],[CUDA_SM])
                PAC_APPEND_FLAG([35],[CUDA_SM])
                PAC_APPEND_FLAG([37],[CUDA_SM])
                ;;

            maxwell)
                PAC_APPEND_FLAG([50],[CUDA_SM])
                PAC_APPEND_FLAG([52],[CUDA_SM])
                PAC_APPEND_FLAG([53],[CUDA_SM])
                ;;

            pascal)
                PAC_APPEND_FLAG([60],[CUDA_SM])
                PAC_APPEND_FLAG([61],[CUDA_SM])
                PAC_APPEND_FLAG([62],[CUDA_SM])
                ;;

            volta)
                PAC_APPEND_FLAG([70],[CUDA_SM])
                PAC_APPEND_FLAG([72],[CUDA_SM])
                ;;

            turing)
                PAC_APPEND_FLAG([75],[CUDA_SM])
                ;;

            ampere)
                PAC_APPEND_FLAG([80],[CUDA_SM])
                ;;

            none)
                ;;

            *)
                PAC_APPEND_FLAG([$sm],[CUDA_SM])
                ;;
          esac
    done
    PAC_POP_FLAG([IFS])

    for sm in ${CUDA_SM} ; do
        if test -z "${CUDA_GENCODE}" ; then
            CUDA_GENCODE="-gencode=arch=compute_${sm},code=sm_${sm}"
        else
            CUDA_GENCODE="${CUDA_GENCODE} -gencode=arch=compute_${sm},code=sm_${sm}"
        fi
    done
    AC_SUBST(CUDA_GENCODE)

    if test -z "${CUDA_GENCODE}" ; then
        AC_MSG_ERROR([--with-cuda-sm not specified; either specify it or disable cuda support])
    fi

    supported_backends="${supported_backends},cuda"
    backend_info="${backend_info}
CUDA backend specific options:
      CUDA GENCODE: ${with_cuda_sm} (${CUDA_GENCODE})"
fi
