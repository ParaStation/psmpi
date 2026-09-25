[#] start of __file__

UCC_ENABLED_DEVICES="psp ch4:ucx"

AC_DEFUN([PAC_SUBCFG_PREREQ_]PAC_SUBCFG_AUTO_SUFFIX,[
    if test "${with_ucc}" = "no" ; then
        pac_have_ucc=no;
    else
        PAC_CHECK_HEADER_LIB_EXPLICIT(ucc,[ucc/api/ucc.h],[ucc],[ucc_init_version])
        if test "$pac_have_ucc" = "yes" ; then
            AC_DEFINE([HAVE_UCC],1,[Define if building ucc])
        fi
    fi
    AM_CONDITIONAL([BUILD_UCC],[test "$pac_have_ucc" = "yes"])

    # Query the highest supported thread level of UCC/UCX
    AC_MSG_CHECKING([for UCC thread level])
    save_psp_LIBS="${LIBS}"
    LIBS="${LIBS} -lucc"
    save_psp_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${with_ucc}/lib:${with_ucc}/lib64"
    AC_RUN_IFELSE(
        [AC_LANG_SOURCE([[
            #include <stdio.h>
            #include <ucc/api/ucc.h>

            int main(int argc, char **argv)
            {
                int retval = -1;
                ucc_lib_h lib;
                ucc_lib_config_h config;
                ucc_lib_params_t params;
                ucc_lib_attr_t lib_attr;
                ucc_status_t status;

                FILE *file = fopen("ucc_thread_level.out", "w");
                if (!file)
                    goto fn_exit;

                params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
                params.thread_mode = UCC_THREAD_MULTIPLE;
                if (ucc_lib_config_read(NULL, NULL, &config) != UCC_OK) {
                    fprintf(file, "# ERROR: ucc_lib_config_read() failed!\n");
                    goto fn_exit;
                }

                status = ucc_init(&params, config, &lib);
                ucc_lib_config_release(config);
                if (status != UCC_OK) {
                    fprintf(file, "# ERROR: ucc_init() failed!\n");
                    goto fn_exit;
                }

                lib_attr.mask = UCC_LIB_ATTR_FIELD_THREAD_MODE;
                if (ucc_lib_get_attr(lib, &lib_attr) != UCC_OK) {
                    fprintf(file, "# ERROR: ucc_lib_get_attr() failed!\n");
		    ucc_finalize(lib);
                    goto fn_exit;
                }

                switch (lib_attr.thread_mode) {
                    case UCC_THREAD_SINGLE:
                        fprintf(file, "UCC_THREAD_SINGLE\n");
                        break;
                    case UCC_THREAD_FUNNELED:
                        fprintf(file, "UCC_THREAD_FUNNELED\n");
                        break;
                    case UCC_THREAD_MULTIPLE:
                        fprintf(file, "UCC_THREAD_MULTIPLE\n");
                        break;
                    default:
                        fprintf(file, "UCC_THREAD_LEVEL_UNKNOWN\n");
                }

                if (ucc_finalize(lib) != UCC_OK) {
                    fprintf(file, "# ERROR: ucc_finalize() failed!\n");
                    goto fn_exit;
                }

                retval = 0;

              fn_exit:
                if (file)
                    fclose(file);
                return retval;
            }
    ]])],
      [ucc_thread_level=$(cat ucc_thread_level.out) && rm -f ucc_thread_level.out],
      [ucc_thread_level="unknown"],
      [ucc_thread_level="cross-compile"]
    )
    LD_LIBRARY_PATH="${save_psp_LD_LIBRARY_PATH}"
    LIBS="${save_psp_LIBS}"
    AC_MSG_RESULT([$ucc_thread_level])
    # report the result via `MPID_COMMON_UCC_MAX_THREAD_LEVEL` to the devices
    if test "$ucc_thread_level" != "unknown" && test "$ucc_thread_level" != "cross-compile" ; then
        MPID_COMMON_UCC_MAX_THREAD_LEVEL="$ucc_thread_level"
    else
        # Note: For debugging purposes, one can check the 'ucc_thread_level.out'
	# artifact and its contents to see what went wrong during the detection.
        MPID_COMMON_UCC_MAX_THREAD_LEVEL=""
    fi

])dnl end PREREQ

AC_DEFUN([PAC_SUBCFG_BODY_]PAC_SUBCFG_AUTO_SUFFIX,[

AM_COND_IF([BUILD_UCC], [
# Check if selected device supports the UCC wrappers
AS_IF([test -n "${DEVICE}"],
[
    AC_MSG_CHECKING([if the device ${DEVICE} supports UCC])
    found_ucc_enabled_device="no"
    for dev in ${UCC_ENABLED_DEVICES}; do
        if test "${dev}" = "${DEVICE}" ; then
            found_ucc_enabled_device="yes"
            AC_MSG_RESULT(yes)
            break
        fi
    done
    if test "${found_ucc_enabled_device}" = "no"; then
        AC_MSG_RESULT(no)
        AC_MSG_ERROR([Device ${DEVICE} does not support UCC. UCC-enabled devices are: ${UCC_ENABLED_DEVICES}])
    fi
])

# Check if CUDA awareness is enabled, and if so, whether UCC is also CUDA-aware
AS_IF([test -n "${GPU_SUPPORT}" -a "x${GPU_SUPPORT}" = "xCUDA" && test -n "${with_ucc}" -a "x${with_ucc}" != "xno"],
[
    AC_PATH_PROG([ucc_info_path], [ucc_info], [no], [${with_ucc}/bin])
    AS_IF([test "x${ucc_info_path}" != "xno"],
    [
        AC_MSG_CHECKING([if UCC is CUDA-aware (HAVE_CUDA)])
        ucc_info_defs=$(${ucc_info_path} -b)
        AC_LANG_PUSH([C])
        AC_COMPILE_IFELSE(  [AC_LANG_PROGRAM(
                                    [[${ucc_info_defs}]],
                                    [[
                                        #if !defined(HAVE_CUDA) || HAVE_CUDA != 1
                                        #error macro not defined
                                        #endif
                                    ]]
                                    )
                            ],[
                                AC_MSG_RESULT(yes)
                            ],[
                                AC_MSG_RESULT(no)
                                AC_MSG_ERROR([CUDA support is requested but no CUDA-aware UCC could be found.])
                            ]
                         )
        AC_LANG_POP([C])
    ],[
        AC_MSG_WARN([ucc_info not found. Cannot check for CUDA awareness of UCC.])
    ])
])
])dnl end AM_COND_IF(BUILD_UCC)

])dnl end _BODY

[#] end of __file__
