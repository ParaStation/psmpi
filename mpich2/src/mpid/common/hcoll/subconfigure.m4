[#] start of __file__

AC_DEFUN([PAC_SUBCFG_PREREQ_]PAC_SUBCFG_AUTO_SUFFIX,[
	if test "${with_hcoll}" = "no" ; then
		have_hcoll=no;
	else
		if test "${with_hcoll}" = "yes" ; then
			with_hcoll="/opt/mellanox/hcoll"
		fi
		PAC_SET_HEADER_LIB_PATH(hcoll)
		PAC_PUSH_FLAG(LIBS)
		export LIBRARY_PATH=$LIBRARY_PATH:${with_hcoll}/lib:${with_hcoll}/../sharp/lib
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${with_hcoll}/lib:${with_hcoll}/../sharp/lib
		PAC_CHECK_HEADER_LIB([hcoll/api/hcoll_api.h],[hcoll],[hcoll_init],[have_hcoll=yes],[have_hcoll=no])
		if test "${have_hcoll}" = "yes" ; then
		   PAC_APPEND_FLAG([-lhcoll],[WRAPPER_LIBS])
		elif test -n "${with_hcoll}" -o -n "${with_hcoll_lib}" -o -n "${with_hcoll_include}" ; then
		   AC_MSG_ERROR(['hcoll/api/hcoll_api.h or libhcoll library not found.'])
		fi
		PAC_POP_FLAG(LIBS)
	fi
	AM_CONDITIONAL([BUILD_HCOLL],[test "$have_hcoll" = "yes"])
])dnl end PREREQ

AC_DEFUN([PAC_SUBCFG_BODY_]PAC_SUBCFG_AUTO_SUFFIX,[
# nothing to do
])dnl end _BODY

[#] end of __file__
