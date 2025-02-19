/*
 * ParaStation
 *
 * Copyright (C) 2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#ifndef _MPID_WIN_INFO_H_
#define _MPID_WIN_INFO_H_

/* sub-struct to become part of win struct via MPID_DEV_WIN_DECL */
struct MPIDI_PSP_Win_info_args {
    int no_locks;
    int accumulate_ordering;
    int accumulate_ops;
    int mpi_accumulate_granularity;
    int same_size;
    int same_disp_unit;
    int alloc_shared_noncontig;
    int wait_on_passive_side;
};

/* special values used for window arguments */
enum MPIDI_PSP_Win_info_args_special_values {
    MPIDI_PSP_WIN_INFO_ARG_invalid = -2,
    MPIDI_PSP_WIN_INFO_ARG_unset = -1,
    MPIDI_PSP_WIN_INFO_ARG_false = 0,
    MPIDI_PSP_WIN_INFO_ARG_true = 1,
};
#define MPIDI_PSP_WIN_INFO_ARG_IS_UNSET(x)	\
    (x == MPIDI_PSP_WIN_INFO_ARG_unset)
#define MPIDI_PSP_WIN_INFO_ARG_IS_VALID(x)	\
    (x != MPIDI_PSP_WIN_INFO_ARG_invalid)
#define MPIDI_PSP_WIN_INFO_ARG_IS_SET_AND_VALID(x)	\
    ((x != MPIDI_PSP_WIN_INFO_ARG_unset) &&		\
      (x != MPIDI_PSP_WIN_INFO_ARG_invalid))

/* possible values for accumulate_ordering */
enum MPIDI_PSP_Win_info_arg_vals_accumulate_ordering {
    MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_none = 0,
    MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_rar = 1,
    MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_raw = 2,
    MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_war = 4,
    MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_waw = 8,
};
#define  MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_all (MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_rar | \
							 MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_raw | \
							 MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_war | \
							 MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_waw)

/* possible values for accumulate_ops */
enum MPIDI_PSP_Win_info_arg_vals_accumulate_ops {
    MPIDI_PSP_WIN_INFO_ARG_accumulate_ops_same_op_no_op = 0,
    MPIDI_PSP_WIN_INFO_ARG_accumulate_ops_same_op = 1,
};

/* macro to set up an enum for common true/false values */
#define MPIDI_PSP_WIN_INFO_ARG_VALS_ENUM_TRUE_FALSE(_arg)		\
    enum MPIDI_PSP_Win_info_arg_vals_ ## _arg {				\
	MPIDI_PSP_WIN_INFO_ARG_ ## _arg ## _true =			\
	    MPIDI_PSP_WIN_INFO_ARG_true,				\
        MPIDI_PSP_WIN_INFO_ARG_ ## _arg ## _false =			\
	    MPIDI_PSP_WIN_INFO_ARG_false,				\
    };
/* use the macro above to define values for further window arguments */
MPIDI_PSP_WIN_INFO_ARG_VALS_ENUM_TRUE_FALSE(no_locks);
MPIDI_PSP_WIN_INFO_ARG_VALS_ENUM_TRUE_FALSE(same_size);
MPIDI_PSP_WIN_INFO_ARG_VALS_ENUM_TRUE_FALSE(same_disp_unit);
MPIDI_PSP_WIN_INFO_ARG_VALS_ENUM_TRUE_FALSE(alloc_shared_noncontig);

/* possible values for wait_on_passive_side (specific to PSP/psmpi) */
enum MPIDI_PSP_Win_info_arg_vals_wait_on_passive_side {
    MPIDI_PSP_WIN_INFO_ARG_wait_on_passive_side_none = 0,
    MPIDI_PSP_WIN_INFO_ARG_wait_on_passive_side_explicit = 1,
};

/* default values as defined by the standard for RMA info keys */
#define MPIDI_PSP_WIN_INFO_ARG_DEFAULT_no_locks           MPIDI_PSP_WIN_INFO_ARG_false
#define MPIDI_PSP_WIN_INFO_ARG_DEFAULT_accumulate_ordering MPIDI_PSP_WIN_INFO_ARG_accumulate_ordering_all
#define MPIDI_PSP_WIN_INFO_ARG_DEFAULT_accumulate_ops     MPIDI_PSP_WIN_INFO_ARG_accumulate_ops_same_op_no_op
#define MPIDI_PSP_WIN_INFO_ARG_DEFAULT_mpi_accumulate_granularity 0
#define MPIDI_PSP_WIN_INFO_ARG_DEFAULT_same_size          MPIDI_PSP_WIN_INFO_ARG_false
#define MPIDI_PSP_WIN_INFO_ARG_DEFAULT_same_disp_unit     MPIDI_PSP_WIN_INFO_ARG_false
#define MPIDI_PSP_WIN_INFO_ARG_DEFAULT_alloc_shared_noncontig MPIDI_PSP_WIN_INFO_ARG_false

/* values specific to PSP/psmpi */
#define MPIDI_PSP_WIN_INFO_ARG_DEFAULT_wait_on_passive_side MPIDI_PSP_WIN_INFO_ARG_wait_on_passive_side_explicit

/* macro for setting an info key/value pair */
#define MPIDI_PSP_INFO_SET(_info, _key, _val)                           \
    do {                                                                \
        mpi_errno = MPIR_Info_set_impl(_info, _key, _val);              \
        MPIR_ERR_CHECK(mpi_errno);                                      \
    } while (0)

/* macro for checking (_flag) and getting an info key (into _valbuf) */
#define MPIDI_PSP_INFO_GET(_info, _key, _valbuf, _flag)                 \
    do {                                                                \
        _flag = 0;                                                      \
        mpi_errno = MPIR_Info_get_impl(_info, _key, MPI_MAX_INFO_VAL,   \
                                       _valbuf, &_flag);                \
        MPIR_ERR_CHECK(mpi_errno);                                      \
    } while (0)

/* macro for applying an argument in win_info_args to the actual */
/* window parameters */
#define MPIDI_PSP_WIN_INFO_APPLY_ARG(_win_info_args, _key,              \
                                           _value, _fallback_cond)      \
    ((MPIDI_PSP_WIN_INFO_ARG_IS_SET_AND_VALID(_win_info_args._key)      \
         && (_win_info_args._key ==                                     \
             MPIDI_PSP_WIN_INFO_ARG_ ## _key ## _ ## _value)) ||        \
     (MPIDI_PSP_WIN_INFO_ARG_IS_UNSET(_win_info_args._key)              \
         && (_fallback_cond)))

/* macro for checking info and setting an argument in win_info_args */
#define MPIDI_PSP_WIN_INFO_GET_ARG(_win_info_args, _info, _key,         \
                                   _val1, _val2, _valbuf, _flag)        \
    do {                                                                \
        MPIDI_PSP_INFO_GET(_info, #_key, _valbuf, _flag);               \
        if (_flag) {                                                    \
            if (strcmp(_valbuf, #_val1) == 0) {                         \
                _win_info_args._key =                                   \
                    MPIDI_PSP_WIN_INFO_ARG_ ## _key ## _ ## _val1;      \
            } else if (strcmp(_valbuf, #_val2) == 0) {                  \
                _win_info_args._key =                                   \
                    MPIDI_PSP_WIN_INFO_ARG_ ## _key ## _ ## _val2;      \
            } else {                                                    \
                _win_info_args._key = MPIDI_PSP_WIN_INFO_ARG_invalid;   \
            }                                                           \
        }                                                               \
    } while (0)

/* macro for checking info and setting an integer value in win_info_args */
#define MPIDI_PSP_WIN_INFO_GET_ARG_INT(_win_info_args, _info, _key,     \
                                       _valbuf, _flag)                  \
    do {                                                                \
        MPIDI_PSP_INFO_GET(_info, #_key, _valbuf, _flag);               \
        if (_flag) {                                                    \
            errno = 0;                                                  \
            char *__endptr;                                             \
            int __value = strtol(_valbuf, &__endptr, 10);               \
            if ((*__endptr == '\0') && (errno == 0) &&                  \
                (__value >= 0)) {                                       \
                _win_info_args._key = __value;                          \
            } else {                                                    \
                _win_info_args._key = MPIDI_PSP_WIN_INFO_ARG_invalid;   \
            }                                                           \
        }                                                               \
    } while (0)

/* macro for checking win_info_args for a default value and setting a */
/* corresponding key/value pair in the given info object */
#define MPIDI_PSP_WIN_INFO_SET_ARG_DEFAULT(_win_info_args, _info, _key, \
                                           _val, ...)                   \
    do {                                                                \
        MPIR_Assert((int)MPIDI_PSP_WIN_INFO_ARG_DEFAULT_ ## _key ==     \
                    (int)MPIDI_PSP_WIN_INFO_ARG_ ## _key ## _ ## _val); \
        if ((_win_info_args._key == MPIDI_PSP_WIN_INFO_ARG_DEFAULT_     \
             ## _key) ||                                                \
            (MPIDI_PSP_WIN_INFO_ARG_IS_UNSET(_win_info_args._key))) {   \
            MPIR_Assert(MPIDI_PSP_WIN_INFO_ARG_IS_VALID(		\
                            _win_info_args._key));                      \
            MPIDI_PSP_INFO_SET(_info, #_key, __VA_ARGS__ #_val);        \
        }                                                               \
    } while (0)

/* macro for checking win_info_args for a specific value and setting a */
/* corresponding key/value pair in the given info object */
#define MPIDI_PSP_WIN_INFO_SET_ARG_SPECIFIC(_win_info_args, _info, _key,\
                                            _val, _condition, ...)      \
    do {                                                                \
        if ((_win_info_args._key == MPIDI_PSP_WIN_INFO_ARG_ ##_key ##   \
             _ ## _val) || (MPIDI_PSP_WIN_INFO_ARG_IS_UNSET(\
                                _win_info_args._key) && (_condition))) {\
            MPIR_Assert(MPIDI_PSP_WIN_INFO_ARG_IS_VALID(\
                            _win_info_args._key));                      \
            MPIDI_PSP_INFO_SET(_info, #_key, __VA_ARGS__ #_val);        \
        }                                                               \
    } while (0)

/* macro to set an alternative string value for a win_info_args member */
#define MPIDI_PSP_WIN_INFO_ARG_DIFFERENT(x) x "\0"

/* macro for checking win_info_args for a default integer value and */
/* setting a corresponding key/value pair in the given info object */
#define MPIDI_PSP_WIN_INFO_SET_ARG_INT_DEFAULT(_win_info_args, _info,   \
                                               _key, _default)          \
    do {                                                                \
        MPIR_Assert((int)MPIDI_PSP_WIN_INFO_ARG_DEFAULT_ ## _key ==     \
                    _default);                                          \
        if ((_win_info_args._key == _default) ||                        \
            (MPIDI_PSP_WIN_INFO_ARG_IS_UNSET(_win_info_args._key))) {   \
            MPIR_Assert(MPIDI_PSP_WIN_INFO_ARG_IS_VALID(		\
                            _win_info_args._key));                      \
            char __str[32];                                             \
            snprintf(__str, 32, "%d", _default);                        \
            MPIDI_PSP_INFO_SET(_info, #_key, __str);                    \
        }                                                               \
    } while (0)

#endif /* _MPID_WIN_INFO_H_ */
