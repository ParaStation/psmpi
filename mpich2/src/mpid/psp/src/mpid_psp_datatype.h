/*
 * ParaStation
 *
 * Copyright (C) 2006-2010 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#ifndef _MPID_PSP_DATATYPE_H_
#define _MPID_PSP_DATATYPE_H_

#include "mpidimpl.h"

typedef struct MPID_PSP_Datatype_info_s
{
	MPI_Datatype	datatype;
	MPID_Datatype	*dtp;
	unsigned	encode_size;
	int		is_predefined; /* == is_builtin || (MPI_FLOAT_INT and co.) */
} MPID_PSP_Datatype_info;


/*
 * Get info about datatype. (initialize *info).
 */
void MPID_PSP_Datatype_get_info(MPI_Datatype datatype, MPID_PSP_Datatype_info *info);


static inline
int MPID_is_predefined_datatype(MPI_Datatype datatype)
{
	return (HANDLE_GET_KIND(datatype) == HANDLE_KIND_BUILTIN) ||
		(datatype == MPI_FLOAT_INT) ||
		(datatype == MPI_DOUBLE_INT) ||
		(datatype == MPI_LONG_INT) ||
		(datatype == MPI_SHORT_INT) ||
		(datatype == MPI_LONG_DOUBLE_INT);
}

/* This is only slightly different to the macro in  mpid/common/datatype/mpid_datatype.h
   in that it does not return MPI_DATATYPE_NULL for composed types like MPI_DOUBLE_INT */
#define MPID_PSP_Datatype_get_basic_type(a,basic_type_) do {            \
    void *ptr;                                                          \
    switch (HANDLE_GET_KIND(a)) {                                       \
        case HANDLE_KIND_DIRECT:                                        \
            ptr = MPID_Datatype_direct+HANDLE_INDEX(a);                 \
            basic_type_ = ((MPID_Datatype *) ptr)->basic_type;          \
            break;                                                      \
        case HANDLE_KIND_INDIRECT:                                      \
            ptr = ((MPID_Datatype *)                                    \
                   MPIU_Handle_get_ptr_indirect(a,&MPID_Datatype_mem)); \
            basic_type_ = ((MPID_Datatype *) ptr)->basic_type;          \
            break;                                                      \
        case HANDLE_KIND_BUILTIN:                                       \
            basic_type_ = a;                                            \
            break;                                                      \
        case HANDLE_KIND_INVALID:                                       \
        default:                                                        \
            basic_type_ = 0;                                            \
            break;                                                      \
    }                                                                   \
} while(0)


static inline
int MPID_PSP_Datatype_is_contig(MPID_PSP_Datatype_info *info)
{
	return !info->dtp || info->dtp->is_contig;
}


/*
 * get the size required to encode the datatype described by info.
 * Use like:
 * encode = malloc(MPID_PSP_Datatype_get_size(info));
 * MPID_PSP_Datatype_encode(info, encode);
 */
static inline
unsigned int MPID_PSP_Datatype_get_size(MPID_PSP_Datatype_info *info)
{
	return info->encode_size;
}


/*
 * Encode the (MPI_Datatype)datatype to *encode. Caller has to allocate at least
 * info->encode_size bytes at encode.
 */
void MPID_PSP_Datatype_encode(MPID_PSP_Datatype_info *info, void *encode);


/*
 * Create a new (MPI_Datatype)new_datatype with refcnt 1. Caller has to call
 * MPID_PSP_Datatype_release(new_datatype) after usage.
 */
MPI_Datatype MPID_PSP_Datatype_decode(void *encode);


void MPID_PSP_Datatype_release(MPI_Datatype datatype);
void MPID_PSP_Datatype_add_ref(MPI_Datatype datatype);

#endif /* _MPID_PSP_DATATYPE_H_ */
