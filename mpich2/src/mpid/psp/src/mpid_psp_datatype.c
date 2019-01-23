/*
 * ParaStation
 *
 * Copyright (C) 2006-2019 ParTec Cluster Competence Center GmbH, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include "mpid_psp_datatype.h"

// This must be the last include before sysmbols are defined:
#include "mpid_visibility.h"


/* private struct for en-/de-coding of MPI_Datatype */
typedef struct MPID_PSP_Datatype_s
{
	MPI_Datatype	datatype;	/* must be the first! */
	MPI_Aint	max_contig_blocks;
	MPI_Aint	size;
	MPI_Aint	extent;
	MPI_Aint	dataloop_size;
	void		*dataloop;  /* Used with Dataloop_update(enc_dataloop, ptrdiff) on remote */
	int		dataloop_depth;
	int		basic_type;
	MPI_Aint	ub, lb, true_ub, true_lb; /* Old MPI1 stuff */

	/* boolean fields: */
	unsigned int	is_contig : 1;
	unsigned int	has_sticky_ub : 1;
	unsigned int	has_sticky_lb : 1;
} MPID_PSP_Datatype;


void MPID_PSP_Datatype_get_info(MPI_Datatype datatype, MPID_PSP_Datatype_info *info)
{
	info->datatype = datatype;

/*	fprintf(stderr, "LINE %d datatype %08x Handlekind: %d\n", __LINE__, datatype, HANDLE_GET_KIND(datatype)); */
	if (HANDLE_GET_KIND(datatype) == HANDLE_KIND_BUILTIN) {
		info->dtp = NULL;
		info->encode_size = (sizeof(datatype) + 7) & (~7);
		info->is_predefined = 1;
		return;
	} else {
		MPIR_Datatype  *dtp;

		MPIR_Datatype_get_ptr(datatype, dtp);
		if (dtp->is_contig) {
			/* ToDo: optimize the case where dtp->is_contig is true: (encode only dtp->size and dtp->true_lb) */
			info->dtp = dtp;
			if (!MPID_is_predefined_datatype(datatype)) {
				info->encode_size = (sizeof(MPID_PSP_Datatype) + info->dtp->dataloop_size + 7) & ~7;
				info->is_predefined = 0;
			} else {
				info->encode_size = (sizeof(datatype) + 7) & (~7);
				info->is_predefined = 1;
			}
		} else {
			info->dtp = dtp;
			info->encode_size = (sizeof(MPID_PSP_Datatype) + info->dtp->dataloop_size + 7) & ~7;
			info->is_predefined = 0;
		}
	}
}


/*
 * Encode the (MPI_Datatype)datatype to *encode. Caller has to allocate at least
 * info->encode_size bytes at encode.
 */
void MPID_PSP_Datatype_encode(MPID_PSP_Datatype_info *info, void *encode)
{
	MPIR_Datatype  *dtp = info->dtp;
	MPID_PSP_Datatype *enc_dtp = (MPID_PSP_Datatype *) encode;

	struct MPIR_Dataloop *enc_dataloop;

	enc_dtp->datatype = info->datatype;

	if (info->is_predefined) {
		return;
	}

	/* Copy datatype info */
	enc_dtp->max_contig_blocks = dtp->max_contig_blocks;
	enc_dtp->size		= dtp->size;
	enc_dtp->extent		= dtp->extent;
	enc_dtp->dataloop_size	= dtp->dataloop_size;
	enc_dtp->dataloop	= dtp->dataloop;
	enc_dtp->dataloop_depth	= dtp->dataloop_depth;
	enc_dtp->basic_type	= dtp->basic_type;
	enc_dtp->ub		= dtp->ub;
	enc_dtp->lb		= dtp->lb;
	enc_dtp->true_ub	= dtp->true_ub;
	enc_dtp->true_lb	= dtp->true_lb;
	/* boolean fields */
	enc_dtp->is_contig	= dtp->is_contig;
	enc_dtp->has_sticky_ub	= dtp->has_sticky_ub;
	enc_dtp->has_sticky_lb	= dtp->has_sticky_lb;

	/* Copy dataloop */
	enc_dataloop =	(struct MPIR_Dataloop *)(void *)(enc_dtp + 1); /* dataloop behind enc_dtp */
	memcpy(enc_dataloop, dtp->dataloop, dtp->dataloop_size);

	return;
}


/*
 * Create a new (MPI_Datatype)new_datatype with refcnt 1. Caller has to call
 * MPIR_Datatype_ptr_release(new_datatype) after usage, if
 * HANDLE_GET_KIND(datatype) != HANDLE_KIND_BUILTIN.
 */
MPI_Datatype MPID_PSP_Datatype_decode(void *encode)
{
	MPIR_Datatype  *dtp;

	MPID_PSP_Datatype *enc_dtp = (MPID_PSP_Datatype *) encode;
	struct MPIR_Dataloop *enc_dataloop;

	if (MPID_is_predefined_datatype(enc_dtp->datatype))
		/* HANDLE_GET_KIND(enc_dtp->datatype) == HANDLE_KIND_BUILTIN)*/ {
		/* nothing to do. */
/*		fprintf(stderr, "#%2d Decode built-in type %08x : %s\n", MPIDI_Process.my_pg_rank,
			enc_dtp->datatype, MPIDU_Datatype_builtin_to_string(enc_dtp->datatype) ? : "<NULL>"); */
		return enc_dtp->datatype;
	}
/*	fprintf(stderr, "#%2d Decode user     type %08x : %s\n", MPIDI_Process.my_pg_rank,
		enc_dtp->datatype, MPIDU_Datatype_builtin_to_string(enc_dtp->datatype) ? : "<NULL>"); */
	/* ToDo: optimize the case where dtp->is_contig is true: (encode only dtp->size and dtp->true_lb) */

	dtp = (MPIR_Datatype *) MPIR_Handle_obj_alloc(&MPIR_Datatype_mem);
	if (!dtp) goto err_alloc_dtp;

	/* Note: handle is filled in by MPIR_Handle_obj_alloc() */
	MPIR_Object_set_ref(dtp, 1);

	dtp->size		= enc_dtp->size;
	dtp->extent		= enc_dtp->extent;
	dtp->ub			= enc_dtp->ub;
	dtp->lb			= enc_dtp->lb;
	dtp->true_ub		= enc_dtp->true_ub;
	dtp->true_lb		= enc_dtp->true_lb;

	dtp->alignsize		= 0;	/* ToDo: Correct value ??? */

	dtp->has_sticky_ub	= enc_dtp->has_sticky_ub;
	dtp->has_sticky_lb	= enc_dtp->has_sticky_lb;

	dtp->is_permanent	= 0;
	dtp->is_committed	= 1;

	dtp->basic_type		= enc_dtp->basic_type;
	dtp->n_builtin_elements = 0;	/* ToDo: Correct value ??? */
	dtp->builtin_element_size = 0;	/* ToDo: Correct value ??? */

	dtp->is_contig		= enc_dtp->is_contig;
	dtp->max_contig_blocks	= enc_dtp->max_contig_blocks;

	dtp->contents		= NULL;

	/* todo: dataloop todo!!! */
	/* set dataloop pointer */
	/* dtp->dataloop = req->dev.dataloop; */
	dtp->dataloop_size	= enc_dtp->dataloop_size;
	dtp->dataloop_depth	= enc_dtp->dataloop_depth;

	dtp->attributes		= 0;
	dtp->name[0]		= 0;
	dtp->cache_id		= 0;


	/*
	 * Copy and update dataloop:
	 */

	enc_dataloop =	(struct MPIR_Dataloop *)(void *)(enc_dtp + 1); /* dataloop behind enc_dtp */
	/* dtp->dataloop will be freed in MPIR_Datatype_free() with MPIR_Dataloop_free(&(ptr->dataloop));
	 * DLOOP_Malloc() == MPL_malloc().
	 */
	dtp->dataloop = (struct MPIR_Dataloop *) MPL_malloc(dtp->dataloop_size, MPL_MEM_DATATYPE);
	if (!dtp->dataloop) goto err_alloc_dataloop;

	memcpy(dtp->dataloop, enc_dataloop, dtp->dataloop_size);

	MPIR_Dataloop_update(dtp->dataloop, ((char *) dtp->dataloop) - ((char *)enc_dtp->dataloop));


	return dtp->handle;

	/* --- */
err_alloc_dataloop:
	MPIR_Datatype_ptr_release(dtp); dtp = NULL;

err_alloc_dtp:
	{
		static int warn = 0;
		if (!warn) {
			fprintf(stderr, "Warning: unhandled error in " __FILE__ ":%d", __LINE__);
			warn = 1;
		}
	}
	return 0;
}


void MPID_PSP_Datatype_release(MPI_Datatype datatype)
{
	if(!MPID_is_predefined_datatype(datatype)) {
		MPIR_Datatype  *dtp;
		MPIR_Datatype_get_ptr(datatype, dtp);
		MPIR_Datatype_ptr_release(dtp);
	}
}


void MPID_PSP_Datatype_add_ref(MPI_Datatype datatype)
{
	if(!MPID_is_predefined_datatype(datatype)) {
		MPIR_Datatype  *dtp;
		MPIR_Datatype_get_ptr(datatype, dtp);
		// TODO: check if MPID_PSP_Datatype_add_ref() is required at all.
		MPIR_Datatype_ptr_add_ref(dtp);
	}
}
