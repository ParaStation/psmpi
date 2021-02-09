/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * DO NOT EDIT: AUTOMATICALLY GENERATED FILE !!
 */

#include <stdio.h>
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri_zei.h"
#include "yaksuri_zei_populate_pupfns.h"

int yaksuri_zei_populate_pupfns(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_zei_type_s *ze = (yaksuri_zei_type_s *) type->backend.ze.priv;
    
    ze->pack = YAKSURI_KERNEL_NULL;
    ze->unpack = YAKSURI_KERNEL_NULL;
    
    switch (type->kind) {
        case YAKSI_TYPE_KIND__HVECTOR:
        switch (type->u.hvector.child->kind) {
            case YAKSI_TYPE_KIND__HVECTOR:
            rc = yaksuri_zei_populate_pupfns_hvector_hvector(type);
            break;
            
            case YAKSI_TYPE_KIND__BLKHINDX:
            rc = yaksuri_zei_populate_pupfns_hvector_blkhindx(type);
            break;
            
            case YAKSI_TYPE_KIND__HINDEXED:
            rc = yaksuri_zei_populate_pupfns_hvector_hindexed(type);
            break;
            
            case YAKSI_TYPE_KIND__CONTIG:
            rc = yaksuri_zei_populate_pupfns_hvector_contig(type);
            break;
            
            case YAKSI_TYPE_KIND__RESIZED:
            rc = yaksuri_zei_populate_pupfns_hvector_resized(type);
            break;
            
            case YAKSI_TYPE_KIND__BUILTIN:
            rc = yaksuri_zei_populate_pupfns_hvector_builtin(type);
            break;
            
            default:
                break;
        }
        break;
        
        case YAKSI_TYPE_KIND__BLKHINDX:
        switch (type->u.blkhindx.child->kind) {
            case YAKSI_TYPE_KIND__HVECTOR:
            rc = yaksuri_zei_populate_pupfns_blkhindx_hvector(type);
            break;
            
            case YAKSI_TYPE_KIND__BLKHINDX:
            rc = yaksuri_zei_populate_pupfns_blkhindx_blkhindx(type);
            break;
            
            case YAKSI_TYPE_KIND__HINDEXED:
            rc = yaksuri_zei_populate_pupfns_blkhindx_hindexed(type);
            break;
            
            case YAKSI_TYPE_KIND__CONTIG:
            rc = yaksuri_zei_populate_pupfns_blkhindx_contig(type);
            break;
            
            case YAKSI_TYPE_KIND__RESIZED:
            rc = yaksuri_zei_populate_pupfns_blkhindx_resized(type);
            break;
            
            case YAKSI_TYPE_KIND__BUILTIN:
            rc = yaksuri_zei_populate_pupfns_blkhindx_builtin(type);
            break;
            
            default:
                break;
        }
        break;
        
        case YAKSI_TYPE_KIND__HINDEXED:
        switch (type->u.hindexed.child->kind) {
            case YAKSI_TYPE_KIND__HVECTOR:
            rc = yaksuri_zei_populate_pupfns_hindexed_hvector(type);
            break;
            
            case YAKSI_TYPE_KIND__BLKHINDX:
            rc = yaksuri_zei_populate_pupfns_hindexed_blkhindx(type);
            break;
            
            case YAKSI_TYPE_KIND__HINDEXED:
            rc = yaksuri_zei_populate_pupfns_hindexed_hindexed(type);
            break;
            
            case YAKSI_TYPE_KIND__CONTIG:
            rc = yaksuri_zei_populate_pupfns_hindexed_contig(type);
            break;
            
            case YAKSI_TYPE_KIND__RESIZED:
            rc = yaksuri_zei_populate_pupfns_hindexed_resized(type);
            break;
            
            case YAKSI_TYPE_KIND__BUILTIN:
            rc = yaksuri_zei_populate_pupfns_hindexed_builtin(type);
            break;
            
            default:
                break;
        }
        break;
        
        case YAKSI_TYPE_KIND__CONTIG:
        switch (type->u.contig.child->kind) {
            case YAKSI_TYPE_KIND__HVECTOR:
            rc = yaksuri_zei_populate_pupfns_contig_hvector(type);
            break;
            
            case YAKSI_TYPE_KIND__BLKHINDX:
            rc = yaksuri_zei_populate_pupfns_contig_blkhindx(type);
            break;
            
            case YAKSI_TYPE_KIND__HINDEXED:
            rc = yaksuri_zei_populate_pupfns_contig_hindexed(type);
            break;
            
            case YAKSI_TYPE_KIND__CONTIG:
            rc = yaksuri_zei_populate_pupfns_contig_contig(type);
            break;
            
            case YAKSI_TYPE_KIND__RESIZED:
            rc = yaksuri_zei_populate_pupfns_contig_resized(type);
            break;
            
            case YAKSI_TYPE_KIND__BUILTIN:
            rc = yaksuri_zei_populate_pupfns_contig_builtin(type);
            break;
            
            default:
                break;
        }
        break;
        
        case YAKSI_TYPE_KIND__RESIZED:
        switch (type->u.resized.child->kind) {
            case YAKSI_TYPE_KIND__HVECTOR:
            rc = yaksuri_zei_populate_pupfns_resized_hvector(type);
            break;
            
            case YAKSI_TYPE_KIND__BLKHINDX:
            rc = yaksuri_zei_populate_pupfns_resized_blkhindx(type);
            break;
            
            case YAKSI_TYPE_KIND__HINDEXED:
            rc = yaksuri_zei_populate_pupfns_resized_hindexed(type);
            break;
            
            case YAKSI_TYPE_KIND__CONTIG:
            rc = yaksuri_zei_populate_pupfns_resized_contig(type);
            break;
            
            case YAKSI_TYPE_KIND__RESIZED:
            rc = yaksuri_zei_populate_pupfns_resized_resized(type);
            break;
            
            case YAKSI_TYPE_KIND__BUILTIN:
            rc = yaksuri_zei_populate_pupfns_resized_builtin(type);
            break;
            
            default:
                break;
        }
        break;
        
        default:
            break;
    }
    
        return rc;
}
