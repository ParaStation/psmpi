/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef TSP_GENTRAN_TYPES_H_INCLUDED
#define TSP_GENTRAN_TYPES_H_INCLUDED

#include "utarray.h"

typedef enum {
    MPII_GENUTIL_VTX_KIND__ISEND,
    MPII_GENUTIL_VTX_KIND__IRECV,
    MPII_GENUTIL_VTX_KIND__IMCAST,
    MPII_GENUTIL_VTX_KIND__ISSEND,
    MPII_GENUTIL_VTX_KIND__REDUCE_LOCAL,
    MPII_GENUTIL_VTX_KIND__LOCALCOPY,
    MPII_GENUTIL_VTX_KIND__SELECTIVE_SINK,
    MPII_GENUTIL_VTX_KIND__SINK,
    MPII_GENUTIL_VTX_KIND__FENCE,
    MPII_GENUTIL_VTX_KIND__LAST,        /* marks the last built-in kind */
} MPII_Genutil_vtx_kind_e;

typedef enum {
    MPII_GENUTIL_VTX_STATE__INIT,
    MPII_GENUTIL_VTX_STATE__ISSUED,
    MPII_GENUTIL_VTX_STATE__COMPLETE,
} MPII_Genutil_vtx_state_e;

typedef struct MPII_Genutil_vtx_t {
    int vtx_kind;
    int vtx_state;
    int vtx_id;

    UT_array *in_vtcs;
    UT_array *out_vtcs;

    int pending_dependencies;

    union {
        struct {
            const void *buf;
            int count;
            MPI_Datatype dt;
            int dest;
            int tag;
            MPIR_Comm *comm;
            MPIR_Request *req;
        } isend;
        struct {
            void *buf;
            int count;
            MPI_Datatype dt;
            int src;
            int tag;
            MPIR_Comm *comm;
            MPIR_Request *req;
        } irecv;
        struct {
            const void *buf;
            int count;
            MPI_Datatype dt;
            UT_array *dests;
            int num_dests;
            int tag;
            MPIR_Comm *comm;
            MPIR_Request **req;
            int last_complete;
        } imcast;
        struct {
            const void *buf;
            int count;
            MPI_Datatype dt;
            int dest;
            int tag;
            MPIR_Comm *comm;
            MPIR_Request *req;
        } issend;
        struct {
            const void *inbuf;
            void *inoutbuf;
            int count;
            MPI_Datatype datatype;
            MPI_Op op;
        } reduce_local;
        struct {
            const void *sendbuf;
            MPI_Aint sendcount;
            MPI_Datatype sendtype;
            void *recvbuf;
            MPI_Aint recvcount;
            MPI_Datatype recvtype;
        } localcopy;
        struct {
            void *data;
        } generic;
    } u;

    struct MPII_Genutil_vtx_t *next;
} MPII_Genutil_vtx_t;

typedef struct {
    UT_array *vtcs;
    int total_vtcs;
    int completed_vtcs;
    int last_fence;

    /* array of buffers allocated for schedule execution */
    UT_array *buffers;

    /* issued vertices linked list */
    struct MPII_Genutil_vtx_t *issued_head;
    struct MPII_Genutil_vtx_t *issued_tail;

    /* list of new type */
    UT_array generic_types;
} MPII_Genutil_sched_t;

typedef MPII_Genutil_vtx_t vtx_t;

typedef int (*MPII_Genutil_sched_issue_fn) (MPII_Genutil_vtx_t * vtxp, int *done);
typedef int (*MPII_Genutil_sched_complete_fn) (MPII_Genutil_vtx_t * vtxp, int *is_completed);
typedef int (*MPII_Genutil_sched_free_fn) (MPII_Genutil_vtx_t * vtxp);

typedef struct {
    int id;
    MPII_Genutil_sched_issue_fn issue_fn;
    MPII_Genutil_sched_complete_fn complete_fn;
    MPII_Genutil_sched_free_fn free_fn;
} MPII_Genutil_vtx_type_t;

#endif /* TSP_GENTRAN_TYPES_H_INCLUDED */