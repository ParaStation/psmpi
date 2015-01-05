/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2014 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 * This file is automatically generated by buildiface
 * DO NOT EDIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <ISO_Fortran_binding.h>
#include <mpi.h>

extern int cdesc_create_datatype(CFI_cdesc_t *cdesc, int oldcount, MPI_Datatype oldtype, MPI_Datatype *newtype);
extern int MPIR_Fortran_array_of_string_f2c(const char* strs_f, char*** strs_c, int str_len, int know_size, int size);
extern int MPIR_Comm_spawn_c(const char *command, char *argv_f, int maxprocs, MPI_Info info, int root,
        MPI_Comm comm, MPI_Comm *intercomm, int* array_of_errcodes, int argv_elem_len);
extern int MPIR_Comm_spawn_multiple_c(int count, char *array_of_commands_f,
        char *array_of_argv_f, const int* array_of_maxprocs,
        const MPI_Info *array_of_info, int root, MPI_Comm comm,
        MPI_Comm *intercomm, int* array_of_errcodes,
        int commands_elem_len, int argv_elem_len);
extern int MPIR_F_sync_reg_cdesc(CFI_cdesc_t* buf);

extern int MPIR_Send_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5);
extern int MPIR_Recv_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5, MPI_Status * x6);
extern int MPIR_Bsend_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5);
extern int MPIR_Ssend_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5);
extern int MPIR_Rsend_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5);
extern int MPIR_Buffer_attach_cdesc(CFI_cdesc_t* x0, int x1);
extern int MPIR_Isend_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Ibsend_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Issend_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Irsend_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Irecv_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Send_init_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Bsend_init_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Ssend_init_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Rsend_init_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Recv_init_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Sendrecv_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, CFI_cdesc_t* x5, int x6, MPI_Datatype x7, int x8, int x9, MPI_Comm x10, MPI_Status * x11);
extern int MPIR_Sendrecv_replace_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, int x5, int x6, MPI_Comm x7, MPI_Status * x8);
extern int MPIR_Pack_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, int * x5, MPI_Comm x6);
extern int MPIR_Unpack_cdesc(CFI_cdesc_t* x0, int x1, int * x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, MPI_Comm x6);
extern int MPIR_Bcast_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, MPI_Comm x4);
extern int MPIR_Gather_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, int x6, MPI_Comm x7);
extern int MPIR_Gatherv_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, const int * x4, const int * x5, MPI_Datatype x6, int x7, MPI_Comm x8);
extern int MPIR_Scatter_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, int x6, MPI_Comm x7);
extern int MPIR_Scatterv_cdesc(CFI_cdesc_t* x0, const int * x1, const int * x2, MPI_Datatype x3, CFI_cdesc_t* x4, int x5, MPI_Datatype x6, int x7, MPI_Comm x8);
extern int MPIR_Allgather_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, MPI_Comm x6);
extern int MPIR_Allgatherv_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, const int * x4, const int * x5, MPI_Datatype x6, MPI_Comm x7);
extern int MPIR_Alltoall_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, MPI_Comm x6);
extern int MPIR_Alltoallv_cdesc(CFI_cdesc_t* x0, const int * x1, const int * x2, MPI_Datatype x3, CFI_cdesc_t* x4, const int * x5, const int * x6, MPI_Datatype x7, MPI_Comm x8);
extern int MPIR_Alltoallw_cdesc(CFI_cdesc_t* x0, const int x1[], const int x2[], const MPI_Datatype x3[], CFI_cdesc_t* x4, const int x5[], const int x6[], const MPI_Datatype x7[], MPI_Comm x8);
extern int MPIR_Exscan_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Op x4, MPI_Comm x5);
extern int MPIR_Reduce_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Op x4, int x5, MPI_Comm x6);
extern int MPIR_Allreduce_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Op x4, MPI_Comm x5);
extern int MPIR_Reduce_scatter_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, const int x2[], MPI_Datatype x3, MPI_Op x4, MPI_Comm x5);
extern int MPIR_Scan_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Op x4, MPI_Comm x5);
extern int MPIR_Accumulate_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, MPI_Aint x4, int x5, MPI_Datatype x6, MPI_Op x7, MPI_Win x8);
extern int MPIR_Get_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, MPI_Aint x4, int x5, MPI_Datatype x6, MPI_Win x7);
extern int MPIR_Put_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, MPI_Aint x4, int x5, MPI_Datatype x6, MPI_Win x7);
extern int MPIR_Win_create_cdesc(CFI_cdesc_t* x0, MPI_Aint x1, int x2, MPI_Info x3, MPI_Comm x4, MPI_Win * x5);
extern int MPIR_Win_attach_cdesc(MPI_Win x0, CFI_cdesc_t* x1, MPI_Aint x2);
extern int MPIR_Win_detach_cdesc(MPI_Win x0, CFI_cdesc_t* x1);
extern int MPIR_Get_accumulate_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, int x6, MPI_Aint x7, int x8, MPI_Datatype x9, MPI_Op x10, MPI_Win x11);
extern int MPIR_Fetch_and_op_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, MPI_Datatype x2, int x3, MPI_Aint x4, MPI_Op x5, MPI_Win x6);
extern int MPIR_Compare_and_swap_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, CFI_cdesc_t* x2, MPI_Datatype x3, int x4, MPI_Aint x5, MPI_Win x6);
extern int MPIR_Rput_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, MPI_Aint x4, int x5, MPI_Datatype x6, MPI_Win x7, MPI_Request * x8);
extern int MPIR_Rget_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, MPI_Aint x4, int x5, MPI_Datatype x6, MPI_Win x7, MPI_Request * x8);
extern int MPIR_Raccumulate_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, MPI_Aint x4, int x5, MPI_Datatype x6, MPI_Op x7, MPI_Win x8, MPI_Request * x9);
extern int MPIR_Rget_accumulate_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, int x6, MPI_Aint x7, int x8, MPI_Datatype x9, MPI_Op x10, MPI_Win x11, MPI_Request * x12);
extern int MPIR_Free_mem_cdesc(CFI_cdesc_t* x0);
extern int MPIR_Get_address_cdesc(CFI_cdesc_t* x0, MPI_Aint * x1);
extern int MPIR_Pack_external_cdesc(const char x0[], CFI_cdesc_t* x1, int x2, MPI_Datatype x3, CFI_cdesc_t* x4, MPI_Aint x5, MPI_Aint * x6);
extern int MPIR_Unpack_external_cdesc(const char x0[], CFI_cdesc_t* x1, MPI_Aint x2, MPI_Aint * x3, CFI_cdesc_t* x4, int x5, MPI_Datatype x6);
extern int MPIR_Reduce_local_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Op x4);
extern int MPIR_Reduce_scatter_block_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Op x4, MPI_Comm x5);
extern int MPIR_Imrecv_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, MPI_Message * x3, MPI_Request * x4);
extern int MPIR_Mrecv_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, MPI_Message * x3, MPI_Status * x4);
extern int MPIR_Ibcast_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, MPI_Comm x4, MPI_Request * x5);
extern int MPIR_Igather_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, int x6, MPI_Comm x7, MPI_Request * x8);
extern int MPIR_Igatherv_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, const int x4[], const int x5[], MPI_Datatype x6, int x7, MPI_Comm x8, MPI_Request * x9);
extern int MPIR_Iscatter_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, int x6, MPI_Comm x7, MPI_Request * x8);
extern int MPIR_Iscatterv_cdesc(CFI_cdesc_t* x0, const int x1[], const int x2[], MPI_Datatype x3, CFI_cdesc_t* x4, int x5, MPI_Datatype x6, int x7, MPI_Comm x8, MPI_Request * x9);
extern int MPIR_Iallgather_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, MPI_Comm x6, MPI_Request * x7);
extern int MPIR_Iallgatherv_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, const int x4[], const int x5[], MPI_Datatype x6, MPI_Comm x7, MPI_Request * x8);
extern int MPIR_Ialltoall_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, MPI_Comm x6, MPI_Request * x7);
extern int MPIR_Ialltoallv_cdesc(CFI_cdesc_t* x0, const int x1[], const int x2[], MPI_Datatype x3, CFI_cdesc_t* x4, const int x5[], const int x6[], MPI_Datatype x7, MPI_Comm x8, MPI_Request * x9);
extern int MPIR_Ialltoallw_cdesc(CFI_cdesc_t* x0, const int x1[], const int x2[], const MPI_Datatype x3[], CFI_cdesc_t* x4, const int x5[], const int x6[], const MPI_Datatype x7[], MPI_Comm x8, MPI_Request * x9);
extern int MPIR_Ireduce_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Op x4, int x5, MPI_Comm x6, MPI_Request * x7);
extern int MPIR_Iallreduce_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Op x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Ireduce_scatter_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, const int x2[], MPI_Datatype x3, MPI_Op x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Ireduce_scatter_block_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Op x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Iscan_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Op x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Iexscan_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Op x4, MPI_Comm x5, MPI_Request * x6);
extern int MPIR_Ineighbor_allgather_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, MPI_Comm x6, MPI_Request * x7);
extern int MPIR_Ineighbor_allgatherv_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, const int x4[], const int x5[], MPI_Datatype x6, MPI_Comm x7, MPI_Request * x8);
extern int MPIR_Ineighbor_alltoall_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, MPI_Comm x6, MPI_Request * x7);
extern int MPIR_Ineighbor_alltoallv_cdesc(CFI_cdesc_t* x0, const int x1[], const int x2[], MPI_Datatype x3, CFI_cdesc_t* x4, const int x5[], const int x6[], MPI_Datatype x7, MPI_Comm x8, MPI_Request * x9);
extern int MPIR_Ineighbor_alltoallw_cdesc(CFI_cdesc_t* x0, const int x1[], const MPI_Aint x2[], const MPI_Datatype x3[], CFI_cdesc_t* x4, const int x5[], const MPI_Aint x6[], const MPI_Datatype x7[], MPI_Comm x8, MPI_Request * x9);
extern int MPIR_Neighbor_allgather_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, MPI_Comm x6);
extern int MPIR_Neighbor_allgatherv_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, const int x4[], const int x5[], MPI_Datatype x6, MPI_Comm x7);
extern int MPIR_Neighbor_alltoall_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, CFI_cdesc_t* x3, int x4, MPI_Datatype x5, MPI_Comm x6);
extern int MPIR_Neighbor_alltoallv_cdesc(CFI_cdesc_t* x0, const int x1[], const int x2[], MPI_Datatype x3, CFI_cdesc_t* x4, const int x5[], const int x6[], MPI_Datatype x7, MPI_Comm x8);
extern int MPIR_Neighbor_alltoallw_cdesc(CFI_cdesc_t* x0, const int x1[], const MPI_Aint x2[], const MPI_Datatype x3[], CFI_cdesc_t* x4, const int x5[], const MPI_Aint x6[], const MPI_Datatype x7[], MPI_Comm x8);
extern int MPIR_File_read_at_cdesc(MPI_File x0, MPI_Offset x1, CFI_cdesc_t* x2, int x3, MPI_Datatype x4, MPI_Status * x5);
extern int MPIR_File_read_at_all_cdesc(MPI_File x0, MPI_Offset x1, CFI_cdesc_t* x2, int x3, MPI_Datatype x4, MPI_Status * x5);
extern int MPIR_File_write_at_cdesc(MPI_File x0, MPI_Offset x1, CFI_cdesc_t* x2, int x3, MPI_Datatype x4, MPI_Status * x5);
extern int MPIR_File_write_at_all_cdesc(MPI_File x0, MPI_Offset x1, CFI_cdesc_t* x2, int x3, MPI_Datatype x4, MPI_Status * x5);
extern int MPIR_File_iread_at_cdesc(MPI_File x0, MPI_Offset x1, CFI_cdesc_t* x2, int x3, MPI_Datatype x4, MPIO_Request * x5);
extern int MPIR_File_iwrite_at_cdesc(MPI_File x0, MPI_Offset x1, CFI_cdesc_t* x2, int x3, MPI_Datatype x4, MPIO_Request * x5);
extern int MPIR_File_read_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Status * x4);
extern int MPIR_File_read_all_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Status * x4);
extern int MPIR_File_write_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Status * x4);
extern int MPIR_File_write_all_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Status * x4);
extern int MPIR_File_iread_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPIO_Request * x4);
extern int MPIR_File_iwrite_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPIO_Request * x4);
extern int MPIR_File_read_shared_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Status * x4);
extern int MPIR_File_write_shared_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Status * x4);
extern int MPIR_File_iread_shared_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPIO_Request * x4);
extern int MPIR_File_iwrite_shared_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPIO_Request * x4);
extern int MPIR_File_read_ordered_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Status * x4);
extern int MPIR_File_write_ordered_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Status * x4);
extern int MPIR_File_read_at_all_begin_cdesc(MPI_File x0, MPI_Offset x1, CFI_cdesc_t* x2, int x3, MPI_Datatype x4);
extern int MPIR_File_read_at_all_end_cdesc(MPI_File x0, CFI_cdesc_t* x1, MPI_Status * x2);
extern int MPIR_File_write_at_all_begin_cdesc(MPI_File x0, MPI_Offset x1, CFI_cdesc_t* x2, int x3, MPI_Datatype x4);
extern int MPIR_File_write_at_all_end_cdesc(MPI_File x0, CFI_cdesc_t* x1, MPI_Status * x2);
extern int MPIR_File_read_all_begin_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3);
extern int MPIR_File_read_all_end_cdesc(MPI_File x0, CFI_cdesc_t* x1, MPI_Status * x2);
extern int MPIR_File_write_all_begin_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3);
extern int MPIR_File_write_all_end_cdesc(MPI_File x0, CFI_cdesc_t* x1, MPI_Status * x2);
extern int MPIR_File_read_ordered_begin_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3);
extern int MPIR_File_read_ordered_end_cdesc(MPI_File x0, CFI_cdesc_t* x1, MPI_Status * x2);
extern int MPIR_File_write_ordered_begin_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3);
extern int MPIR_File_write_ordered_end_cdesc(MPI_File x0, CFI_cdesc_t* x1, MPI_Status * x2);
