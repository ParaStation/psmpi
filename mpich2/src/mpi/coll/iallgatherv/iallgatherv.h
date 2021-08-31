/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef IALLGATHERV_H_INCLUDED
#define IALLGATHERV_H_INCLUDED

#include "mpiimpl.h"

int MPII_Iallgatherv_is_displs_ordered(int size, const int recvcounts[], const int displs[]);

#endif /* IALLGATHERV_H_INCLUDED */
