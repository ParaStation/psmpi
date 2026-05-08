/*
 * ParaStation
 *
 * Copyright (C) 2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#ifndef _MPID_PSP_TOPO_H_
#define _MPID_PSP_TOPO_H_

#include "mpidimpl.h"

#if 0
/* Initialize topology information */
int MPIDI_PSP_topo_init(MPIDI_PSP_topo_level_t ** topo_levels);

/* Update the topology levels in comm */
int MPIDI_PSP_update_topo_level(MPIR_Comm * comm);

/* Check if a PG already knows a topo level of a specific degree */
int MPIDI_PSP_check_pg_for_level(int degree, MPIDI_PG_t * pg, MPIDI_PSP_topo_level_t ** level);

/* Return 1 if all processes in comm have the same badge than the calling process in a
 * given topology level, return 0 otherwise */
int MPIDI_PSP_comm_is_flat_on_level(MPIR_Comm * comm, MPIDI_PSP_topo_level_t * level);

/* To provide a data structure that supports any number of hierarchy levels in an MSA,
 * a linked list of topology levels (MPIDI_PSP_topo_level_t) can be created.
 * Each level in this list is defined by a "degree" (i.e., the height of the level) and
 * a "badge table" (i.e., integer values in an array that represent the group memberships
 * index by the process group ranks).
 * Accordingly, each list always refers to a specific process group (pg), which means that
 * in the case of multiple process groups (i.e. when dynamic process management comes into
 * play), there will also be multiple such lists.
 *
 * The following functions are provided to manage the relationships between the process
 * groups and the topology levels.
 */

/* Return the number of topology levels attached as a list to a given process group (pg). */
int MPIDI_PSP_get_num_topology_levels(MPIDI_PG_t * pg);

/* Pack the all the topology levels and their badge tables of a given process group (pg)
 * into an opaque buffer of integer values so that it can be exchanged between processes.
 * The buffer is allocated within this function and its size and address are returned.
 * It's the caller's task to release the buffer again after use.
 */
void MPIDI_PSP_pack_topology_badges(int **pack_msg, int *msg_size, MPIDI_PG_t * pg);

/* Counterpart to MPIDI_PSP_pack_topology_badges() (see above). */
void MPIDI_PSP_unpack_topology_badges(int *pack_msg, int pg_size, int num_levels,
                                      MPIDI_PSP_topo_level_t ** levels);

/* By calling MPIDI_PSP_add_topo_level_to_pg() (see above) in a loop, this function
 * attaches multiple levels of a given list (levels) to the given process group (pg).
 */
int MPIDI_PSP_add_topo_levels_to_pg(MPIDI_PG_t * pg, MPIDI_PSP_topo_level_t * level);

/* A flat level is some kind of a dummy level for a given degree, where all badges
 * in the array would have the same value. This function attaches such a level to
 * a given process group (pg).
 */
int MPIDI_PSP_add_flat_level_to_pg(MPIDI_PG_t * pg, int degree);
#endif

#endif /* _MPID_PSP_TOPO_H_ */
