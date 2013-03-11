/*
 * ParaStation
 *
 * Copyright (C) 2007 ParTec Cluster Competence Center GmbH, Munich
 *
 * All rights reserved.
 *
 * Author:	Jens Hauke <hauke@par-tec.com>
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <signal.h>
#include <math.h>

#include "pmi.h"

static int mypid = 0;
#define PMICALL(func) do {						\
	int pmi_errno = (func);						\
	if (pmi_errno != PMI_SUCCESS) {					\
		fprintf(stderr,"PMI@%6d: " #func			\
			" = %d\n=============continue with error\n",	\
			mypid, pmi_errno);				\
	}								\
} while (0)



char *pg_id = NULL;
static
void init_pg_id(void)
{
	int pg_id_sz;
	if (pg_id) return;

	/* obtain the id of the process group */

	PMICALL(PMI_Get_id_length_max(&pg_id_sz));

	pg_id = malloc(pg_id_sz + 1);
	if (!pg_id) { perror("malloc"); exit(1); }

	PMICALL(PMI_Get_id(pg_id, pg_id_sz));
}


int main(int argc, char **argv)
{

	int pg_rank = -99;
	int pg_size = -99;
	int pg_id_sz = -99;
	int has_parent = -99;
	int appnum = -99;
	int universe_size = -99;

	char key[100];
	char val[100];
	mypid = getpid();

	printf("%6d PMI_Init (stdout=%p)\n", mypid, stdout);

	PMICALL(PMI_Init(&has_parent));
	printf("%6d PMI_Init(&has_parent) : has_parent = %d\n", mypid, has_parent);

	PMICALL(PMI_Get_rank(&pg_rank));
	printf("%6d PMI_Get_rank(&pg_rank) : pg_rank = %d\n", mypid, pg_rank);

	PMICALL(PMI_Get_size(&pg_size));
	printf("%6d PMI_Get_size(&pg_size) : pg_size = %d\n", mypid, pg_size);

	PMICALL(PMI_Get_appnum(&appnum));
	printf("%6d PMI_Get_appnum(&appnum) : appnum = %d\n", mypid, appnum);

	init_pg_id();
	printf("%6d PMI_Get_id(pg_id, pg_id_sz)): pg_id = %s\n", mypid, pg_id);

	printf("%6d PMI_Get_universe_size(&universe_size)...\n", mypid);
	PMICALL(PMI_Get_universe_size(&universe_size));
	printf("%6d PMI_Get_universe_size(&universe_size) : universe_size = %d\n", mypid, universe_size);

	snprintf(key, sizeof(key), "test_rank%d", pg_rank);
	snprintf(val, sizeof(val), "test_rank%d_value", pg_rank);
	printf("%6d PMI_KVS_Put(,key=%s,val=%s)\n", mypid, key, val);
	PMICALL(PMI_KVS_Put(pg_id, key, val));

	printf("%6d PMI_KVS_Commit() (stdout=%p)\n", mypid, stdout);
//	system("ls -la /proc/self/fd");

	PMICALL(PMI_KVS_Commit(pg_id));

	PMICALL(PMI_Barrier());

	for (int i = 0; i < pg_size; i++) {
		snprintf(key, sizeof(key), "test_rank%d", i);
		strcpy(val, "***noval***");
		printf("%6d PMI_KVS_Get(pg_id, key=%s, val, sizeof(val))\n", mypid, key);
		PMICALL(PMI_KVS_Get(pg_id, key, val, sizeof(val)));
		printf("%6d PMI_KVS_Get(pg_id, key=%s, val, sizeof(val)) : val = %s\n", mypid, key, val);
	}

	printf("%6d PMI_Finalize()\n", mypid);

	fflush(stdout);
	sleep(1);
	PMICALL(PMI_Finalize());
//	sleep(10);

	return 0;
}
