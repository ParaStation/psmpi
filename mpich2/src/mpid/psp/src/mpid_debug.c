/*
 * ParaStation
 *
 * Copyright (C) 2006-2021 ParTec Cluster Competence Center GmbH, Munich
 * Copyright (C) 2021-2022 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#if defined(__GNUC__)
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <math.h>
#include <unistd.h>
#include <execinfo.h>
#include <sys/mman.h>
#include <sys/types.h>
#include "mpid_debug.h"

/* Obtain a backtrace and print it to `stdout'. */
static void print_trace(void)
{
	void *array[10];
	int size;
	char **strings;
	int i;
	pid_t pid = getpid();

	size = backtrace (array, 10);
	strings = backtrace_symbols (array, size);

	printf ("(%6d): Obtained %d stack frames.\n", pid, size);

	for (i = 0; i < size; i++)
		printf ("(%6d): %s\n", pid, strings[i]);

	/* backtrace_symbols_fd (array, size, 1); */
	free (strings);
}

static
void sig_segv(int sig)
{
	print_trace();
	exit(1);
}

static
void mpid_debug_init_gnuc(unsigned debug_level)
{
	if (debug_level > 0) {
		signal(SIGSEGV, sig_segv);
	}
}

#else /* !defined(__GNUC__) */
#include "mpid_debug.h"

static
void mpid_debug_init_gnuc(unsigned debug_level)
{
}

#endif /* !defined(__GNUC__) */

#include <stdlib.h>

#include "mpidimpl.h"

#ifdef PSCOM_ALLIN
static
char* MPIDI_PSP_pscom_allin_get_plugin_list_as_string()
{
	static char pscom_allin_plugin_list[] =
#ifdef PSCOM_ALLIN_OPENIB
		"openib,"
#endif
#ifdef PSCOM_ALLIN_PSM2
		"psm2,"
#endif
#ifdef PSCOM_ALLIN_UCP
		"ucp,"
#endif
		"";
	if (sizeof(pscom_allin_plugin_list) > 1) {
		pscom_allin_plugin_list[sizeof(pscom_allin_plugin_list) - 2] = '\0';
	}

	return pscom_allin_plugin_list;
}
#endif

char* MPIDI_PSP_get_psmpi_version_string()
{
	static char* psmpi_version_string = NULL;
	static char psmpi_version_string_pattern[] =
		"Version(PSMPI): %s (%s)+confset(%s)"
#ifdef MPICH_IS_THREADED
				"+threaded"
#endif
#ifdef MPIDI_PSP_WITH_CUDA_AWARENESS
				"+cuda"
#endif
#ifdef USE_PMI1_API
				"+pmi"
#elif defined USE_PMIX_API
				"+pmix"
#endif
#ifdef PSCOM_ALLIN
				"+allin(%s)"
#endif
		                "";

	if (!psmpi_version_string) {
		asprintf(&psmpi_version_string, psmpi_version_string_pattern,
			__DATE__, MPIDI_PSP_VC_VERSION, MPIDI_PSP_CONFSET
#ifdef PSCOM_ALLIN
			, MPIDI_PSP_pscom_allin_get_plugin_list_as_string()
#endif
		);
	}

	return psmpi_version_string;
}

void mpid_debug_init(void)
{
	/* evaluate environment variables */
	pscom_env_get_uint(&MPIDI_Process.env.debug_level, "PSP_DEBUG");
	pscom_env_get_uint(&MPIDI_Process.env.debug_version, "PSP_DEBUG_VERSION");

	/* initialize the signal handler for backtracing */
	mpid_debug_init_gnuc(MPIDI_Process.env.debug_level);

	/* print the version string if requested */
	if ((MPIDI_Process.env.debug_level > 2) ||
	    (MPIDI_Process.env.debug_version))
	{
		fprintf(stderr, "# %s\n", MPIDI_PSP_get_psmpi_version_string());
	}
}

const char *mpid_msgtype_str(enum MPID_PSP_MSGTYPE msg_type)
{
	switch (msg_type) {
	case MPID_PSP_MSGTYPE_DATA:		return "DATA";
	case MPID_PSP_MSGTYPE_DATA_REQUEST_ACK:	return "DATA_REQUEST_ACK";
	case MPID_PSP_MSGTYPE_DATA_ACK:		return "DATA_ACK";
	case MPID_PSP_MSGTYPE_CANCEL_DATA_ACK:	return "CANCEL_DATA_ACK";
	case MPID_PSP_MSGTYPE_CANCEL_DATA_REQUEST_ACK: return "CANCEL_DATA_REQUEST_ACK";
	case MPID_PSP_MSGTYPE_RMA_PUT:		return "RMA_PUT";

	case MPID_PSP_MSGTYPE_RMA_GET_REQ:	return "RMA_GET_REQ";
	case MPID_PSP_MSGTYPE_RMA_GET_ANSWER:	return "RMA_GET_ANSWER";
	case MPID_PSP_MSGTYPE_RMA_ACCUMULATE:	return "RMA_ACCUMULATE";

	case MPID_PSP_MSGTYPE_RMA_SYNC:         return "RMA_SYNC";

	case MPID_PSP_MSGTYPE_RMA_LOCK_SHARED_REQUEST: return "RMA_LOCK_SHARED_REQUEST";
	case MPID_PSP_MSGTYPE_RMA_LOCK_EXCLUSIVE_REQUEST: return "RMA_LOCK_EXCLUSIVE_REQUEST";
	case MPID_PSP_MSGTYPE_RMA_LOCK_ANSWER:	return "RMA_LOCK_ANSWER";
	case MPID_PSP_MSGTYPE_RMA_UNLOCK_REQUEST: return "RMA_UNLOCK_REQUEST";
	case MPID_PSP_MSGTYPE_RMA_UNLOCK_ANSWER: return "RMA_UNLOCK_ANSWER";

	case MPID_PSP_MSGTYPE_DATA_CANCELLED:	return "DATA_CANCELLED";
	case MPID_PSP_MSGTYPE_MPROBE_RESERVED_REQUEST: return "MPROBE_RESERVED_REQUEST";
	case MPID_PSP_MSGTYPE_MPROBE_RESERVED_REQUEST_ACK: return "MPID_PSP_MSGTYPE_MPROBE_RESERVED_REQUEST_ACK";

	case MPID_PSP_MSGTYPE_RMA_FLUSH_REQUEST:          return "RMA_FLUSH_REQUEST";
	case MPID_PSP_MSGTYPE_RMA_FLUSH_ANSWER:           return "RMA_FLUSH_ANSWER";

	case MPID_PSP_MSGTYPE_RMA_INTERNAL_LOCK_REQUEST:  return "RMA_LOCK_SHARED_REQUEST";
	case MPID_PSP_MSGTYPE_RMA_INTERNAL_LOCK_ANSWER:   return "RMA_LOCK_ANSWER";
	case MPID_PSP_MSGTYPE_RMA_INTERNAL_UNLOCK_REQUEST:return "RMA_UNLOCK_REQUEST";
	case MPID_PSP_MSGTYPE_RMA_INTERNAL_UNLOCK_ANSWER: return "RMA_UNLOCK_ANSWER";

	case MPID_PSP_MSGTYPE_FINALIZE_TOKEN:   return "FINALIZE_TOKEN";
	}
	return "UNKNOWN";
}
