/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <string.h>
#include "queue/zm_queue.h"

int zm_queue_if;

struct zm_queue_name_pair {
    int type;
    const char *name;
};

static struct zm_queue_name_pair name_pairs[] = {
    { ZM_GLQUEUE_IF, "gl" },
    { ZM_MSQUEUE_IF, "ms" },
    { ZM_SWPQUEUE_IF, "swp" },
    { ZM_FAQUEUE_IF, "fa" },
    { -1, NULL } /* name == NULL indicates the end of the list */
};

int zm_queue_parse_name(const char *name)
{
    int i;
    for (i = 0; name_pairs[i].name != NULL; i++) {
        /* Compare first few characters only -- so all of "gl", "glq", "glqueue" work */
        if (strncmp(name, name_pairs[i].name, strlen(name_pairs[i].name)) == 0)
            return name_pairs[i].type;
    }
    return -1; /* Matching queue type not found */
}
