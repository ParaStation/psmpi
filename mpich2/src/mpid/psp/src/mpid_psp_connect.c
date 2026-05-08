/*
 * ParaStation
 *
 * Copyright (C) 2024-2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"
#include "mpl.h"
#include "mpiimpl.h"

#define KEY_SETTINGS_CHECK "psmpi-settings"

struct InitMsg {
    int from_rank;
};

static
const char *direct_connect_to_str(int direct_connect)
{
    if (!direct_connect) {
        return "ondemand";
    } else {
        return "direct";
    }
}

static
const char *pm_to_str(void)
{
    switch (MPIR_CVAR_PMI_VERSION) {
        case MPIR_CVAR_PMI_VERSION_1:
            return "pmi";
        case MPIR_CVAR_PMI_VERSION_2:
            return "pmi2";
        case MPIR_CVAR_PMI_VERSION_x:
            return "pmix";
        default:
            MPIR_Assert(0);
            return "error";
    }
}

/* Prepare the psmpi settings check, return settings string */
static
int prep_settings_check(char **settings)
{
    int mpi_errno = MPI_SUCCESS;
    char *s = NULL;
    char *key = NULL;

    /* Settings check is only reasonable for 2 or more processes */
    if (MPIDI_Process.env.debug_settings && (MPIDI_Process.my_pg_size >= 2)) {
        int max_len_value;
        int max_len_key;
        const char *direct_connect = direct_connect_to_str(MPIDI_Process.env.enable_direct_connect);
        const char *pm = pm_to_str();

        /* Prepare settings string including psmpi version, PM interface, direct connect */
        max_len_value = MPIR_pmi_max_val_size();
        s = MPL_malloc(max_len_value, MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!(s), mpi_errno, MPI_ERR_OTHER, "**nomem");
        snprintf(s, max_len_value, "%s-%s-%s", MPIDI_PSP_VC_VERSION, pm, direct_connect);

        /* Encode the rank in the key so that each process uses a unique key (needed for PMI) */
        max_len_key = MPIR_pmi_max_key_size();
        key = MPL_malloc(max_len_key, MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!(key), mpi_errno, MPI_ERR_OTHER, "**nomem");
        snprintf(key, max_len_key, "%s.rank-%d", KEY_SETTINGS_CHECK, MPIDI_Process.my_pg_rank);

        mpi_errno = MPIR_pmi_kvs_put(key, s);
        MPIR_ERR_CHECK(mpi_errno);

        *settings = s;
    }

  fn_exit:
    MPL_free(key);
    return mpi_errno;
  fn_fail:
    MPL_free(s);
    goto fn_exit;
}

/* Do optional settings check, return error if the check fails. This check
 * ensures at runtime that all processes use the same settings of psmpi.
 * This can be relevant, e.g., in case of MSA runs where there might be
 * different module trees */
static
int do_settings_check(char *settings, int *granks, int size)
{
    int mpi_errno = MPI_SUCCESS;
    int max_len_value = MPIR_pmi_max_val_size();
    int max_len_key = MPIR_pmi_max_key_size();
    int pg_rank = MPIDI_Process.my_pg_rank;
    char *s = NULL;
    char *key = NULL;
    int diffs = 0;

    /* All processes compare their settings to that of all other processes */
    if (MPIDI_Process.env.debug_settings && (size >= 2)) {
        /* Make sure that settings is a non-null string */
        MPIR_Assert(settings != NULL);

        s = MPL_malloc(max_len_value, MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!(s), mpi_errno, MPI_ERR_OTHER, "**nomem");
        key = MPL_malloc(max_len_key, MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!(key), mpi_errno, MPI_ERR_OTHER, "**nomem");

        for (int i = 0; i < size; i++) {
            int dest = granks ? granks[i] : i;
            /* Skip self */
            if (dest == pg_rank) {
                continue;
            }

            memset(s, 0, max_len_value);
            memset(key, 0, max_len_key);

            /* Use key for rank dest */
            snprintf(key, max_len_key, "%s.rank-%d", KEY_SETTINGS_CHECK, dest);

            mpi_errno = MPIR_pmi_kvs_get(dest, key, s, max_len_value);
            MPIR_ERR_CHECK(mpi_errno);

            if (strcmp(s, settings)) {
                if (diffs == 0) {
                    /* Print error msg on first diff */
                    fprintf(stderr,
                            "MPI error: different psmpi settings: own rank %d:'%s' != rank %d:'%s'\n",
                            pg_rank, settings, dest, s);
                }
                diffs++;
            }
        }
    }
    MPIR_ERR_CHKANDJUMP1(diffs > 0, mpi_errno, MPI_ERR_OTHER, "**psp|settings_check",
                         "**psp|settings_check %d", diffs);

  fn_exit:
    MPL_free(s);
    MPL_free(key);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* atexit handler to clean up connection mapping */
static
void free_grank2con_mapping(void)
{
    MPL_direct_free(MPIDI_Process.grank2con);
}

/* set connection */
static
void grank2con_set(int dest_grank, pscom_connection_t * con)
{
    int pg_size = MPIDI_Process.my_pg_size;

    MPIR_Assert(dest_grank < pg_size);

    MPIDI_Process.grank2con[dest_grank] = con;

    /* Update con in the world group vcr and cons if world group already exists
     * i.e, it is not the first time that connections are created through a new communicator
     * and we are adding a new connection */
    if (MPIDI_Process.my_pg != NULL) {
        MPIDI_Process.my_pg->vcr[dest_grank]->con = con;
        MPIDI_Process.my_pg->cons[dest_grank] = con;
    }
}

/* return connection */
static
pscom_connection_t *grank2con_get(int dest_grank)
{
    int pg_size = MPIDI_Process.my_pg_size;

    MPIR_Assert(dest_grank < pg_size);

    return MPIDI_Process.grank2con[dest_grank];
}

/* Initialize global connection map */
int MPIDI_PSP_grank2con_mapping_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    int i;
    int pg_size = MPIDI_Process.my_pg_size;

    if (MPIDI_Process.grank2con) {
        /* Re-init, connections kept open, insert them back to my_pg */
        if (MPIDI_Process.my_pg != NULL) {
            for (i = 0; i < pg_size; i++) {
                MPIDI_Process.my_pg->vcr[i]->con = grank2con_get(i);
                MPIDI_Process.my_pg->cons[i] = grank2con_get(i);
            }
        }
        MPIR_Assert(MPIDI_Process.env.enable_keep_connections >= 1);
        goto fn_exit;
    }

    if (MPIDI_Process.env.enable_keep_connections) {
        /* Use direct mem allocation because memory is freed in atexit handler */
        MPIDI_Process.grank2con = MPL_direct_malloc(sizeof(MPIDI_Process.grank2con[0]) * pg_size);
    } else {
        MPIDI_Process.grank2con =
            MPL_malloc(sizeof(MPIDI_Process.grank2con[0]) * pg_size, MPL_MEM_OBJECT);
    }
    MPIR_ERR_CHKANDJUMP(!MPIDI_Process.grank2con, mpi_errno, MPI_ERR_OTHER, "**nomem");

    if (MPIDI_Process.env.enable_keep_connections) {
        atexit(free_grank2con_mapping);
    }

    for (i = 0; i < pg_size; i++) {
        grank2con_set(i, NULL);
    }
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* pscom callback for io_done of connection init message (direct connect mode)*/
static
void cb_io_done_init_msg(pscom_request_t * req)
{
    if (pscom_req_successful(req)) {
        pscom_connection_t *old_connection;

        struct InitMsg *init_msg = (struct InitMsg *) req->data;

        old_connection = grank2con_get(init_msg->from_rank);
        if (old_connection) {
            if (old_connection == req->connection) {
                /* Loopback connection */
                ;
            } else {
                /* Already connected??? */
                fprintf(stderr,
                        "Second connection from %s as rank %i (previous connection from %s). Closing second.\n",
                        pscom_con_info_str(&old_connection->remote_con_info), init_msg->from_rank,
                        pscom_con_info_str(&req->connection->remote_con_info));
                pscom_close_connection(req->connection);
            }
        } else {
            /* register connection */
            grank2con_set(init_msg->from_rank, req->connection);
        }
    } else {
        pscom_close_connection(req->connection);
    }
    pscom_request_free(req);
}

/* pscom callback for accepted connections/ init message (direct connect mode) */
static
void mpid_con_accept(pscom_connection_t * new_connection)
{
    pscom_request_t *req;
    req = pscom_request_create(0, sizeof(struct InitMsg));

    req->xheader_len = 0;
    req->data_len = sizeof(struct InitMsg);
    req->data = req->user;
    req->connection = new_connection;
    req->ops.io_done = cb_io_done_init_msg;

    pscom_post_recv(req);
}

/* Wait for incoming connection from src rank */
static
void do_wait(int src)
{
    /* printf("Accepting (rank %d to %d).\n", src, MPIDI_Process.my_pg_rank); */
    while (!grank2con_get(src)) {
        pscom_wait_any();
    }
}


/* Mark send of init message as completed */
static
void init_send_done(pscom_req_state_t state, void *priv)
{
    int *send_done = (int *) priv;
    *send_done = 1;
}

/* Open new pscom connection and connect to dest */
static
int do_connect(pscom_socket_t * socket, int dest, char *ep_str, pscom_connection_t ** con)
{
    int mpi_errno = MPI_SUCCESS;
    pscom_connection_t *_con;
    pscom_err_t rc;

    /* printf("Connecting (rank %d to %d) (%s)\n", MPIDI_Process.my_pg_rank, dest, ep_str); */
    _con = pscom_open_connection(socket);
    MPIR_ERR_CHKANDJUMP(!_con, mpi_errno, MPI_ERR_OTHER, "**psp|openconn");

#if MPID_PSP_HAVE_PSCOM_ABI_5
    uint64_t flags =
        MPIDI_Process.env.enable_direct_connect ? PSCOM_CON_FLAG_DIRECT : PSCOM_CON_FLAG_ONDEMAND;
    rc = pscom_connect(_con, ep_str, dest, flags);
#else
    rc = pscom_connect_socket_str(_con, ep_str);
#endif
    MPIR_ERR_CHKANDJUMP1((rc != PSCOM_SUCCESS), mpi_errno, MPI_ERR_OTHER,
                         "**psp|connect", "**psp|connect %d", rc);

    grank2con_set(dest, _con);

    if (con) {
        *con = _con;
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Direct connect: Create connection to dest, send init message and wait for completion */
static
int do_connect_direct(pscom_socket_t * socket, int dest, char *ep_str)
{
    int mpi_errno = MPI_SUCCESS;
    pscom_connection_t *con;
    struct InitMsg init_msg;
    int init_msg_sent = 0;

    /* open pscom connection and connect */
    mpi_errno = do_connect(socket, dest, ep_str, &con);
    MPIR_ERR_CHECK(mpi_errno);

    /* send the initialization message and wait for its completion */
    init_msg.from_rank = MPIDI_Process.my_pg_rank;
    pscom_send_inplace(con, NULL, 0, &init_msg, sizeof(init_msg), init_send_done, &init_msg_sent);

    while (!init_msg_sent) {
        pscom_wait_any();
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Connect all processes in direct mode */
static
int connect_direct(pscom_socket_t * socket, int *granks, int size, int rank, char **ep_strs)
{
    int mpi_errno = MPI_SUCCESS;
    int i;

    /* connect ranks rank..(rank + size/2) */
    for (i = 0; i <= size / 2; i++) {
        int dest = (rank + i) % size;
        int src = (rank + size - i) % size;
        /* ep_strs array has size elements, where size <= MPIDI_Process.my_pg_size.
         * Hence, indexing has to happen relative to loop index i
         * and not relative to global granks. */
        char *dest_ep = ep_strs[dest];
        if (granks) {
            dest = granks[dest];
            src = granks[src];
        }

        if (!i || (rank / i) % 2) {
            /* connect, accept */
            if (!grank2con_get(dest)) {
                mpi_errno = do_connect_direct(socket, dest, dest_ep);
                MPIR_ERR_CHECK(mpi_errno);
            }
            if (!i || src != dest) {
                if (!grank2con_get(src)) {
                    do_wait(src);
                }
            }
        } else {
            /* accept, connect */
            if (!grank2con_get(src)) {
                do_wait(src);
            }
            if (src != dest) {
                if (!grank2con_get(dest)) {
                    mpi_errno = do_connect_direct(socket, dest, dest_ep);
                    MPIR_ERR_CHECK(mpi_errno);
                }
            }
        }
    }

    /* Wait for all missing connections: (already done?) */
    for (i = 0; i < size; i++) {
        int dest = granks ? granks[i] : i;
        while (!grank2con_get(dest)) {
            pscom_wait_any();
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Connect all processes in ondemand mode */
static
int connect_ondemand(pscom_socket_t * socket, int *granks, int size, char **ep_strs)
{
    int mpi_errno = MPI_SUCCESS;
    int i;

    /* Create all connections */
    for (i = 0; i < size; i++) {
        int dest = granks ? granks[i] : i;
        if (!grank2con_get(dest)) {
            mpi_errno = do_connect(socket, dest, ep_strs[i], NULL);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_PSP_socket_get_ep_str(pscom_socket_t * socket, char **ep_str)
{
    int mpi_errno = MPI_SUCCESS;

#if MPID_PSP_HAVE_PSCOM_ABI_5
    char *_ep_str = NULL;
    pscom_err_t rc = PSCOM_SUCCESS;
    rc = pscom_socket_get_ep_str(socket, &_ep_str);
    MPIR_ERR_CHKANDJUMP1(rc != PSCOM_SUCCESS, mpi_errno, MPI_ERR_OTHER, "**psp|getepstr",
                         "**psp|getepstr %s", pscom_err_str(rc));
#else
    const char *_ep_str = NULL;
    if (MPIDI_Process.env.enable_direct_connect) {
        _ep_str = pscom_listen_socket_str(socket);
    } else {
        _ep_str = pscom_listen_socket_ondemand_str(socket);
    }
#endif

    MPIR_Assert(ep_str);
    if (_ep_str) {
        *ep_str = MPL_strdup(_ep_str);
        MPIR_ERR_CHKANDJUMP(!(*ep_str), mpi_errno, MPI_ERR_OTHER, "**nomem");
    } else {
        *ep_str = NULL;
    }

#if MPID_PSP_HAVE_PSCOM_ABI_5
    pscom_socket_free_ep_str(_ep_str);
#endif

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Exchange endpoint strings of all processes in the granks array
 * granks == NULL means: world comm
 *
 * The resulting ep_strs array of strings has 'size' elements. Elements are NULL
 * for processes to which we are already connected, i.e., if there is already a
 * connection stored in grank2con.
 */
static
int exchange_ep_strs(MPIR_Comm * comm, pscom_socket_t * socket, int *granks, int size,
                     char ***ep_strs)
{
    int mpi_errno = MPI_SUCCESS;
    char *key = NULL;
    char *val = NULL;
    int max_len_value = MPIR_pmi_max_val_size();
    int max_len_key = MPIR_pmi_max_key_size();
    const char *base_key = "psp-conn";
    int i;
    int pg_rank = MPIDI_Process.my_pg_rank;
    char *ep_str = NULL;
    char *settings = NULL;
    char **_ep_strs = NULL;

    mpi_errno = MPIDI_PSP_socket_get_ep_str(socket, &ep_str);
    MPIR_ERR_CHECK(mpi_errno);

    /* For only one process there is no need to exchange any connection infos */
    if (size > 1) {
        mpi_errno = prep_settings_check(&settings);
        MPIR_ERR_CHECK(mpi_errno);

        if (ep_str) {
            key = MPL_calloc(max_len_key, sizeof(char), MPL_MEM_STRINGS);
            MPIR_ERR_CHKANDJUMP(!key, mpi_errno, MPI_ERR_OTHER, "**nomem");

            /* Create KVS key for this rank */
            snprintf(key, max_len_key, "%s%i", base_key, pg_rank);

            /* PMI(x)_put and PMI(x)_commit() */
            mpi_errno = MPIR_pmi_kvs_put(key, ep_str);
            MPIR_ERR_CHECK(mpi_errno);
        }

        if (MPIDI_Process.env.debug_settings || ep_str) {
            if (granks && (size < MPIDI_Process.my_pg_size)) {
                mpi_errno = MPIR_pmi_barrier_group(granks, size, comm->stringtag);
            } else {
                /* Use world barrier for world comm and comms that have size of world comm */
                mpi_errno = MPIR_pmi_barrier();
            }
            MPIR_ERR_CHECK(mpi_errno);
        } else if (MPIDI_Process.env.enable_lightweight_init_barrier) {
            if (granks && (size < MPIDI_Process.my_pg_size)) {
                mpi_errno = MPIR_pmi_barrier_only_group(granks, size, comm->stringtag);
            } else {
                /* Use lightweight world barrier for world comm and comms that have size of world comm */
                mpi_errno = MPIR_pmi_barrier_only();
            }
            MPIR_ERR_CHECK(mpi_errno);
        }

        mpi_errno = do_settings_check(settings, granks, size);
        MPIR_ERR_CHECK(mpi_errno);
    }

    _ep_strs = MPL_calloc(size, sizeof(char *), MPL_MEM_STRINGS);
    MPIR_ERR_CHKANDJUMP(!_ep_strs, mpi_errno, MPI_ERR_OTHER, "**nomem");

    /* Get endpoints from other processes in comm */
    for (i = 0; i < size; i++) {
        int dest = granks ? granks[i] : i;
        if (ep_str) {
            /* Skip if we are already connected to dest */
            if (grank2con_get(dest)) {
                continue;
            }

            if (!val) {
                val = MPL_calloc(max_len_value, sizeof(char), MPL_MEM_STRINGS);
                MPIR_ERR_CHKANDJUMP(!val, mpi_errno, MPI_ERR_OTHER, "**nomem");
            } else {
                memset(val, 0, max_len_value);
            }

            if (dest != pg_rank) {
                /* Erase any content from the (previously used) key */
                memset(key, 0, max_len_key);
                /* Create KVS key for rank "dest" */
                snprintf(key, max_len_key, "%s%i", base_key, dest);
                /* "dest" is the source who published the information */
                mpi_errno = MPIR_pmi_kvs_get(dest, key, val, max_len_value);
                MPIR_ERR_CHECK(mpi_errno);

                /* Make sure we got non-null string back from the KVS */
                MPIR_Assert(val != NULL);
            } else {
                /* Myself: Dont use KVS because this fails for singleton case */
                MPIR_Assert(strlen(ep_str) < max_len_value);
                strncpy(val, ep_str, max_len_value);
            }
            _ep_strs[i] = MPL_strdup(val);
            MPIR_ERR_CHKANDJUMP(!_ep_strs[i], mpi_errno, MPI_ERR_OTHER, "**nomem");

        } else {
            _ep_strs[i] = NULL;
        }
        MPIR_ERR_CHECK(mpi_errno);
    }

    *ep_strs = _ep_strs;

  fn_exit:
    MPL_free(val);
    MPL_free(key);
    MPL_free(ep_str);
    MPL_free(settings);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_PSP_connection_init(MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;
    pscom_err_t rc;
    pscom_socket_t *socket = MPIDI_Process.socket;
    static int first_init = 1;
    bool fast_path = true;
    int *granks = NULL;
    int size = 0, rank = -1;
    char **ep_strs = NULL;

    /* This function is collective over comm, get the granks of the processes in
     * comm so that we know who is in comm for all following steps.
     *
     * If comm is NULL (world comm), granks will be NULL. */
    mpi_errno = MPIDI_PSP_comm_get_granks(comm, &granks, &size, &rank);
    MPIR_ERR_CHECK(mpi_errno);

    /* Check if we have to do something or if connections to all processes of
     * the comm are already available */
    for (int i = 0; i < size; i++) {
        int dest = granks ? granks[i] : i;
        if (!grank2con_get(dest)) {
            fast_path = false;  /* There is at least one connection missing */
            break;
        }
    }

    if (fast_path) {
        goto fn_exit;
    }

    if (first_init) {
        /* Listen on any port, we don't have contact infos yet */
        rc = pscom_listen(socket, PSCOM_ANYPORT);
        MPIR_ERR_CHKANDJUMP1((rc != PSCOM_SUCCESS), mpi_errno, MPI_ERR_OTHER,
                             "**psp|listen_anyport", "**psp|listen_anyport %s", pscom_err_str(rc));
        first_init = 0;
    } else {
        /* Start to listen again for incoming connections on the port assigned
         * in previous call of pscom_listen */
#ifdef PSCOM_HAS_LISTEN_SUSPEND_RESUME
        pscom_resume_listen(socket);
#else
        int port = 0;
        char *ep_str = NULL;
        mpi_errno = MPIDI_PSP_socket_get_ep_str(socket, &ep_str);
        MPIR_ERR_CHECK(mpi_errno);

        /* Extract port number from ep_str (element after delimiter ':') */
        char *elem = strtok(ep_str, ":");
        elem = strtok(NULL, ":");
        port = atoi(elem);
        MPL_free(ep_str);

        /* Note: This is not safe because the listen port from the first initialization may
         * be used differently by now. The port is returned to the OS by pscom_stop_listen()
         * before.
         * Compile with a newer pscom version for a safe listen suspend/ resume solution. */
        rc = pscom_listen(socket, port);
        MPIR_ERR_CHKANDJUMP1((rc != PSCOM_SUCCESS), mpi_errno, MPI_ERR_OTHER,
                             "**psp|listen_anyport", "**psp|listen_anyport %s", pscom_err_str(rc));
#endif
    }

    /* Distribute any missing contact information and store endpoint strings */
    mpi_errno = exchange_ep_strs(comm, socket, granks, size, &ep_strs);
    MPIR_ERR_CHECK(mpi_errno);

    if (MPIDI_Process.env.enable_direct_connect) {
        mpi_errno = connect_direct(socket, granks, size, rank, ep_strs);
    } else {
        mpi_errno = connect_ondemand(socket, granks, size, ep_strs);
    }
    MPIR_ERR_CHECK(mpi_errno);

#ifdef PSCOM_HAS_LISTEN_SUSPEND_RESUME
    /* Suspend listening for incoming connections (keep the assigned port) */
    pscom_suspend_listen(socket);
#else
    /* Stop listening for incoming connections */
    pscom_stop_listen(socket);
#endif

    MPID_enable_receive_dispach(socket);

  fn_exit:
    if (ep_strs) {
        for (int i = 0; i < size; i++) {
            if (ep_strs[i]) {
                MPL_free(ep_strs[i]);
            }
        }
        MPL_free(ep_strs);
    }
    MPL_free(granks);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Initialize the global pscom socket */
int MPIDI_PSP_socket_init(void)
{
    int mpi_errno = MPI_SUCCESS;

    if (!MPIDI_Process.socket) {
        /* First init: open new socket */
        pscom_socket_t *socket;
#if MPID_PSP_HAVE_PSCOM_ABI_5
        uint64_t flags = PSCOM_SOCK_FLAG_INTRA_JOB;
        socket = pscom_open_socket(0, 0, MPIDI_Process.my_pg_rank, flags);
#else
        socket = pscom_open_socket(0, 0);
#endif
        MPIR_ERR_CHKANDJUMP(!socket, mpi_errno, MPI_ERR_OTHER, "**psp|opensocket");

        if (MPIDI_Process.env.enable_direct_connect) {
            socket->ops.con_accept = mpid_con_accept;
        }

        {
            char name[10];
            snprintf(name, sizeof(name), "r%07u", (unsigned) MPIDI_Process.my_pg_rank % 100000000);
            pscom_socket_set_name(socket, name);
        }

        MPIDI_Process.socket = socket;
    }

    if (MPIDI_Process.smp_node_id <= MPIDI_PSP_NODE_ID_UNDEFINED) {
        /* If no smp_node_id is set explicitly, use the pscom's node_id for this:
         * (...which is an int and might be negative. However, since we know that it actually
         * corresponds to the IPv4 address of the node, it is safe to force the most significant
         * bit to be unset so that it is positive and can thus also be used as a split color.)
         */
        MPIDI_Process.smp_node_id =
            (int) ((unsigned) MPIDI_Process.socket->local_con_info.node_id & (unsigned) 0x7fffffff);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Check if there are missing connections for an array of remote lpids */
int MPIDI_PG_check_missing_remote_cons(MPIR_Comm * comm_ptr, MPIR_Comm * peer_comm_ptr,
                                       int root, int remote_leader, int peer_tag, int remote_size,
                                       MPIR_Lpid * remote_lpids, int *flag)
{
    int mpi_errno = MPI_SUCCESS;
    int coll_attr = MPIR_COLL_ATTR_SYNC;
    int all_found_local = 1;
    int all_found_remote = 0;

    /* Check if we have a connection for each remote lpid */
    for (int i = 0; i < remote_size; i++) {
        int world_idx = MPIR_LPID_WORLD_INDEX(remote_lpids[i]);
        int grank = MPIR_LPID_WORLD_RANK(remote_lpids[i]);
        MPIDI_PG_t *pg = NULL;
        MPIDI_PG_get(world_idx, &pg);
        MPIR_Assert(pg != NULL);

        /* Check if a connection for grank is available in the pg */
        if ((pg->vcr[grank] == NULL) || (pg->vcr[grank]->con == NULL)) {
            all_found_local = 0;
        }
    }

    /* See if everyone in local comm is happy: */
    mpi_errno = MPIR_Allreduce(MPI_IN_PLACE, &all_found_local, 1, MPIR_INT_INTERNAL, MPI_LAND,
                               comm_ptr, coll_attr);
    MPIR_ERR_CHECK(mpi_errno);

    /* See if remote procs are happy, too: */
    if (comm_ptr->rank == root) {
        mpi_errno = MPIC_Sendrecv(&all_found_local, 1, MPIR_INT_INTERNAL, remote_leader, peer_tag,
                                  &all_found_remote, 1, MPIR_INT_INTERNAL, remote_leader, peer_tag,
                                  peer_comm_ptr, MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);
    }

    /* Check if we can stop this here because all procs involved are happy: */
    mpi_errno = MPIR_Bcast(&all_found_remote, 1, MPIR_INT_INTERNAL, root, comm_ptr, coll_attr);
    MPIR_ERR_CHECK(mpi_errno);

    if (all_found_local && all_found_remote) {
        /* Oh Happy Day! :-) We have all remote connections without further ado!
         * (Quite likely we are dealing here with a non-spawn case...)
         */
        *flag = 0;
    } else {
        *flag = 1;
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


/* Get all ep strings of the remote endpoints:
 * - Step 1: root gathers all ep strings of comm
 * - Step 2: root and peer exchange the ep strings of comm
 */
static
int MPIDI_PSP_get_remote_endpoints(MPIR_Comm * peer_comm_ptr, MPIR_Comm * comm_ptr, int root,
                                   int remote_leader, int peer_tag, char *ep_str,
                                   char **remote_ep_strs, MPI_Aint ** remote_ep_strs_displs,
                                   int *_remote_size)
{
    int mpi_errno = MPI_SUCCESS;
    int coll_attr = MPIR_COLL_ATTR_SYNC;
    MPI_Aint ep_strlen = 0;

    char *ep_strs_local = NULL;
    MPI_Aint *ep_strs_local_sizes = NULL;
    MPI_Aint *ep_strs_local_displs = NULL;
    MPI_Aint ep_strs_local_total_size = 0;

    char *ep_strs_remote = NULL;
    MPI_Aint *ep_strs_remote_sizes = NULL;
    MPI_Aint *ep_strs_remote_displs = NULL;
    MPI_Aint ep_strs_remote_total_size = 0;
    int local_size = comm_ptr->local_size;
    int remote_size = 0;
    int i;

    MPIR_Assert(ep_str != NULL);
    ep_strlen = strlen(ep_str) + 1;     /* +1 to account for NULL terminator */

    /* Step 1 - Root gathers all ep strings in comm */

    if (comm_ptr->rank == root) {
        ep_strs_local_sizes = (MPI_Aint *) MPL_calloc(local_size, sizeof(MPI_Aint), MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!ep_strs_local_sizes, mpi_errno, MPI_ERR_OTHER, "**nomem");
        ep_strs_local_displs = (MPI_Aint *) MPL_calloc(local_size, sizeof(MPI_Aint), MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!ep_strs_local_displs, mpi_errno, MPI_ERR_OTHER, "**nomem");
    }

    /* Gather size of all ep strings from ranks in comm */
    mpi_errno = MPID_Gather(&ep_strlen, 1, MPIR_AINT_INTERNAL, ep_strs_local_sizes, 1,
                            MPIR_AINT_INTERNAL, root, comm_ptr, coll_attr);
    MPIR_ERR_CHECK(mpi_errno);

    if (comm_ptr->rank == root) {
        /* Calculate displacement vector and allocate contiguous memory block for ep strings */
        for (i = 0; i < local_size; i++) {
            if (i == 0) {
                ep_strs_local_displs[i] = 0;
            } else {
                ep_strs_local_displs[i] = ep_strs_local_sizes[i - 1] + ep_strs_local_displs[i - 1];
            }
            ep_strs_local_total_size += ep_strs_local_sizes[i];
        }

        MPIR_Assert(ep_strs_local_total_size > 0);
        ep_strs_local =
            (char *) MPL_calloc(ep_strs_local_total_size, sizeof(char), MPL_MEM_STRINGS);
        MPIR_ERR_CHKANDJUMP(!ep_strs_local, mpi_errno, MPI_ERR_OTHER, "**nomem");
    }

    /* Gather all ep strings from ranks in comm */
    mpi_errno = MPID_Gatherv(ep_str, ep_strlen, MPIR_CHAR_INTERNAL, ep_strs_local,
                             ep_strs_local_sizes, ep_strs_local_displs, MPIR_CHAR_INTERNAL,
                             root, comm_ptr, coll_attr);
    MPIR_ERR_CHECK(mpi_errno);

    /* Step 2 - Root and peer exchange ep strings */

    if (comm_ptr->rank == root) {
        MPIR_Assert(ep_strs_local_sizes != NULL);

        /* Exchange comm size with remote peer */
        mpi_errno = MPIC_Sendrecv(&local_size, 1, MPIR_INT_INTERNAL, remote_leader, peer_tag,
                                  &remote_size, 1, MPIR_INT_INTERNAL, remote_leader, peer_tag,
                                  peer_comm_ptr, MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);

        MPIR_Assert(remote_size > 0);
        ep_strs_remote_sizes =
            (MPI_Aint *) MPL_malloc(remote_size * sizeof(MPI_Aint), MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!ep_strs_remote_sizes, mpi_errno, MPI_ERR_OTHER, "**nomem");
        ep_strs_remote_displs =
            (MPI_Aint *) MPL_malloc(remote_size * sizeof(MPI_Aint), MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!ep_strs_remote_displs, mpi_errno, MPI_ERR_OTHER, "**nomem");

        /* Exchange array of ep string sizes with remote peer  */
        mpi_errno = MPIC_Sendrecv(ep_strs_local_sizes, local_size, MPIR_AINT_INTERNAL,
                                  remote_leader, peer_tag,
                                  ep_strs_remote_sizes, remote_size, MPIR_AINT_INTERNAL,
                                  remote_leader, peer_tag,
                                  peer_comm_ptr, MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);

        /* Calculate total remote size and displacements */
        for (i = 0; i < remote_size; i++) {
            if (i == 0) {
                ep_strs_remote_displs[i] = 0;
            } else {
                ep_strs_remote_displs[i] =
                    ep_strs_remote_sizes[i - 1] + ep_strs_remote_displs[i - 1];
            }
            ep_strs_remote_total_size += ep_strs_remote_sizes[i];
        }

        /* Allocate memory for remote ep strings based on the received sizes */
        MPIR_Assert(ep_strs_remote_total_size > 0);
        ep_strs_remote =
            (char *) MPL_calloc(ep_strs_remote_total_size, sizeof(char), MPL_MEM_STRINGS);
        MPIR_ERR_CHKANDJUMP(!ep_strs_remote, mpi_errno, MPI_ERR_OTHER, "**nomem");

        /* Exchange ep strings with remote peer */
        mpi_errno = MPIC_Sendrecv(ep_strs_local, ep_strs_local_total_size, MPIR_CHAR_INTERNAL,
                                  remote_leader, peer_tag,
                                  ep_strs_remote, ep_strs_remote_total_size, MPIR_CHAR_INTERNAL,
                                  remote_leader, peer_tag,
                                  peer_comm_ptr, MPI_STATUS_IGNORE, coll_attr);
        MPIR_ERR_CHECK(mpi_errno);
    }

    mpi_errno = MPIR_Bcast(&remote_size, 1, MPIR_INT_INTERNAL, root, comm_ptr, coll_attr);
    MPIR_ERR_CHECK(mpi_errno);
    MPIR_Assert(!MPIR_COLL_ATTR_HAS_ERR(coll_attr));

    mpi_errno = MPIR_Bcast(&ep_strs_remote_total_size, 1, MPIR_AINT_INTERNAL, root, comm_ptr,
                           coll_attr);
    MPIR_ERR_CHECK(mpi_errno);
    MPIR_Assert(!MPIR_COLL_ATTR_HAS_ERR(coll_attr));
    MPIR_Assert(remote_size > 0);
    MPIR_Assert(ep_strs_remote_total_size > 0);

    if (comm_ptr->rank != root) {
        ep_strs_remote_displs =
            (MPI_Aint *) MPL_malloc(remote_size * sizeof(MPI_Aint), MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!ep_strs_remote_displs, mpi_errno, MPI_ERR_OTHER, "**nomem");

        ep_strs_remote =
            (char *) MPL_calloc(ep_strs_remote_total_size, sizeof(char), MPL_MEM_STRINGS);
        MPIR_ERR_CHKANDJUMP(!ep_strs_remote, mpi_errno, MPI_ERR_OTHER, "**nomem");
    }

    mpi_errno = MPIR_Bcast(ep_strs_remote_displs, remote_size, MPIR_AINT_INTERNAL, root, comm_ptr,
                           coll_attr);
    MPIR_ERR_CHECK(mpi_errno);
    MPIR_Assert(!MPIR_COLL_ATTR_HAS_ERR(coll_attr));

    mpi_errno = MPIR_Bcast(ep_strs_remote, ep_strs_remote_total_size, MPIR_CHAR_INTERNAL, root,
                           comm_ptr, coll_attr);
    MPIR_ERR_CHECK(mpi_errno);
    MPIR_Assert(!MPIR_COLL_ATTR_HAS_ERR(coll_attr));

    /* Set output values */
    *_remote_size = remote_size;
    *remote_ep_strs_displs = ep_strs_remote_displs;
    *remote_ep_strs = ep_strs_remote;

  fn_exit:
    MPL_free(ep_strs_local_displs);
    MPL_free(ep_strs_local_sizes);
    MPL_free(ep_strs_local);
    MPL_free(ep_strs_remote_sizes);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_PSP_connect_remote(MPIR_Comm * peer_comm_ptr, MPIR_Comm * comm_ptr, int root,
                             int remote_leader, int peer_tag, MPIR_Lpid * remote_lpids)
{
    int mpi_errno = MPI_SUCCESS;
    pscom_err_t rc = PSCOM_SUCCESS;
    pscom_socket_t *inter_socket = NULL;        /* Inter job socket */

    char *ep_str = NULL;
    char *remote_ep_strs = NULL;
    MPI_Aint *remote_ep_strs_displs = NULL;
    int remote_size = 0;

    /* Open an inter-job socket and return ep str */
    mpi_errno = MPID_PSP_open_all_sockets(&ep_str, &inter_socket);
    MPIR_ERR_CHECK(mpi_errno);

    MPIR_Assert(ep_str != NULL);
    MPIR_Assert(inter_socket != NULL);

    /* Get remote endpoints */
    mpi_errno = MPIDI_PSP_get_remote_endpoints(peer_comm_ptr, comm_ptr, root, remote_leader,
                                               peer_tag, ep_str, &remote_ep_strs,
                                               &remote_ep_strs_displs, &remote_size);
    MPIR_ERR_CHECK(mpi_errno);

    for (int i = 0; i < remote_size; i++) {
        int world_idx = MPIR_LPID_WORLD_INDEX(remote_lpids[i]);
        int grank = MPIR_LPID_WORLD_RANK(remote_lpids[i]);
        MPIDI_PG_t *pg = NULL;
        MPIDI_PG_get(world_idx, &pg);
        MPIR_Assert(pg != NULL);

        if ((pg->vcr[grank] != NULL) && (pg->vcr[grank]->con != NULL)) {
            continue;   /* already connected to this lpid */
        }

        char *remote_ep;
        pscom_connection_t *con = pscom_open_connection(inter_socket);
        MPIR_ERR_CHKANDJUMP(!con, mpi_errno, MPI_ERR_OTHER, "**psp|openconn");

        /* Displacement determines the ep string to connect to */
        remote_ep = remote_ep_strs + remote_ep_strs_displs[i];

#if MPID_PSP_HAVE_PSCOM_ABI_5
        uint64_t flags = PSCOM_CON_FLAG_ONDEMAND;
        rc = pscom_connect(con, remote_ep, PSCOM_RANK_UNDEFINED, flags);
#else
        rc = pscom_connect_socket_str(con, remote_ep);
#endif
        MPIR_ERR_CHKANDJUMP1((rc != PSCOM_SUCCESS), mpi_errno, MPI_ERR_OTHER,
                             "**psp|connect", "**psp|connect %d", rc);

        /* Add new connection to pg connection table */
        if (pg->vcr[grank] != NULL) {
            /* Update the connection in existing vcr (likely from my_pg) */
            pg->vcr[grank]->con = con;
            pg->cons[grank] = con;      /* for 'lazy disconnect' feature */
        } else {
            /* Create new vcr */
            MPIDI_VC_t *new_vcr = MPIDI_VC_Create(pg, grank, con, remote_lpids[i]);
            MPIR_ERR_CHKANDJUMP(!new_vcr, mpi_errno, MPI_ERR_OTHER, "**nomem");
        }

#if 0
        /* Sanity check and connection warm-up: */
        if (MPIDI_Process.env.enable_direct_connect_spawn) {
            int remote_world_id;
            bool flip_sendrecv = !(MPIDI_Process.my_pg->id_num < pg->id_num);
            int contig;
            size_t data_sz;
            MPIR_Datatype *dtp;
            MPI_Aint true_lb;
            MPIDI_Datatype_get_info(1, MPIR_INT_INTERNAL, contig, data_sz, dtp, true_lb);

            /* Avoid compiler warnings about unused variables: */
            (void) contig;
            (void) true_lb;

            /* We use the newly created pscom connection. The receive is is blocking;
             * We need to be careful with deadlocks here since progress in the pscom
             * is not triggered explicitly. */
            if (!flip_sendrecv) {
                pscom_send(con, NULL, 0, (void *) &(MPIDI_Process.my_pg->world_idx), data_sz);
                rc = pscom_recv_from(con, NULL, 0, (void *) &remote_world_id, data_sz);
                MPIR_Assert(rc == PSCOM_SUCCESS);
            } else {
                rc = pscom_recv_from(con, NULL, 0, (void *) &remote_world_id, data_sz);
                MPIR_Assert(rc == PSCOM_SUCCESS);
                pscom_send(con, NULL, 0, (void *) &(MPIDI_Process.my_pg->world_idx), data_sz);
            }

            MPIR_ERR_CHECK(mpi_errno);
            MPIR_Assert(remote_world_id == 0);  /* world idx of my pg must be 0 */
        }
#endif
    }

    pscom_stop_listen(inter_socket);

  fn_exit:
    MPL_free(ep_str);
    MPL_free(remote_ep_strs);
    MPL_free(remote_ep_strs_displs);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
