/*
 * ParaStation
 *
 * Copyright (C) 2024-2025 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"
#include "mpl.h"
#include "mpiimpl.h"

#define MAX_KEY_LENGTH 50
#define KEY_SETTINGS_CHECK "psmpi-settings"

struct InitMsg {
    int from_rank;
};

/* set endpoint string of rank */
static
int grank2ep_str_set(int rank, char *ep_str)
{
    int mpi_errno = MPI_SUCCESS;
    int pg_size = MPIDI_Process.my_pg_size;

    assert(rank < pg_size);

    if (ep_str) {
        /* Use direct strdup because endpoint strings are freed in atexit handler */
        assert(MPIDI_Process.grank2ep_str[rank] == NULL);
        MPIDI_Process.grank2ep_str[rank] = MPL_direct_strdup(ep_str);
        MPIR_ERR_CHKANDJUMP(!MPIDI_Process.grank2ep_str[rank], mpi_errno, MPI_ERR_OTHER, "**nomem");
    } else {
        MPIDI_Process.grank2ep_str[rank] = NULL;
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* return endpoint string of rank */
static
char *grank2ep_str_get(int rank)
{
    int pg_size = MPIDI_Process.my_pg_size;

    assert(rank < pg_size);

    return MPIDI_Process.grank2ep_str[rank];
}

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
int do_settings_check(char *settings)
{
    int mpi_errno = MPI_SUCCESS;
    int max_len_value = MPIR_pmi_max_val_size();
    int max_len_key = MPIR_pmi_max_key_size();
    int pg_rank = MPIDI_Process.my_pg_rank;
    int pg_size = MPIDI_Process.my_pg_size;
    char *s = NULL;
    char *key = NULL;
    int diffs = 0;

    /* Make sure that settings is a non-null string */
    MPIR_Assert(settings != NULL);

    /* All processes compare their settings to that of all other processes */
    if (MPIDI_Process.env.debug_settings && (pg_size >= 2)) {
        s = MPL_malloc(max_len_value, MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!(s), mpi_errno, MPI_ERR_OTHER, "**nomem");
        key = MPL_malloc(max_len_key, MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(!(key), mpi_errno, MPI_ERR_OTHER, "**nomem");

        for (int i = 0; i < pg_size; i++) {
            if (i == pg_rank) {
                continue;
            }
            memset(s, 0, max_len_value);
            memset(key, 0, max_len_key);

            /* Use key for rank i */
            snprintf(key, max_len_key, "%s.rank-%d", KEY_SETTINGS_CHECK, i);

            mpi_errno = MPIR_pmi_kvs_get(i, key, s, max_len_value);
            MPIR_ERR_CHECK(mpi_errno);

            if (strcmp(s, settings)) {
                if (diffs == 0) {
                    /* Print error msg on first diff */
                    fprintf(stderr,
                            "MPI error: different psmpi settings: own rank %d:'%s' != rank %d:'%s'\n",
                            pg_rank, settings, i, s);
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

/* atexit handler to clean up endpoint strings */
static
void free_grank2ep_str_mapping(void)
{
    if (MPIDI_Process.grank2ep_str) {
        for (int i = 0; i < MPIDI_Process.my_pg_size; i++) {
            MPL_direct_free(MPIDI_Process.grank2ep_str[i]);
        }
        MPL_direct_free(MPIDI_Process.grank2ep_str);
    }
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

    assert(dest_grank < pg_size);

    MPIDI_Process.grank2con[dest_grank] = con;
}

/* return connection */
static
pscom_connection_t *grank2con_get(int dest_grank)
{
    int pg_size = MPIDI_Process.my_pg_size;

    assert(dest_grank < pg_size);

    return MPIDI_Process.grank2con[dest_grank];
}

/* Initialize global connection map */
static
int init_grank2con_mapping(void)
{
    int mpi_errno = MPI_SUCCESS;
    int i;
    int pg_size = MPIDI_Process.my_pg_size;

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
void do_wait(int pg_rank, int src)
{
    /* printf("Accepting (rank %d to %d).\n", src, pg_rank); */
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
int do_connect(pscom_socket_t * socket, int pg_rank, int dest, char *ep_str,
               pscom_connection_t ** con)
{
    int mpi_errno = MPI_SUCCESS;
    pscom_connection_t *_con;
    pscom_err_t rc;

    /* printf("Connecting (rank %d to %d) (%s)\n", pg_rank, dest, ep_str); */
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
int do_connect_direct(pscom_socket_t * socket, int pg_rank, int dest, char *ep_str)
{
    int mpi_errno = MPI_SUCCESS;
    pscom_connection_t *con;
    struct InitMsg init_msg;
    int init_msg_sent = 0;

    /* open pscom connection and connect */
    mpi_errno = do_connect(socket, pg_rank, dest, ep_str, &con);
    MPIR_ERR_CHECK(mpi_errno);

    /* send the initialization message and wait for its completion */
    init_msg.from_rank = pg_rank;
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
int connect_direct(pscom_socket_t * socket)
{
    int mpi_errno = MPI_SUCCESS;
    int i;
    int pg_rank = MPIDI_Process.my_pg_rank;
    int pg_size = MPIDI_Process.my_pg_size;

    /* connect ranks pg_rank..(pg_rank + pg_size/2) */
    for (i = 0; i <= pg_size / 2; i++) {
        int dest = (pg_rank + i) % pg_size;
        int src = (pg_rank + pg_size - i) % pg_size;

        if (!i || (pg_rank / i) % 2) {
            /* connect, accept */
            mpi_errno = do_connect_direct(socket, pg_rank, dest, grank2ep_str_get(dest));
            MPIR_ERR_CHECK(mpi_errno);
            if (!i || src != dest) {
                do_wait(pg_rank, src);
            }
        } else {
            /* accept, connect */
            do_wait(pg_rank, src);
            if (src != dest) {
                mpi_errno = do_connect_direct(socket, pg_rank, dest, grank2ep_str_get(dest));
                MPIR_ERR_CHECK(mpi_errno);
            }
        }
    }

    /* Wait for all connections: (already done?) */
    for (i = 0; i < pg_size; i++) {
        while (!grank2con_get(i)) {
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
int connect_ondemand(pscom_socket_t * socket)
{
    int mpi_errno = MPI_SUCCESS;
    int i;
    int pg_rank = MPIDI_Process.my_pg_rank;
    int pg_size = MPIDI_Process.my_pg_size;

    /* Create all connections */
    for (i = 0; i < pg_size; i++) {
        mpi_errno = do_connect(socket, pg_rank, i, grank2ep_str_get(i), NULL);
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static
int get_ep_str(pscom_socket_t * socket, char **ep_str)
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
    /* For now, getting NULL here is an error. In the future psmpi will get
     * support for cases where we don't get an endpoint string from pscom. */
    MPIR_ERR_CHKANDJUMP1(!_ep_str, mpi_errno, MPI_ERR_OTHER, "**psp|nullendpoint",
                         "**psp|nullendpoint %s", socket->local_con_info.name);

    MPIR_Assert(ep_str);
    *ep_str = MPL_strdup(_ep_str);
    MPIR_ERR_CHKANDJUMP(!(*ep_str), mpi_errno, MPI_ERR_OTHER, "**nomem");

#if MPID_PSP_HAVE_PSCOM_ABI_5
    pscom_socket_free_ep_str(_ep_str);
#endif

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Exchange endpoint strings of all processes */
static
int exchange_ep_strs(pscom_socket_t * socket)
{
    int mpi_errno = MPI_SUCCESS;
    char key[MAX_KEY_LENGTH];
    const char *base_key = "psp-conn";
    int i;
    int pg_rank = MPIDI_Process.my_pg_rank;
    int pg_size = MPIDI_Process.my_pg_size;
    char *ep_str = NULL;
    char *settings = NULL;

    mpi_errno = get_ep_str(socket, &ep_str);
    MPIR_ERR_CHECK(mpi_errno);

    /* For only one process there is no need to exchange any connection infos */
    if (pg_size > 1) {

        mpi_errno = prep_settings_check(&settings);
        MPIR_ERR_CHECK(mpi_errno);

        /* Create KVS key for this rank */
        snprintf(key, MAX_KEY_LENGTH, "%s%i", base_key, pg_rank);

        /* PMI(x)_put and PMI(x)_commit() */
        mpi_errno = MPIR_pmi_kvs_put(key, ep_str);
        MPIR_ERR_CHECK(mpi_errno);

        mpi_errno = MPIR_pmi_barrier();
        MPIR_ERR_CHECK(mpi_errno);

        mpi_errno = do_settings_check(settings);
        MPIR_ERR_CHECK(mpi_errno);
    }

    /* Get endpoints from other processes */
    for (i = 0; i < pg_size; i++) {
        char val[100];

        if (i != pg_rank) {
            /* Erase any content from the (previously used) key */
            memset(key, 0, sizeof(key));
            /* Create KVS key for rank "i" */
            snprintf(key, MAX_KEY_LENGTH, "%s%i", base_key, i);
            /*"i" is the source who published the information */
            mpi_errno = MPIR_pmi_kvs_get(i, key, val, sizeof(val));
            MPIR_ERR_CHECK(mpi_errno);
        } else {
            /* myself: Dont use PMI_KVS_Get, because this fail
             * in the case of no pm (SINGLETON_INIT_BUT_NO_PM) */
            strcpy(val, ep_str);
        }

        mpi_errno = grank2ep_str_set(i, val);
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    MPL_free(ep_str);
    MPL_free(settings);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Initialize all connections (either direct connect mode or ondemand) */
static
int InitConnections(pscom_socket_t * socket)
{
    int mpi_errno = MPI_SUCCESS;
    pscom_err_t rc;

    if (MPIDI_Process.grank2con) {
        /* If the connection map is available, we are in a re-init and kept
         * the connections alive, nothing to do here */
        MPIR_Assert(MPIDI_Process.env.enable_keep_connections >= 1);
        goto fn_exit;
    }

    if (!MPIDI_Process.grank2ep_str) {
        /* Listen on any port, we don't have contact infos yet */
        rc = pscom_listen(socket, PSCOM_ANYPORT);
        MPIR_ERR_CHKANDJUMP1((rc != PSCOM_SUCCESS), mpi_errno, MPI_ERR_OTHER,
                             "**psp|listen_anyport", "**psp|listen_anyport %s", pscom_err_str(rc));

        /* Use direct mem allocation because endpoint strings are freed in atexit handler */
        MPIDI_Process.grank2ep_str =
            MPL_direct_malloc(MPIDI_Process.my_pg_size * sizeof(*MPIDI_Process.grank2ep_str));
        MPIR_ERR_CHKANDJUMP(!MPIDI_Process.grank2ep_str, mpi_errno, MPI_ERR_OTHER, "**nomem");
        for (int i = 0; i < MPIDI_Process.my_pg_size; i++) {
            grank2ep_str_set(i, NULL);
        }
        /* free endpoint strings in atexit handler */
        atexit(free_grank2ep_str_mapping);

        /* Distribute and store endpoint strings */
        mpi_errno = exchange_ep_strs(socket);
        MPIR_ERR_CHECK(mpi_errno);
    } else {
        /* Start to listen again for incoming connections on the port assigned
         * in previous call of pscom_listen */
#ifdef PSCOM_HAS_LISTEN_SUSPEND_RESUME
        pscom_resume_listen(socket);
#else
        int port = 0;
        char *ep_str = MPL_strdup(grank2ep_str_get(MPIDI_Process.my_pg_rank));
        MPIR_ERR_CHKANDJUMP(!ep_str, mpi_errno, MPI_ERR_OTHER, "**nomem");

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

    mpi_errno = init_grank2con_mapping();
    MPIR_ERR_CHECK(mpi_errno);

    if (MPIDI_Process.env.enable_direct_connect) {
        mpi_errno = connect_direct(socket);
    } else {
        mpi_errno = connect_ondemand(socket);
    }
    MPIR_ERR_CHECK(mpi_errno);

#ifdef PSCOM_HAS_LISTEN_SUSPEND_RESUME
    /* Suspend listening for incoming connections (keep the assigned port) */
    pscom_suspend_listen(socket);
#else
    /* Stop listening for incoming connections */
    pscom_stop_listen(socket);
#endif

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Initialize global pscom socket and connections */
int MPIDI_PSP_connection_init(void)
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

    mpi_errno = InitConnections(MPIDI_Process.socket);
    MPIR_ERR_CHECK(mpi_errno);

    MPID_enable_receive_dispach(MPIDI_Process.socket);  /* ToDo: move MPID_enable_receive_dispach to bg thread */

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
