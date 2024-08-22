/*
 * ParaStation
 *
 * Copyright (C) 2024 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpidimpl.h"
#include "mpl.h"
#include "mpiimpl.h"

#define MAX_KEY_LENGTH 50

struct InitMsg {
    int from_rank;
};

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

/* Initialize global listen addresses (port mapping) field */
static
int init_grank_port_mapping(void)
{
    int mpi_errno = MPI_SUCCESS;
    int i;
    int pg_size = MPIDI_Process.my_pg_size;

    MPIDI_Process.grank2con =
        MPL_malloc(sizeof(MPIDI_Process.grank2con[0]) * pg_size, MPL_MEM_OBJECT);
    MPIR_ERR_CHKANDJUMP(!MPIDI_Process.grank2con, mpi_errno, MPI_ERR_OTHER, "**nomem");

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
int do_connect(pscom_socket_t * socket, int pg_rank, int dest, char *dest_addr,
               pscom_connection_t ** con)
{
    int mpi_errno = MPI_SUCCESS;
    pscom_connection_t *_con;
    pscom_err_t rc;

    /* printf("Connecting (rank %d to %d) (%s)\n", pg_rank, dest, dest_addr); */
    _con = pscom_open_connection(socket);
    if (!_con) {
        MPIR_ERR_SETANDJUMP(mpi_errno, MPI_ERR_OTHER, "**psp|openconn");
    }
    rc = pscom_connect_socket_str(_con, dest_addr);
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
int do_connect_direct(pscom_socket_t * socket, int pg_rank, int dest, char *dest_addr)
{
    int mpi_errno = MPI_SUCCESS;
    pscom_connection_t *con;
    struct InitMsg init_msg;
    int init_msg_sent = 0;

    /* open pscom connection and connect */
    mpi_errno = do_connect(socket, pg_rank, dest, dest_addr, &con);
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
int connect_direct(pscom_socket_t * socket, char **psp_port)
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
            mpi_errno = do_connect_direct(socket, pg_rank, dest, psp_port[dest]);
            MPIR_ERR_CHECK(mpi_errno);
            if (!i || src != dest) {
                do_wait(pg_rank, src);
            }
        } else {
            /* accept, connect */
            do_wait(pg_rank, src);
            if (src != dest) {
                mpi_errno = do_connect_direct(socket, pg_rank, dest, psp_port[dest]);
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
int connect_ondemand(pscom_socket_t * socket, char **psp_port)
{
    int mpi_errno = MPI_SUCCESS;
    int i;
    int pg_rank = MPIDI_Process.my_pg_rank;
    int pg_size = MPIDI_Process.my_pg_size;

    /* Create all connections */
    for (i = 0; i < pg_size; i++) {
        mpi_errno = do_connect(socket, pg_rank, i, psp_port[i], NULL);
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Exchange connection information (listen addresses) of all processes via KVS */
static
int exchange_conn_info(pscom_socket_t * socket, unsigned int ondemand, char **psp_port)
{
    int mpi_errno = MPI_SUCCESS;
    char key[MAX_KEY_LENGTH];
    const char *base_key = "psp-conn";
    int i;
    int pg_rank = MPIDI_Process.my_pg_rank;
    int pg_size = MPIDI_Process.my_pg_size;
    char *listen_socket = NULL;

    if (!ondemand) {
        listen_socket = MPL_strdup(pscom_listen_socket_str(socket));
    } else {
        listen_socket = MPL_strdup(pscom_listen_socket_ondemand_str(socket));
    }
    MPIR_ERR_CHKANDJUMP(!listen_socket, mpi_errno, MPI_ERR_OTHER, "**nomem");

    /* For only one process there is no need to exchange any connection infos */
    if (pg_size > 1) {


        /* Create KVS key for this rank */
        snprintf(key, MAX_KEY_LENGTH, "%s%i", base_key, pg_rank);

        /* PMI(x)_put and PMI(x)_commit() */
        mpi_errno = MPIR_pmi_kvs_put(key, listen_socket);
        MPIR_ERR_CHECK(mpi_errno);

        mpi_errno = MPIR_pmi_barrier();
        MPIR_ERR_CHECK(mpi_errno);
    }

    /* Get portlist */
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
            strcpy(val, listen_socket);
        }

        psp_port[i] = MPL_strdup(val);
        MPIR_ERR_CHKANDJUMP(!(psp_port[i]), mpi_errno, MPI_ERR_OTHER, "**nomem");
    }

  fn_exit:
    MPL_free(listen_socket);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Initialize all connections (either direct connect mode or ondemand) */
static
int InitConnections(pscom_socket_t * socket, unsigned int ondemand)
{
    int mpi_errno = MPI_SUCCESS;
    int pg_size = MPIDI_Process.my_pg_size;
    char **psp_port = NULL;

    psp_port = MPL_malloc(pg_size * sizeof(*psp_port), MPL_MEM_OBJECT);
    MPIR_ERR_CHKANDJUMP(!psp_port, mpi_errno, MPI_ERR_OTHER, "**nomem");

    /* Distribute my contact information and fill in port list */
    mpi_errno = exchange_conn_info(socket, ondemand, psp_port);
    MPIR_ERR_CHECK(mpi_errno);

    mpi_errno = init_grank_port_mapping();
    MPIR_ERR_CHECK(mpi_errno);

    if (!ondemand) {
        mpi_errno = connect_direct(socket, psp_port);
    } else {
        mpi_errno = connect_ondemand(socket, psp_port);
    }
    MPIR_ERR_CHECK(mpi_errno);

    /* ToDo: */
    pscom_stop_listen(socket);

  fn_exit:
    if (psp_port) {
        for (int i = 0; i < pg_size; i++) {
            MPL_free(psp_port[i]);
            psp_port[i] = NULL;
        }
        MPL_free(psp_port);
    }
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Initialize global pscom socket and connections */
int MPIDI_PSP_connection_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    pscom_socket_t *socket = NULL;
    pscom_err_t rc;

    socket = pscom_open_socket(0, 0);

    if (!MPIDI_Process.env.enable_ondemand) {
        socket->ops.con_accept = mpid_con_accept;
    }

    {
        char name[10];
        snprintf(name, sizeof(name), "r%07u", (unsigned) MPIDI_Process.my_pg_rank % 100000000);
        pscom_socket_set_name(socket, name);
    }

    rc = pscom_listen(socket, PSCOM_ANYPORT);
    MPIR_ERR_CHKANDJUMP1((rc != PSCOM_SUCCESS), mpi_errno, MPI_ERR_OTHER,
                         "**psp|listen_anyport", "**psp|listen_anyport %d", rc);

    mpi_errno = InitConnections(socket, MPIDI_Process.env.enable_ondemand);
    MPIR_ERR_CHECK(mpi_errno);

    MPID_enable_receive_dispach(socket);        /* ToDo: move MPID_enable_receive_dispach to bg thread */
    MPIDI_Process.socket = socket;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
