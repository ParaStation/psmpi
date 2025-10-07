/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifdef ENABLE_PMI1

MPID_Thread_mutex_t pmi1_spawn_nb_mutex;

static int pmi1_init(int *has_parent, int *rank, int *size, int *appnum)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;
    int mutex_err;
    static int pmi1_init_count = 0;
    pmi1_init_count++;

    pmi_errno = PMI_Init(has_parent);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_init", "**pmi_init %d", pmi_errno);
    pmi_errno = PMI_Get_rank(rank);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_get_rank", "**pmi_get_rank %d", pmi_errno);
    pmi_errno = PMI_Get_size(size);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_get_size", "**pmi_get_size %d", pmi_errno);
    pmi_errno = PMI_Get_appnum(appnum);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_get_appnum", "**pmi_get_appnum %d", pmi_errno);

    int pmi_max_kvs_name_length;
    pmi_errno = PMI_KVS_Get_name_length_max(&pmi_max_kvs_name_length);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_kvs_get_name_length_max",
                         "**pmi_kvs_get_name_length_max %d", pmi_errno);
    pmi_kvs_name = (char *) MPL_malloc(pmi_max_kvs_name_length, MPL_MEM_OTHER);
    pmi_errno = PMI_KVS_Get_my_name(pmi_kvs_name, pmi_max_kvs_name_length);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_kvs_get_my_name", "**pmi_kvs_get_my_name %d", pmi_errno);

    pmi_errno = PMI_KVS_Get_key_length_max(&pmi_max_key_size);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_kvs_get_key_length_max",
                         "**pmi_kvs_get_key_length_max %d", pmi_errno);
    pmi_errno = PMI_KVS_Get_value_length_max(&pmi_max_val_size);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_kvs_get_value_length_max",
                         "**pmi_kvs_get_value_length_max %d", pmi_errno);

    if (pmi1_init_count == 1) {
        MPID_Thread_mutex_create(&pmi1_spawn_nb_mutex, &mutex_err);
        MPIR_Assert(mutex_err == 0);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static void pmi1_exit(void)
{
    int mutex_err;
    MPID_Thread_mutex_destroy(&pmi1_spawn_nb_mutex, &mutex_err);
    PMI_Finalize();
}

static void pmi1_abort(int exit_code, const char *error_msg)
{
    /* abort messages may contain spaces. Since there is no specific use case for error_msg,
     * simplify it with a stub message. */
    PMI_Abort(exit_code, "abort");
}

static int pmi1_put(const char *key, const char *val)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;

    pmi_errno = PMI_KVS_Put(pmi_kvs_name, key, val);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_kvs_put", "**pmi_kvs_put %d", pmi_errno);
    pmi_errno = PMI_KVS_Commit(pmi_kvs_name);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_kvs_commit", "**pmi_kvs_commit %d", pmi_errno);
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int pmi1_get(int src, const char *key, char *val, int val_size)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;

    /* src is not used in PMI1 */
    pmi_errno = PMI_KVS_Get(pmi_kvs_name, key, val, val_size);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_kvs_get", "**pmi_kvs_get %d", pmi_errno);
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int pmi1_get_parent(const char *key, char *val, int val_size)
{
    return pmi1_get(-1, key, val, val_size);
}

static bool pmi1_get_jobattr(const char *key, char *valbuf)
{
    int pmi_errno = PMI_KVS_Get(pmi_kvs_name, key, valbuf, pmi_max_val_size);
    if (pmi_errno != PMI_SUCCESS) {
        return false;
    }

    /* we either get "unavailable" or a valid filename */
    if (strcmp(valbuf, "unavailable") == 0) {
        return false;
    }

    return true;
}

static int pmi1_barrier(void)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;

    pmi_errno = PMI_Barrier();
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_barrier", "**pmi_barrier %d", pmi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int pmi1_barrier_local(void)
{
    return pmi1_barrier();
}

static int pmi1_optimized_put(const char *key, const char *val, int is_local)
{
    return pmi1_put(key, val);
}

static int pmi1_optimized_get(int src, const char *key, char *val, int valsize, int is_local)
{
    return pmi1_get(src, key, val, valsize);
}

static int pmi1_optional_bcast_barrier(MPIR_PMI_DOMAIN domain)
{
    return pmi1_barrier();
}

static int pmi1_get_universe_size(int *universe_size)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;

    pmi_errno = PMI_Get_universe_size(universe_size);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_get_universe_size", "**pmi_get_universe_size %d", pmi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

typedef struct pmi1_spawn_nb_fn_params {
    int count;
    char **commands;
    char ***argvs;
    int *maxprocs;
    int *info_keyval_sizes;
    INFO_TYPE **info_keyval_vectors;
    int num_preput_keyval;
    INFO_TYPE *preput_vector;
    int *pmi_errcodes;
    MPIR_Request *req;
} pmi1_spawn_nb_fn_params_t;

static void pmi1_spawn_nb_fn(void *data)
{
    int pmi_errno;
    int mutex_err;
    pmi1_spawn_nb_fn_params_t *p = (pmi1_spawn_nb_fn_params_t *) data;

    MPID_Thread_mutex_lock(&pmi1_spawn_nb_mutex, &mutex_err);
    MPIR_Assert(mutex_err == 0);
    pmi_errno = PMI_Spawn_multiple(p->count, (const char **) p->commands, (const char ***) p->argvs,
                                   p->maxprocs, p->info_keyval_sizes,
                                   (const PMI_keyval_t **) p->info_keyval_vectors,
                                   p->num_preput_keyval, (const PMI_keyval_t *) p->preput_vector,
                                   p->pmi_errcodes);
    MPID_Thread_mutex_unlock(&pmi1_spawn_nb_mutex, &mutex_err);
    MPIR_Assert(mutex_err == 0);

    /* Set status of the request */
    if (pmi_errno == PMI_SUCCESS) {
        p->req->status.MPI_ERROR = MPI_SUCCESS;
    } else {
        p->req->status.MPI_ERROR = MPI_ERR_SPAWN;
    }

    /* Mark request as completed */
    MPIR_cc_set(p->req->cc_ptr, 0);

    MPL_free(data);
}

static int pmi1_spawn(int count, char *commands[], char **argvs[], const int maxprocs[],
                      MPIR_Info * info_ptrs[], int num_preput_keyval,
                      struct MPIR_PMI_KEYVAL *preput_keyvals, int *pmi_errcodes, MPIR_Request * req)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;
    INFO_TYPE *preput_vector = NULL;

    int *info_keyval_sizes = NULL;
    INFO_TYPE **info_keyval_vectors = NULL;
    mpi_errno = get_info_kv_vectors(count, info_ptrs, &info_keyval_vectors, &info_keyval_sizes);
    MPIR_ERR_CHECK(mpi_errno);

    if (num_preput_keyval > 0) {
        preput_vector = MPL_malloc(num_preput_keyval * sizeof(INFO_TYPE), MPL_MEM_BUFFER);
        MPIR_ERR_CHKANDJUMP(!preput_vector, mpi_errno, MPI_ERR_OTHER, "**nomem");
        for (int i = 0; i < num_preput_keyval; i++) {
            INFO_TYPE_KEY(preput_vector[i]) = preput_keyvals[i].key;
            INFO_TYPE_VAL(preput_vector[i]) = preput_keyvals[i].val;
        }
    }

    if (req && !req->u.spawn.blocking) {

        MPL_thread_id_t thread_id;
        int thread_err = 0;
        pmi1_spawn_nb_fn_params_t *thread_params;

        thread_params = MPL_malloc(sizeof(pmi1_spawn_nb_fn_params_t), MPL_MEM_OTHER);
        thread_params->count = count;
        thread_params->commands = commands;
        thread_params->argvs = argvs;
        thread_params->maxprocs = (int *) maxprocs;
        thread_params->info_keyval_sizes = info_keyval_sizes;
        thread_params->info_keyval_vectors = info_keyval_vectors;
        thread_params->num_preput_keyval = num_preput_keyval;
        thread_params->preput_vector = preput_vector;
        thread_params->pmi_errcodes = pmi_errcodes;
        thread_params->req = req;

        MPID_Thread_create(pmi1_spawn_nb_fn, thread_params, &thread_id, &thread_err);

        if (thread_err != 0) {
            /* Mark request completed with error state */
            req->status.MPI_ERROR = MPI_ERR_SPAWN;
            MPIR_cc_set(req->cc_ptr, 0);
        }

        MPIR_ERR_CHKANDJUMP1(thread_err != 0, mpi_errno, MPI_ERR_OTHER,
                             "**pmi_spawn_multiple", "**pmi_spawn_multiple %d", thread_err);
    } else {
        pmi_errno = PMI_Spawn_multiple(count, (const char **) commands, (const char ***) argvs,
                                       maxprocs,
                                       info_keyval_sizes, (const PMI_keyval_t **) info_keyval_vectors,
                                       num_preput_keyval, (const PMI_keyval_t *) preput_vector,
                                       pmi_errcodes);

        if (req) {
            /* Set status of the request */
            if (pmi_errno == PMI_SUCCESS) {
                req->status.MPI_ERROR = MPI_SUCCESS;
            } else {
                req->status.MPI_ERROR = MPI_ERR_SPAWN;
            }

            /* Mark request as completed */
            MPIR_cc_set(req->cc_ptr, 0);
        }

        MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                             "**pmi_spawn_multiple", "**pmi_spawn_multiple %d", pmi_errno);
    }

  fn_exit:
    free_pmi_keyvals(info_keyval_vectors, count, info_keyval_sizes);
    if (num_preput_keyval > 0) {
        MPL_free(preput_vector);
    }
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int pmi1_publish(const char name[], const char port[])
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;
    pmi_errno = PMI_Publish_name(name, port);
    MPIR_ERR_CHKANDJUMP1(pmi_errno, mpi_errno, MPI_ERR_NAME, "**namepubnotpub",
                         "**namepubnotpub %s", name);
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int pmi1_lookup(const char name[], char port[], int port_len)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;
#ifdef PMI_MAX_PORT_NAME
    int maxlen = PMI_MAX_PORT_NAME;
#else
    int maxlen = MPI_MAX_PORT_NAME;
#endif
    char *tmpbuf = NULL;

    if (port_len >= maxlen) {
        pmi_errno = PMI_Lookup_name(name, port);
    } else {
        /* allocate a temporary buffer for safety */
        tmpbuf = MPL_malloc(maxlen, MPL_MEM_OTHER);
        pmi_errno = PMI_Lookup_name(name, tmpbuf);
        if (pmi_errno == PMI_SUCCESS) {
            int mpl_err = MPL_strncpy(port, tmpbuf, port_len);
            MPIR_ERR_CHKANDJUMP1(mpl_err, mpi_errno, MPI_ERR_NAME, "**namepubtrunc",
                                 "**namepubtrunc %s", name);
        }
    }
    MPIR_ERR_CHKANDJUMP1(pmi_errno, mpi_errno, MPI_ERR_NAME, "**namepubnotfound",
                         "**namepubnotfound %s", name);

  fn_exit:
    if (tmpbuf) {
        MPL_free(tmpbuf);
    }
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int pmi1_unpublish(const char name[])
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;
    pmi_errno = PMI_Unpublish_name(name);
    MPIR_ERR_CHKANDJUMP1(pmi_errno, mpi_errno, MPI_ERR_SERVICE, "**namepubnotunpub",
                         "**namepubnotunpub %s", name);
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#else /* ENABLE_PMI1 */

static int pmi1_init(int *has_parent, int *rank, int *size, int *appnum)
{
    return MPI_ERR_INTERN;
}

static void pmi1_exit(void)
{
    MPIR_Assert(0);
}

static void pmi1_abort(int exit_code, const char *error_msg)
{
    MPIR_Assert(0);
}

static int pmi1_put(const char *key, const char *val)
{
    return MPI_ERR_INTERN;
}

static int pmi1_get(int src, const char *key, char *val, int val_size)
{
    return MPI_ERR_INTERN;
}

static int pmi1_get_parent(const char *key, char *val, int val_size)
{
    return MPI_ERR_INTERN;
}

static bool pmi1_get_jobattr(const char *key, char *valbuf)
{
    MPIR_Assert(0);
    return false;
}

static int pmi1_barrier(void)
{
    return MPI_ERR_INTERN;
}

static int pmi1_barrier_local(void)
{
    return MPI_ERR_INTERN;
}

static int pmi1_optimized_put(const char *key, const char *val, int is_local)
{
    return MPI_ERR_INTERN;
}

static int pmi1_optimized_get(int src, const char *key, char *val, int valsize, int is_local)
{
    return MPI_ERR_INTERN;
}

static int pmi1_optional_bcast_barrier(MPIR_PMI_DOMAIN domain)
{
    return MPI_ERR_INTERN;
}

static int pmi1_get_universe_size(int *universe_size)
{
    return MPI_ERR_INTERN;
}

static int pmi1_spawn(int count, char *commands[], char **argvs[], const int maxprocs[],
                      MPIR_Info * info_ptrs[], int num_preput_keyval,
                      struct MPIR_PMI_KEYVAL *preput_keyvals, int *pmi_errcodes, MPIR_Request * req)
{
    return MPI_ERR_INTERN;
}

static int pmi1_publish(const char name[], const char port[])
{
    return MPI_ERR_INTERN;
}

static int pmi1_lookup(const char name[], char port[], int port_len)
{
    return MPI_ERR_INTERN;
}

static int pmi1_unpublish(const char name[])
{
    return MPI_ERR_INTERN;
}

#endif /* ENABLE_PMI1 */
