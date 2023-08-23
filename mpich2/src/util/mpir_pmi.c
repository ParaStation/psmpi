/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <mpir_pmi.h>
#include <mpir_pset.h>
#include <mpiimpl.h>
#include "mpir_nodemap.h"

/*
=== BEGIN_MPI_T_CVAR_INFO_BLOCK ===

categories:
    - name        : NODEMAP
      description : cvars that control behavior of nodemap

cvars:
    - name        : MPIR_CVAR_NOLOCAL
      category    : NODEMAP
      alt-env     : MPIR_CVAR_NO_LOCAL
      type        : boolean
      default     : false
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        If true, force all processes to operate as though all processes
        are located on another node.  For example, this disables shared
        memory communication hierarchical collectives.

    - name        : MPIR_CVAR_ODD_EVEN_CLIQUES
      category    : NODEMAP
      alt-env     : MPIR_CVAR_EVEN_ODD_CLIQUES
      type        : boolean
      default     : false
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        If true, odd procs on a node are seen as local to each other, and even
        procs on a node are seen as local to each other.  Used for debugging on
        a single machine. Deprecated in favor of MPIR_CVAR_NUM_CLIQUES.

    - name        : MPIR_CVAR_NUM_CLIQUES
      category    : NODEMAP
      type        : int
      default     : 1
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        Specify the number of cliques that should be used to partition procs on
        a local node. Procs with the same clique number are seen as local to
        each other. Used for debugging on a single machine.

    - name        : MPIR_CVAR_CLIQUES_BY_BLOCK
      category    : NODEMAP
      type        : boolean
      default     : false
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        Specify to divide processes into cliques by uniform blocks. The default
        is to divide in round-robin fashion. Used for debugging on a single machine.

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/

#ifdef USE_PMI2_SLURM
#define INFO_TYPE PMI2U_Info
#define INFO_TYPE_KEY(kv) (kv).key
#define INFO_TYPE_VAL(kv) (kv).value

#elif defined(USE_PMI2_CRAY)
#define INFO_TYPE PMI_keyval_t
#define INFO_TYPE_KEY(kv) (kv).key
#define INFO_TYPE_VAL(kv) (kv).val

#elif defined(USE_PMI1_API)
#define INFO_TYPE PMI_keyval_t
#define INFO_TYPE_KEY(kv) (kv).key
#define INFO_TYPE_VAL(kv) (kv).val

#else
#define INFO_TYPE PMI2_keyval_t
#define INFO_TYPE_KEY(kv) (kv).key
#define INFO_TYPE_VAL(kv) (kv).val

#endif

static int build_nodemap(int *nodemap, int sz, int *num_nodes);
static int build_locality(void);

#ifdef USE_PMIX_API
#if PMIX_NUMERIC_VERSION >= 0x00040203
static void pset_define_callback(size_t refid, pmix_status_t status, const pmix_proc_t * source,
                                 pmix_info_t * info, size_t ninfo, pmix_info_t * results,
                                 size_t nresults, pmix_event_notification_cbfunc_fn_t cbfunc,
                                 void *cbdata);
static void pset_delete_callback(size_t refid, pmix_status_t status, const pmix_proc_t * source,
                                 pmix_info_t * info, size_t ninfo, pmix_info_t * results,
                                 size_t nresults, pmix_event_notification_cbfunc_fn_t cbfunc,
                                 void *cbdata);

/* Identifier of pset define event handler */
static int rc_pset_define_handler;

/* Identifier of pset delete event handler */
static int rc_pset_delete_handler;
#endif /* PMIx min version 4.2.3 */
static void pmix_not_supported(const char *elem, char *error_str, int strlen);
#endif /*PMIX API */

static int pmi_version = 1;
static int pmi_subversion = 1;

static int pmi_max_key_size;
static int pmi_max_val_size;

#ifdef USE_PMI1_API
static int pmi_max_kvs_name_length;
static char *pmi_kvs_name;
#elif defined USE_PMI2_API
static char *pmi_jobid;
#elif defined USE_PMIX_API
static pmix_proc_t pmix_proc;
static pmix_proc_t pmix_wcproc;
static pmix_proc_t pmix_parent;
#endif

static char *hwloc_topology_xmlfile;

static void MPIR_pmi_finalize_on_exit(void)
{
#ifdef USE_PMI1_API
    PMI_Finalize();
#elif defined USE_PMI2_API
    PMI2_Finalize();
#elif defined USE_PMIX_API
    PMIx_Finalize(NULL, 0);
#endif
}

int MPIR_pmi_init(void)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;
    static bool pmi_connected = false;

    /* See if the user wants to override our default values */
    MPL_env2int("PMI_VERSION", &pmi_version);
    MPL_env2int("PMI_SUBVERSION", &pmi_subversion);

    int has_parent, rank, size, appnum;
#ifdef USE_PMI1_API
    pmi_errno = PMI_Init(&has_parent);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_init", "**pmi_init %d", pmi_errno);
    pmi_errno = PMI_Get_rank(&rank);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_get_rank", "**pmi_get_rank %d", pmi_errno);
    pmi_errno = PMI_Get_size(&size);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_get_size", "**pmi_get_size %d", pmi_errno);
    pmi_errno = PMI_Get_appnum(&appnum);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_get_appnum", "**pmi_get_appnum %d", pmi_errno);

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

#elif defined USE_PMI2_API
    pmi_max_key_size = PMI2_MAX_KEYLEN;
    pmi_max_val_size = PMI2_MAX_VALLEN;

    pmi_errno = PMI2_Init(&has_parent, &size, &rank, &appnum);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI2_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_init", "**pmi_init %d", pmi_errno);

    pmi_jobid = (char *) MPL_malloc(PMI2_MAX_VALLEN, MPL_MEM_OTHER);
    pmi_errno = PMI2_Job_GetId(pmi_jobid, PMI2_MAX_VALLEN);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI2_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_job_getid", "**pmi_job_getid %d", pmi_errno);

#elif defined USE_PMIX_API
    pmi_max_key_size = PMIX_MAX_KEYLEN;
    pmi_max_val_size = 1024;    /* this is what PMI2_MAX_VALLEN currently set to */

    pmix_value_t *pvalue = NULL;

    pmi_errno = PMIx_Init(&pmix_proc, NULL, 0);
    if (pmi_errno == PMIX_ERR_UNREACH) {
        /* no pmi server, assume we are a singleton */
        rank = 0;
        size = 1;
        appnum = 0;
        has_parent = 0;
        goto singleton_out;
    }
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_init", "**pmix_init %d", pmi_errno);

    /*check overflow because of type mismatch between pmix_proc_t.rank (uint32_t) and MPIR_Process_t.rank (int) */
    MPIR_Assert(pmix_proc.rank <= INT_MAX);

    rank = (int) pmix_proc.rank;
    PMIX_PROC_CONSTRUCT(&pmix_wcproc);
    MPL_strncpy(pmix_wcproc.nspace, pmix_proc.nspace, PMIX_MAX_NSLEN);
    pmix_wcproc.rank = PMIX_RANK_WILDCARD;

    pmi_errno = PMIx_Get(&pmix_wcproc, PMIX_JOB_SIZE, NULL, 0, &pvalue);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_get", "**pmix_get %d", pmi_errno);
    size = pvalue->data.uint32;
    PMIX_VALUE_RELEASE(pvalue);

    PMIX_PROC_CONSTRUCT(&pmix_parent);
    pmi_errno = PMIx_Get(&pmix_proc, PMIX_PARENT_ID, NULL, 0, &pvalue);
    if (pmi_errno == PMIX_ERR_NOT_FOUND) {
        has_parent = 0; /* process not spawned */
    } else if (pmi_errno == PMIX_SUCCESS) {
        has_parent = 1; /* spawned process */
        PMIX_PROC_LOAD(&pmix_parent, pvalue->data.proc->nspace, pvalue->data.proc->rank);
        PMIX_VALUE_RELEASE(pvalue);
    } else {
        MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER, "**pmix_get", "**pmix_get %s",
                             PMIx_Error_string(pmi_errno));
    }

    /* Get the appnum */
    pmi_errno = PMIx_Get(&pmix_proc, PMIX_APPNUM, NULL, 0, &pvalue);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_get", "**pmix_get %s", PMIx_Error_string(pmi_errno));
    MPIR_Assert(pvalue->data.uint32 <= INT_MAX);        /* overflow check */
    appnum = (int) pvalue->data.uint32;
    PMIX_VALUE_RELEASE(pvalue);

  singleton_out:

#endif

    if (!pmi_connected) {
        /* Register finalization of PM connection in exit handler */
        mpi_errno = atexit(MPIR_pmi_finalize_on_exit);
        MPIR_ERR_CHKANDJUMP1(mpi_errno != 0, mpi_errno, MPI_ERR_OTHER,
                             "**atexit_pmi_finalize", "**atexit_pmi_finalize %d", mpi_errno);

        pmi_connected = true;
    }

    MPIR_Process.has_parent = has_parent;
    MPIR_Process.rank = rank;
    MPIR_Process.size = size;
    MPIR_Process.appnum = appnum;

    MPIR_Process.node_map = (int *) MPL_malloc(size * sizeof(int), MPL_MEM_ADDRESS);

    mpi_errno = build_nodemap(MPIR_Process.node_map, size, &MPIR_Process.num_nodes);
    MPIR_ERR_CHECK(mpi_errno);

    /* allocate and populate MPIR_Process.node_local_map and MPIR_Process.node_root_map */
    mpi_errno = build_locality();

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

void MPIR_pmi_finalize(void)
{
    /* Finalize of PM interface happens in exit handler,
     * here: free allocated memory */
#ifdef USE_PMI1_API
    MPL_free(pmi_kvs_name);
#elif defined(USE_PMI2_API)
    MPL_free(pmi_jobid);
#elif defined(USE_PMIX_API)
    /* pmix_proc does not need free */
#endif

    MPL_free(MPIR_Process.node_map);
    MPL_free(MPIR_Process.node_root_map);
    MPL_free(MPIR_Process.node_local_map);

    MPL_free(hwloc_topology_xmlfile);
}

void MPIR_pmi_abort(int exit_code, const char *error_msg)
{
#ifdef USE_PMI1_API
    PMI_Abort(exit_code, error_msg);
#elif defined(USE_PMI2_API)
    PMI2_Abort(TRUE, error_msg);
#elif defined(USE_PMIX_API)
    PMIx_Abort(exit_code, error_msg, NULL, 0);
#endif
}

/* This function is currently unused in MPICH because we always call
 * PMI functions from a single thread or within a critical section.
 */
int MPIR_pmi_set_threaded(int is_threaded)
{
#if defined(USE_PMI2_API) && !defined(USE_PMI2_SLURM) && !defined(USE_PMI2_CRAY)
    PMI2_Set_threaded(is_threaded);
#endif
    return MPI_SUCCESS;
}

/* getters for internal constants */
int MPIR_pmi_max_key_size(void)
{
    return pmi_max_key_size;
}

int MPIR_pmi_max_val_size(void)
{
    return pmi_max_val_size;
}

char *MPIR_pmi_get_hwloc_xmlfile(void)
{
    char *valbuf = NULL;

    /* try to get hwloc topology file */
    if (hwloc_topology_xmlfile == NULL && MPIR_Process.local_size > 1) {
        valbuf = MPL_malloc(pmi_max_val_size, MPL_MEM_OTHER);
        if (!valbuf) {
            goto fn_exit;
        }
#ifdef USE_PMI1_API
        int pmi_errno = PMI_KVS_Get(pmi_kvs_name, "PMI_hwloc_xmlfile", valbuf, pmi_max_val_size);
        if (pmi_errno != MPI_SUCCESS) {
            goto fn_exit;
        }

        /* we either get "unavailable" or a valid filename */
        if (strcmp(valbuf, "unavailable") != 0) {
            hwloc_topology_xmlfile = MPL_strdup(valbuf);
        }
#elif defined USE_PMI2_API
        int found;
        int pmi_errno = PMI2_Info_GetJobAttr("PMI_hwloc_xmlfile", valbuf, pmi_max_val_size,
                                             &found);
        if (pmi_errno != MPI_SUCCESS) {
            MPL_free(valbuf);
            goto fn_exit;
        }

        if (found) {
            hwloc_topology_xmlfile = MPL_strdup(valbuf);
        }
#endif
    }

  fn_exit:
    MPL_free(valbuf);
    return hwloc_topology_xmlfile;
}

const char *MPIR_pmi_job_id(void)
{
#ifdef USE_PMI1_API
    return (const char *) pmi_kvs_name;
#elif defined USE_PMI2_API
    return (const char *) pmi_jobid;
#elif defined USE_PMIX_API
    return (const char *) pmix_proc.nspace;
#endif
}

/* wrapper functions */
int MPIR_pmi_kvs_put(const char *key, const char *val)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;

#ifdef USE_PMI1_API
    pmi_errno = PMI_KVS_Put(pmi_kvs_name, key, val);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_kvs_put", "**pmi_kvs_put %d", pmi_errno);
    pmi_errno = PMI_KVS_Commit(pmi_kvs_name);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_kvs_commit", "**pmi_kvs_commit %d", pmi_errno);
#elif defined(USE_PMI2_API)
    pmi_errno = PMI2_KVS_Put(key, val);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI2_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_kvsput", "**pmi_kvsput %d", pmi_errno);
#elif defined(USE_PMIX_API)
    pmix_value_t value;
    value.type = PMIX_STRING;
    value.data.string = (char *) val;
    pmi_errno = PMIx_Put(PMIX_GLOBAL, key, &value);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_put", "**pmix_put %d", pmi_errno);
    pmi_errno = PMIx_Commit();
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_commit", "**pmix_commit %d", pmi_errno);
#endif

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* NOTE: src is a hint, use src = -1 if not known */
int MPIR_pmi_kvs_get(int src, const char *key, char *val, int val_size)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;

#ifdef USE_PMI1_API
    /* src is not used in PMI1 */
    pmi_errno = PMI_KVS_Get(pmi_kvs_name, key, val, val_size);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_kvs_get", "**pmi_kvs_get %d", pmi_errno);
#elif defined(USE_PMI2_API)
    if (src < 0)
        src = PMI2_ID_NULL;
    int out_len;
    pmi_errno = PMI2_KVS_Get(pmi_jobid, src, key, val, val_size, &out_len);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI2_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_kvsget", "**pmi_kvsget %d", pmi_errno);
#elif defined(USE_PMIX_API)
    pmix_value_t *pvalue;
    if (src < 0) {
        pmi_errno = PMIx_Get(NULL, key, NULL, 0, &pvalue);
    } else {
        pmix_proc_t proc;
        PMIX_PROC_CONSTRUCT(&proc);
        proc.rank = src;

        pmi_errno = PMIx_Get(&proc, key, NULL, 0, &pvalue);
    }
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_get", "**pmix_get %d", pmi_errno);
    MPL_strncpy(val, pvalue->data.string, val_size);
    PMIX_VALUE_RELEASE(pvalue);
#endif

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_pmi_kvs_parent_get(const char *key, char *val, int val_size)
{
    int mpi_errno = MPI_SUCCESS;

    /* Process needs to have a parent to use this function */
    if (!MPIR_Process.has_parent) {
        mpi_errno = MPI_ERR_INTERN;
        goto fn_fail;
    }
#ifdef USE_PMI1_API
    mpi_errno = MPIR_pmi_kvs_get(-1, key, val, val_size);
    MPIR_ERR_CHECK(mpi_errno);
#elif defined(USE_PMI2_API)
    mpi_errno = MPIR_pmi_kvs_get(PMI2_ID_NULL, key, val, val_size);
    MPIR_ERR_CHECK(mpi_errno);
#elif defined(USE_PMIX_API)
    int pmi_errno = PMIX_SUCCESS;
    pmix_value_t *pvalue;
    pmi_errno = PMIx_Get(&pmix_parent, key, NULL, 0, &pvalue);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER, "**pmix_get",
                         "**pmix_get %s", PMIx_Error_string(pmi_errno));
    MPL_strncpy(val, pvalue->data.string, val_size);
    PMIX_VALUE_RELEASE(pvalue);
#endif

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* ---- utils functions ---- */

int MPIR_pmi_barrier(void)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;

#ifdef USE_PMI1_API
    pmi_errno = PMI_Barrier();
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_barrier", "**pmi_barrier %d", pmi_errno);
#elif defined(USE_PMI2_API)
    pmi_errno = PMI2_KVS_Fence();
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI2_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_kvsfence", "**pmi_kvsfence %d", pmi_errno);
    /* Get a non-existent key, it only returns after every process called fence */
    int out_len;
    PMI2_KVS_Get(pmi_jobid, PMI2_ID_NULL, "-NONEXIST-KEY", NULL, 0, &out_len);
#elif defined(USE_PMIX_API)
    pmix_info_t *info;
    PMIX_INFO_CREATE(info, 1);
    int flag = 1;
    PMIX_INFO_LOAD(info, PMIX_COLLECT_DATA, &flag, PMIX_BOOL);

    /* use global wildcard proc set */
    pmi_errno = PMIx_Fence(&pmix_wcproc, 1, info, 1);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_fence", "**pmix_fence %d", pmi_errno);
    PMIX_INFO_FREE(info, 1);
#endif

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_pmi_barrier_local(void)
{
#if defined(USE_PMIX_API)
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;
    int local_size = MPIR_Process.local_size;
    pmix_proc_t *procs = NULL;
    pmix_info_t *info;
    int flag = 1;

    PMIX_PROC_CREATE(procs, local_size);
    PMIX_INFO_CREATE(info, 1);
    for (int i = 0; i < local_size; i++) {
        PMIX_LOAD_PROCID(&(procs[i]), pmix_proc.nspace, MPIR_Process.node_local_map[i]);
    }
    PMIX_INFO_LOAD(info, PMIX_COLLECT_DATA, &flag, PMIX_BOOL);

    pmi_errno = PMIx_Fence(procs, local_size, info, 1);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER, "**pmix_fence",
                         "**pmix_fence %d", pmi_errno);
  fn_exit:
    PMIX_INFO_FREE(info, 1);
    PMIX_PROC_FREE(procs, local_size);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
#else
    /* If local barrier is not supported (PMI1 and PMI2), simply fallback */
    return MPIR_pmi_barrier();
#endif
}

#if defined(USE_PMI1_API) || defined(USE_PMI2_API)
/* declare static functions used in bcast/allgather */
static void encode(int size, const char *src, char *dest);
static void decode(int size, const char *src, char *dest);


/* is_local is a hint that we optimize for node local access when we can */
static int optimized_put(const char *key, const char *val, int is_local)
{
    int mpi_errno = MPI_SUCCESS;
#if defined(USE_PMI1_API)
    mpi_errno = MPIR_pmi_kvs_put(key, val);
    MPIR_ERR_CHECK(mpi_errno);
#elif defined(USE_PMI2_API)
    if (!is_local) {
        mpi_errno = MPIR_pmi_kvs_put(key, val);
    } else {
        int pmi_errno = PMI2_Info_PutNodeAttr(key, val);
        MPIR_ERR_CHKANDJUMP(pmi_errno != PMI2_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                            "**pmi_putnodeattr");
    }
#elif defined(USE_PMIX_API)
    int pmi_errno;
    pmix_value_t value;
    value.type = PMIX_STRING;
    value.data.string = (char *) val;
    pmi_errno = PMIx_Put(is_local ? PMIX_LOCAL : PMIX_GLOBAL, key, &value);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_put", "**pmix_put %d", pmi_errno);
    pmi_errno = PMIx_Commit();
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_commit", "**pmix_commit %d", pmi_errno);
#endif

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int optimized_get(int src, const char *key, char *val, int valsize, int is_local)
{
#if defined(USE_PMI1_API)
    return MPIR_pmi_kvs_get(src, key, val, valsize);
#elif defined(USE_PMI2_API)
    if (is_local) {
        int mpi_errno = MPI_SUCCESS;
        int found;
        int pmi_errno = PMI2_Info_GetNodeAttr(key, val, valsize, &found, TRUE);
        if (pmi_errno != PMI2_SUCCESS) {
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**pmi_getnodeattr");
        } else if (!found) {
            MPIR_ERR_SET(mpi_errno, MPI_ERR_OTHER, "**pmi_getnodeattr");
        }
        return mpi_errno;
    } else {
        return MPIR_pmi_kvs_get(src, key, val, valsize);
    }
#else
    return MPIR_pmi_kvs_get(src, key, val, valsize);
#endif
}
#endif

/* higher-level binary put/get:
 * 1. binary encoding/decoding
 * 2. chops long values into multiple segments
 * 3. uses optimized_put/get for the case of node-level access
 */
static int put_ex(const char *key, const void *buf, int bufsize, int is_local)
{
    int mpi_errno = MPI_SUCCESS;
#if defined(USE_PMI1_API) || defined(USE_PMI2_API)
    char *val = MPL_malloc(pmi_max_val_size, MPL_MEM_OTHER);
    /* reserve some spaces for '\0' and maybe newlines
     * (depends on pmi implementations, and may not be sufficient) */
    int segsize = (pmi_max_val_size - 2) / 2;
    if (bufsize < segsize) {
        encode(bufsize, buf, val);
        mpi_errno = optimized_put(key, val, is_local);
        MPIR_ERR_CHECK(mpi_errno);
    } else {
        int num_segs = bufsize / segsize;
        if (bufsize % segsize > 0) {
            num_segs++;
        }
        MPL_snprintf(val, pmi_max_val_size, "segments=%d", num_segs);
        mpi_errno = MPIR_pmi_kvs_put(key, val);
        MPIR_ERR_CHECK(mpi_errno);
        for (int i = 0; i < num_segs; i++) {
            char seg_key[50];
            sprintf(seg_key, "%s-seg-%d/%d", key, i + 1, num_segs);
            int n = segsize;
            if (i == num_segs - 1) {
                n = bufsize - segsize * (num_segs - 1);
            }
            encode(n, (char *) buf + i * segsize, val);
            mpi_errno = optimized_put(seg_key, val, is_local);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }
#elif defined(USE_PMIX_API)
    int pmi_errno;
    pmix_value_t value;
    value.type = PMIX_BYTE_OBJECT;
    value.data.bo.bytes = (char *) buf;
    value.data.bo.size = bufsize;
    pmi_errno = PMIx_Put(is_local ? PMIX_LOCAL : PMIX_GLOBAL, key, &value);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_put", "**pmix_put %d", pmi_errno);
    pmi_errno = PMIx_Commit();
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_commit", "**pmix_commit %d", pmi_errno);
#endif

  fn_exit:
#if defined(USE_PMI1_API) || defined(USE_PMI2_API)
    MPL_free(val);
#endif
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int get_ex(int src, const char *key, void *buf, int *p_size, int is_local)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_Assert(p_size);
    MPIR_Assert(*p_size > 0);
    int bufsize = *p_size;
#if defined(USE_PMI1_API) || defined(USE_PMI2_API)
    char *val = MPL_malloc(pmi_max_val_size, MPL_MEM_OTHER);
    int segsize = (pmi_max_val_size - 1) / 2;

    int got_size;

    mpi_errno = optimized_get(src, key, val, pmi_max_val_size, is_local);
    MPIR_ERR_CHECK(mpi_errno);
    if (strncmp(val, "segments=", 9) == 0) {
        int num_segs = atoi(val + 9);
        got_size = 0;
        for (int i = 0; i < num_segs; i++) {
            char seg_key[50];
            sprintf(seg_key, "%s-seg-%d/%d", key, i + 1, num_segs);
            mpi_errno = optimized_get(src, seg_key, val, pmi_max_val_size, is_local);
            MPIR_ERR_CHECK(mpi_errno);
            int n = strlen(val) / 2;    /* 2-to-1 decode */
            if (i < num_segs - 1) {
                MPIR_Assert(n == segsize);
            } else {
                MPIR_Assert(n <= segsize);
            }
            decode(n, val, (char *) buf + i * segsize);
            got_size += n;
        }
    } else {
        int n = strlen(val) / 2;        /* 2-to-1 decode */
        decode(n, val, (char *) buf);
        got_size = n;
    }
    MPIR_Assert(got_size <= bufsize);
    if (got_size < bufsize) {
        ((char *) buf)[got_size] = '\0';
    }

    *p_size = got_size;

#elif defined(USE_PMIX_API)
    int pmi_errno;
    pmix_value_t *pvalue;
    if (src < 0) {
        pmi_errno = PMIx_Get(NULL, key, NULL, 0, &pvalue);
    } else {
        pmix_proc_t proc;
        PMIX_PROC_CONSTRUCT(&proc);
        proc.rank = src;

        pmi_errno = PMIx_Get(&proc, key, NULL, 0, &pvalue);
    }
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_get", "**pmix_get %d", pmi_errno);
    MPIR_Assert(pvalue->type == PMIX_BYTE_OBJECT);
    MPIR_Assert(pvalue->data.bo.size <= bufsize);

    memcpy(buf, pvalue->data.bo.bytes, pvalue->data.bo.size);
    *p_size = pvalue->data.bo.size;

    PMIX_VALUE_RELEASE(pvalue);
#endif

  fn_exit:
#if defined(USE_PMI1_API) || defined(USE_PMI2_API)
    MPL_free(val);
#endif
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int optional_bcast_barrier(MPIR_PMI_DOMAIN domain)
{
#if defined(USE_PMI1_API)
    /* unless bcast is skipped altogether */
    if (domain == MPIR_PMI_DOMAIN_ALL && MPIR_Process.size == 1) {
        return MPI_SUCCESS;
    } else if (domain == MPIR_PMI_DOMAIN_NODE_ROOTS && MPIR_Process.num_nodes == 1) {
        return MPI_SUCCESS;
    } else if (domain == MPIR_PMI_DOMAIN_LOCAL && MPIR_Process.size == MPIR_Process.num_nodes) {
        return MPI_SUCCESS;
    }
#elif defined(USE_PMI2_API)
    if (domain == MPIR_PMI_DOMAIN_ALL && MPIR_Process.size == 1) {
        return MPI_SUCCESS;
    } else if (domain == MPIR_PMI_DOMAIN_NODE_ROOTS && MPIR_Process.num_nodes == 1) {
        return MPI_SUCCESS;
    } else if (domain == MPIR_PMI_DOMAIN_LOCAL) {
        /* PMI2 local uses Put/GetNodeAttr, no need for barrier */
        return MPI_SUCCESS;
    }
#elif defined(USE_PMIx_API)
    /* PMIx will block/wait, so barrier unnecessary */
    return MPI_SUCCESS;
#endif
    return MPIR_pmi_barrier();
}

int MPIR_pmi_bcast(void *buf, int bufsize, MPIR_PMI_DOMAIN domain)
{
    int mpi_errno = MPI_SUCCESS;

    int rank = MPIR_Process.rank;
    int local_node_id = MPIR_Process.node_map[rank];
    int node_root = MPIR_Process.node_root_map[local_node_id];
    int is_node_root = (node_root == rank);

    int in_domain, is_root, is_local, bcast_size;
    if (domain == MPIR_PMI_DOMAIN_NODE_ROOTS && !is_node_root) {
        in_domain = 0;
    } else {
        in_domain = 1;
    }
    if (rank == 0 || (domain == MPIR_PMI_DOMAIN_LOCAL && is_node_root)) {
        is_root = 1;
    } else {
        is_root = 0;
    }
    is_local = (domain == MPIR_PMI_DOMAIN_LOCAL);

    bcast_size = MPIR_Process.size;
    if (domain == MPIR_PMI_DOMAIN_NODE_ROOTS) {
        bcast_size = MPIR_Process.num_nodes;
    } else if (domain == MPIR_PMI_DOMAIN_LOCAL) {
        bcast_size = MPIR_Process.local_size;
    }
    if (bcast_size == 1) {
        in_domain = 0;
    }

    char key[50];
    int root;
    static int bcast_seq = 0;

    if (!in_domain) {
        /* PMI_Barrier may require all process to participate */
        mpi_errno = optional_bcast_barrier(domain);
        MPIR_ERR_CHECK(mpi_errno);
    } else {
        MPIR_Assert(buf);
        MPIR_Assert(bufsize > 0);

        bcast_seq++;

        root = 0;
        if (domain == MPIR_PMI_DOMAIN_LOCAL) {
            root = node_root;
        }
        /* add root to the key since potentially we may have multiple root(s)
         * on a single node due to odd-even-cliques */
        sprintf(key, "-bcast-%d-%d", bcast_seq, root);

        if (is_root) {
            mpi_errno = put_ex(key, buf, bufsize, is_local);
            MPIR_ERR_CHECK(mpi_errno);
        }

        mpi_errno = optional_bcast_barrier(domain);
        MPIR_ERR_CHECK(mpi_errno);

        if (!is_root) {
            int got_size = bufsize;
            mpi_errno = get_ex(root, key, buf, &got_size, is_local);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_pmi_allgather(const void *sendbuf, int sendsize, void *recvbuf, int recvsize,
                       MPIR_PMI_DOMAIN domain)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_Assert(domain != MPIR_PMI_DOMAIN_LOCAL);

    int local_node_id = MPIR_Process.node_map[MPIR_Process.rank];
    int is_node_root = (MPIR_Process.node_root_map[local_node_id] == MPIR_Process.rank);
    int in_domain = 1;
    if (domain == MPIR_PMI_DOMAIN_NODE_ROOTS && !is_node_root) {
        in_domain = 0;
    }

    static int allgather_seq = 0;
    allgather_seq++;

    char key[50];
    sprintf(key, "-allgather-%d-%d", allgather_seq, MPIR_Process.rank);

    if (in_domain) {
        mpi_errno = put_ex(key, sendbuf, sendsize, 0);
        MPIR_ERR_CHECK(mpi_errno);
    }
#ifndef USE_PMIX_API
    /* PMIx will wait, so barrier unnecessary */
    mpi_errno = MPIR_pmi_barrier();
    MPIR_ERR_CHECK(mpi_errno);
#endif

    if (in_domain) {
        int domain_size = MPIR_Process.size;
        if (domain == MPIR_PMI_DOMAIN_NODE_ROOTS) {
            domain_size = MPIR_Process.num_nodes;
        }
        for (int i = 0; i < domain_size; i++) {
            int rank = i;
            if (domain == MPIR_PMI_DOMAIN_NODE_ROOTS) {
                rank = MPIR_Process.node_root_map[i];
            }
            sprintf(key, "-allgather-%d-%d", allgather_seq, rank);
            int got_size = recvsize;
            mpi_errno = get_ex(rank, key, (unsigned char *) recvbuf + i * recvsize, &got_size, 0);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* This version assumes shm_buf is shared across local procs. Each process
 * participate in the gather part by distributing the task over local procs.
 *
 * NOTE: the behavior is different with MPIR_pmi_allgather when domain is
 * MPIR_PMI_DOMAIN_NODE_ROOTS. With MPIR_pmi_allgather, only the root_nodes participate.
 */
int MPIR_pmi_allgather_shm(const void *sendbuf, int sendsize, void *shm_buf, int recvsize,
                           MPIR_PMI_DOMAIN domain)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_Assert(domain != MPIR_PMI_DOMAIN_LOCAL);

    int rank = MPIR_Process.rank;
    int size = MPIR_Process.size;
    int local_size = MPIR_Process.local_size;
    int local_rank = MPIR_Process.local_rank;
    int local_node_id = MPIR_Process.node_map[rank];
    int node_root = MPIR_Process.node_root_map[local_node_id];
    int is_node_root = (node_root == MPIR_Process.rank);

    static int allgather_shm_seq = 0;
    allgather_shm_seq++;

    char key[50];
    sprintf(key, "-allgather-shm-%d-%d", allgather_shm_seq, rank);

    /* in roots-only, non-roots would skip the put */
    if (domain != MPIR_PMI_DOMAIN_NODE_ROOTS || is_node_root) {
        mpi_errno = put_ex(key, (unsigned char *) sendbuf, sendsize, 0);
        MPIR_ERR_CHECK(mpi_errno);
    }

    mpi_errno = MPIR_pmi_barrier();
    MPIR_ERR_CHECK(mpi_errno);

    /* Each rank need get val from "size" ranks, divide the task evenly over local ranks */
    if (domain == MPIR_PMI_DOMAIN_NODE_ROOTS) {
        size = MPIR_Process.num_nodes;
    }
    int per_local_rank = size / local_size;
    if (per_local_rank * local_size < size) {
        per_local_rank++;
    }
    int start = local_rank * per_local_rank;
    int end = start + per_local_rank;
    if (end > size) {
        end = size;
    }
    for (int i = start; i < end; i++) {
        int src = i;
        if (domain == MPIR_PMI_DOMAIN_NODE_ROOTS) {
            src = MPIR_Process.node_root_map[i];
        }
        sprintf(key, "-allgather-shm-%d-%d", allgather_shm_seq, src);
        int got_size = recvsize;
        mpi_errno = get_ex(src, key, (unsigned char *) shm_buf + i * recvsize, &got_size, 0);
        MPIR_ERR_CHECK(mpi_errno);
        MPIR_Assert(got_size <= recvsize);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_pmi_get_universe_size(int *universe_size)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;

#ifdef USE_PMI1_API
    pmi_errno = PMI_Get_universe_size(universe_size);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_get_universe_size", "**pmi_get_universe_size %d", pmi_errno);
#elif defined(USE_PMI2_API)
    char val[PMI2_MAX_VALLEN];
    int found = 0;
    char *endptr;

    pmi_errno = PMI2_Info_GetJobAttr("universeSize", val, sizeof(val), &found);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI2_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_getjobattr", "**pmi_getjobattr %d", pmi_errno);
    if (!found) {
        *universe_size = MPIR_UNIVERSE_SIZE_NOT_AVAILABLE;
    } else {
        *universe_size = strtol(val, &endptr, 0);
        MPIR_ERR_CHKINTERNAL(endptr - val != strlen(val), mpi_errno, "can't parse universe size");
    }
#elif defined(USE_PMIX_API)
    pmix_value_t *pvalue = NULL;

    pmi_errno = PMIx_Get(&pmix_wcproc, PMIX_UNIV_SIZE, NULL, 0, &pvalue);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_get", "**pmix_get %d", pmi_errno);
    *universe_size = pvalue->data.uint32;
    PMIX_VALUE_RELEASE(pvalue);
#endif
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

char *MPIR_pmi_get_failed_procs(void)
{
    char *failed_procs_string = NULL;

    failed_procs_string = MPL_malloc(pmi_max_val_size, MPL_MEM_OTHER);
    MPIR_Assert(failed_procs_string);
#ifdef USE_PMI1_API
    int pmi_errno;
    pmi_errno = PMI_KVS_Get(pmi_kvs_name, "PMI_dead_processes",
                            failed_procs_string, pmi_max_val_size);
    if (pmi_errno != PMI_SUCCESS)
        goto fn_fail;
#elif defined(USE_PMI2_API)
    int out_len;
    int pmi_errno;
    pmi_errno = PMI2_KVS_Get(pmi_jobid, PMI2_ID_NULL, "PMI_dead_processes",
                             failed_procs_string, pmi_max_val_size, &out_len);
    if (pmi_errno != PMI2_SUCCESS)
        goto fn_fail;
#elif defined(USE_PMIX_API)
    goto fn_fail;
#endif

  fn_exit:
    return failed_procs_string;
  fn_fail:
    /* FIXME: appropriate error messages here? */
    MPL_free(failed_procs_string);
    failed_procs_string = NULL;
    goto fn_exit;
}

/* static functions only for MPIR_pmi_spawn_multiple */
#if defined(USE_PMI1_API) || defined(USE_PMI2_API)
static int mpi_to_pmi_keyvals(MPIR_Info * info_ptr, INFO_TYPE ** kv_ptr, int *nkeys_ptr);
static void free_pmi_keyvals(INFO_TYPE ** kv, int size, int *counts);
#elif defined(USE_PMIX_API)
static int pmix_build_job_info(MPIR_Info * info_ptr, pmix_info_t ** pmix_job_info,
                               size_t * njob_info, char **path);
static int pmix_add_to_info(MPIR_Info * info_ptr, const char *key, const char *pmix_key,
                            MPIR_Info * target_ptr, int *key_found, size_t * counter, char **value);
static int mpi_to_pmix_keyvals(MPIR_Info * info_ptr, int ninfo, pmix_info_t ** pmix_info);
#endif

/* NOTE: MPIR_pmi_spawn_multiple is to be called by a single root spawning process */
int MPIR_pmi_spawn_multiple(int count, char *commands[], char **argvs[],
                            const int maxprocs[], MPIR_Info * info_ptrs[],
                            int num_preput_keyval, struct MPIR_PMI_KEYVAL *preput_keyvals,
                            int *pmi_errcodes)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;

#if defined(USE_PMI1_API) || defined(USE_PMI2_API)

    int *info_keyval_sizes = NULL;
    INFO_TYPE **info_keyval_vectors = NULL;
#if defined(USE_PMI2_SLURM)
    const INFO_TYPE **preput_vector = NULL;
    INFO_TYPE *preput_vector_array = NULL;
#else
    INFO_TYPE *preput_vector = NULL;
#endif

    info_keyval_sizes = (int *) MPL_malloc(count * sizeof(int), MPL_MEM_BUFFER);
    MPIR_ERR_CHKANDJUMP(!info_keyval_sizes, mpi_errno, MPI_ERR_OTHER, "**nomem");

    info_keyval_vectors = (INFO_TYPE **) MPL_malloc(count * sizeof(INFO_TYPE *), MPL_MEM_BUFFER);
    MPIR_ERR_CHKANDJUMP(!info_keyval_vectors, mpi_errno, MPI_ERR_OTHER, "**nomem");

    if (!info_ptrs) {
        for (int i = 0; i < count; i++) {
            info_keyval_vectors[i] = 0;
            info_keyval_sizes[i] = 0;
        }
    } else {
        for (int i = 0; i < count; i++) {
            mpi_errno = mpi_to_pmi_keyvals(info_ptrs[i], &info_keyval_vectors[i],
                                           &info_keyval_sizes[i]);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

    if (num_preput_keyval > 0) {
#if defined(USE_PMI2_SLURM)
        preput_vector = MPL_malloc(num_preput_keyval * sizeof(INFO_TYPE *), MPL_MEM_BUFFER);
        MPIR_ERR_CHKANDJUMP(!preput_vector, mpi_errno, MPI_ERR_OTHER, "**nomem");
        preput_vector_array = MPL_malloc(num_preput_keyval * sizeof(INFO_TYPE), MPL_MEM_BUFFER);
        MPIR_ERR_CHKANDJUMP(!preput_vector_array, mpi_errno, MPI_ERR_OTHER, "**nomem");
        for (int i = 0; i < num_preput_keyval; i++) {
            INFO_TYPE_KEY(preput_vector_array[i]) = (char *) preput_keyvals[i].key;
            INFO_TYPE_VAL(preput_vector_array[i]) = preput_keyvals[i].val;
            preput_vector[i] = &preput_vector_array[i];
        }
#else
        preput_vector = MPL_malloc(num_preput_keyval * sizeof(INFO_TYPE), MPL_MEM_BUFFER);
        MPIR_ERR_CHKANDJUMP(!preput_vector, mpi_errno, MPI_ERR_OTHER, "**nomem");
        for (int i = 0; i < num_preput_keyval; i++) {
            INFO_TYPE_KEY(preput_vector[i]) = preput_keyvals[i].key;
            INFO_TYPE_VAL(preput_vector[i]) = preput_keyvals[i].val;
        }
#endif
    }
#endif

#ifdef USE_PMI1_API
#ifdef NO_PMI_SPAWN_MULTIPLE
    /* legacy bgq system does not have PMI_Spawn_multiple */
    MPIR_ERR_SETANDJUMP1(mpi_errno, MPI_ERR_OTHER,
                         "**pmi_spawn_multiple", "**pmi_spawn_multiple %d", 0);
#else
    pmi_errno = PMI_Spawn_multiple(count, (const char **) commands, (const char ***) argvs,
                                   maxprocs,
                                   info_keyval_sizes,
                                   (const INFO_TYPE **) info_keyval_vectors,
                                   num_preput_keyval, preput_vector, pmi_errcodes);

    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_spawn_multiple", "**pmi_spawn_multiple %d", pmi_errno);
#endif
#elif defined(USE_PMI2_API)
    int *argcs = MPL_malloc(count * sizeof(int), MPL_MEM_DYNAMIC);
    MPIR_Assert(argcs);

    /* compute argcs array */
    for (int i = 0; i < count; ++i) {
        argcs[i] = 0;
        if (argvs != NULL && argvs[i] != NULL) {
            while (argvs[i][argcs[i]]) {
                ++argcs[i];
            }
        }
    }

    pmi_errno = PMI2_Job_Spawn(count, (const char **) commands,
                               argcs, (const char ***) argvs,
                               maxprocs,
                               info_keyval_sizes,
                               (const INFO_TYPE **) info_keyval_vectors,
                               num_preput_keyval, preput_vector, NULL, 0, pmi_errcodes);
    MPL_free(argcs);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI2_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi_spawn_multiple", "**pmi_spawn_multiple %d", pmi_errno);
#elif defined(USE_PMIX_API)
    /* not supported yet */
    MPIR_Assert(0);
#endif

  fn_exit:
#if defined(USE_PMI1_API) || defined(USE_PMI2_API)
    if (info_keyval_vectors) {
        free_pmi_keyvals(info_keyval_vectors, count, info_keyval_sizes);
        MPL_free(info_keyval_vectors);
    }

    MPL_free(info_keyval_sizes);
    if (num_preput_keyval > 0) {
#if defined(USE_PMI2_SLURM)
        MPL_free(preput_vector_array);
        MPL_free(preput_vector);
#else
        MPL_free(preput_vector);
#endif
    }
#endif

    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_pmi_publish(const char name[], const char port[])
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;

#ifdef USE_PMI2_API
    /* release the global CS for PMI calls */
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    pmi_errno = PMI2_Nameserv_publish(name, NULL, port);
    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
#elif defined(USE_PMIX_API)
    pmix_info_t *info;
    PMIX_INFO_CREATE(info, 1);
    MPL_strncpy(info[0].key, name, PMIX_MAX_KEYLEN);
    info[0].value.type = PMIX_STRING;
    info[0].value.data.string = MPL_direct_strdup(port);
    pmi_errno = PMIx_Publish(info, 1);
    PMIX_INFO_FREE(info, 1);
#else
    pmi_errno = PMI_Publish_name(name, port);
#endif
    MPIR_ERR_CHKANDJUMP1(pmi_errno, mpi_errno, MPI_ERR_NAME, "**namepubnotpub",
                         "**namepubnotpub %s", name);

  fn_fail:
    return mpi_errno;
}

int MPIR_pmi_lookup(const char name[], char port[])
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;

#ifdef USE_PMI2_API
    /* release the global CS for PMI calls */
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    pmi_errno = PMI2_Nameserv_lookup(name, NULL, port, MPI_MAX_PORT_NAME);
    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
#elif defined(USE_PMIX_API)
    pmix_pdata_t *pdata;
    PMIX_PDATA_CREATE(pdata, 1);
    MPL_strncpy(pdata[0].key, name, PMIX_MAX_KEYLEN);
    pmi_errno = PMIx_Lookup(pdata, 1, NULL, 0);
    if (pmi_errno == PMIX_SUCCESS) {
        MPL_strncpy(port, pdata[0].value.data.string, MPI_MAX_PORT_NAME);
    }
    PMIX_PDATA_FREE(pdata, 1);
#else
    pmi_errno = PMI_Lookup_name(name, port);
#endif
    MPIR_ERR_CHKANDJUMP1(pmi_errno, mpi_errno, MPI_ERR_NAME, "**namepubnotfound",
                         "**namepubnotfound %s", name);

  fn_fail:
    return mpi_errno;
}

int MPIR_pmi_unpublish(const char name[])
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;

#ifdef USE_PMI2_API
    /* release the global CS for PMI calls */
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    pmi_errno = PMI2_Nameserv_unpublish(name, NULL);
    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
#elif defined(USE_PMIX_API)
    char *keys[2] = { (char *) name, NULL };
    pmi_errno = PMIx_Unpublish(keys, NULL, 0);
#else
    pmi_errno = PMI_Unpublish_name(name);
#endif
    MPIR_ERR_CHKANDJUMP1(pmi_errno, mpi_errno, MPI_ERR_SERVICE, "**namepubnotunpub",
                         "**namepubnotunpub %s", name);

  fn_fail:
    return mpi_errno;
}

/* ---- static functions ---- */

/* The following static function declares are only for build_nodemap() */
static int get_option_no_local(void);
static int get_option_num_cliques(void);
static int build_nodemap_nolocal(int *nodemap, int sz, int *num_nodes);
static int build_nodemap_roundrobin(int num_cliques, int *nodemap, int sz, int *num_nodes);
static int build_nodemap_byblock(int num_cliques, int *nodemap, int sz, int *num_nodes);

#ifdef USE_PMI1_API
static int build_nodemap_pmi1(int *nodemap, int sz);
static int build_nodemap_fallback(int *nodemap, int sz);
#elif defined(USE_PMI2_API)
static int build_nodemap_pmi2(int *nodemap, int sz);
#elif defined(USE_PMIX_API)
static int build_nodemap_pmix(int *nodemap, int sz);
#endif

/* TODO: if the process manager promises persistent node_id across multiple spawns,
 *       we can use the node id to check intranode processes across comm worlds.
 *       Currently we don't do this check and all dynamic processes are treated as
 *       inter-node. When we add the optimization, we should switch off the flag
 *       when appropriate environment variable from process manager is set.
 */
static bool do_normalize_nodemap = true;

static int build_nodemap(int *nodemap, int sz, int *num_nodes)
{
    int mpi_errno = MPI_SUCCESS;

    if (sz == 1 || get_option_no_local()) {
        mpi_errno = build_nodemap_nolocal(nodemap, sz, num_nodes);
        goto fn_exit;
    }
#ifdef USE_PMI1_API
    mpi_errno = build_nodemap_pmi1(nodemap, sz);
#elif defined(USE_PMI2_API)
    mpi_errno = build_nodemap_pmi2(nodemap, sz);
#elif defined(USE_PMIX_API)
    mpi_errno = build_nodemap_pmix(nodemap, sz);
#endif
    MPIR_ERR_CHECK(mpi_errno);

    if (do_normalize_nodemap) {
        /* node ids from process manager may not start from 0 or has gaps.
         * Normalize it since most of the code assume a contiguous node id range */
        int max_id = -1;
        for (int i = 0; i < sz; i++) {
            if (max_id < nodemap[i]) {
                max_id = nodemap[i];
            }
        }
        int *nodeids = MPL_malloc((max_id + 1) * sizeof(int), MPL_MEM_OTHER);
        for (int i = 0; i < max_id + 1; i++) {
            nodeids[i] = -1;
        }
        int next_node_id = 0;
        for (int i = 0; i < sz; i++) {
            int old_id = nodemap[i];
            if (nodeids[old_id] == -1) {
                nodeids[old_id] = next_node_id;
                next_node_id++;
            }
            nodemap[i] = nodeids[old_id];
        }
        *num_nodes = next_node_id;
        MPL_free(nodeids);
    }

    /* local cliques */
    int num_cliques = get_option_num_cliques();
    if (num_cliques > sz) {
        num_cliques = sz;
    }
    if (*num_nodes == 1 && num_cliques > 1) {
        if (MPIR_CVAR_CLIQUES_BY_BLOCK) {
            mpi_errno = build_nodemap_byblock(num_cliques, nodemap, sz, num_nodes);
        } else {
            mpi_errno = build_nodemap_roundrobin(num_cliques, nodemap, sz, num_nodes);
        }
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int get_option_no_local(void)
{
    /* Used for debugging only.  This disables communication over shared memory */
#ifdef ENABLE_NO_LOCAL
    return 1;
#else
    return MPIR_CVAR_NOLOCAL;
#endif
}

static int get_option_num_cliques(void)
{
    /* Used for debugging on a single machine: split procs into num_cliques nodes.
     * If ODD_EVEN_CLIQUES were enabled, split procs into 2 nodes.
     */
    if (MPIR_CVAR_NUM_CLIQUES > 1) {
        return MPIR_CVAR_NUM_CLIQUES;
    } else {
        return MPIR_CVAR_ODD_EVEN_CLIQUES ? 2 : 1;
    }
}

int MPIR_pmi_has_local_cliques(void)
{
    return (get_option_num_cliques() > 1);
}

/* one process per node */
int build_nodemap_nolocal(int *nodemap, int sz, int *num_nodes)
{
    for (int i = 0; i < sz; ++i) {
        nodemap[i] = i;
    }
    *num_nodes = sz;
    return MPI_SUCCESS;
}

/* assign processes to num_cliques nodes in a round-robin fashion */
static int build_nodemap_roundrobin(int num_cliques, int *nodemap, int sz, int *num_nodes)
{
    for (int i = 0; i < sz; ++i) {
        nodemap[i] = i % num_cliques;
    }
    *num_nodes = (sz >= num_cliques) ? num_cliques : sz;
    return MPI_SUCCESS;
}

/* assign processes to num_cliques nodes by uniform block */
static int build_nodemap_byblock(int num_cliques, int *nodemap, int sz, int *num_nodes)
{
    int block_size = sz / num_cliques;
    int remainder = sz % num_cliques;
    /* The first `remainder` cliques have size `block_size + 1` */
    int middle = (block_size + 1) * remainder;
    for (int i = 0; i < sz; ++i) {
        if (i < middle) {
            nodemap[i] = i / (block_size + 1);
        } else {
            nodemap[i] = (i - remainder) / block_size;
        }
    }
    *num_nodes = (sz >= num_cliques) ? num_cliques : sz;
    return MPI_SUCCESS;
}

#ifdef USE_PMI1_API

/* build nodemap based on allgather hostnames */
/* FIXME: migrate the function */
static int build_nodemap_fallback(int *nodemap, int sz)
{
    return MPIR_NODEMAP_build_nodemap_fallback(sz, MPIR_Process.rank, nodemap);
}

/* build nodemap using PMI1 process_mapping or fallback with hostnames */
static int build_nodemap_pmi1(int *nodemap, int sz)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;
    int did_map = 0;
    if (pmi_version == 1 && pmi_subversion == 1) {
        char *process_mapping = MPL_malloc(pmi_max_val_size, MPL_MEM_ADDRESS);
        pmi_errno = PMI_KVS_Get(pmi_kvs_name, "PMI_process_mapping",
                                process_mapping, pmi_max_val_size);
        if (pmi_errno == PMI_SUCCESS && strcmp(process_mapping, "") != 0) {
            int mpl_err = MPL_rankmap_str_to_array(process_mapping, sz, nodemap);
            MPIR_ERR_CHKINTERNAL(mpl_err, mpi_errno,
                                 "unable to populate node ids from PMI_process_mapping");
            did_map = 1;
        }
        MPL_free(process_mapping);
    }
    if (!did_map) {
        mpi_errno = build_nodemap_fallback(nodemap, sz);
    }
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#elif defined USE_PMI2_API

/* build nodemap using PMI2 process_mapping or error */
static int build_nodemap_pmi2(int *nodemap, int sz)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;
    char process_mapping[PMI2_MAX_VALLEN];
    int found;

    pmi_errno = PMI2_Info_GetJobAttr("PMI_process_mapping", process_mapping, PMI2_MAX_VALLEN,
                                     &found);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMI2_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmi2_info_getjobattr", "**pmi2_info_getjobattr %d", pmi_errno);
    MPIR_ERR_CHKINTERNAL(!found, mpi_errno, "PMI_process_mapping attribute not found");

    int mpl_err;
    mpl_err = MPL_rankmap_str_to_array(process_mapping, sz, nodemap);
    MPIR_ERR_CHKINTERNAL(mpl_err, mpi_errno,
                         "unable to populate node ids from PMI_process_mapping");
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#elif defined USE_PMIX_API

/* build nodemap using PMIx_Resolve_nodes */
int build_nodemap_pmix(int *nodemap, int sz)
{
    int mpi_errno = MPI_SUCCESS;
    int pmi_errno;
    char *nodelist = NULL, *node = NULL;
    pmix_proc_t *procs = NULL;
    size_t nprocs, node_id = 0;

    pmi_errno = PMIx_Resolve_nodes(pmix_proc.nspace, &nodelist);
    MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_resolve_nodes", "**pmix_resolve_nodes %d", pmi_errno);
    MPIR_Assert(nodelist);

    node = strtok(nodelist, ",");
    while (node) {
        pmi_errno = PMIx_Resolve_peers(node, pmix_proc.nspace, &procs, &nprocs);
        MPIR_ERR_CHKANDJUMP1(pmi_errno != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                             "**pmix_resolve_peers", "**pmix_resolve_peers %d", pmi_errno);
        for (int i = 0; i < nprocs; i++) {
            nodemap[procs[i].rank] = node_id;
        }
        node_id++;
        node = strtok(NULL, ",");
    }
    /* PMIx latest adds pmix_free. We should switch to that at some point */
    MPL_external_free(nodelist);
    PMIX_PROC_FREE(procs, nprocs);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#endif

/* allocate and populate MPIR_Process.node_local_map and MPIR_Process.node_root_map */
static int build_locality(void)
{
    int local_rank = -1;
    int local_size = 0;
    int *node_root_map, *node_local_map;

    int rank = MPIR_Process.rank;
    int size = MPIR_Process.size;
    int *node_map = MPIR_Process.node_map;
    int num_nodes = MPIR_Process.num_nodes;
    int local_node_id = node_map[rank];

    node_root_map = MPL_malloc(num_nodes * sizeof(int), MPL_MEM_ADDRESS);
    for (int i = 0; i < num_nodes; i++) {
        node_root_map[i] = -1;
    }

    for (int i = 0; i < size; i++) {
        int node_id = node_map[i];
        if (node_root_map[node_id] < 0) {
            node_root_map[node_id] = i;
        }
        if (node_id == local_node_id) {
            local_size++;
        }
    }

    node_local_map = MPL_malloc(local_size * sizeof(int), MPL_MEM_ADDRESS);
    int j = 0;
    for (int i = 0; i < size; i++) {
        int node_id = node_map[i];
        if (node_id == local_node_id) {
            node_local_map[j] = i;
            if (i == rank) {
                local_rank = j;
            }
            j++;
        }
    }

    MPIR_Process.node_root_map = node_root_map;
    MPIR_Process.node_local_map = node_local_map;
    MPIR_Process.local_size = local_size;
    MPIR_Process.local_rank = local_rank;

    return MPI_SUCCESS;
}

#if defined(USE_PMI1_API) || defined(USE_PMI2_API)
/* similar to functions in mpl/src/str/mpl_argstr.c, but much simpler */
static int hex(unsigned char c)
{
    if (c >= '0' && c <= '9') {
        return c - '0';
    } else if (c >= 'a' && c <= 'f') {
        return 10 + c - 'a';
    } else if (c >= 'A' && c <= 'F') {
        return 10 + c - 'A';
    } else {
        MPIR_Assert(0);
        return -1;
    }
}

static void encode(int size, const char *src, char *dest)
{
    for (int i = 0; i < size; i++) {
        MPL_snprintf(dest, 3, "%02X", (unsigned char) *src);
        src++;
        dest += 2;
    }
}

static void decode(int size, const char *src, char *dest)
{
    for (int i = 0; i < size; i++) {
        *dest = (char) (hex(src[0]) << 4) + hex(src[1]);
        src += 2;
        dest++;
    }
}

/* static functions used in MPIR_pmi_spawn_multiple */
static int mpi_to_pmi_keyvals(MPIR_Info * info_ptr, INFO_TYPE ** kv_ptr, int *nkeys_ptr)
{
    char key[MPI_MAX_INFO_KEY];
    INFO_TYPE *kv = 0;
    int nkeys = 0, vallen, flag, mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_ENTER;

    if (!info_ptr || info_ptr->handle == MPI_INFO_NULL)
        goto fn_exit;

    MPIR_Info_get_nkeys_impl(info_ptr, &nkeys);

    if (nkeys == 0)
        goto fn_exit;

    kv = (INFO_TYPE *) MPL_malloc(nkeys * sizeof(INFO_TYPE), MPL_MEM_BUFFER);

    for (int i = 0; i < nkeys; i++) {
        mpi_errno = MPIR_Info_get_nthkey_impl(info_ptr, i, key);
        MPIR_ERR_CHECK(mpi_errno);
        MPIR_Info_get_valuelen_impl(info_ptr, key, &vallen, &flag);

        char *s_val;
        s_val = (char *) MPL_malloc(vallen + 1, MPL_MEM_BUFFER);
        MPIR_Info_get_impl(info_ptr, key, vallen + 1, s_val, &flag);

        INFO_TYPE_KEY(kv[i]) = MPL_strdup(key);
        INFO_TYPE_VAL(kv[i]) = s_val;
    }

  fn_exit:
    *kv_ptr = kv;
    *nkeys_ptr = nkeys;
    MPIR_FUNC_EXIT;
    return mpi_errno;

  fn_fail:
    goto fn_exit;
}

static void free_pmi_keyvals(INFO_TYPE ** kv, int size, int *counts)
{
    MPIR_FUNC_ENTER;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < counts[i]; j++) {
            /* cast the "const" away */
            MPL_free((char *) INFO_TYPE_KEY(kv[i][j]));
            MPL_free((char *) INFO_TYPE_VAL(kv[i][j]));
        }
        MPL_free(kv[i]);
    }

    MPIR_FUNC_EXIT;
}
#elif defined(USE_PMIX_API)
/* Add a specific key/value pair from an MPIR_Info object to a target MPIR_info object */
static
int pmix_add_to_info(MPIR_Info * info_ptr, const char *key, const char *pmix_key,
                     MPIR_Info * target_ptr, int *key_found, size_t * counter, char **value)
{
    int mpi_errno = MPI_SUCCESS;
    int flag;
    char val[MPI_MAX_INFO_VAL];

    mpi_errno = MPIR_Info_get_impl(info_ptr, key, MPI_MAX_INFO_VAL, val, &flag);
    MPIR_ERR_CHECK(mpi_errno);

    if (flag) {
        /* Add pmix_key/ value pair to target info */
        mpi_errno = MPIR_Info_set_impl(target_ptr, pmix_key, val);
        MPIR_ERR_CHECK(mpi_errno);
        if (key_found) {
            *key_found = 1;
        }
        if (value) {
            *value = MPL_malloc(MPI_MAX_INFO_VAL, MPL_MEM_OTHER);
            MPL_strncpy(*value, val, MPI_MAX_INFO_VAL);
        }
        (*counter)++;
    } else {
        if (key_found) {
            *key_found = 0;
        }
        if (value) {
            *value = NULL;
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
#endif

/**
 * @brief   Register a process for PMIx process set events
 *
 * @return  int MPI_SUCCESS or MPI_ERR_OTHER
 */
int MPIR_pmi_register_process_set_event_handlers(void)
{

    int mpi_errno = MPI_SUCCESS;
#ifdef USE_PMIX_API
#if PMIX_NUMERIC_VERSION >= 0x00040203
    /* Set the PMIX codes of the events for which handlers shall be registered */
    pmix_status_t code_pset_define[1] = { PMIX_PROCESS_SET_DEFINE };
    pmix_status_t code_pset_delete[1] = { PMIX_PROCESS_SET_DELETE };

    /* Give handlers a name so we can identify them more easily */
    pmix_info_t info_pset_define[1];
    pmix_info_t info_pset_delete[1];

    PMIX_INFO_LOAD(&info_pset_define[0], PMIX_EVENT_HDLR_NAME, "Process-Set-Define", PMIX_STRING);
    PMIX_INFO_LOAD(&info_pset_delete[0], PMIX_EVENT_HDLR_NAME, "Process-Set-Delete", PMIX_STRING);

    /* Register event handlers for PMIx process set define and delete events
     * in a blocking way (last two parameters are NULL) and treat errors */
    rc_pset_define_handler =
        (int) PMIx_Register_event_handler(code_pset_define, 1, info_pset_define, 1,
                                          pset_define_callback, NULL, NULL);
    if (rc_pset_define_handler < 0) {
        MPIR_ERR_CHKANDJUMP1(rc_pset_define_handler != PMIX_SUCCESS, mpi_errno,
                             MPI_ERR_OTHER, "**pmix_register_event_handler",
                             "**pmix_register_event_handler %d", rc_pset_define_handler);
    }
    rc_pset_delete_handler =
        (int) PMIx_Register_event_handler(code_pset_delete, 1, info_pset_delete, 1,
                                          pset_delete_callback, NULL, NULL);
    if (rc_pset_delete_handler < 0) {
        MPIR_ERR_CHKANDJUMP1(rc_pset_delete_handler != PMIX_SUCCESS, mpi_errno,
                             MPI_ERR_OTHER, "**pmix_register_event_handler",
                             "**pmix_register_event_handler %d", rc_pset_delete_handler);
    }

    /* Release PMIx info objects */
    PMIX_INFO_DESTRUCT(&info_pset_define[0]);
    PMIX_INFO_DESTRUCT(&info_pset_delete[0]);
#endif /* PMIx min version 4.2.3 */
#endif /* PMIX API */

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;

}


/**
 * @brief   Deregister a process from PMIx process set events
 *
 * @return  int MPI_SUCCESS or MPI_ERR_OTHER
 */
int MPIR_pmi_deregister_process_set_event_handlers(void)
{
    int mpi_errno = MPI_SUCCESS;
#ifdef USE_PMIX_API
#if PMIX_NUMERIC_VERSION >= 0x00040203
    /* Deregister PMIx event handler for pset define and delete events
     * in a blocking way (last two parameters are NULL) and treat errors */
    pmix_status_t rc;
    rc = PMIx_Deregister_event_handler((size_t) rc_pset_define_handler, NULL, NULL);
    MPIR_ERR_CHKANDJUMP1(rc != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_deregister_event_handler", "**pmix_deregister_event_handler %d",
                         rc);
    rc_pset_define_handler = 0;

    rc = PMIx_Deregister_event_handler((size_t) rc_pset_delete_handler, NULL, NULL);
    MPIR_ERR_CHKANDJUMP1(rc != PMIX_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                         "**pmix_deregister_event_handler", "**pmix_deregister_event_handler %d",
                         rc);
    rc_pset_delete_handler = 0;
#endif /* PMIx min version 4.2.3 */
#endif /* PMIX API */
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#ifdef USE_PMIX_API
#if PMIX_NUMERIC_VERSION >= 0x00040203
/**
 * @brief   Compare two integers. Helper function for qsort algorithm.
 *
 * @param   a   Integer to compare
 * @param   b   Integer to compate
 * @return  int Result of comparison; 0: equal, <0: b>a, >0: a>b
 */
static
int compareint(void const *a, void const *b)
{
    return *(const int *) a - *(const int *) b;
}

/**
 * @brief Callback for PMIx process set define event handler (for function prototype see also PMIx standard document)
 *
 * @param refid     Registration number of the handler being called.
 * @param status    Status associated with the operation.
 * @param source    Identifier of the process that generated the event. If the source if the SMS, then the namespace will be empty and the rank will be PMIX_RANK_UNDEF.
 * @param info      Information describing the event. This argument will be NULL if no additional information was provided by the event generator.
 * @param ninfo     Number of elements in the info array.
 * @param results   Aggregated results from prior event handlers servicing this event. This argument will be NULL if this is the first handler servicing the event or if no prior handlers provided results.
 * @param nresults  Number of elements in the results array.
 * @param cbfunc    Callback function to be executed upon completion of the handler's operation and prior to handler return (function reference).
 * @param cbdata    Callback data to be passed to cbfunc (memory reference).
 */
static
void pset_define_callback(size_t refid, pmix_status_t status, const pmix_proc_t * source,
                          pmix_info_t * info, size_t ninfo, pmix_info_t * results, size_t nresults,
                          pmix_event_notification_cbfunc_fn_t cbfunc, void *cbdata)
{

    pmix_value_t *pset_name_val = NULL;
    char *pset_name = NULL;
    pmix_value_t *membership_val = NULL;
    pmix_proc_t *membership = NULL;
    int *members = NULL;
    int size = 0;
    bool name_key_found = false;
    bool membership_key_found = false;
    bool process_in_pset = false;
    MPIR_Pset *new_pset = NULL;
    pmix_status_t cb_status = PMIX_EVENT_ACTION_COMPLETE;

    /* Extract pset name and membership from the info object */
    for (size_t i = 0; i < ninfo; i++) {
        if (strncasecmp((char *) info[i].key, PMIX_PSET_NAME, strlen(PMIX_PSET_NAME)) == 0) {
            name_key_found = true;
            pset_name_val = PMIx_Value_create(1);
            PMIx_Value_xfer(pset_name_val, &(info[i].value));
            pset_name = pset_name_val->data.string;
        } else if (strncasecmp((char *) info[i].key, PMIX_PSET_MEMBERS, strlen(PMIX_PSET_MEMBERS))
                   == 0) {
            membership_key_found = true;
            membership_val = PMIx_Value_create(1);
            PMIx_Value_xfer(membership_val, &(info[i].value));
            membership = (pmix_proc_t *) membership_val->data.darray->array;

            /* Overflow check */
            MPIR_Assert(membership_val->data.darray->size <= INT_MAX);
            size = (int) membership_val->data.darray->size;
        }

        if (name_key_found && membership_key_found) {
            break;
        }
    }

    if (!name_key_found || !membership_key_found || pset_name == NULL || membership == NULL) {
        cb_status = PMIX_EVENT_NO_ACTION_TAKEN;
        goto fn_exit;
    }

    if (size == 0) {
        /* Prevent adding empty psets (no error) */
        MPL_DBG_MSG_FMT(MPIR_DBG_INIT, TYPICAL, (MPL_DBG_FDEST,
                                                 "PMIx pset define callback: pset '%s' has size 0 and is not added.",
                                                 pset_name));
        goto fn_exit;
    }

    /* Create sorted membership array for new pset (required for group creation from pset) */
    members = MPL_malloc(size * sizeof(int), MPL_MEM_OTHER);
    if (members == NULL) {
        cb_status = PMIX_EVENT_NO_ACTION_TAKEN;
        goto fn_exit;
    }
    for (int k = 0; k < size; k++) {
        /* TODO is it ok to ignore the PMIx namespace here? */
        pmix_proc_t proc = membership[k];

        /* Check overflow because of type mismatch between pmix_proc_t.rank (uint32_t) and MPIR_Process_t.rank (int) */
        MPIR_Assert(proc.rank <= INT_MAX);

        if ((int) proc.rank == MPIR_Process.rank) {
            process_in_pset = true;
        }
        members[k] = (int) proc.rank;
    }

    if (!process_in_pset) {
        /* Prevent adding a pset in which the process is not a member (no error) */
        MPL_DBG_MSG_FMT(MPIR_DBG_INIT, TYPICAL, (MPL_DBG_FDEST,
                                                 "PMIx pset define callback: not a member of pset '%s', pset not added.",
                                                 pset_name));
        goto fn_exit;
    }
    qsort(members, (size_t) size, sizeof(int), compareint);

    /* Create the new pset and add it to the global pmix pset array */
    new_pset = MPL_malloc(sizeof(MPIR_Pset), MPL_MEM_OTHER);
    new_pset->uri = pset_name;
    new_pset->size = size;
    new_pset->is_valid = true;
    new_pset->members = members;

    int add_status = MPIR_Pset_array_add(MPIR_Process.pm_pset_array, new_pset);
    if (add_status == MPI_ERR_OTHER) {
        cb_status = PMIX_EVENT_NO_ACTION_TAKEN;
        MPL_DBG_MSG_FMT(MPIR_DBG_INIT, TYPICAL, (MPL_DBG_FDEST,
                                                 "PMIx pset define callback: pset '%s' already exists, not added again.",
                                                 pset_name));
        goto fn_exit;
    }

  fn_exit:
    if (pset_name_val != NULL)
        PMIx_Value_destruct(pset_name_val);
    if (membership_val != NULL)
        PMIx_Value_destruct(membership_val);
    if (members != NULL)
        MPL_free(members);
    if (new_pset != NULL)
        MPL_free(new_pset);

    /* If a callback function is provided, we have to call it.
     * We cannot treat errors here, but we can give a hint via cbfunc status that something went wrong */
    if (NULL != cbfunc) {
        cbfunc(cb_status, NULL, 0, NULL, NULL, cbdata);
    }

    return;
}

/**
 * @brief Callback for PMIx process set delete event handler (for function prototype see also PMIx standard document)
 *
 * @param refid     Registration number of the handler being called.
 * @param status    Status associated with the operation.
 * @param source    Identifier of the process that generated the event. If the source if the SMS, then the namespace will be empty and the rank will be PMIX_RANK_UNDEF.
 * @param info      Information describing the event. This argument will be NULL if no additional information was provided by the event generator.
 * @param ninfo     Number of elements in the info array.
 * @param results   Aggregated results from prior event handlers servicing this event. This argument will be NULL if this is the first handler servicing the event or if no prior handlers provided results.
 * @param nresults  Number of elements in the results array.
 * @param cbfunc    Callback function to be executed upon completion of the handler's operation and prior to handler return (function reference).
 * @param cbdata    Callback data to be passed to cbfunc (memory reference).
 */
static
void pset_delete_callback(size_t refid, pmix_status_t status, const pmix_proc_t * source,
                          pmix_info_t * info, size_t ninfo, pmix_info_t * results, size_t nresults,
                          pmix_event_notification_cbfunc_fn_t cbfunc, void *cbdata)
{

    pmix_value_t *pset_name_val = NULL;
    char *pset_name = NULL;
    bool key_found = false;
    pmix_status_t cb_status = PMIX_EVENT_ACTION_COMPLETE;

    /* Extract pset name from the info object */
    for (size_t i = 0; i < ninfo; i++) {
        if (strncasecmp((char *) info[i].key, PMIX_PSET_NAME, strlen(PMIX_PSET_NAME)) == 0) {
            key_found = true;
            pset_name_val = PMIx_Value_create(1);
            PMIx_Value_xfer(pset_name_val, &(info[i].value));
            pset_name = pset_name_val->data.string;
            break;
        }
    }

    if (!key_found) {
        cb_status = PMIX_EVENT_NO_ACTION_TAKEN;
        goto fn_exit;
    }

    /* Search in known PMIx psets for pset_name and invalidate the pset */
    int ret = MPIR_Pset_array_invalidate(MPIR_Process.pm_pset_array, pset_name);
    if (ret == MPI_ERR_OTHER) {
        /* Pset not found in this process (not an error!) */
        MPL_DBG_MSG_FMT(MPIR_DBG_INIT, TYPICAL, (MPL_DBG_FDEST,
                                                 "PMIx pset delete callback: pset '%s' not found.",
                                                 pset_name));
    }

  fn_exit:
    if (pset_name_val)
        PMIx_Value_destruct(pset_name_val);

    if (NULL != cbfunc) {
        /* If a callback function is provided, we have to call it
         * We cannot treat an error here, but we can give a hint via cbfunc status that something went wrong */
        cbfunc(cb_status, NULL, 0, NULL, NULL, cbdata);
    }
}

#endif /* PMIx min version 4.2.3 */


static
int pmix_build_job_info(MPIR_Info * info_ptr, pmix_info_t ** pmix_job_info, size_t * njob_info,
                        char **path)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Info *mpi_job_info;
    size_t ninfo = 0;

    if (info_ptr == NULL) {
        goto fn_exit;
    }

    mpi_errno = MPIR_Info_alloc(&mpi_job_info);
    MPIR_ERR_CHECK(mpi_errno);

    /* path - standard key */
    mpi_errno = pmix_add_to_info(info_ptr, "path", PMIX_PREFIX, mpi_job_info, NULL, &ninfo, path);
    MPIR_ERR_CHECK(mpi_errno);

    /* FIXME: There is currently no mapping of the standard key `soft` to a
     * PMIx key supported by PMIx_Spawn. Once PMIx_Spawn supports soft spawning
     * we should add the key `soft` here. */

    /* PMIX_ALLOC_ID - non-standard key
     * A string identifier (provided by the host environment) for the resulting allocation
     * from a successful PMIx_Allocation_request */
    mpi_errno =
        pmix_add_to_info(info_ptr, "PMIX_ALLOC_ID", PMIX_ALLOC_ID, mpi_job_info, NULL, &ninfo,
                         NULL);
    MPIR_ERR_CHECK(mpi_errno);

    if (ninfo > 0) {
        mpi_errno = mpi_to_pmix_keyvals(mpi_job_info, ninfo, pmix_job_info);
        MPIR_ERR_CHECK(mpi_errno);
        *njob_info = ninfo;
    } else {
        *njob_info = 0;
        *pmix_job_info = NULL;
    }

    mpi_errno = MPIR_Info_free_impl(mpi_job_info);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static
int mpi_to_pmix_keyvals(MPIR_Info * info_ptr, int ninfo, pmix_info_t ** pmix_info)
{
    int mpi_errno = MPI_SUCCESS;
    if (ninfo > 0) {
        PMIX_INFO_CREATE(*pmix_info, ninfo);
        MPIR_ERR_CHKANDJUMP(!(*pmix_info), mpi_errno, MPI_ERR_OTHER, "**nomem");
        for (int k = 0; k < ninfo; k++) {
            char key[MPI_MAX_INFO_KEY];
            char val[MPI_MAX_INFO_VAL];
            int flag;
            mpi_errno = MPIR_Info_get_nthkey_impl(info_ptr, k, key);
            MPIR_ERR_CHECK(mpi_errno);
            mpi_errno = MPIR_Info_get_impl(info_ptr, key, MPI_MAX_INFO_VAL, val, &flag);
            MPIR_ERR_CHECK(mpi_errno);
            PMIX_INFO_LOAD(&((*pmix_info)[k]), key, val, PMIX_STRING);
        }
    }
  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


static
void pmix_not_supported(const char *elem, char *error_str, int strlen)
{
    int pmi_errno = PMIX_SUCCESS;
    pmix_value_t *rm_name = NULL;
    pmix_value_t *rm_version = NULL;
    char *name = NULL;
    char *version = NULL;

    /* Try to get infos about PMIx Host (name and version) */
    pmi_errno = PMIx_Get(&pmix_wcproc, PMIX_RM_NAME, NULL, 0, &rm_name);
    if (pmi_errno == PMIX_SUCCESS) {
        name = MPL_strdup(rm_name->data.string);
        PMIX_VALUE_RELEASE(rm_name);
    }

    pmi_errno = PMIx_Get(&pmix_wcproc, PMIX_RM_VERSION, NULL, 0, &rm_version);
    if (pmi_errno == PMIX_SUCCESS) {
        version = MPL_strdup(rm_version->data.string);
        PMIX_VALUE_RELEASE(rm_version);
    }

    /* Create a comprehensible error message based on the infos that
     * could be obtained about the PMIx Host */
    if (name && version) {
        MPL_snprintf(error_str, strlen, "%s not supported by PMIx Host %s version %s",
                     elem, name, version);
    } else if (name) {
        MPL_snprintf(error_str, strlen, "%s not supported by PMIx Host %s", elem, name);
    } else {
        MPL_snprintf(error_str, strlen, "%s not supported by PMIx Host", elem);
    }

    if (name) {
        MPL_free(name);
    }
    if (version) {
        MPL_free(version);
    }
}
#endif /* PMIX API */
