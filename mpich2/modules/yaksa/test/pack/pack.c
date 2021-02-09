/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>
#include "yaksa_config.h"
#include "yaksa.h"
#include "dtpools.h"
#include "pack-common.h"

uintptr_t maxbufsize = 512 * 1024 * 1024;

enum {
    PACK_ORDER__UNSET,
    PACK_ORDER__NORMAL,
    PACK_ORDER__REVERSE,
    PACK_ORDER__RANDOM,
};

enum {
    OVERLAP__NONE,
    OVERLAP__REGULAR,
    OVERLAP__IRREGULAR,
};

#define MAX_DTP_BASESTRLEN (1024)

static int verbose = 0;

#define dprintf(...)                            \
    do {                                        \
        if (verbose)                            \
            printf(__VA_ARGS__);                \
    } while (0)

static void swap_segments(uintptr_t * starts, uintptr_t * lengths, int x, int y)
{
    uintptr_t tmp = starts[x];
    starts[x] = starts[y];
    starts[y] = tmp;

    tmp = lengths[x];
    lengths[x] = lengths[y];
    lengths[y] = tmp;
}

int device_id = 0;
int device_stride = 0;

char typestr[MAX_DTP_BASESTRLEN + 1] = { 0 };

int seed = -1;
int basecount = -1;
int iters = -1;
int max_segments = -1;
int pack_order = PACK_ORDER__UNSET;
int overlap = -1;
mem_type_e sbuf_memtype = MEM_TYPE__UNREGISTERED_HOST;
mem_type_e dbuf_memtype = MEM_TYPE__UNREGISTERED_HOST;
mem_type_e tbuf_memtype = MEM_TYPE__UNREGISTERED_HOST;
DTP_pool_s *dtp;

void *runtest(void *arg);
void *runtest(void *arg)
{
    DTP_obj_s sobj, dobj;
    int rc;
    uintptr_t tid = (uintptr_t) arg;

    uintptr_t *segment_starts = (uintptr_t *) malloc(max_segments * sizeof(uintptr_t));
    uintptr_t *segment_lengths = (uintptr_t *) malloc(max_segments * sizeof(uintptr_t));

    for (int i = 0; i < iters; i++) {
        dprintf("==== iter %d ====\n", i);

        /* create the source object */
        rc = DTP_obj_create(dtp[tid], &sobj, maxbufsize);
        assert(rc == DTP_SUCCESS);

        char *sbuf_h = NULL, *sbuf_d = NULL;
        pack_alloc_mem(sobj.DTP_bufsize, sbuf_memtype, (void **) &sbuf_h, (void **) &sbuf_d);
        assert(sbuf_h);
        assert(sbuf_d);

        if (verbose) {
            char *desc;
            rc = DTP_obj_get_description(sobj, &desc);
            assert(rc == DTP_SUCCESS);
            dprintf("==> sbuf_h %p, sbuf_d %p, sobj (count: %zu):\n%s\n", sbuf_h, sbuf_d,
                    sobj.DTP_type_count, desc);
            free(desc);
        }

        rc = DTP_obj_buf_init(sobj, sbuf_h, 0, 1, basecount);
        assert(rc == DTP_SUCCESS);

        uintptr_t ssize;
        rc = yaksa_type_get_size(sobj.DTP_datatype, &ssize);
        assert(rc == YAKSA_SUCCESS);


        /* create the destination object */
        rc = DTP_obj_create(dtp[tid], &dobj, maxbufsize);
        assert(rc == DTP_SUCCESS);

        char *dbuf_h, *dbuf_d;
        pack_alloc_mem(dobj.DTP_bufsize, dbuf_memtype, (void **) &dbuf_h, (void **) &dbuf_d);
        assert(dbuf_h);
        assert(dbuf_d);

        if (verbose) {
            char *desc;
            rc = DTP_obj_get_description(dobj, &desc);
            assert(rc == DTP_SUCCESS);
            dprintf("==> dbuf_h %p, dbuf_d %p, dobj (count: %zu):\n%s\n", dbuf_h, dbuf_d,
                    dobj.DTP_type_count, desc);
            free(desc);
        }

        rc = DTP_obj_buf_init(dobj, dbuf_h, -1, -1, basecount);
        assert(rc == DTP_SUCCESS);

        uintptr_t dsize;
        rc = yaksa_type_get_size(dobj.DTP_datatype, &dsize);
        assert(rc == YAKSA_SUCCESS);


        /* the source and destination objects should have the same
         * signature */
        assert(ssize * sobj.DTP_type_count == dsize * dobj.DTP_type_count);


        /* pack from the source object to a temporary buffer and
         * unpack into the destination object */

        /* figure out the lengths and offsets of each segment */
        uintptr_t type_size;
        rc = yaksa_type_get_size(dtp[tid].DTP_base_type, &type_size);
        assert(rc == YAKSA_SUCCESS);

        int segments = max_segments;
        while (((ssize * sobj.DTP_type_count) / type_size) % segments)
            segments--;

        uintptr_t offset = 0;
        for (int j = 0; j < segments; j++) {
            segment_starts[j] = offset;

            uintptr_t eqlength = ssize * sobj.DTP_type_count / segments;

            /* make sure eqlength is a multiple of type_size */
            eqlength = (eqlength / type_size) * type_size;

            if (overlap == OVERLAP__NONE) {
                segment_lengths[j] = eqlength;
                offset += eqlength;
            } else if (overlap == OVERLAP__REGULAR) {
                if (offset + 2 * eqlength <= ssize * sobj.DTP_type_count)
                    segment_lengths[j] = 2 * eqlength;
                else
                    segment_lengths[j] = eqlength;
                offset += eqlength;
            } else {
                if (j == segments - 1) {
                    if (ssize * sobj.DTP_type_count > offset)
                        segment_lengths[j] = ssize * sobj.DTP_type_count - offset;
                    else
                        segment_lengths[j] = 0;
                    segment_lengths[j] += rand() % eqlength;
                } else {
                    segment_lengths[j] = rand() % (ssize * sobj.DTP_type_count - offset + eqlength);
                }

                offset += ((rand() % (segment_lengths[j] + 1)) / type_size) * type_size;
                if (offset > ssize * sobj.DTP_type_count)
                    offset = ssize * sobj.DTP_type_count;
            }
        }

        /* update the order in which we access the segments */
        if (pack_order == PACK_ORDER__NORMAL) {
            /* nothing to do */
        } else if (pack_order == PACK_ORDER__REVERSE) {
            for (int j = 0; j < segments / 2; j++) {
                swap_segments(segment_starts, segment_lengths, j, segments - j - 1);
            }
        } else if (pack_order == PACK_ORDER__RANDOM) {
            for (int j = 0; j < 1000; j++) {
                int x = rand() % segments;
                int y = rand() % segments;
                swap_segments(segment_starts, segment_lengths, x, y);
            }
        }

        /* the actual pack/unpack loop */
        pack_copy_content(sbuf_h, sbuf_d, sobj.DTP_bufsize, sbuf_memtype);
        pack_copy_content(dbuf_h, dbuf_d, dobj.DTP_bufsize, dbuf_memtype);

        void *tbuf_h, *tbuf_d;
        uintptr_t tbufsize = ssize * sobj.DTP_type_count;
        pack_alloc_mem(tbufsize, tbuf_memtype, &tbuf_h, &tbuf_d);

        yaksa_info_t pack_info, unpack_info;
        pack_get_ptr_attr(sbuf_d + sobj.DTP_buf_offset, tbuf_d, &pack_info);
        pack_get_ptr_attr(tbuf_d, dbuf_d + dobj.DTP_buf_offset, &unpack_info);

        for (int j = 0; j < segments; j++) {
            uintptr_t actual_pack_bytes;
            yaksa_request_t request;

            rc = yaksa_ipack(sbuf_d + sobj.DTP_buf_offset, sobj.DTP_type_count, sobj.DTP_datatype,
                             segment_starts[j], tbuf_d, segment_lengths[j], &actual_pack_bytes,
                             pack_info, &request);
            assert(rc == YAKSA_SUCCESS);
            assert(actual_pack_bytes <= segment_lengths[j]);

            if (j == segments - 1) {
                DTP_obj_free(sobj);
            }

            rc = yaksa_request_wait(request);
            assert(rc == YAKSA_SUCCESS);

            uintptr_t actual_unpack_bytes;
            rc = yaksa_iunpack(tbuf_d, actual_pack_bytes, dbuf_d + dobj.DTP_buf_offset,
                               dobj.DTP_type_count, dobj.DTP_datatype, segment_starts[j],
                               &actual_unpack_bytes, unpack_info, &request);
            assert(rc == YAKSA_SUCCESS);
            assert(actual_pack_bytes == actual_unpack_bytes);

            rc = yaksa_request_wait(request);
            assert(rc == YAKSA_SUCCESS);
        }

        if (pack_info) {
            rc = yaksa_info_free(pack_info);
            assert(rc == YAKSA_SUCCESS);
        }

        if (unpack_info) {
            rc = yaksa_info_free(unpack_info);
            assert(rc == YAKSA_SUCCESS);
        }

        pack_copy_content(dbuf_d, dbuf_h, dobj.DTP_bufsize, dbuf_memtype);
        rc = DTP_obj_buf_check(dobj, dbuf_h, 0, 1, basecount);
        assert(rc == DTP_SUCCESS);


        /* free allocated buffers and objects */
        pack_free_mem(sbuf_memtype, sbuf_h, sbuf_d);
        pack_free_mem(dbuf_memtype, dbuf_h, dbuf_d);
        pack_free_mem(tbuf_memtype, tbuf_h, tbuf_d);

        DTP_obj_free(dobj);
    }

    free(segment_lengths);
    free(segment_starts);

    return NULL;
}

int main(int argc, char **argv)
{
    int num_threads = 1;

    while (--argc && ++argv) {
        if (!strcmp(*argv, "-datatype")) {
            --argc;
            ++argv;
            strncpy(typestr, *argv, MAX_DTP_BASESTRLEN);
        } else if (!strcmp(*argv, "-count")) {
            --argc;
            ++argv;
            basecount = atoi(*argv);
        } else if (!strcmp(*argv, "-seed")) {
            --argc;
            ++argv;
            seed = atoi(*argv);
        } else if (!strcmp(*argv, "-iters")) {
            --argc;
            ++argv;
            iters = atoi(*argv);
        } else if (!strcmp(*argv, "-segments")) {
            --argc;
            ++argv;
            max_segments = atoi(*argv);
        } else if (!strcmp(*argv, "-ordering")) {
            --argc;
            ++argv;
            if (!strcmp(*argv, "normal"))
                pack_order = PACK_ORDER__NORMAL;
            else if (!strcmp(*argv, "reverse"))
                pack_order = PACK_ORDER__REVERSE;
            else if (!strcmp(*argv, "random"))
                pack_order = PACK_ORDER__RANDOM;
            else {
                fprintf(stderr, "unknown packing order %s\n", *argv);
                exit(1);
            }
        } else if (!strcmp(*argv, "-overlap")) {
            --argc;
            ++argv;
            if (!strcmp(*argv, "none"))
                overlap = OVERLAP__NONE;
            else if (!strcmp(*argv, "regular"))
                overlap = OVERLAP__REGULAR;
            else if (!strcmp(*argv, "irregular"))
                overlap = OVERLAP__IRREGULAR;
            else {
                fprintf(stderr, "unknown overlap type %s\n", *argv);
                exit(1);
            }
        } else if (!strcmp(*argv, "-sbuf-memtype")) {
            --argc;
            ++argv;
            if (!strcmp(*argv, "unreg-host"))
                sbuf_memtype = MEM_TYPE__UNREGISTERED_HOST;
            else if (!strcmp(*argv, "reg-host"))
                sbuf_memtype = MEM_TYPE__REGISTERED_HOST;
            else if (!strcmp(*argv, "managed"))
                sbuf_memtype = MEM_TYPE__MANAGED;
            else if (!strcmp(*argv, "device"))
                sbuf_memtype = MEM_TYPE__DEVICE;
            else {
                fprintf(stderr, "unknown buffer type %s\n", *argv);
                exit(1);
            }
        } else if (!strcmp(*argv, "-dbuf-memtype")) {
            --argc;
            ++argv;
            if (!strcmp(*argv, "unreg-host"))
                dbuf_memtype = MEM_TYPE__UNREGISTERED_HOST;
            else if (!strcmp(*argv, "reg-host"))
                dbuf_memtype = MEM_TYPE__REGISTERED_HOST;
            else if (!strcmp(*argv, "managed"))
                dbuf_memtype = MEM_TYPE__MANAGED;
            else if (!strcmp(*argv, "device"))
                dbuf_memtype = MEM_TYPE__DEVICE;
            else {
                fprintf(stderr, "unknown buffer type %s\n", *argv);
                exit(1);
            }
        } else if (!strcmp(*argv, "-tbuf-memtype")) {
            --argc;
            ++argv;
            if (!strcmp(*argv, "unreg-host"))
                tbuf_memtype = MEM_TYPE__UNREGISTERED_HOST;
            else if (!strcmp(*argv, "reg-host"))
                tbuf_memtype = MEM_TYPE__REGISTERED_HOST;
            else if (!strcmp(*argv, "managed"))
                tbuf_memtype = MEM_TYPE__MANAGED;
            else if (!strcmp(*argv, "device"))
                tbuf_memtype = MEM_TYPE__DEVICE;
            else {
                fprintf(stderr, "unknown buffer type %s\n", *argv);
                exit(1);
            }
        } else if (!strcmp(*argv, "-device-start-id")) {
            --argc;
            ++argv;
            device_id = atoi(*argv);
        } else if (!strcmp(*argv, "-device-stride")) {
            --argc;
            ++argv;
            device_stride = atoi(*argv);
        } else if (!strcmp(*argv, "-verbose")) {
            verbose = 1;
        } else if (!strcmp(*argv, "-num-threads")) {
            --argc;
            ++argv;
            num_threads = atoi(*argv);
        } else {
            fprintf(stderr, "unknown argument %s\n", *argv);
            exit(1);
        }
    }
    if (strlen(typestr) == 0 || basecount <= 0 || seed < 0 || iters <= 0 || max_segments < 0 ||
        pack_order == PACK_ORDER__UNSET || overlap < 0 || num_threads <= 0) {
        fprintf(stderr, "Usage: ./pack {options}\n");
        fprintf(stderr, "   -datatype    base datatype to use, e.g., int\n");
        fprintf(stderr, "   -count       number of base datatypes in the signature\n");
        fprintf(stderr, "   -seed        random seed (changes the datatypes generated)\n");
        fprintf(stderr, "   -iters       number of iterations\n");
        fprintf(stderr, "   -segments    number of segments to chop the packing into\n");
        fprintf(stderr, "   -ordering  packing order of segments (normal, reverse, random)\n");
        fprintf(stderr, "   -overlap     should packing overlap (none, regular, irregular)\n");
        fprintf(stderr, "   -sbuf-memtype memory type (unreg-host, reg-host, device)\n");
        fprintf(stderr, "   -dbuf-memtype memory type (unreg-host, reg-host, device)\n");
        fprintf(stderr, "   -tbuf-memtype memory type (unreg-host, reg-host, device)\n");
        fprintf(stderr, "   -device-start-id  ID of the device for the first allocation\n");
        fprintf(stderr, "   -device-stride    difference between consecutive device allocations\n");
        fprintf(stderr, "   -verbose     verbose output\n");
        fprintf(stderr, "   -num-threads number of threads to spawn\n");
        exit(1);
    }

    yaksa_init(NULL);
    pack_init_devices();

    dtp = (DTP_pool_s *) malloc(num_threads * sizeof(DTP_pool_s));
    for (uintptr_t i = 0; i < num_threads; i++) {
        int rc = DTP_pool_create(typestr, basecount, seed + i, &dtp[i]);
        assert(rc == DTP_SUCCESS);
    }

    pthread_t *threads = (pthread_t *) malloc(num_threads * sizeof(pthread_t));

    for (uintptr_t i = 0; i < num_threads; i++)
        pthread_create(&threads[i], NULL, runtest, (void *) i);

    for (uintptr_t i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);

    free(threads);

    for (uintptr_t i = 0; i < num_threads; i++) {
        int rc = DTP_pool_free(dtp[i]);
        assert(rc == DTP_SUCCESS);
    }
    free(dtp);

    pack_finalize_devices();
    yaksa_finalize();

    return 0;
}
