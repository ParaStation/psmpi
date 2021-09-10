#! /usr/bin/env python3
##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

import sys
import argparse

sys.path.append('maint/')
import yutils

sys.path.append('src/backend/')
import gencomm


########################################################################################
##### Global variables
########################################################################################

num_paren_open = 0
blklens = [ "generic" ]
builtin_types = [ "_Bool", "char", "wchar_t", "int8_t", "int16_t", "int32_t", "int64_t", "float", "double" ]
builtin_maps = {
    "YAKSA_TYPE__LONG_DOUBLE": "double",
}


########################################################################################
##### Type-specific functions
########################################################################################

## hvector routines
def hvector(suffix, dtp, b, last):
    global s
    global idx
    global need_extent
    yutils.display(OUTFILE, "intptr_t stride%d = %s->u.hvector.stride;\n" % (suffix, dtp))
    if (need_extent == True):
        yutils.display(OUTFILE, "uintptr_t extent%d = %s->extent;\n" % (suffix, dtp))
    if (last != 1):
        s += " + x%d * stride%d + x%d * extent%d" % (idx, suffix, idx + 1, suffix + 1)
        need_extent = True
    else:
        s += " + x%d * stride%d + x%d * sizeof(%s)" % (idx, suffix, idx + 1, b)
        need_extent = False
    idx = idx + 2

## blkhindx routines
def blkhindx(suffix, dtp, b, last):
    global s
    global idx
    global need_extent
    yutils.display(OUTFILE, "intptr_t *array_of_displs%d = %s->u.blkhindx.array_of_displs;\n" % (suffix, dtp))
    if (need_extent == True):
        yutils.display(OUTFILE, "uintptr_t extent%d = %s->extent;\n" % (suffix, dtp))
    if (last != 1):
        s += " + array_of_displs%d[x%d] + x%d * extent%d" % \
             (suffix, idx, idx + 1, suffix + 1)
        need_extent = True
    else:
        s += " + array_of_displs%d[x%d] + x%d * sizeof(%s)" % (suffix, idx, idx + 1, b)
        need_extent = False
    idx = idx + 2

## hindexed routines
def hindexed(suffix, dtp, b, last):
    global s
    global idx
    global need_extent
    yutils.display(OUTFILE, "intptr_t *array_of_displs%d = %s->u.hindexed.array_of_displs;\n" % (suffix, dtp))
    if (need_extent == True):
        yutils.display(OUTFILE, "uintptr_t extent%d = %s->extent;\n" % (suffix, dtp))
    if (last != 1):
        s += " + array_of_displs%d[x%d] + x%d * extent%d" % \
             (suffix, idx, idx + 1, suffix + 1)
        need_extent = True
    else:
        s += " + array_of_displs%d[x%d] + x%d * sizeof(%s)" % (suffix, idx, idx + 1, b)
        need_extent = False
    idx = idx + 2

## contig routines
def contig(suffix, dtp, b, last):
    global s
    global idx
    global need_extent
    yutils.display(OUTFILE, "intptr_t stride%d = %s->u.contig.child->extent;\n" % (suffix, dtp))
    if (need_extent == True):
        yutils.display(OUTFILE, "uintptr_t extent%d = %s->extent;\n" % (suffix, dtp))
    need_extent = False
    s += " + x%d * stride%d" % (idx, suffix)
    idx = idx + 1

# resized routines
def resized(suffix, dtp, b, last):
    global need_extent
    if (need_extent == True):
        yutils.display(OUTFILE, "uintptr_t extent%d = %s->extent;\n" % (suffix, dtp))
    need_extent = False


########################################################################################
##### Core kernels
########################################################################################
def generate_kernels(b, darray):
    global need_extent
    global s
    global idx

    # we don't need pup kernels for basic types
    if (len(darray) == 0):
        return

    for func in "pack","unpack":
        ##### figure out the function name to use
        funcprefix = "%s_" % func
        for d in darray:
            funcprefix = funcprefix + "%s_" % d
        funcprefix = funcprefix + b.replace(" ", "_")

        ##### generate the CUDA kernel
        yutils.display(OUTFILE, "__global__ void yaksuri_cudai_kernel_%s(const void *inbuf, void *outbuf, uintptr_t count, const yaksuri_cudai_md_s *__restrict__ md)\n" % funcprefix)
        yutils.display(OUTFILE, "{\n")
        yutils.display(OUTFILE, "const char *__restrict__ sbuf = (const char *) inbuf;\n");
        yutils.display(OUTFILE, "char *__restrict__ dbuf = (char *) outbuf;\n");
        yutils.display(OUTFILE, "uintptr_t extent = md->extent;\n")
        yutils.display(OUTFILE, "uintptr_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n")
        yutils.display(OUTFILE, "uintptr_t res = idx;\n")
        yutils.display(OUTFILE, "uintptr_t inner_elements = md->num_elements;\n")
        yutils.display(OUTFILE, "\n")

        yutils.display(OUTFILE, "if (idx >= (count * inner_elements))\n")
        yutils.display(OUTFILE, "    return;\n")
        yutils.display(OUTFILE, "\n")

        # copy loop
        idx = 0
        md = "md"
        for d in darray:
            if (d == "hvector" or d == "blkhindx" or d == "hindexed" or \
                d == "contig"):
                yutils.display(OUTFILE, "uintptr_t x%d = res / inner_elements;\n" % idx)
                idx = idx + 1
                yutils.display(OUTFILE, "res %= inner_elements;\n")
                yutils.display(OUTFILE, "inner_elements /= %s->u.%s.count;\n" % (md, d))
                yutils.display(OUTFILE, "\n")

            if (d == "hvector" or d == "blkhindx"):
                yutils.display(OUTFILE, "uintptr_t x%d = res / inner_elements;\n" % idx)
                idx = idx + 1
                yutils.display(OUTFILE, "res %= inner_elements;\n")
                yutils.display(OUTFILE, "inner_elements /= %s->u.%s.blocklength;\n" % (md, d))
            elif (d == "hindexed"):
                yutils.display(OUTFILE, "uintptr_t x%d;\n" % idx)
                yutils.display(OUTFILE, "for (int i = 0; i < %s->u.%s.count; i++) {\n" % (md, d))
                yutils.display(OUTFILE, "    uintptr_t in_elems = %s->u.%s.array_of_blocklengths[i] *\n" % (md, d))
                yutils.display(OUTFILE, "                         %s->u.%s.child->num_elements;\n" % (md, d))
                yutils.display(OUTFILE, "    if (res < in_elems) {\n")
                yutils.display(OUTFILE, "        x%d = i;\n" % idx)
                yutils.display(OUTFILE, "        res %= in_elems;\n")
                yutils.display(OUTFILE, "        inner_elements = %s->u.%s.child->num_elements;\n" % (md, d))
                yutils.display(OUTFILE, "        break;\n")
                yutils.display(OUTFILE, "    } else {\n")
                yutils.display(OUTFILE, "        res -= in_elems;\n")
                yutils.display(OUTFILE, "    }\n")
                yutils.display(OUTFILE, "}\n")
                idx = idx + 1
                yutils.display(OUTFILE, "\n")

            md = "%s->u.%s.child" % (md, d)

        yutils.display(OUTFILE, "uintptr_t x%d = res;\n" % idx)
        yutils.display(OUTFILE, "\n")

        dtp = "md"
        s = "x0 * extent"
        idx = 1
        x = 1
        need_extent = False
        for d in darray:
            if (x == len(darray)):
                last = 1
            else:
                last = 0
            getattr(sys.modules[__name__], d)(x, dtp, b, last)
            x = x + 1
            dtp = dtp + "->u.%s.child" % d

        if (func == "pack"):
            yutils.display(OUTFILE, "*((%s *) (void *) (dbuf + idx * sizeof(%s))) = *((const %s *) (const void *) (sbuf + %s));\n"
                           % (b, b, b, s))
        else:
            yutils.display(OUTFILE, "*((%s *) (void *) (dbuf + %s)) = *((const %s *) (const void *) (sbuf + idx * sizeof(%s)));\n"
                           % (b, s, b, b))

        yutils.display(OUTFILE, "}\n\n")

        # generate the host function
        yutils.display(OUTFILE, "void yaksuri_cudai_%s(const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s *md, int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device)\n" % funcprefix)
        yutils.display(OUTFILE, "{\n")
        yutils.display(OUTFILE, "void *args[] = { &inbuf, &outbuf, &count, &md };\n")
        yutils.display(OUTFILE, "cudaError_t cerr = cudaLaunchKernel((const void *) yaksuri_cudai_kernel_%s,\n" % funcprefix)
        yutils.display(OUTFILE, "    dim3(n_blocks_x, n_blocks_y, n_blocks_z), dim3(n_threads), args, 0, yaksuri_cudai_global.stream[device]);\n")
        yutils.display(OUTFILE, "YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);\n")
        yutils.display(OUTFILE, "}\n\n")


########################################################################################
##### main function
########################################################################################
if __name__ == '__main__':
    ##### parse user arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pup-max-nesting', type=int, default=3, help='maximum nesting levels to generate')
    args = parser.parse_args()
    if (args.pup_max_nesting < 0):
        parser.print_help()
        print
        print("===> ERROR: pup-max-nesting must be positive")
        sys.exit(1)

    ##### generate the core pack/unpack kernels (single level)
    for b in builtin_types:
        for d in gencomm.derived_types:
            filename = "src/backend/cuda/pup/yaksuri_cudai_pup_%s_%s.cu" % (d, b.replace(" ","_"))
            yutils.copyright_c(filename)
            OUTFILE = open(filename, "a")
            yutils.display(OUTFILE, "#include <string.h>\n")
            yutils.display(OUTFILE, "#include <stdint.h>\n")
            yutils.display(OUTFILE, "#include <wchar.h>\n")
            yutils.display(OUTFILE, "#include <assert.h>\n")
            yutils.display(OUTFILE, "#include <cuda.h>\n")
            yutils.display(OUTFILE, "#include <cuda_runtime.h>\n")
            yutils.display(OUTFILE, "#include \"yaksuri_cudai_base.h\"\n")
            yutils.display(OUTFILE, "#include \"yaksuri_cudai_pup.h\"\n")
            yutils.display(OUTFILE, "\n")

            emptylist = [ ]
            emptylist.append(d)
            generate_kernels(b, emptylist)
            emptylist.pop()

            OUTFILE.close()

    ##### generate the core pack/unpack kernels (more than one level)
    darraylist = [ ]
    yutils.generate_darrays(gencomm.derived_types, darraylist, args.pup_max_nesting - 2)
    for b in builtin_types:
        for d1 in gencomm.derived_types:
            for d2 in gencomm.derived_types:
                filename = "src/backend/cuda/pup/yaksuri_cudai_pup_%s_%s_%s.cu" % (d1, d2, b.replace(" ","_"))
                yutils.copyright_c(filename)
                OUTFILE = open(filename, "a")
                yutils.display(OUTFILE, "#include <string.h>\n")
                yutils.display(OUTFILE, "#include <stdint.h>\n")
                yutils.display(OUTFILE, "#include <wchar.h>\n")
                yutils.display(OUTFILE, "#include <assert.h>\n")
                yutils.display(OUTFILE, "#include <cuda.h>\n")
                yutils.display(OUTFILE, "#include <cuda_runtime.h>\n")
                yutils.display(OUTFILE, "#include \"yaksuri_cudai_base.h\"\n")
                yutils.display(OUTFILE, "#include \"yaksuri_cudai_pup.h\"\n")
                yutils.display(OUTFILE, "\n")

                for darray in darraylist:
                    darray.append(d1)
                    darray.append(d2)
                    generate_kernels(b, darray)
                    darray.pop()
                    darray.pop()

                OUTFILE.close()

    ##### generate the core pack/unpack kernel declarations
    filename = "src/backend/cuda/pup/yaksuri_cudai_pup.h"
    yutils.copyright_c(filename)
    OUTFILE = open(filename, "a")
    yutils.display(OUTFILE, "#ifndef YAKSURI_CUDAI_PUP_H_INCLUDED\n")
    yutils.display(OUTFILE, "#define YAKSURI_CUDAI_PUP_H_INCLUDED\n")
    yutils.display(OUTFILE, "\n")
    yutils.display(OUTFILE, "#include <string.h>\n")
    yutils.display(OUTFILE, "#include <stdint.h>\n")
    yutils.display(OUTFILE, "\n")
    yutils.display(OUTFILE, "#ifdef __cplusplus\n")
    yutils.display(OUTFILE, "extern \"C\"\n")
    yutils.display(OUTFILE, "{\n")
    yutils.display(OUTFILE, "#endif\n")
    yutils.display(OUTFILE, "\n")

    darraylist = [ ]
    yutils.generate_darrays(gencomm.derived_types, darraylist, args.pup_max_nesting)
    for b in builtin_types:
        for darray in darraylist:
            # we don't need pup kernels for basic types
            if (len(darray) == 0):
                continue

            for func in "pack","unpack":
                ##### figure out the function name to use
                s = "void yaksuri_cudai_%s_" % func
                for d in darray:
                    s = s + "%s_" % d
                s = s + b.replace(" ", "_")
                OUTFILE.write("%s" % s)
                OUTFILE.write("(const void *inbuf, ")
                OUTFILE.write("void *outbuf, ")
                OUTFILE.write("uintptr_t count, ")
                OUTFILE.write("yaksuri_cudai_md_s *md, ")
                OUTFILE.write("int n_threads, ")
                OUTFILE.write("int n_blocks_x, ")
                OUTFILE.write("int n_blocks_y, ")
                OUTFILE.write("int n_blocks_z, ")
                OUTFILE.write("int device);\n")

    yutils.display(OUTFILE, "\n")
    yutils.display(OUTFILE, "#ifdef __cplusplus\n")
    yutils.display(OUTFILE, "}\n")
    yutils.display(OUTFILE, "#endif\n")
    yutils.display(OUTFILE, "\n")
    yutils.display(OUTFILE, "#endif  /* YAKSURI_CUDAI_PUP_H_INCLUDED */\n")
    OUTFILE.close()

    ##### generate the pup makefile
    filename = "src/backend/cuda/pup/Makefile.pup.mk"
    yutils.copyright_makefile(filename)
    OUTFILE = open(filename, "a")
    yutils.display(OUTFILE, "libyaksa_la_SOURCES += \\\n")
    for b in builtin_types:
        for d1 in gencomm.derived_types:
            yutils.display(OUTFILE, "\tsrc/backend/cuda/pup/yaksuri_cudai_pup_%s_%s.cu \\\n" % \
                           (d1, b.replace(" ","_")))
            for d2 in gencomm.derived_types:
                yutils.display(OUTFILE, "\tsrc/backend/cuda/pup/yaksuri_cudai_pup_%s_%s_%s.cu \\\n" % \
                               (d1, d2, b.replace(" ","_")))
    yutils.display(OUTFILE, "\tsrc/backend/cuda/pup/yaksuri_cudai_pup.c\n")
    yutils.display(OUTFILE, "\n")
    yutils.display(OUTFILE, "noinst_HEADERS += \\\n")
    yutils.display(OUTFILE, "\tsrc/backend/cuda/pup/yaksuri_cudai_pup.h\n")
    OUTFILE.close()

    ##### generate the switching logic to select pup functions
    gencomm.populate_pupfns(args.pup_max_nesting, "cuda", blklens, builtin_types, builtin_maps)
