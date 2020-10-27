ParaStation MPI
===============

Contents
--------
* Introduction
* Requirements
* Installation
    - Prerequisites
    - Configuration
    - Build
    - Environment
* Alternative Installation
* Session Parameters
    - Debugging
    - Feature Activation
    - Statistical Analysis
* Test Suite

Introduction
------------
ParaStation MPI is an implementation of the Message-Passing Interface (MPI)
Standard. Being an [MPICH](https://www.mpich.org)-derivate it bases on the most
recent MPICH-3.3 implementation. ParaStation MPI relies on the low-level
communication layer [pscom](https://github.com/parastation/pscom) for the
implementation of the psp device on the ADI-Layer of the MPICH architecture.
In doing so, ParaStation MPI provides full MPI-3 compliance including recent
additions to the RMA interface. Furthermore, the MPICH-related Process Manager
Interface (PMI) is supported enabling a compatibility to a variety of process
managers.

Requirements
------------
* C compiler with C99 support
* C++ compiler for applications written in C++ (optional)
* Fortran compiler for applications written in Fortran (optional)
* Copy of the [pscom](https://github.com/parastation/pscom) library
* Copy of the [psmgmt](https://github.com/parastation/psmgmt) process manager (optional)

Installation
------------

### Prerequisites
1. A working installation of the pscom library
   (except for the [Alternative Installation](#Alternative-Installation))
2. You need to install psmgmt if you do not want to use the Hydra process
   manager that comes with MPICH.

### Configure ParaStation MPI
Download the source code from GitHub:
````
$ git clone https://github.com/ParaStation/psmpi.git psmpi
````

We encourage you *not* to build ParaStation MPI directly within the main source
folder:
````
$ cd /path/to/psmpi
$ mkdir build && cd build
````

ParaStation MPI relies on Autotools as the build system requiring a configure
step. The build system already provides so-called "confsets" that pass the
necessary configuration arguments for a particular installation type to the
underlying MPICH build system. It is strongly recommended to rely on these
confsets for a proper installation! Currently, the following confsets are
provided:
```
none       : Do not configure mpich.
             Prepare only for tar,rpm and srpm build
default    : like gcc

gcc        : Use Gnu compiler (gcc)
intel      : Use intel compiler (icc)
icc        : Like 'intel'
pgi        : Portland group compiler (pgcc)
nvhpc      : Nvidia hpc compiler (nvc)
psc        : Pathscale compiler (pathcc)
cellgcc    : ppu-gcc
cellxlc    : ppuxlc

devel      : With error checking and debug info (gcc)
user       : No predefined options
ch3|ch4    : original mpich ch3/ch4 device (no parastation)
```

The following example configures ParaStation MPI for using the gcc compiler:
```
$ ../configure --prefix=/path/to/installation/dir --with-confset=default
```
#### Optional configure arguments
| Argument                    | Description                                       |
------------------------------|---------------------------------------------------|
| `--with-cuda`               | Enable CUDA awareness                             |
| `--with-hydra`              | Use MPICH's process manager Hydra                 |
| `--with-threading`          | Enable multi-thread support                       |
| `--with-topology-awareness` | Enable topology/hierarchy-aware collectives       |
| `--with-session-statistics` | Enable the collection of statistical information  |
| `--with-hcoll[=PATH]`       | Enable hcoll support [PATH to hcoll installation] |
| `--with-hwloc[=PATH]`       | Enable hwloc in MPICH/Hydra [built-in or PATH]    |

### Build ParaStation MPI
For a successful build, your environment has to include the path to your pscom
installation, e.g., by setting the
`LIBRARY_PATH` and `CPATH` environment variables:
```
$ export LIBRARY_PATH=/path/to/pscom/installation/lib[64]:${LIBRARY_PATH}
$ export CPATH=/path/to/pscom/installation/include:${CPATH}
```

Now, ParaStation MPI can be built and installed in accordance with the
configuration arguments:
```
$ make -j8 && make install
```

### Prepare the environment
To use ParaStation MPI for building and running MPI applications, it is
advisable to adjust the environment properly. This can be done, e.g.,  by using
the following bash script:
```
#!/bin/bash

if [ $# -eq 0 ]; then
    echo "ERROR: Please provide the path to ParaStation MPI. Abort!"
    exit -1
fi

PARASTATION_MPI=`realpath ${1}`

export PATH="${PARASTATION_MPI}/bin${PATH:+:}${PATH}"
export CPATH="${PARASTATION_MPI}/include${CPATH:+:}${CPATH}"
export LD_LIBRARY_PATH="${PARASTATION_MPI}/lib${LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${PARASTATION_MPI}/lib${LIBRARY_PATH:+:}${LIBRARY_PATH}"
```

Alternative Installation
------------------------
Instead of relying on the pscom as a shared-library, ParaStation MPI can be
optionally compiled as a single shared-object by directly using the pscom
sources. For doing so, the pscom source files are required:
```
$ git clone https://github.com/ParaStation/pscom.git pscom
```

After downloading the psmpi sources, the same
configuration parameters apply as discussed above (see [Configure](#Configure-ParaStation-MPI)).
Additionally, you will need to add the `--with-pscom-allin` flag, e.g.:
```
$ ../configure --prefix=/path/to/installation/dir --with-confset=default --with-pscom-allin=/path/to/pscom/sources
```

By default, the pscom4openib as well as the pscom4psm plugins are included
firmly into ParaStation MPI if `--with-pscom-allin` is set and the related
low-level drivers are found. For specifying the plugins to be built-in
explicitly, the `--with-pscom-builtin[=list]` option can be used.


Session Parameters
------------------

### Debugging

| Environment Variable        | Description                                          |
------------------------------|------------------------------------------------------|
| `PSP_DEBUG=0`               | only fatal conditions (like detected bugs)           |
| `PSP_DEBUG=1`               | fatal conditions + errors (default)                  |
| `PSP_DEBUG=2`               |  + warnings                                          |
| `PSP_DEBUG=3`               |  + information                                       |
| `PSP_DEBUG=4`               |  + debug                                             |
| `PSP_DEBUG=5`               |  + verbose debug                                     |
| `PSP_DEBUG=6`               |  + tracing calls                                     |
| `PSP_DEBUG_VERSION=1`       | Show always the pscom version (info)                 |
| `PSP_DEBUG_CONTYPE=1`       | Show connection types (info)                         |

### Feature Activation

| Environment Variable        | Description                                          |
------------------------------|------------------------------------------------------|
| `PSP_CUDA=1`                | Enable/Disable CUDA awareness (default = 0)          |
| `PSP_HCOLL=1`               | Enable/Disable HCOLL support (default = 0)           |
| `PSP_SMP_AWARENESS=1`       | Take locality information into account (default = 1) |
| `PSP_SMP_AWARE_COLLOPS=1`   | Enable/Disable SMP-aware collectives (default = 0)   |
| `PSP_MSA_AWARENESS=1`       | Take topology information into account (default = 0) |
| `PSP_MSA_AWARE_COLLOPS=1`   | Enable/Disable MSA-aware collectives (default = 0)   |

### Statistical Analysis

| Environment Variable        | Description                                          |
------------------------------|------------------------------------------------------|
| `PSP_HISTOGRAM=1`           | Enable the collection of statistical data            |
| `PSP_HISTOGRAM_MIN=x`       | Set the lower message size limit for the histogram   |
| `PSP_HISTOGRAM_MAX=y`       | Set the upper message size limit for the histogram   |
| `PSP_HISTOGRAM_SHIFT=z`     | Bit shift for the number of bins of the histogram    |
| `PSP_HISTOGRAM_CONTYPE=con` | Limit the histogram to a particular connection type  |


Test Suite
----------
MPICH has a test suite that can also be used and is even extended by ParaStation MPI.
````
$ make test
````
...will run the complete test suite comprising tests in the following subfolders in `mpich2/test/mpi/`:

````
parastation
cuda
attr
coll
comm
cxx
datatype
errhan
errors
f77
f90
group
impls
info
io
init
mpi_t
pt2pt
rma
spawn
topo
````

However, you can also have only tests of a certain subdirectory to be executed by specifying a `TESTDIR`, for example:
````
$ make test TESTDIR=pt2pt
````

In addition, a `TESTSET` can be specified to further restrict the number of tests.
The tests that belong to a certain test set are stated as a list in a file with the same name within the above listed subdirectories.
Currently, only `ps-test-minimal` (minimal list of tests) and `testlist` (containing all tests) are valid test sets.
So, for example, the following invocation runs all tests belonging to `ps-test-minimal` within all subdirectories:

````
$ make test TESTSET=ps-test-minimal
````
And if, for example, the test set `ps-test-minimal` should only be executed for tests within the subfolder `pt2pt`, then the following invocation is the means of choice:
````
$ make test TESTSET=ps-test-minimal TESTDIR=pt2pt
````
