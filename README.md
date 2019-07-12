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
psc        : Pathscale compiler (pathcc)
cellgcc    : ppu-gcc
cellxlc    : ppuxlc
cuda       : With CUDA support

devel      : With error checking and debug info (gcc)
cuda-devel : Same as 'devel' but with CUDA support
user       : No predefined options
ch3        : original mpich ch3 device (no parastation)
```

The following example configures ParaStation MPI for using the gcc compiler:
```
$ ../configure --prefix=/path/to/installation/dir --with-confset=default
```
#### Optional configure arguments
| Argument                    | Description                                 |
------------------------------|---------------------------------------------|
| `--with-hydra`              | Use MPICH's process manager hydra           |
| `--with-threading`          | Enable multi-thread support                 |
| `--with-topology-awareness` | Enable topology/hierarchy-aware collectives |


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
