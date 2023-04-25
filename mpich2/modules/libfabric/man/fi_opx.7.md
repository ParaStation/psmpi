---
layout: page
title: fi_opx(7)
tagline: Libfabric Programmer's Manual
---
{%include JB/setup %}

# NAME

fi_opx \- The Omni-Path Express Fabric Provider

# OVERVIEW

The OPX provider is a native implementation of the libfabric interfaces
that makes direct use of Omni-Path fabrics as well as libfabric
acceleration features.
The purpose of this provider is to show the scalability and
performance of libfabric, providing an "extreme scale" development
environment for applications and middleware using the libfabric API, and
to support a functional and performant version of MPI on Omni-Path fabrics.

# SUPPORTED FEATURES

The OPX provider supports most features defined for the libfabric API.

Key features include:

Endpoint types 
: The Omni-Path HFI hardware is connectionless and reliable.
  The OPX provider only supports the *FI_EP_RDM* endpoint type.

Capabilities 
: Supported capabilities include *FI_MSG*, *FI_RMA, *FI_TAGGED*, *FI_ATOMIC*,
  *FI_NAMED_RX_CTX*, *FI_SOURCE*, *FI_SEND*, *FI_RECV*, *FI_MULTI_RECV*, 
  *FI_DIRECTED_RECV*, *FI_SOURCE*.

  Notes on *FI_DIRECTED_RECV* capability: The immediate data which is sent
  within the "senddata" call to support *FI_DIRECTED_RECV* for OPX 
  must be exactly 4 bytes, which OPX uses to completely identify the
  source address to an exascale\-level number of ranks for tag matching on
  the recv and can be managed within the MU packet.
  Therefore the domain attribute "cq_data_size" is set to 4 which is the OFI
  standard minimum.

Modes 
: Two modes are defined: *FI_CONTEXT2* and *FI_ASYNC_IOV*.
  The OPX provider requires *FI_CONTEXT2*.

Additional features
: Supported additional features include *FABRIC_DIRECT*, *scalable endpoints*,
  and *counters*.

Progress 
: Only *FI_PROGRESS_MANUAL* is supported.

Address vector 
: Only the *FI_AV_MAP* address vector format is supported.

Memory registration modes 
: Only *FI_MR_SCALABLE* is supported.

# UNSUPPORTED FEATURES

Endpoint types 
: Unsupported endpoint types include *FI_EP_DGRAM* and *FI_EP_MSG*.

Capabilities 
: The OPX provider does not support *FI_RMA_EVENT* and *FI_TRIGGER* 
  capabilities.

Address vector 
: The OPX provider does not support the *FI_AV_TABLE* address vector 
  format. This may be added in the future.

# LIMITATIONS

As OPX is under development this list of limitations is subject
to change.

It runs under the following MPI versions:

Intel MPI from Parallel Studio 2020, update 4.
Intel MPI from OneAPI 2021, update 3.
Open MPI 4.1.2a1 (Older version of Open MPI will not work).
MPICH 3.4.2.

Currently, this provider is PIO-only. SDMA is not supported
at this time.

Usage:

If using with OpenMPI 4.1.x, disable UCX and openib transports.
OPX is not compatible with Open MPI 4.1.x PML/BTL.
DMA, RDMA and SDMA are not implemented.
Performance falls off when using message sizes larger than 
1 MTU (4K max size). 
Shared memory is not cleaned up after an application crashes. Use
"rm -rf /dev/shm/*" to remove old shared-memory files.

# RUNTIME PARAMETERS

*FI_OPX_UUID*
: OPX requires a unique ID for each job. In order for all processes in a
  job to communicate with each other, they require to use the same UUID.
  This variable can be set with FI_OPX_UUID=${RANDOM} 
  The default UUID is 00112233445566778899aabbccddeeff.

*FI_OPX_RELIABILITY_SERVICE_USEC_MAX*
: This setting controls how frequently the reliability/replay function
  will issue PING requests to a remote connection. Reducing this value
  may improve performance at the expense of increased traffic on the OPX 
  fabric.
  Default setting is 500.

*FI_OPX_RELIABILITY_SERVICE_PRE_ACK_RATE*
: This setting controls how frequently a receiving rank will send ACKs
  for packets it has received without being prompted through a PING request.
  A non-zero value N tells the receiving rank to send an ACK for the
  last N packets every Nth packet. Used in conjunction with an increased
  value for FI_OPX_RELIABILITY_SERVICE_USEC_MAX may improve performance.

  Valid values are 0 (disabled) and powers of 2 in the range of 1-32,768, inclusive.

  Default setting is 64.

*FI_OPX_HFI_SELECT*
: Controls how OPX chooses which HFI to use when opening a context.
  Has two forms:
  - `<hfi-unit>` Force OPX provider to use `hfi-unit`.
  - `<selector1>[,<selector2>[,...,<selectorN>]]` Select HFI based on first matching `selector`

  Where `selector` is one of the following forms:
  - `default` to use the default logic
  - `fixed:<hfi-unit>` to fix to one `hfi-unit`
  - `<selector-type>:<hfi-unit>:<selector-data>`

  The above fields have the following meaning:
  - `selector-type` The selector criteria the caller opening the context is evaluated against.
  - `hfi-unit` The HFI to use if the caller matches the selector.
  - `selector-data` Data the caller must match (e.g. NUMA node ID).

  Where `selector-type` is one of the following:
  - `numa` True when caller is local to the NUMA node ID given by `selector-data`.
  - `core` True when caller is local to the CPU core given by `selector-data`.

  And `selector-data` is one of the following:
  - `value` The specific value to match
  - `<range-start>-<range-end>` Matches with any value in that range

  In the second form, when opening a context, OPX uses the `hfi-unit` of the
  first-matching selector. Selectors are evaluated left-to-right. OPX will
  return an error if the caller does not match any selector.

  In either form, it is an error if the specified or selected HFI is not in the
  Active state. In this case, OPX will return an error and execution will not
  continue.

  With this option, it is possible to cause OPX to try to open more contexts on
  an HFI than there are free contexts on that HFI. In this case, one or more of
  the context-opening calls will fail and OPX will return an error.
  For the second form, as which HFI is selected depends on properties of the
  caller, deterministic HFI selection requires deterministic caller properties.
  E.g.  for the `numa` selector, if the caller can migrate between NUMA domains,
  then HFI selection will not be deterministic.

  The logic used will always be the first valid in a selector list. For example, `default` and 
  `fixed` will match all callers, so if either are in the beginning of a selector list, you will
  only use `fixed` or `default` regardles of if there are any more selectors.

  Examples:
  - `FI_OPX_HFI_SELECT=1` all callers will open contexts on HFI 0.
  - `FI_OPX_HFI_SELECT=numa:0:0,numa:1:1,numa:0:2,numa:1:3` callers local to NUMA nodes 0 and 2 will use HFI 0, callers local to NUMA domains 1 and 3 will use HFI 1.
  - `FI_OPX_HFI_SELECT=numa:0:0-3,default` callers local to NUMA nodes 0 thru 3 (including 0 and 3) will use HFI 0, and all else will use default selection logic.
  - `FI_OPX_HFI_SELECT=core:1:0,fixed:0` callers local to CPU core 0 will use HFI 1, and all others will use HFI 0.
  - `FI_OPX_HFI_SELECT=default,core:1:0` all callers will use default HFI selection logic.

# SEE ALSO

[`fabric`(7)](fabric.7.html),
[`fi_provider`(7)](fi_provider.7.html),
[`fi_getinfo`(7)](fi_getinfo.7.html),
