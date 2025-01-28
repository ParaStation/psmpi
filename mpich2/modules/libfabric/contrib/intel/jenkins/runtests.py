import argparse
import os
import sys
sys.path.append(f"{os.environ['WORKSPACE']}/ci_resources/configs/{os.environ['CLUSTER']}")
import cloudbees_config
import subprocess
import run
import common
import shlex

class ParseDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

parser = argparse.ArgumentParser()
parser.add_argument('--build_hw', help="HW type for build",
                    choices=['water', 'grass', 'fire', 'electric', 'daos',\
                                'gpu', 'ucx'])
parser.add_argument('--prov', help="core provider", choices=['verbs', \
                     'tcp', 'udp', 'sockets', 'shm', 'psm3', 'ucx'])
parser.add_argument('--util', help="utility provider", choices=['rxd', 'rxm'])
parser.add_argument('--ofi_build_mode', help="specify the build configuration",\
                    choices = ['reg', 'dbg', 'dl'], default='reg')
parser.add_argument('--test', help="specify test to execute", \
                    choices = ['all', 'shmem', 'IMB', 'osu', 'oneccl', \
                               'mpichtestsuite', 'fabtests', 'onecclgpu', \
                               'fi_info', 'daos', 'multinode', 'dmabuf'])

parser.add_argument('--imb_grp', help="IMB test group 1:[MPI1, P2P], \
                    2:[EXT, IO], 3:[NBC, RMA, MT]", choices=['1', '2', '3'])
parser.add_argument('--way', help="direction to run with device option",
                    choices=['h2d', 'd2d', 'xd2d'], default=None)
parser.add_argument('--user_env', help="Run with additional environment " \
                    "variables", nargs='*', action=ParseDict, default={})
parser.add_argument('--mpi', help="Select mpi to use for middlewares",
                    choices=['impi', 'mpich', 'ompi'], default='impi')
parser.add_argument('--log_file', help="Full path to log file",
                    default=os.environ['DEFAULT_LOG_LOCATION'], type=str)
parser.add_argument('--weekly', help="run weekly", default=False, type=bool)

args = parser.parse_args()
build_hw = args.build_hw
args_core = args.prov
args_util = args.util
user_env = args.user_env
log_file = args.log_file
weekly = args.weekly

if (args.ofi_build_mode):
    ofi_build_mode = args.ofi_build_mode
else:
    ofi_build_mode='reg'

if (args.test):
    run_test = args.test
else:
    run_test = 'all'

if (args.imb_grp):
    imb_group = args.imb_grp
else:
    imb_group = '1'

mpi = args.mpi
way = args.way

hosts = []
if 'slurm' in os.environ['FABRIC']:
    slurm_nodes = os.environ['SLURM_JOB_NODELIST'] # example cb[1-4,11]
    common.run_command(shlex.split(f"sinfo --Format=Features -n {slurm_nodes}"))
    if int(os.environ['SLURM_NNODES']) == 1:
        hosts.append(slurm_nodes)
    else:
        prefix = slurm_nodes[0:slurm_nodes.find('[')]
        nodes = slurm_nodes[slurm_nodes.find('[') + 1 :
                            slurm_nodes.find(']')].split(',') # ['1-4', '11']
        for item in nodes: # ['1-4', '11'] -> ['cb1', 'cb2', 'cb3', 'cb4', 'cb11']
            if '-' in item:
                rng = item.split('-')
                node_list = list(range(int(rng[0]), int(rng[1]) + 1))
                for node in node_list:
                    hosts.append(f'{prefix}{node}')
            else:
                hosts.append(f'{prefix}{item}')
else:
    node = (os.environ['NODE_NAME']).split('_')[0]
    hosts = [node]
    for host in cloudbees_config.node_map[node]:
        hosts.append(host)
    print(f"hosts = {hosts}")

print(common.cloudbees_log_start_string)

#this script is executed from /tmp
#this is done since some mpi tests
#look for a valid location before running
# the test on the secondary host(client)
# but jenkins only creates a valid path on
# the primary host (server/test node)

os.chdir('/tmp/')

if(args_core):
    if (run_test == 'all' or run_test == 'fi_info'):
        run.fi_info_test(build_hw, args_core, hosts, ofi_build_mode,
                         user_env, log_file, util=args.util)

    if (run_test == 'all' or run_test == 'fabtests'):
        run.fabtests(build_hw, args_core, hosts, ofi_build_mode, user_env,
                     log_file, args_util, way)

    if (run_test == 'all' or run_test == 'shmem'):
        run.shmemtest(build_hw, args_core, hosts, ofi_build_mode, user_env,
                      log_file, args_util)

    if (run_test == 'all' or run_test == 'oneccl'):
        run.oneccltest(build_hw, args_core, hosts, ofi_build_mode, user_env,
                       log_file, args_util)

    if (run_test == 'all' or run_test == 'onecclgpu'):
        run.oneccltestgpu(build_hw, args_core, hosts, ofi_build_mode,
                          user_env, log_file, args_util)

    if (run_test == 'all' or run_test == 'daos'):
        run.daos_cart_tests(build_hw, args_core, hosts, ofi_build_mode,
                            user_env, log_file, args_util)

    if (run_test == 'all' or run_test == 'multinode'):
        run.multinodetest(build_hw, args_core, hosts, ofi_build_mode,
                          user_env, log_file, args_util)

    if (run_test == 'all' or run_test == 'mpichtestsuite'):
        run.mpich_test_suite(build_hw, args_core, hosts, mpi,
                             ofi_build_mode, user_env, log_file,
                             args_util, weekly)

    if (run_test == 'all' or run_test == 'IMB'):
        run.intel_mpi_benchmark(build_hw, args_core, hosts, mpi,
                                ofi_build_mode, imb_group,
                                user_env, log_file, args_util)

    if (run_test == 'all' or run_test == 'osu'):
        run.osu_benchmark(build_hw, args_core, hosts, mpi,
                          ofi_build_mode, user_env, log_file,
                          args_util)

    if (run_test == 'all' or run_test == 'dmabuf'):
        run.dmabuftests(build_hw, args_core, hosts, ofi_build_mode,
                        user_env, log_file, args_util)
else:
    print("Error : Specify a core provider to run tests")
