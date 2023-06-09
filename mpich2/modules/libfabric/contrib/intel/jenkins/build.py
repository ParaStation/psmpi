import os
import sys

# add jenkins config location to PATH
sys.path.append(os.environ['CI_SITE_CONFIG'])

import ci_site_config
import argparse
import subprocess
import shlex
import common
import re
import shutil

def build_libfabric(libfab_install_path, mode, cluster=None):

    if (os.path.exists(libfab_install_path) != True):
        os.makedirs(libfab_install_path)

    config_cmd = ['./configure', f'--prefix={libfab_install_path}']
    enable_prov_val = 'yes'

    if (mode == 'dbg'):
        config_cmd.append('--enable-debug')
    elif (mode == 'dl'):
        enable_prov_val='dl'

    if (cluster == 'daos'):
        prov_list = common.daos_prov_list
    elif (cluster == 'dsa'):
        prov_list = common.dsa_prov_list
    else:
        prov_list = common.default_prov_list

    for prov in prov_list:
       config_cmd.append(f'--enable-{prov}={enable_prov_val}')

    for op in common.common_disable_list:
         config_cmd.append(f'--enable-{op}=no')

    if (cluster == 'default'):
        for op in common.default_enable_list:
            config_cmd.append(f'--enable-{op}')

    common.run_command(['./autogen.sh'])
    common.run_command(shlex.split(" ".join(config_cmd)))
    common.run_command(['make','clean'])
    common.run_command(['make', '-j32'])
    common.run_command(['make','install'])


def build_fabtests(libfab_install_path, mode):

    os.chdir(f'{workspace}/fabtests')
    if (mode == 'dbg'):
        config_cmd = ['./configure', '--enable-debug',
                      f'--prefix={libfab_install_path}',
                      f'--with-libfabric={libfab_install_path}']
    else:
        config_cmd = ['./configure', f'--prefix={libfab_install_path}',
                      f'--with-libfabric={libfab_install_path}']

    common.run_command(['./autogen.sh'])
    common.run_command(config_cmd)
    common.run_command(['make','clean'])
    common.run_command(['make', '-j32'])
    common.run_command(['make', 'install'])

def copy_build_dir(install_path):
    shutil.copytree(ci_site_config.build_dir,
                    f'{install_path}/ci_middlewares')

def copy_file(file_name):
    if (os.path.exists(f'{workspace}/{file_name}')):
            shutil.copyfile(f'{workspace}/{file_name}',
                            f'{install_path}/log_dir/{file_name}')

def log_dir(install_path, release=False):
    if (os.path.exists(f'{install_path}/log_dir') != True):
         os.makedirs(f'{install_path}/log_dir')
    if (release):
        copy_file('Makefile.am.diff')
        copy_file('configure.ac.diff')
        copy_file('release_num.txt')

if __name__ == "__main__":
#read Jenkins environment variables
    # In Jenkins,  JOB_NAME  = 'ofi_libfabric/master' vs BRANCH_NAME = 'master'
    # job name is better to use to distinguish between builds of different
    # jobs but with same branch name.
    jobname = os.environ['JOB_NAME']
    buildno = os.environ['BUILD_NUMBER']
    workspace = os.environ['WORKSPACE']

    parser = argparse.ArgumentParser()
    parser.add_argument('--build_item', help="build libfabric or fabtests",
                        choices=['libfabric', 'fabtests', 'builddir', 'logdir'])

    parser.add_argument('--ofi_build_mode', help="select buildmode debug or dl", \
                        choices=['dbg', 'dl'])

    parser.add_argument('--build_cluster', help="build libfabric on specified cluster", \
                        choices=['daos', 'dsa'], default='default')
    parser.add_argument('--release', help="This job is likely testing a "\
                        "release and will be checked into a git tree.",
                        action='store_true')

    args = parser.parse_args()
    build_item = args.build_item
    cluster = args.build_cluster
    release = args.release

    if (args.ofi_build_mode):
        ofi_build_mode = args.ofi_build_mode
    else:
        ofi_build_mode = 'reg'

    install_path = f'{ci_site_config.install_dir}/{jobname}/{buildno}'
    libfab_install_path = f'{ci_site_config.install_dir}/{jobname}/{buildno}/{ofi_build_mode}'

    p = re.compile('mpi*')

    if (build_item == 'libfabric'):
        build_libfabric(libfab_install_path, ofi_build_mode, cluster)

    elif (build_item == 'fabtests'):
        build_fabtests(libfab_install_path, ofi_build_mode)

    elif (build_item == 'builddir'):
        copy_build_dir(install_path)

    elif (build_item == 'logdir'):
        log_dir(install_path, release)
