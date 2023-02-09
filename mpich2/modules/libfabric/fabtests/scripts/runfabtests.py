#!/usr/bin/env python3
#
# Copyright (c) 2021-2022 Amazon.com, Inc. or its affiliates. All rights reserved.
#
# This software is available to you under a choice of one of two
# licenses.  You may choose to be licensed under the terms of the GNU
# General Public License (GPL) Version 2, available from the file
# COPYING in the main directory of this source tree, or the
# BSD license below:
#
#     Redistribution and use in source and binary forms, with or
#     without modification, are permitted provided that the following
#     conditions are met:
#
#      - Redistributions of source code must retain the above
#        copyright notice, this list of conditions and the following
#        disclaimer.
#
#      - Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials
#        provided with the distribution.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

def get_option_longform(option_name, option_params):
    '''
        get the long form command line option name of an option
    '''
    return option_params.get("longform", "--" + option_name.replace("_", "-"))

def get_ubertest_test_type(fabtests_testsets):
    test_list = fabtests_testsets.split(",")

    for test in test_list:
        if test == "quick" or test == "ubertest_quick" or test == "ubertest":
            return "quick"

        if test == "all" or test == "ubertest_all":
            return "all"

        if test == "verify" or test == "ubertest_verify":
            return "verify"

    return None

def fabtests_testsets_to_pytest_markers(fabtests_testsets):
    test_set = set()
    test_list = fabtests_testsets.split(",")

    # use set() to remove duplicate test set
    for test in test_list:
        if test == "quick":
            test_set.add("unit")
            test_set.add("functional")
            test_set.add("short")
            test_set.add("ubertest_quick")
        elif test =="ubertest":
            test_set.add("ubertest_quick")
        elif test == "all":
            test_set.add("unit")
            test_set.add("functional")
            test_set.add("standard")
            test_set.add("multinode")
            test_set.add("ubertest_all")
        elif test == "verify":
            test_set.add("ubertest_verify")
        else:
            test_set.add(test)

    markers = None
    for test in test_set:
        if markers is None:
            markers = test[:]
        else:
            markers += " or " + test

    return markers

def get_default_exclusion_file(fabtests_args):
    import os
    test_configs_dir = os.path.abspath(os.path.join(get_pytest_root_dir(), "..", "test_configs"))
    exclusion_file = os.path.join(test_configs_dir, fabtests_args.provider,
                                  fabtests_args.provider + ".exclude")
    if not os.path.exists(exclusion_file):
        return None

    return exclusion_file

def get_default_ubertest_config_file(fabtests_args):
    import os
 
    test_configs_dir = os.path.abspath(os.path.join(get_pytest_root_dir(), "..", "test_configs"))
    provider = fabtests_args.provider
    if provider.find(";") != -1:
        core,util = fabtests_args.provider.split(";")
        cfg_file = os.path.join(test_configs_dir, util, core + ".test")
    else:
        core = fabtests_args.provider
        ubertest_test_type = get_ubertest_test_type(fabtests_args.testsets)
        if not ubertest_test_type:
            return None

        cfg_file = os.path.join(test_configs_dir, core, ubertest_test_type + ".test")

    if not os.path.exists(cfg_file):
        return None

    return cfg_file

def add_common_arguments(parser, shared_options):
    import builtins

    for option_name in shared_options.keys():
        option_params = shared_options[option_name]
        option_longform = get_option_longform(option_name, option_params)
        option_shortform = option_params.get("shortform")
        option_type = option_params["type"]
        option_helpmsg = option_params["help"]
        option_default = option_params.get("default")
        if option_type == "int" and not (option_default is None):
            option_default = int(option_default)

        if option_shortform:
            forms = [option_shortform, option_longform]
        else:
            forms = [option_longform]

        if option_type == "bool" or option_type == "boolean":
            parser.add_argument(*forms,
                                dest=option_name, action="store_true",
                                help=option_helpmsg, default=option_default)
        else:
            assert option_type == "str" or option_type == "int"
            parser.add_argument(*forms,
                                dest=option_name, type=getattr(builtins, option_type),
                                help=option_helpmsg, default=option_default)

def fabtests_args_to_pytest_args(fabtests_args, shared_options):
    import os

    pytest_args = []

    pytest_args.append("--provider=" + fabtests_args.provider)
    pytest_args.append("--server-id=" + fabtests_args.server_id)
    pytest_args.append("--client-id=" + fabtests_args.client_id)

    # -v make pytest to print 1 line for each test
    pytest_args.append("-v")

    pytest_verbose_options = {
            0 : "-rN",      # print no extra information
            1 : "-rfE",     # print extra information for failed test(s)
            2 : "-rfEsx",   # print extra information for failed/skipped test(s)
            3 : "-rA",      # print extra information for all test(s) (failed/skipped/passed)
        }

    pytest_args.append(pytest_verbose_options[fabtests_args.verbose])

    verbose_fail = fabtests_args.verbose > 0
    if verbose_fail:
        # Use short python trace back because it show captured stdout of failed tests
        pytest_args.append("--tb=short")
    else:
        pytest_args.append("--tb=no")

    markers = fabtests_testsets_to_pytest_markers(fabtests_args.testsets)
    pytest_args.append("-m")
    pytest_args.append(markers)

    if fabtests_args.expression:
        pytest_args.append("-k")
        pytest_args.append(fabtests_args.expression)

    if fabtests_args.html:
        pytest_args.append("--html")
        pytest_args.append(os.path.abspath(fabtests_args.html))
        pytest_args.append("--self-contained-html")

    if fabtests_args.junit_xml:
        pytest_args.append("--junit-xml")
        pytest_args.append(os.path.abspath(fabtests_args.junit_xml))
        pytest_args.append("--self-contained-html")

    # add options shared between runfabtests.py and libfabric pytest
    for option_name in shared_options.keys():
        option_params = shared_options[option_name]
        option_longform = get_option_longform(option_name, option_params)
        option_type = option_params["type"]
 
        if not hasattr(fabtests_args, option_name):
            continue

        option_value = getattr(fabtests_args, option_name)
        if (option_value is None):
            continue

        if option_type == "bool" or option_type == "boolean":
            assert option_value
            pytest_args.append(get_option_longform(option_name, option_params))
        else:
            assert option_type == "str" or option_type == "int"
            pytest_args.append(get_option_longform(option_name, option_params) + "=" + str(option_value))

    if not hasattr(fabtests_args, "exclusion_file") or not fabtests_args.exclusion_file:
        default_exclusion_file = get_default_exclusion_file(fabtests_args)
        if default_exclusion_file:
            pytest_args.append("--exclusion-file=" + default_exclusion_file)

    if not hasattr(fabtests_args, "ubertest_config_file") or not fabtests_args.ubertest_config_file:
        default_ubertest_config_file = get_default_ubertest_config_file(fabtests_args)
        if default_ubertest_config_file:
            pytest_args.append("--ubertest-config-file=" + default_ubertest_config_file)

    return pytest_args

def get_pytest_root_dir():
    '''
        find the pytest root directory according the location of runfabtests.py
    '''
    import os
    import sys
    script_path = os.path.abspath(sys.argv[0])
    script_dir = os.path.dirname(script_path)
    if os.path.basename(script_dir) == "bin":
        # runfabtests.py is part of a fabtests installation
        pytest_root_dir = os.path.abspath(os.path.join(script_dir, "..", "share", "fabtests", "pytest"))
    elif os.path.basename(script_dir) == "scripts":
        # runfabtests.py is part of a fabtests source code
        pytest_root_dir = os.path.abspath(os.path.join(script_dir, "..", "pytest"))
    else:
        raise RuntimeError("Error: runfabtests.py is under directory {}, "
                "which is neither part of fabtests installation "
                "nor part of fabetsts source code".format(script_dir))

    if not os.path.exists(pytest_root_dir):
        raise RuntimeError("Deduced pytest root directory {} does not exist!".format(pytest_root_dir))

    return pytest_root_dir

def get_pytest_relative_case_dir(fabtests_args, pytest_root_dir):
    '''
        the directory that contains test cases, relative to pytest_root_dir
    '''
    import os

    # provider's own test directory (if exists) overrides default
    pytest_case_dir = os.path.join(pytest_root_dir, fabtests_args.provider)
    if os.path.exists(pytest_case_dir):
        return fabtests_args.provider

    assert os.path.exists(os.path.join(pytest_root_dir, "default"))
    return "default"

def main():
    import os
    import sys
    import yaml
    import pytest
    import argparse

    pytest_root_dir = get_pytest_root_dir()

    # pytest/options.yaml contains the definition of a list of options that are
    # shared between runfabtests.py and pytest
    option_yaml = os.path.join(pytest_root_dir, "options.yaml")
    if not os.path.exists(option_yaml):
        print("Error: option definition yaml file {} not found!".format(option_yaml))
        exit(1)

    shared_options = yaml.safe_load(open(option_yaml))

    parser = argparse.ArgumentParser(description="libfabric integration test runner")

    parser.add_argument("provider", type=str, help="libfabric provider")
    parser.add_argument("server_id", type=str, help="server ip or hostname")
    parser.add_argument("client_id", type=str, help="client ip or hostname")
    parser.add_argument("-t", dest="testsets", type=str, default="quick",
                        help="test set(s): all,quick,unit,functional,standard,short,ubertest (default quick)")
    parser.add_argument("-v", dest="verbose", action="count", default=0,
                        help="verbosity level"
                             "-v: print extra info for failed test(s)"
                             "-vv: print extra info of failed/skipped test(s)"
                             "-vvv: print extra info of failed/skipped/passed test(s)")
    parser.add_argument("--expression", type=str,
                        help="only run tests which match the given substring expression.")
    parser.add_argument("--html", type=str, help="path to generated html report")
    parser.add_argument("--junit-xml", type=str, help="path to generated junit xml report")

    add_common_arguments(parser, shared_options)

    fabtests_args = parser.parse_args()
    pytest_args = fabtests_args_to_pytest_args(fabtests_args, shared_options)

    os.chdir(pytest_root_dir)

    pytest_args.append(get_pytest_relative_case_dir(fabtests_args, pytest_root_dir))

    pytest_command = "cd " + pytest_root_dir + "; pytest"
    for arg in pytest_args:
        if arg.find(' ') != -1:
            arg = "'" + arg + "'"
        pytest_command += " " + arg
    print(pytest_command)

    # actually running tests
    exit(pytest.main(pytest_args))

main()
