#!/usr/bin/env python3
#
# ParaStation
#
# Copyright (C) 2010-2021 ParTec Cluster Competence Center GmbH, Munich
# Copyright (C) 2021-2025 ParTec AG, Munich
#
# This file may be distributed under the terms of the Q Public License
# as defined in the file LICENSE.QPL included in the packaging of this
# file.

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(
        description="Create a ParaStation MPI release tarball.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Show the output of the individual steps.",
    )
    parser.add_argument(
        "-r",
        "--ref",
        help="Checkout this ref before packaging.",
    )
    parser.add_argument("-p", "--path", default="./", help="Path to the psmpi repo.")
    parser.add_argument(
        "-s",
        "--suffix",
        help="A custom suffix appended to the version string.",
    )

    args = parser.parse_args()

    # Redirect output to /dev/null unless '--verbose' is given
    end_marker = "\n" if args.verbose else ""

    print("===> Preparing temporary working directory... ", end="")
    release_dir = tempfile.TemporaryDirectory()
    cloned_repo_path = os.path.join(release_dir.name, "psmpi-clone.git")
    print("done")

    # Clone the repository to run `vcversion` in a clean checkout
    print(
        f"===> Cloning git repo from '{args.path}' to '{cloned_repo_path}'... ",
        end=end_marker,
    )
    subprocess.run(
        ["git", "clone", args.path, cloned_repo_path],
        stdout=sys.stdout if args.verbose else subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    print("done")

    if args.ref:
        print(f"===> Checking out '{args.ref}'... ", end=end_marker)
        subprocess.run(
            ["git", "checkout", args.ref],
            stdout=sys.stdout if args.verbose else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            cwd=cloned_repo_path,
        )
        print("done")
    else:
        args.ref = "HEAD"

    print("===> Determine version... ", end="")
    vcversion_cmd = ["./scripts/vcversion", "--git"]
    if args.suffix is not None:
        vcversion_cmd.extend(["--suffix", args.suffix])

    version = (
        subprocess.run(
            vcversion_cmd,
            stdout=subprocess.PIPE,
            cwd=cloned_repo_path,
        )
        .stdout.decode("utf-8")
        .rstrip()
    )
    packaged_repo_path = os.path.join(release_dir.name, f"psmpi-{version}")
    print("done")

    print("===> Exporting code from git... ", end="")

    # Create a git archive as basis fo the tarball
    ps_archive = subprocess.Popen(
        ["git", "archive", args.ref, f"--prefix=psmpi-{version}/"],
        stdout=subprocess.PIPE,
        cwd=cloned_repo_path,
    )

    # Directly extract the git archive created above
    ps_extract = subprocess.Popen(
        ["tar", "-xav", f"--directory={release_dir.name}"],
        stdin=ps_archive.stdout,
        stdout=sys.stdout if args.verbose else subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        cwd=cloned_repo_path,
    )

    ps_archive.stdout.close()
    ps_extract.communicate()
    print("done")

    print("===> Running autotools... ", end=end_marker, flush=True)
    subprocess.run(
        ["./autogen.sh"],
        stdout=sys.stdout if args.verbose else subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        cwd=packaged_repo_path,
    )
    print("done")

    print("===> Removing unnecessary files... ", end="")
    for directory in glob.glob(
        f"{packaged_repo_path}/mpich2/**/autom4te.cache", recursive=True
    ):
        shutil.rmtree(directory)

    for e in [
        "README.vin",
        "maint/config.log",
        "maint/config.status",
        "unusederr.txt",
    ]:
        os.remove(f"{packaged_repo_path}/mpich2/{e}")
    print("done")

    print("===> Creating the tarball... ", end="")
    with tarfile.open(f"psmpi-{version}.tar.gz", "w:gz") as tar:
        tar.add(packaged_repo_path, arcname=os.path.basename(packaged_repo_path))
    print("done")


if __name__ == "__main__":
    main()
