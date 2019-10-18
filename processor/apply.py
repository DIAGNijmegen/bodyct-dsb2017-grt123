#!/usr/bin/python2

from __future__ import print_function

import sys
import os
import argparse
import subprocess
import glob


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a set of images using the DSB2017 winner solution "
        "(see https://github.com/lfz/DSB2017)."
    )

    parser.add_argument(
        "--input-dir",
        default="/input",
        help="The default location from where to read input images (mhd or "
        "individual dicom-filled directories are allowed).",
    )
    parser.add_argument(
        "--output-dir",
        default="/output",
        help="The default location where to store the result preditions.csv file",
    )

    parser.add_argument(
        "--n-cpus",
        default=6,
        type=int,
        help="Number of CPUs to use for preprocessing the input data.",
    )
    parser.add_argument(
        "--n-gpus",
        default=1,
        type=int,
        help="Number of GPUs to use for processing data.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        parser.exit(1, "input-dir does not exist\n")
    if not os.path.isdir(args.output_dir):
        parser.exit(1, "output-dir does not exist\n")

    new_env = dict(os.environ)
    new_env["N_PREPROCESSING_TASKS"] = str(args.n_cpus)
    new_env["N_GPUS"] = str(args.n_gpus)
    main_input_dir = os.path.abspath(args.input_dir)
    main_output_dir = os.path.abspath(args.output_dir)
    pattern = os.path.join(main_input_dir, "*")
    output_dirs = []
    is_a_directory = True
    for candidate in glob.glob(pattern):
        if os.path.isdir(candidate):
            new_env["INPUT_DIR"] = os.path.abspath(candidate)
            output_dir = os.path.join(main_output_dir, os.path.basename(candidate))
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            new_env["OUTPUT_DIR"] = os.path.abspath(output_dir)
            output_dirs.append(new_env["OUTPUT_DIR"])
            print(
                "input dir = {}, output dir = {}".format(
                    new_env["INPUT_DIR"], new_env["OUTPUT_DIR"]
                )
            )
            subprocess.check_call(
                [sys.executable, "main.py"],
                env=new_env,
                cwd=os.path.join(THIS_DIR, "DSB2017"),
            )
        else:
            is_a_directory = False
    if not is_a_directory:
        new_env["INPUT_DIR"] = os.path.abspath(args.input_dir)
        new_env["OUTPUT_DIR"] = os.path.abspath(args.output_dir)
        subprocess.check_call(
            [sys.executable, "main.py"],
            env=new_env,
            cwd=os.path.join(THIS_DIR, "DSB2017"),
        )

    print("--- Processing completed")
