#!/usr/bin/python2

from __future__ import print_function

import sys
import os
import shutil
import argparse
import subprocess
import glob
import json
from xmlreport import LungCadReport
from xml.etree import ElementTree

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_json_report(xmlfile):
    filename = os.path.basename(xmlfile).replace('.xml', '')
    xmlreport = LungCadReport.from_xml(ElementTree.parse(str(xmlfile)))

    report = {
        "entity": filename,
        "metrics": {
            "cancer_score": float(xmlreport.cancerinfo.casecancerprobability),
            "nodules": [dict(x=f.x, y=f.y, z=f.z,
                             diameter_mm=f.diameter_mm,
                             volume_mm3=f.volume_mm3,
                             probability=f.probability,
                             cancerprobability=f.cancerprobability)
                        for f in xmlreport.findings],
        },
        "error_messages": [],
    }
    return report


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
    parser.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        help="Disable cleanup of intermediate results "
             "(preparation files & bounding boxes).",
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

    new_env["INPUT_DIR"] = main_input_dir
    new_env["OUTPUT_DIR"] = main_output_dir
    subprocess.check_call(
        [sys.executable, "main.py"],
        env=new_env,
        cwd=os.path.join(THIS_DIR, "DSB2017"),
    )
    print("--- Parsing result reports and writing results.json")
    output_json = []
    for output_xml in glob.glob(os.path.join(main_output_dir, "*.xml")):
        print(output_xml)
        output_json.append(extract_json_report(xmlfile=output_xml))

    with open(os.path.join(args.output_dir, "results.json"), 'wb') as f:
        json.dump(output_json, f, encoding='utf-8', indent=4)

    if args.cleanup:
        print("--- Cleanup")
        bbox_dir = os.path.join(main_output_dir, "bbox")
        prep_dir = os.path.join(main_output_dir, "prep")
        for directory in [bbox_dir, prep_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)

    print("--- Processing completed")
