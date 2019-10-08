#!/usr/bin/python2

from __future__ import print_function

import sys
import os
import argparse
import subprocess
import json
import csv
import glob


def process_crop_rects_and_prediction(output_dir):
    crop_rects_path = os.path.join(output_dir, "crop_rects.json")
    predictions_csv_path = os.path.join(output_dir, "prediction.csv")
    cancer_score = -1.0
    with open(predictions_csv_path, 'rb') as f:
        class PandasDialect(csv.Dialect):
            delimiter = ','
            quotechar = '"'
            doublequote = True
            escapechar = None
            skipinitialspace = True
            lineterminator = '\a'
            quoting = csv.QUOTE_NONE
            strict = True

        csvr = csv.reader(f, PandasDialect())
        csvr.next()  # skip header
        cancer_score = csvr.next()[1]
    folder_name = os.path.basename(output_dir)
    output_file_path = os.path.join(output_dir,
                                    'detected_nodules_in_world_and_voxel_coordinates.json')
    with open(output_file_path, 'rb') as f:
        modified_json_output = json.load(f)
    print("Combining {}".format(folder_name))
    os.remove(crop_rects_path)
    os.remove(output_file_path)
    return ({
        "entity": folder_name,
        "metrics": {
            "cancer_score": float(cancer_score),
            "nodules": modified_json_output.get(folder_name, []),
        },
        "error_messages": [],
    })


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Process a set of images using the DSB2017 winner solution '
        '(see https://github.com/lfz/DSB2017).')

    parser.add_argument(
        '--input-dir',
        default='/input',
        help=
        'The default location from where to read input images (mhd or '
        'individual dicom-filled directories are allowed).')
    parser.add_argument(
        '--output-dir',
        default='/output',
        help='The default location where to store the result preditions.csv file')

    parser.add_argument(
        '--n-cpus',
        default=6,
        type=int,
        help="Number of CPUs to use for preprocessing the input data.")
    parser.add_argument(
        '--n-gpus',
        default=1,
        type=int,
        help="Number of GPUs to use for processing data.")

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
    pattern = os.path.join(main_input_dir, '*')
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
            print("input dir = {}, output dir = {}".format(new_env["INPUT_DIR"], new_env["OUTPUT_DIR"]))
            subprocess.check_call(
                [sys.executable, "main.py"],
                env=new_env,
                cwd=os.path.join(THIS_DIR, "DSB2017"))
        else:
            is_a_directory = False
    if not is_a_directory:
        new_env["INPUT_DIR"] = os.path.abspath(args.input_dir)
        new_env["OUTPUT_DIR"] = os.path.abspath(args.output_dir)
        subprocess.check_call(
            [sys.executable, "main.py"],
            env=new_env,
            cwd=os.path.join(THIS_DIR, "DSB2017"))

    print("--- Processing completed, merging data files")
    if not is_a_directory:
        crop_rects_path = os.path.join(args.output_dir, "crop_rects.json")
        with open(crop_rects_path, 'rb') as f:
            crop_rects = json.load(f)

        output_json = []

        predictions_csv_path = os.path.join(args.output_dir, "prediction.csv")
        with open(predictions_csv_path, 'rb') as f:
            class PandasDialect(csv.Dialect):
                delimiter = ','
                quotechar = '"'
                doublequote = True
                escapechar = None
                skipinitialspace = True
                lineterminator = '\a'
                quoting = csv.QUOTE_NONE
                strict = True


            csvr = csv.reader(f, PandasDialect())
            csvr.next()  # skip header
            for filename, cancer_score in (x for x in csvr if len(x) == 2):
                output_file_path = os.path.join(args.output_dir,
                                                'detected_nodules_in_world_and_voxel_coordinates.json')
                with open(output_file_path, 'rb') as f:
                    modified_json_output = json.load(f)
                print("Combining", filename)
                output_json.append({
                    "entity": filename,
                    "metrics": {
                        "cancer_score": float(cancer_score),
                        "nodules": modified_json_output.get(filename, []),
                    },
                    "error_messages": [],
                })

        with open(os.path.join(args.output_dir, "results.json"), 'wb') as f:
            json.dump(output_json, f, encoding='utf-8', indent=4)
        os.remove(crop_rects_path)
        os.remove(output_file_path)
    else:
        output_json = []
        for output_dir in output_dirs:
            output_json.append(process_crop_rects_and_prediction(output_dir))
        with open(os.path.join(args.output_dir, "results.json"), 'wb') as f:
            json.dump(output_json, f, encoding='utf-8', indent=4)
