#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

# Resolve and change to source directory so all COLMAP paths can be relative.
# This avoids Unicode-path issues with COLMAP's C++/SQLite on Windows.
src = os.path.abspath(args.source_path)
os.makedirs(src, exist_ok=True)
os.chdir(src)

if not os.path.isdir("input"):
    logging.error("Input directory does not exist: %s", os.path.join(src, "input"))
    exit(1)

if not args.skip_matching:
    os.makedirs("distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = (
        colmap_command + " feature_extractor"
        + " --database_path distorted/database.db"
        + " --image_path input"
        + " --ImageReader.single_camera 1"
        + " --ImageReader.camera_model " + args.camera
        + " --SiftExtraction.use_gpu " + str(use_gpu)
    )
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error("Feature extraction failed with code %d. Exiting.", exit_code)
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = (
        colmap_command + " exhaustive_matcher"
        + " --database_path distorted/database.db"
        + " --SiftMatching.use_gpu " + str(use_gpu)
    )
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error("Feature matching failed with code %d. Exiting.", exit_code)
        exit(exit_code)

    ### Bundle adjustment
    mapper_cmd = (
        colmap_command + " mapper"
        + " --database_path distorted/database.db"
        + " --image_path input"
        + " --output_path distorted/sparse"
        + " --Mapper.ba_global_function_tolerance=0.000001"
    )
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error("Mapper failed with code %d. Exiting.", exit_code)
        exit(exit_code)

### Image undistortion
img_undist_cmd = (
    colmap_command + " image_undistorter"
    + " --image_path input"
    + " --input_path distorted/sparse/0"
    + " --output_path ."
    + " --output_type COLMAP"
)
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error("Image undistortion failed with code %d. Exiting.", exit_code)
    exit(exit_code)

# Move sparse files into sparse/0/
os.makedirs("sparse/0", exist_ok=True)
for file in os.listdir("sparse"):
    if file == '0':
        continue
    shutil.move(os.path.join("sparse", file), os.path.join("sparse", "0", file))

if args.resize:
    print("Copying and resizing...")

    os.makedirs("images_2", exist_ok=True)
    os.makedirs("images_4", exist_ok=True)
    os.makedirs("images_8", exist_ok=True)
    files = os.listdir("images")
    for file in files:
        source_file = os.path.join("images", file)

        destination_file = os.path.join("images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error("50% resize failed with code %d. Exiting.", exit_code)
            exit(exit_code)

        destination_file = os.path.join("images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error("25% resize failed with code %d. Exiting.", exit_code)
            exit(exit_code)

        destination_file = os.path.join("images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error("12.5% resize failed with code %d. Exiting.", exit_code)
            exit(exit_code)

print("Done.")
