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

# Normalise source_path and build all sub-paths via os.path.join
src = os.path.abspath(args.source_path)
input_dir = os.path.join(src, "input")
distorted_dir = os.path.join(src, "distorted")
distorted_sparse_dir = os.path.join(distorted_dir, "sparse")
distorted_db = os.path.join(distorted_dir, "database.db")
sparse_dir = os.path.join(src, "sparse")
sparse0_dir = os.path.join(sparse_dir, "0")
images_dir = os.path.join(src, "images")

# Wrap a path in double quotes for shell safety
def q(p):
    return '"' + p + '"'

if not args.skip_matching:
    os.makedirs(distorted_sparse_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        logging.error("Input directory does not exist: %s", input_dir)
        exit(1)

    ## Feature extraction
    feat_extracton_cmd = (
        colmap_command + " feature_extractor"
        + " --database_path " + q(distorted_db)
        + " --image_path " + q(input_dir)
        + " --ImageReader.single_camera 1"
        + " --ImageReader.camera_model " + args.camera
        + " --SiftExtraction.use_gpu " + str(use_gpu)
    )
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = (
        colmap_command + " exhaustive_matcher"
        + " --database_path " + q(distorted_db)
        + " --SiftMatching.use_gpu " + str(use_gpu)
    )
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    mapper_cmd = (
        colmap_command + " mapper"
        + " --database_path " + q(distorted_db)
        + " --image_path " + q(input_dir)
        + " --output_path " + q(distorted_sparse_dir)
        + " --Mapper.ba_global_function_tolerance=0.000001"
    )
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
img_undist_cmd = (
    colmap_command + " image_undistorter"
    + " --image_path " + q(input_dir)
    + " --input_path " + q(os.path.join(distorted_sparse_dir, "0"))
    + " --output_path " + q(src)
    + " --output_type COLMAP"
)
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Image undistortion failed with code {exit_code}. Exiting.")
    exit(exit_code)

# Move sparse files into sparse/0/
os.makedirs(sparse0_dir, exist_ok=True)
for file in os.listdir(sparse_dir):
    if file == '0':
        continue
    shutil.move(os.path.join(sparse_dir, file), os.path.join(sparse0_dir, file))

if args.resize:
    print("Copying and resizing...")

    os.makedirs(os.path.join(src, "images_2"), exist_ok=True)
    os.makedirs(os.path.join(src, "images_4"), exist_ok=True)
    os.makedirs(os.path.join(src, "images_8"), exist_ok=True)
    files = os.listdir(images_dir)
    for file in files:
        source_file = os.path.join(images_dir, file)

        destination_file = os.path.join(src, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + q(destination_file))
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(src, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + q(destination_file))
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(src, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + q(destination_file))
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
