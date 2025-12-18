#!/usr/bin/env python3
"""
3D Gaussian Splatting with Mip-Filtering for Mip-NeRF 360 Dataset
Training Script for RTX 3060 GPU
Author: 3DGS Expert
Date: 2025-12-18
"""

import os
import torch
import numpy as np
import cv2
import math
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import subprocess

# 自定义模块
from dataloader import MipNeRF360Dataset
from gaussian_model import GaussianModel
from optimizer import Optimizer
from render import render
from loss import LossFunction
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting Training with Mip-Filtering")

    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset (e.g., ./archive/360_v2)")
    parser.add_argument("--scene", type=str, default="bicycle",
                        help="Scene name (bicycle, bonsai, counter, garden, etc.)")
    parser.add_argument("--output_path", type=str, default="./output",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--iterations", type=int, default=30000,
                        help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training (reduce if OOM)")
    parser.add_argument("--resolution", type=int, default=1,
                        help="Image scale factor: 1,2,4,8 (1 for full resolution)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--sh_degree", type=int, default=3,
                        help="Spherical harmonics degree")
    parser.add_argument("--densify_until", type=int, default=15000,
                        help="Iterations until densification stops")
    parser.add_argument("--opacity_reset_interval", type=int, default=3000,
                        help="Opacity reset interval")
    parser.add_argument("--densification_interval", type=int, default=100,
                        help="Densification interval")
    parser.add_argument("--position_lr_max_steps", type=int, default=30000,
                        help="Position learning rate max steps")
    parser.add_argument("--position_lr_init", type=float, default=0.00016,
                        help="Initial position learning rate")
    parser.add_argument("--position_lr_final", type=float, default=0.0000016,
                        help="Final position learning rate")
    parser.add_argument("--feature_lr", type=float, default=0.0025,
                        help="Feature learning rate")
    parser.add_argument("--opacity_lr", type=float, default=0.05,
                        help="Opacity learning rate")
    parser.add_argument("--scaling_lr", type=float, default=0.005,
                        help="Scaling learning rate")
    parser.add_argument("--rotation_lr", type=float, default=0.001,
                        help="Rotation learning rate")
    parser.add_argument("--lambda_dssim", type=float, default=0.2,
                        help="Lambda for DSSIM loss")
    parser.add_argument("--percent_dense", type=float, default=0.01,
                        help="Percent dense for densification")
    parser.add_argument("--mip_filter", action="store_true", default=True,
                        help="Enable Mip-Filtering for anti-aliasing")
    parser.add_argument("--mip_levels", type=int, default=3,
                        help="Number of Mip levels")

    return parser.parse_args()


def setup_experiment(args):
    """Setup experiment directory and logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.scene}_{timestamp}"
    exp_dir = os.path.join(args.output_path, exp_name)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "renders"), exist_ok=True)

    # Save configuration
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    return exp_dir


def initialize_gaussians(dataset, args):
    """Initialize Gaussian model from COLMAP point cloud"""
    print("Initializing Gaussians from COLMAP...")

    # Load COLMAP point cloud
    colmap_path = os.path.join(dataset.data_path, "sparse/0")
    if os.path.exists(colmap_path):
        print(f"Loading COLMAP points from {colmap_path}")
        # Load point cloud using pycolmap
        import pycolmap
        reconstruction = pycolmap.Reconstruction(colmap_path)

        # Extract 3D points
        points3D = []
        colors = []
        for point3D_id, point3D in reconstruction.points3D.items():
            points3D.append(point3D.xyz)
            colors.append(point3D.color / 255.0)

        if len(points3D) > 0:
            points3D = np.array(points3D)
            colors = np.array(colors)
            print(f"Loaded {len(points3D)} points from COLMAP")

            # Initialize Gaussian model
            gaussians = GaussianModel(args.sh_degree)
            gaussians.create_from_pcd(points3D, colors)
            return gaussians

    # Fallback: Initialize random Gaussians
    print("COLMAP not found, initializing random Gaussians...")
    gaussians = GaussianModel(args.sh_degree)

    # Create random points around origin
    num_points = 10000
    points = np.random.random((num_points, 3)) * 2 - 1  # [-1, 1]
    colors = np.random.random((num_points, 3)) * 0.8 + 0.1  # [0.1, 0.9]

    gaussians.create_from_pcd(points, colors)
    return gaussians


def main():
    args = parse_args()

    # Setup
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Setup experiment directory
    exp_dir = setup_experiment(args)
    print(f"Experiment directory: {exp_dir}")

    # Load dataset
    dataset = MipNeRF360Dataset(
        base_path=args.data_path,
        scene=args.scene,
        resolution=args.resolution,
        device=device
    )
    print(f"Dataset loaded: {len(dataset)} training images")

    # Initialize Gaussian model
    gaussians = initialize_gaussians(dataset, args)
    gaussians.training_setup(args)
    gaussians.to(device)

    # Initialize optimizer
    optimizer = Optimizer(gaussians, args)

    # Initialize loss function
    loss_fn = LossFunction(lambda_dssim=args.lambda_dssim)

    # Training loop
    print("\nStarting training...")
    progress_bar = tqdm(range(1, args.iterations + 1), desc="Training")

    for iteration in progress_bar:
        # Sample random camera
        camera = dataset.get_random_camera()

        # Render with Mip-Filtering if enabled
        render_pkg = render(
            camera,
            gaussians,
            args,
            mip_filter=args.mip_filter,
            mip_levels=args.mip_levels
        )

        # Compute loss
        gt_image = camera.original_image.to(device)
        loss = loss_fn(render_pkg["render"], gt_image)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step(iteration)
        optimizer.zero_grad(set_to_none=True)

        # Densification and pruning
        if iteration < args.densify_until and iteration % args.densification_interval == 0:
            gaussians.densify_and_prune(
                args.percent_dense,
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                iteration
            )

        # Opacity reset
        if iteration % args.opacity_reset_interval == 0:
            gaussians.reset_opacity()

        # Logging
        if iteration % 500 == 0:
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Gaussians": gaussians.get_xyz.shape[0]
            })

            # Save checkpoint
            if iteration % 5000 == 0:
                checkpoint_path = os.path.join(
                    exp_dir,
                    "checkpoints",
                    f"iteration_{iteration}.pth"
                )
                gaussians.save_ply(checkpoint_path.replace(".pth", ".ply"))
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': gaussians.capture_state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)

                # Render test view
                if iteration % 10000 == 0:
                    test_camera = dataset.get_test_camera(0)
                    with torch.no_grad():
                        test_render = render(
                            test_camera,
                            gaussians,
                            args,
                            mip_filter=args.mip_filter
                        )

                    render_path = os.path.join(
                        exp_dir,
                        "renders",
                        f"iter_{iteration}.png"
                    )
                    save_image(test_render["render"], render_path)

        # Clear cache periodically
        if iteration % 1000 == 0:
            torch.cuda.empty_cache()

    # Final save
    final_ply = os.path.join(exp_dir, f"{args.scene}_final.ply")
    final_checkpoint = os.path.join(exp_dir, f"{args.scene}_final.pth")

    gaussians.save_ply(final_ply)
    torch.save({
        'iteration': args.iterations,
        'model_state_dict': gaussians.capture_state_dict(),
        'args': vars(args)
    }, final_checkpoint)

    print(f"\nTraining completed!")
    print(f"Final model saved to: {final_ply}")
    print(f"Checkpoint saved to: {final_checkpoint}")

    # Generate reconstruction report
    generate_report(exp_dir, args, gaussians, dataset)


if __name__ == "__main__":
    main()