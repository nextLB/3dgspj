#!/usr/bin/env python3
"""
3D Gaussian Splatting Inference with Novel View Synthesis
Author: 3DGS Expert
Date: 2025-12-18
"""

import os
import torch
import numpy as np
import cv2
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import imageio
from PIL import Image
import matplotlib.pyplot as plt

# 自定义模块
from dataloader import MipNeRF360Dataset, Camera
from gaussian_model import GaussianModel
from render import render
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting Inference")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file (.pth)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to dataset (optional, for camera paths)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene name (optional)")
    parser.add_argument("--output_path", type=str, default="./inference_output",
                        help="Output directory")
    parser.add_argument("--num_views", type=int, default=120,
                        help="Number of novel views to generate")
    parser.add_argument("--trajectory", type=str, default="circle",
                        choices=["circle", "spiral", "original", "custom"],
                        help="Camera trajectory type")
    parser.add_argument("--resolution", type=tuple, default=(800, 800),
                        help="Output resolution (width, height)")
    parser.add_argument("--fov", type=float, default=60.0,
                        help="Field of view in degrees")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS for output video")
    parser.add_argument("--mip_filter", action="store_true", default=True,
                        help="Enable Mip-Filtering for anti-aliasing")
    parser.add_argument("--save_ply", action="store_true", default=True,
                        help="Save point cloud as PLY")
    parser.add_argument("--save_video", action="store_true", default=True,
                        help="Save video of novel views")
    parser.add_argument("--save_images", action="store_true", default=True,
                        help="Save individual images")
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="Launch interactive viewer")

    return parser.parse_args()


def load_checkpoint(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model arguments
    if 'args' in checkpoint:
        args = argparse.Namespace(**checkpoint['args'])
    else:
        # Default args if not saved
        args = argparse.Namespace()
        args.sh_degree = 3
        args.mip_filter = True
        args.mip_levels = 3

    # Initialize Gaussian model
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_state_dict(checkpoint['model_state_dict'])
    gaussians.to(device)
    gaussians.eval()

    print(f"Model loaded: {gaussians.get_xyz.shape[0]} Gaussians")
    print(f"Checkpoint iteration: {checkpoint.get('iteration', 'unknown')}")

    return gaussians, args


def generate_camera_path(dataset, args):
    """Generate camera trajectory for novel view synthesis"""
    print(f"Generating {args.trajectory} camera trajectory...")

    cameras = []

    if args.trajectory == "circle":
        # Circular path around scene
        radius = 2.0
        center = torch.tensor([0.0, 0.0, 0.0])

        for i in range(args.num_views):
            angle = 2 * np.pi * i / args.num_views

            # Camera position
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 0.5  # Slight elevation

            position = torch.tensor([x, y, z]) + center

            # Look at center
            forward = center - position
            forward = forward / torch.norm(forward)

            # Create camera
            camera = Camera(
                R=look_at_matrix(position, center),
                T=position.unsqueeze(1),
                fx=args.fov,
                fy=args.fov,
                cx=args.resolution[0] // 2,
                cy=args.resolution[1] // 2,
                width=args.resolution[0],
                height=args.resolution[1],
                image_name=f"novel_{i:04d}"
            )
            cameras.append(camera)

    elif args.trajectory == "spiral":
        # Spiral path
        for i in range(args.num_views):
            t = i / args.num_views
            radius = 2.0 * (1 - t * 0.5)  # Slowly descend

            angle = 2 * np.pi * 4 * t  # 4 full rotations

            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 1.5 - t  # Descend from 1.5 to 0.5

            position = torch.tensor([x, y, z])
            center = torch.tensor([0.0, 0.0, 0.0])

            camera = Camera(
                R=look_at_matrix(position, center),
                T=position.unsqueeze(1),
                fx=args.fov,
                fy=args.fov,
                cx=args.resolution[0] // 2,
                cy=args.resolution[1] // 2,
                width=args.resolution[0],
                height=args.resolution[1],
                image_name=f"spiral_{i:04d}"
            )
            cameras.append(camera)

    elif args.trajectory == "original" and dataset is not None:
        # Use original camera poses
        cameras = [dataset.get_camera(i) for i in range(min(args.num_views, len(dataset)))]
        # Adjust resolution if needed
        for cam in cameras:
            cam.width, cam.height = args.resolution

    return cameras


def render_novel_views(gaussians, cameras, args, device):
    """Render novel views from camera trajectory"""
    print(f"Rendering {len(cameras)} novel views...")

    renders = []
    depths = []
    normals = []

    with torch.no_grad():
        for i, camera in enumerate(tqdm(cameras, desc="Rendering")):
            # Move camera to device
            camera.to(device)

            # Render
            render_pkg = render(
                camera,
                gaussians,
                args,
                mip_filter=args.mip_filter,
                bg_color=torch.tensor([0.0, 0.0, 0.0], device=device)
            )

            # Get rendered image
            rendered_image = render_pkg["render"]
            rendered_image = torch.clamp(rendered_image, 0, 1)

            renders.append(rendered_image.cpu())

            # Optional: extract depth and normals
            if "depth" in render_pkg:
                depth = render_pkg["depth"]
                depths.append(depth.cpu())

            if "normal" in render_pkg:
                normal = render_pkg["normal"]
                normals.append(normal.cpu())

    return renders, depths, normals


def save_results(renders, cameras, args, output_dir):
    """Save rendered results to disk"""
    os.makedirs(output_dir, exist_ok=True)

    # Save individual images
    if args.save_images:
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        print(f"Saving images to {images_dir}")
        for i, render_img in enumerate(tqdm(renders, desc="Saving images")):
            # Convert to numpy and save
            img_np = (render_img.numpy() * 255).astype(np.uint8)

            # Handle different tensor shapes
            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)

            img_path = os.path.join(images_dir, f"render_{i:04d}.png")
            cv2.imwrite(img_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # Save video
    if args.save_video and len(renders) > 1:
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        print(f"Saving video to {video_dir}")

        # Prepare frames
        frames = []
        for render_img in renders:
            img_np = (render_img.numpy() * 255).astype(np.uint8)
            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)
            frames.append(img_np)

        # Save as video
        video_path = os.path.join(video_dir, f"{args.trajectory}_trajectory.mp4")

        # Use imageio or OpenCV
        try:
            import imageio
            imageio.mimsave(video_path, frames, fps=args.fps)
        except:
            # Fallback to OpenCV
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))

            for frame in frames:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            video_writer.release()

        print(f"Video saved: {video_path}")

    # Save camera trajectory
    trajectory_path = os.path.join(output_dir, "camera_trajectory.json")
    trajectory_data = []

    for i, cam in enumerate(cameras):
        trajectory_data.append({
            "index": i,
            "position": cam.T.squeeze().tolist() if hasattr(cam, 'T') else [0, 0, 0],
            "rotation": cam.R.tolist() if hasattr(cam, 'R') else np.eye(3).tolist(),
            "fov": args.fov,
            "resolution": args.resolution
        })

    with open(trajectory_path, 'w') as f:
        json.dump(trajectory_data, f, indent=2)


def create_interactive_viewer(gaussians, output_dir):
    """Create interactive HTML viewer for 3D Gaussians"""
    print("Creating interactive viewer...")

    viewer_dir = os.path.join(output_dir, "viewer")
    os.makedirs(viewer_dir, exist_ok=True)

    # Export Gaussians for viewer
    ply_path = os.path.join(viewer_dir, "gaussians.ply")
    gaussians.save_ply(ply_path)

    # Create simple HTML viewer
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>3D Gaussian Splatting Viewer</title>
    <style>
        body { margin: 0; overflow: hidden; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            z-index: 100;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/@pmndrs/gsplat@latest"></script>
</head>
<body>
    <div id="info">
        <div>3D Gaussian Splatting Viewer</div>
        <div>Number of Gaussians: """ + str(gaussians.get_xyz.shape[0]) + """</div>
        <div>Use mouse to rotate, scroll to zoom</div>
    </div>
    <script type="importmap">
    {
        "imports": {
            "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
            "@react-three/fiber": "https://unpkg.com/@react-three/fiber@8.16.5/dist/fiber.three.legacy.js",
            "@react-three/drei": "https://unpkg.com/@react-three/drei@9.112.30/core/drei.three.legacy.js"
        }
    }
    </script>
    <script type="module">
        import * as THREE from 'three';
        import { WebGLRenderer, PerspectiveCamera, Scene, AmbientLight } from 'three';
        import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
        import { SplatBuffer } from 'https://cdn.jsdelivr.net/npm/@pmndrs/gsplat@latest';

        // Setup renderer
        const renderer = new WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // Setup scene
        const scene = new THREE.Scene();

        // Setup camera
        const camera = new PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 0, 5);

        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Lighting
        const ambientLight = new AmbientLight(0xffffff, 1.0);
        scene.add(ambientLight);

        // Load Gaussian splats
        const loader = new SplatBuffer();
        loader.load('gaussians.ply').then((splatBuffer) => {
            scene.add(splatBuffer);

            // Auto-rotate for visualization
            controls.autoRotate = true;
            controls.autoRotateSpeed = 0.5;
        });

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        // Handle resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>
"""

    html_path = os.path.join(viewer_dir, "index.html")
    with open(html_path, 'w') as f:
        f.write(html_content)

    print(f"Interactive viewer created at: {html_path}")
    print("Open this file in a modern web browser to view the 3D reconstruction")


def main():
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_path, f"inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save inference arguments
    with open(os.path.join(output_dir, "inference_args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Load trained model
    gaussians, model_args = load_checkpoint(args.checkpoint, device)

    # Merge args
    for key, value in vars(model_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Load dataset for camera poses if provided
    dataset = None
    if args.data_path and args.scene:
        try:
            dataset = MipNeRF360Dataset(
                base_path=args.data_path,
                scene=args.scene,
                resolution=1,  # Full resolution for inference
                device=device
            )
            print(f"Dataset loaded for camera reference: {len(dataset)} images")
        except:
            print("Warning: Could not load dataset, using synthetic camera path")

    # Generate camera path
    cameras = generate_camera_path(dataset, args)

    # Render novel views
    renders, depths, normals = render_novel_views(gaussians, cameras, args, device)

    # Save results
    save_results(renders, cameras, args, output_dir)

    # Save point cloud
    if args.save_ply:
        ply_path = os.path.join(output_dir, "reconstruction.ply")
        gaussians.save_ply(ply_path)
        print(f"Point cloud saved: {ply_path}")

    # Create interactive viewer
    if args.interactive:
        create_interactive_viewer(gaussians, output_dir)

    # Generate reconstruction report
    report_path = os.path.join(output_dir, "reconstruction_report.txt")
    with open(report_path, 'w') as f:
        f.write("3D Gaussian Splatting Reconstruction Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Number of Gaussians: {gaussians.get_xyz.shape[0]}\n")
        f.write(f"Number of Novel Views: {len(renders)}\n")
        f.write(f"Resolution: {args.resolution}\n")
        f.write(f"Trajectory: {args.trajectory}\n")
        f.write(f"Output Directory: {output_dir}\n\n")

        f.write("Files Generated:\n")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(('.png', '.mp4', '.ply', '.html', '.json', '.txt')):
                    rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                    f.write(f"  - {rel_path}\n")

    print(f"\nInference completed!")
    print(f"All outputs saved to: {output_dir}")
    print(f"Report saved to: {report_path}")

    if args.interactive:
        print("\nTo view the interactive reconstruction:")
        print(f"Open file://{os.path.join(output_dir, 'viewer', 'index.html')} in your web browser")


if __name__ == "__main__":
    main()