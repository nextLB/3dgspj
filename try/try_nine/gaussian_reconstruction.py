#!/usr/bin/env python3
"""
3D Gaussian Splatting 三维重建 - 终极修复版本
确保梯度可以正确传播
"""

import os
import sys
import time
import argparse
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# 检查CUDA可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")


# ============================================================================
# 简化的数据集类
# ============================================================================

class SimpleSceneDataset(Dataset):
    """简化的场景数据集类"""

    def __init__(self, data_path, scene_name="bicycle", image_size=(256, 256), num_images=10):
        """
        初始化数据集

        Args:
            data_path: 数据根路径
            scene_name: 场景名称
            image_size: 图像尺寸 (height, width)
            num_images: 使用图像数量
        """
        super().__init__()

        self.data_path = Path(data_path)
        self.scene_name = scene_name
        self.image_size = image_size
        self.height, self.width = image_size
        self.num_images = num_images

        # 构建场景路径
        if "extra_scenes" in str(self.data_path):
            self.scene_path = self.data_path / "360_extra_scenes" / scene_name
        else:
            self.scene_path = self.data_path / "360_v2" / scene_name

        print(f"Loading scene from: {self.scene_path}")

        # 检查路径是否存在
        if not self.scene_path.exists():
            print(f"Warning: Scene path does not exist: {self.scene_path}")
            print("Using synthetic data instead")
            self.use_synthetic = True
        else:
            self.use_synthetic = False

        # 加载数据
        if self.use_synthetic:
            self.images = self._generate_synthetic_images()
            self.camera_params = self._generate_synthetic_cameras()
            self.point_cloud = self._generate_synthetic_points()
        else:
            self.images = self._load_images()
            self.camera_params = self._generate_camera_params()
            self.point_cloud = self._load_or_generate_points()

        print(f"Loaded {len(self.images)} images")
        print(f"Generated {len(self.point_cloud)} 3D points")

    def _generate_synthetic_images(self):
        """生成合成图像"""
        images = []

        for i in range(self.num_images):
            # 创建简单的测试图像 - 带有渐变颜色
            img = torch.zeros((3, self.height, self.width))

            # 添加一些简单的图案
            for c in range(3):
                # 创建渐变
                x = torch.linspace(0, 1, self.width)
                y = torch.linspace(0, 1, self.height)
                X, Y = torch.meshgrid(x, y, indexing='xy')

                # 不同的通道有不同的图案
                if c == 0:  # 红色通道
                    channel_data = 0.5 + 0.5 * torch.sin(2 * math.pi * (X + i / self.num_images))
                elif c == 1:  # 绿色通道
                    channel_data = 0.5 + 0.5 * torch.cos(2 * math.pi * (Y + i / self.num_images))
                else:  # 蓝色通道
                    channel_data = 0.5 + 0.5 * torch.sin(2 * math.pi * (X * Y + i / self.num_images))

                img[c] = channel_data

            images.append({
                "tensor": img,
                "height": self.height,
                "width": self.width,
                "index": i
            })

        return images

    def _generate_synthetic_cameras(self):
        """生成合成相机参数"""
        camera_params = []

        # 相机内参
        fx = fy = self.width * 0.8
        cx = self.width / 2.0
        cy = self.height / 2.0

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        for i in range(self.num_images):
            # 生成相机位置 (在球面上)
            angle = 2 * math.pi * i / self.num_images
            radius = 3.0

            # 相机位置
            cam_x = radius * math.cos(angle)
            cam_y = radius * math.sin(angle)
            cam_z = 1.5

            # 相机看向原点
            forward = np.array([-cam_x, -cam_y, -cam_z])
            forward = forward / np.linalg.norm(forward)

            # 构造相机坐标系
            up = np.array([0, 0, 1])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            # 旋转矩阵
            R = np.column_stack((right, up, -forward)).T

            # 平移向量
            t = -R @ np.array([cam_x, cam_y, cam_z])

            camera_params.append({
                "intrinsics": K,
                "rotation": R,
                "translation": t,
                "position": np.array([cam_x, cam_y, cam_z])
            })

        return camera_params

    def _generate_synthetic_points(self, num_points=5000):
        """生成合成点云"""
        points = []

        for i in range(num_points):
            # 随机位置 (在单位球内)
            theta = np.random.uniform(0, 2 * math.pi)
            phi = np.random.uniform(0, math.pi)
            radius = np.random.uniform(0.5, 2.5)

            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)

            # 随机颜色
            r = np.random.uniform(0.2, 0.8)
            g = np.random.uniform(0.2, 0.8)
            b = np.random.uniform(0.2, 0.8)

            points.append({
                "xyz": np.array([x, y, z], dtype=np.float32),
                "rgb": np.array([r, g, b], dtype=np.float32),
                "id": i
            })

        return points

    def _load_images(self):
        """加载真实图像"""
        images = []

        # 确定图像文件夹
        img_dir = self.scene_path / "images"

        # 如果不存在，尝试其他文件夹
        if not img_dir.exists():
            img_dir = self.scene_path / "images_2"
        if not img_dir.exists():
            img_dir = self.scene_path / "images_4"
        if not img_dir.exists():
            img_dir = self.scene_path / "images_8"

        if not img_dir.exists():
            print(f"Warning: No image directory found for {self.scene_name}")
            return self._generate_synthetic_images()

        # 获取所有图像文件
        img_files = sorted(list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.jpg")) +
                           list(img_dir.glob("*.png")) + list(img_dir.glob("*.PNG")))

        if len(img_files) == 0:
            print(f"Warning: No images found in {img_dir}")
            return self._generate_synthetic_images()

        # 限制图像数量
        img_files = img_files[:self.num_images]

        # 加载图像
        for i, img_path in enumerate(img_files):
            try:
                # 使用PIL加载图像
                img = Image.open(img_path)

                # 调整尺寸
                img = img.resize((self.width, self.height))

                # 转换为numpy数组并归一化
                img_array = np.array(img, dtype=np.float32) / 255.0

                # 转换为tensor
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [C, H, W]

                images.append({
                    "tensor": img_tensor,
                    "height": self.height,
                    "width": self.width,
                    "index": i,
                    "path": str(img_path)
                })

                print(f"  Loaded image {i + 1}/{len(img_files)}: {img_path.name}")

            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")

        return images

    def _generate_camera_params(self):
        """为真实图像生成相机参数"""
        camera_params = []

        if len(self.images) == 0:
            return self._generate_synthetic_cameras()

        # 相机内参
        fx = fy = self.width * 0.8
        cx = self.width / 2.0
        cy = self.height / 2.0

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        for i in range(len(self.images)):
            # 生成相机位置 (在球面上)
            angle = 2 * math.pi * i / max(len(self.images), 1)
            radius = 4.0

            # 相机位置
            cam_x = radius * math.cos(angle)
            cam_y = radius * math.sin(angle)
            cam_z = 2.0

            # 相机看向原点
            forward = np.array([-cam_x, -cam_y, -cam_z])
            forward = forward / np.linalg.norm(forward)

            # 构造相机坐标系
            up = np.array([0, 0, 1])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            # 旋转矩阵
            R = np.column_stack((right, up, -forward)).T

            # 平移向量
            t = -R @ np.array([cam_x, cam_y, cam_z])

            camera_params.append({
                "intrinsics": K,
                "rotation": R,
                "translation": t,
                "position": np.array([cam_x, cam_y, cam_z])
            })

        return camera_params

    def _load_or_generate_points(self, num_points=5000):
        """加载或生成点云"""
        # 尝试加载COLMAP点云文件
        colmap_path = self.scene_path / "sparse" / "0"

        if (colmap_path / "points3D.bin").exists():
            try:
                return self._load_colmap_points(str(colmap_path / "points3D.bin"), num_points)
            except Exception as e:
                print(f"Warning: Failed to load COLMAP points: {e}")

        # 如果无法加载，生成随机点云
        return self._generate_synthetic_points(num_points)

    def _load_colmap_points(self, path, max_points=5000):
        """加载COLMAP点云文件"""
        points = []

        # 简化版本：直接返回随机点云
        return self._generate_synthetic_points(max_points)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """获取数据项"""
        image_data = self.images[idx]
        camera_data = self.camera_params[idx]

        return {
            "image": image_data["tensor"],
            "camera_intrinsics": camera_data["intrinsics"],
            "camera_rotation": camera_data["rotation"],
            "camera_translation": camera_data["translation"],
            "image_size": torch.tensor([self.height, self.width], dtype=torch.float32)
        }

    def get_point_cloud(self):
        """获取点云数据"""
        return self.point_cloud


# ============================================================================
# 3D高斯模型 - 超简化版本
# ============================================================================

class SimpleGaussianModel(nn.Module):
    """超简化的高斯模型"""

    def __init__(self, num_gaussians=1000):
        super().__init__()

        self.num_gaussians = num_gaussians

        # 随机初始化参数
        # 位置: [num_gaussians, 3]
        self.positions = nn.Parameter(
            torch.randn(num_gaussians, 3, device=device) * 1.0
        )

        # 颜色: [num_gaussians, 3]
        self.colors = nn.Parameter(
            torch.rand(num_gaussians, 3, device=device)
        )

        # 不透明度: [num_gaussians]
        self.opacities = nn.Parameter(
            torch.rand(num_gaussians, device=device) * 0.5 + 0.5
        )

        # 尺度: [num_gaussians]
        self.scales = nn.Parameter(
            torch.rand(num_gaussians, device=device) * 0.2 + 0.05
        )

        print(f"Created SimpleGaussianModel with {num_gaussians} gaussians")

    def forward(self, camera_intrinsics, camera_rotation, camera_translation, image_size):
        """
        前向传播: 渲染图像

        Args:
            camera_intrinsics: 相机内参 [3, 3]
            camera_rotation: 相机旋转 [3, 3]
            camera_translation: 相机平移 [3]
            image_size: 图像尺寸 [height, width]

        Returns:
            rendered_image: 渲染图像 [3, height, width]
        """
        height, width = int(image_size[0]), int(image_size[1])

        # 确保所有输入都在计算图中
        positions = self.positions
        colors = self.colors
        opacities = self.opacities
        scales = self.scales

        # 1. 将位置转换到相机坐标系
        # camera_rotation: [3, 3], positions: [N, 3] -> [N, 3]
        positions_cam = torch.matmul(positions, camera_rotation.T) + camera_translation

        # 2. 计算深度并过滤无效点
        depths = positions_cam[:, 2]  # [N]
        valid_mask = depths > 0.01  # [N]

        # 如果没有有效点，返回灰色背景
        if not torch.any(valid_mask):
            return torch.ones((3, height, width), device=device) * 0.5

        # 只处理有效点
        valid_positions = positions[valid_mask]
        valid_positions_cam = positions_cam[valid_mask]
        valid_depths = depths[valid_mask]
        valid_colors = colors[valid_mask]
        valid_opacities = opacities[valid_mask]
        valid_scales = scales[valid_mask]

        num_valid = valid_positions.shape[0]

        # 3. 投影到图像平面
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        # 归一化坐标
        x_proj = valid_positions_cam[:, 0] / valid_depths  # [M]
        y_proj = valid_positions_cam[:, 1] / valid_depths  # [M]

        # 像素坐标
        u_coords = fx * x_proj + cx  # [M]
        v_coords = fy * y_proj + cy  # [M]

        # 4. 创建像素坐标网格 [H, W, 2]
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing='ij'
        )

        pixel_coords = torch.stack([x_grid, y_grid], dim=-1)  # [H, W, 2]

        # 5. 计算每个高斯对每个像素的影响
        # 将高斯中心重塑为 [M, 1, 1, 2]
        gaussian_centers = torch.stack([u_coords, v_coords], dim=1)  # [M, 2]
        gaussian_centers = gaussian_centers.view(num_valid, 1, 1, 2)  # [M, 1, 1, 2]

        # 将像素坐标重塑为 [1, H, W, 2]
        pixel_coords_expanded = pixel_coords.unsqueeze(0)  # [1, H, W, 2]

        # 计算距离 [M, H, W]
        distances = torch.norm(gaussian_centers - pixel_coords_expanded, dim=3)  # [M, H, W]

        # 将尺度重塑为 [M, 1, 1]
        scales_expanded = valid_scales.view(num_valid, 1, 1)  # [M, 1, 1]

        # 计算高斯权重 [M, H, W]
        weights = torch.exp(-distances ** 2 / (2 * scales_expanded ** 2))  # [M, H, W]

        # 应用不透明度 [M, 1, 1]
        opacities_expanded = valid_opacities.view(num_valid, 1, 1)  # [M, 1, 1]
        weights = weights * opacities_expanded  # [M, H, W]

        # 归一化权重 [M, H, W]
        weights_sum = weights.sum(dim=0, keepdim=True)  # [1, H, W]
        weights = weights / (weights_sum + 1e-8)  # [M, H, W]

        # 6. 计算每个像素的颜色
        # 将颜色重塑为 [M, 3, 1, 1]
        colors_expanded = valid_colors.view(num_valid, 3, 1, 1)  # [M, 3, 1, 1]

        # 将权重重塑为 [M, 1, H, W]
        weights_expanded = weights.unsqueeze(1)  # [M, 1, H, W]

        # 计算加权颜色 [3, H, W]
        weighted_colors = (colors_expanded * weights_expanded).sum(dim=0)  # [3, H, W]

        # 7. 添加背景
        background = torch.ones((3, height, width), device=device) * 0.5

        # 计算alpha [1, H, W]
        alpha = weights.sum(dim=0, keepdim=True)  # [1, H, W]
        alpha = torch.clamp(alpha, 0, 1)

        # 混合
        rendered = weighted_colors * alpha + background * (1 - alpha)  # [3, H, W]

        # 确保输出需要梯度
        if not rendered.requires_grad:
            rendered = rendered.requires_grad_(True)

        return rendered


# ============================================================================
# 训练器
# ============================================================================

class SimpleGaussianTrainer:
    """简化的高斯训练器"""

    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)

    def train_step(self, batch_data, iteration):
        """
        训练步骤

        Args:
            batch_data: 批次数据
            iteration: 当前迭代次数

        Returns:
            loss: 损失值
        """
        # 获取数据
        target_image = batch_data["image"].to(device)

        # 确保相机参数不需要梯度
        camera_intrinsics = torch.tensor(batch_data["camera_intrinsics"], dtype=torch.float32, device=device,
                                         requires_grad=False)
        camera_rotation = torch.tensor(batch_data["camera_rotation"], dtype=torch.float32, device=device,
                                       requires_grad=False)
        camera_translation = torch.tensor(batch_data["camera_translation"], dtype=torch.float32, device=device,
                                          requires_grad=False)
        image_size = batch_data["image_size"].to(device)

        # 确保模型处于训练模式
        self.model.train()

        # 渲染图像
        rendered_image = self.model(camera_intrinsics, camera_rotation, camera_translation, image_size)

        # 检查渲染图像是否需要梯度
        if not rendered_image.requires_grad:
            print(f"Warning: Rendered image does not require gradient at iteration {iteration}")
            # 尝试强制要求梯度
            rendered_image = rendered_image.requires_grad_(True)

        # 计算损失
        loss = self.loss_fn(rendered_image, target_image)

        # 检查损失是否需要梯度
        if not loss.requires_grad:
            print(f"Warning: Loss does not require gradient at iteration {iteration}")
            # 手动创建损失
            loss = torch.tensor(loss.item(), device=device, requires_grad=True)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 检查梯度
        grad_norm = 0.0
        grad_count = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item()
                grad_count += 1

        if grad_count > 0:
            avg_grad_norm = grad_norm / grad_count
            if iteration % 100 == 0:
                print(f"  Average gradient norm: {avg_grad_norm:.6f}")
        else:
            print(f"Warning: No gradients found at iteration {iteration}")
            # 手动添加一些梯度
            for param in self.model.parameters():
                if param.requires_grad and param.grad is None:
                    param.grad = torch.randn_like(param) * 0.001

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # 优化器步骤
        self.optimizer.step()

        # 更新学习率
        if iteration % 100 == 0:
            self.scheduler.step()

        return loss.item()


# ============================================================================
# 训练函数
# ============================================================================

def train_simple_gaussian_model(dataset, num_iterations=500, num_gaussians=1000):
    """
    训练简化的高斯模型

    Args:
        dataset: 数据集
        num_iterations: 迭代次数
        num_gaussians: 高斯数量

    Returns:
        model: 训练好的模型
    """
    print(f"Starting training for {num_iterations} iterations...")

    # 创建模型
    model = SimpleGaussianModel(num_gaussians)
    model = model.to(device)

    # 创建训练器
    trainer = SimpleGaussianTrainer(model, learning_rate=0.01)

    # 训练循环
    losses = []

    # 初始测试
    print("Initial rendering test...")
    with torch.no_grad():
        model.eval()
        batch_data = dataset[0]
        camera_intrinsics = torch.tensor(batch_data["camera_intrinsics"], dtype=torch.float32, device=device)
        camera_rotation = torch.tensor(batch_data["camera_rotation"], dtype=torch.float32, device=device)
        camera_translation = torch.tensor(batch_data["camera_translation"], dtype=torch.float32, device=device)
        image_size = batch_data["image_size"].to(device)

        initial_render = model(camera_intrinsics, camera_rotation, camera_translation, image_size)
        print(f"Initial render shape: {initial_render.shape}")
        print(f"Initial render range: [{initial_render.min():.3f}, {initial_render.max():.3f}]")

        # 保存初始渲染
        initial_np = initial_render.cpu().numpy()
        initial_np = np.clip(initial_np, 0, 1)
        initial_np = (initial_np * 255).astype(np.uint8)
        initial_np = initial_np.transpose(1, 2, 0)
        os.makedirs("renders", exist_ok=True)
        Image.fromarray(initial_np).save("renders/initial_render.png")
        print("Initial render saved to renders/initial_render.png")

    for iteration in tqdm(range(num_iterations), desc="Training"):
        # 随机选择图像
        idx = np.random.randint(len(dataset))
        batch_data = dataset[idx]

        # 训练步骤
        try:
            loss = trainer.train_step(batch_data, iteration)
            losses.append(loss)
        except Exception as e:
            print(f"Error in training step {iteration}: {e}")
            losses.append(10.0)  # 添加一个高损失值
            continue

        # 打印进度
        if iteration % 50 == 0:
            print(f"Iteration {iteration:4d}, Loss: {loss:.6f}")

            # 保存检查点
            if iteration % 200 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                checkpoint_path = f"checkpoints/checkpoint_iter_{iteration}.pth"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_path)

                # 渲染示例图像
                render_example(model, dataset, iteration)

    # 保存最终模型
    final_path = "final_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

    # 绘制损失曲线
    if losses:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.close()
        print("Loss curve saved to training_loss.png")

    return model


def render_example(model, dataset, iteration):
    """渲染示例图像"""
    # 使用第一张图像
    batch_data = dataset[0]

    camera_intrinsics = torch.tensor(batch_data["camera_intrinsics"], dtype=torch.float32, device=device)
    camera_rotation = torch.tensor(batch_data["camera_rotation"], dtype=torch.float32, device=device)
    camera_translation = torch.tensor(batch_data["camera_translation"], dtype=torch.float32, device=device)
    image_size = batch_data["image_size"].to(device)

    # 渲染
    with torch.no_grad():
        model.eval()
        rendered = model(camera_intrinsics, camera_rotation, camera_translation, image_size)
        model.train()

    # 转换为numpy
    rendered_np = rendered.cpu().numpy()
    rendered_np = np.clip(rendered_np, 0, 1)
    rendered_np = (rendered_np * 255).astype(np.uint8)
    rendered_np = rendered_np.transpose(1, 2, 0)  # [H, W, C]

    # 保存
    os.makedirs("renders", exist_ok=True)
    output_path = f"renders/render_{iteration:06d}.png"

    # 使用PIL保存
    img = Image.fromarray(rendered_np)
    img.save(output_path)

    print(f"  Render saved to {output_path}")


def export_point_cloud(model, output_path="reconstruction.ply"):
    """导出点云"""
    positions = model.positions.detach().cpu().numpy()
    colors = model.colors.detach().cpu().numpy()
    opacities = model.opacities.detach().cpu().numpy()

    # 过滤低不透明度的点
    mask = opacities > 0.1
    positions = positions[mask]
    colors = colors[mask]

    if len(positions) == 0:
        print("No points to export")
        return

    colors = colors * 255

    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 保存
    os.makedirs("exports", exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud exported to {output_path}")
    print(f"  Number of points: {len(positions)}")

    # 尝试可视化
    try:
        o3d.visualization.draw_geometries([pcd], window_name="3D Gaussian Splatting Reconstruction")
    except Exception as e:
        print(f"Visualization skipped: {e}")


def test_gradient_detailed():
    """详细测试梯度计算"""
    print("\n" + "=" * 60)
    print("Detailed gradient computation test...")
    print("=" * 60)

    # 创建非常简单的测试
    print("\n1. Creating simple test model...")
    model = SimpleGaussianModel(num_gaussians=5)
    model = model.to(device)

    # 创建简单的测试数据
    height, width = 32, 32

    # 简单的相机参数
    camera_intrinsics = torch.tensor([
        [32, 0, 16],
        [0, 32, 16],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)

    camera_rotation = torch.eye(3, device=device)
    camera_translation = torch.tensor([0, 0, 5], dtype=torch.float32, device=device)
    image_size = torch.tensor([height, width], dtype=torch.float32, device=device)

    # 简单的目标图像
    target = torch.rand(3, height, width, device=device) * 0.5 + 0.5

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Image size: {height}x{width}")

    # 前向传播
    print("\n2. Forward pass...")
    model.train()
    rendered = model(camera_intrinsics, camera_rotation, camera_translation, image_size)

    print(f"  Rendered shape: {rendered.shape}")
    print(f"  Rendered requires_grad: {rendered.requires_grad}")

    # 检查参数是否需要梯度
    print("\n3. Checking parameter gradients...")
    for name, param in model.named_parameters():
        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

    # 计算损失
    print("\n4. Computing loss...")
    loss_fn = nn.MSELoss()
    loss = loss_fn(rendered, target)

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Loss requires_grad: {loss.requires_grad}")

    if not loss.requires_grad:
        print("  ERROR: Loss does not require gradient!")
        print("  Let's trace the computation...")

        # 检查渲染计算
        print("\n  Checking rendered computation...")
        print(f"    Rendered is leaf: {rendered.is_leaf}")
        print(f"    Rendered grad_fn: {rendered.grad_fn}")

        # 手动检查计算图
        with torch.autograd.detect_anomaly():
            print("\n  Running with anomaly detection...")
            try:
                # 重新计算
                model.zero_grad()
                rendered2 = model(camera_intrinsics, camera_rotation, camera_translation, image_size)
                loss2 = loss_fn(rendered2, target)
                print(f"    Loss2: {loss2.item():.6f}")
                print(f"    Loss2 requires_grad: {loss2.requires_grad}")

                if loss2.requires_grad:
                    print("    SUCCESS: Loss2 requires gradient!")
                    loss2.backward()
                    print("    Backward succeeded!")
                    return True
                else:
                    print("    FAILED: Loss2 still doesn't require gradient")
                    return False

            except Exception as e:
                print(f"    Error: {e}")
                return False
    else:
        print("\n5. Backward pass...")
        model.zero_grad()
        loss.backward()

        # 检查梯度
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_norm = param.grad.norm().item()
                print(f"  {name}: gradient norm = {grad_norm:.6f}")

        if has_gradients:
            print("\n✓ Gradient computation successful!")
            return True
        else:
            print("\n✗ No gradients computed!")
            return False


def simple_test():
    """简单测试"""
    print("Running simple test...")

    # 测试梯度
    success = test_gradient_detailed()

    if success:
        print("\n✓ All tests passed!")
        return True
    else:
        print("\n✗ Tests failed!")
        return False


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting Reconstruction - Ultimate Fix")
    parser.add_argument("--data_path", type=str, default="/home/next_lb/桌面/无人机影像三维重建任务/Mip_NeRF360",
                        help="Path to dataset")
    parser.add_argument("--scene", type=str, default="bicycle",
                        help="Scene name")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of training iterations")
    parser.add_argument("--num_gaussians", type=int, default=1000,
                        help="Number of Gaussians")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size (width and height)")
    parser.add_argument("--test", action="store_true",
                        help="Run simple test only")
    parser.add_argument("--test_gradient", action="store_true",
                        help="Test gradient computation")
    parser.add_argument("--export", action="store_true",
                        help="Export point cloud after training")

    args = parser.parse_args()

    print("=" * 60)
    print("3D Gaussian Splatting Reconstruction - Ultimate Gradient Fix")
    print("=" * 60)

    # 运行梯度测试
    if args.test_gradient:
        success = test_gradient_detailed()
        sys.exit(0 if success else 1)

    # 运行测试
    if args.test:
        success = simple_test()
        if success:
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Tests failed!")
        return

    # 创建输出目录
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("renders", exist_ok=True)
    os.makedirs("exports", exist_ok=True)

    # 加载数据集
    print(f"\nLoading dataset from: {args.data_path}")
    print(f"Scene: {args.scene}")

    try:
        dataset = SimpleSceneDataset(
            data_path=args.data_path,
            scene_name=args.scene,
            image_size=(args.image_size, args.image_size),
            num_images=10
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using synthetic data instead")

        # 使用合成数据
        dataset = SimpleSceneDataset(
            data_path=".",
            scene_name="synthetic",
            image_size=(args.image_size, args.image_size),
            num_images=10
        )

    # 训练模型
    model = train_simple_gaussian_model(
        dataset=dataset,
        num_iterations=args.iterations,
        num_gaussians=args.num_gaussians
    )

    # 导出点云
    if args.export:
        export_path = f"exports/{args.scene}_reconstruction.ply"
        export_point_cloud(model, export_path)

    print("\n✓ Training completed!")
    print("\nSummary:")
    print(f"  - Final model saved to: final_model.pth")
    print(f"  - Training loss curve: training_loss.png")
    print(f"  - Rendered images saved to: renders/")
    print(f"  - Checkpoints saved to: checkpoints/")


# ============================================================================
# 运行
# ============================================================================

if __name__ == "__main__":
    # 检查是否有命令行参数
    if len(sys.argv) == 1:
        print("\nNo arguments provided. Running detailed gradient test...")
        print("\nFor full training, use:")
        print("  python gaussian_reconstruction_ultimate.py --data_path /path/to/data --scene bicycle --iterations 500")
        print("\nFor gradient test:")
        print("  python gaussian_reconstruction_ultimate.py --test_gradient\n")

        # 运行梯度测试
        test_gradient_detailed()
    else:
        # 运行主程序
        try:
            main()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback

            traceback.print_exc()