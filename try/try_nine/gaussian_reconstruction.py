#!/usr/bin/env python3
"""
3D Gaussian Splatting 三维重建 - 修复版本
适用于Mip_NeRF360数据集
作者: AI助手
"""

import os
import sys
import time
import argparse
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

# 检查CUDA可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")


# ============================================================================
# COLMAP数据读取工具类
# ============================================================================

class ColmapReader:
    """读取COLMAP二进制文件"""

    @staticmethod
    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
        """读取并解包二进制数据"""
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    @staticmethod
    def read_points3D_bin(path):
        """读取points3D.bin文件"""
        points3D = {}

        with open(path, "rb") as fid:
            num_points = ColmapReader.read_next_bytes(fid, 8, "Q")[0]

            for _ in range(num_points):
                point_id = ColmapReader.read_next_bytes(fid, 8, "Q")[0]
                xyz = ColmapReader.read_next_bytes(fid, 24, "ddd")
                rgb = ColmapReader.read_next_bytes(fid, 24, "ddd")  # 实际上是3个double
                error = ColmapReader.read_next_bytes(fid, 8, "d")[0]

                # 读取track长度
                track_length = ColmapReader.read_next_bytes(fid, 8, "Q")[0]

                # 读取track数据 (image_id, point2D_idx)
                track_data = []
                for __ in range(track_length):
                    img_id = ColmapReader.read_next_bytes(fid, 4, "I")[0]
                    point2d_idx = ColmapReader.read_next_bytes(fid, 8, "Q")[0]
                    track_data.append((img_id, point2d_idx))

                # 存储点
                points3D[point_id] = {
                    "xyz": np.array(xyz, dtype=np.float32),
                    "rgb": np.array(rgb, dtype=np.float32),
                    "error": error,
                    "track": track_data
                }

        return points3D

    @staticmethod
    def read_images_bin(path):
        """读取images.bin文件"""
        images = {}

        with open(path, "rb") as fid:
            num_images = ColmapReader.read_next_bytes(fid, 8, "Q")[0]

            for _ in range(num_images):
                image_id = ColmapReader.read_next_bytes(fid, 4, "I")[0]
                qvec = ColmapReader.read_next_bytes(fid, 32, "dddd")
                tvec = ColmapReader.read_next_bytes(fid, 24, "ddd")
                camera_id = ColmapReader.read_next_bytes(fid, 4, "I")[0]

                # 读取图像名称
                image_name = ""
                char = fid.read(1)
                while char != b'\x00':
                    image_name += char.decode('utf-8')
                    char = fid.read(1)

                # 读取关键点数量
                num_points2D = ColmapReader.read_next_bytes(fid, 8, "Q")[0]

                # 读取关键点
                xys = []
                point3D_ids = []
                for __ in range(num_points2D):
                    x, y = ColmapReader.read_next_bytes(fid, 16, "dd")
                    point3D_id = ColmapReader.read_next_bytes(fid, 8, "q")[0]  # q表示有符号64位
                    xys.append((x, y))
                    point3D_ids.append(point3D_id)

                images[image_id] = {
                    "qvec": np.array(qvec, dtype=np.float32),
                    "tvec": np.array(tvec, dtype=np.float32),
                    "camera_id": camera_id,
                    "name": image_name,
                    "xys": np.array(xys, dtype=np.float32),
                    "point3D_ids": np.array(point3D_ids, dtype=np.int64)
                }

        return images

    @staticmethod
    def read_cameras_bin(path):
        """读取cameras.bin文件"""
        cameras = {}

        with open(path, "rb") as fid:
            num_cameras = ColmapReader.read_next_bytes(fid, 8, "Q")[0]

            for _ in range(num_cameras):
                camera_id = ColmapReader.read_next_bytes(fid, 4, "I")[0]
                model_id = ColmapReader.read_next_bytes(fid, 4, "I")[0]
                width = ColmapReader.read_next_bytes(fid, 8, "Q")[0]
                height = ColmapReader.read_next_bytes(fid, 8, "Q")[0]

                # 读取参数
                num_params = ColmapReader.read_next_bytes(fid, 8, "Q")[0]
                params = ColmapReader.read_next_bytes(fid, 8 * num_params, "d" * num_params)

                cameras[camera_id] = {
                    "model": model_id,
                    "width": width,
                    "height": height,
                    "params": np.array(params, dtype=np.float32)
                }

        return cameras


# ============================================================================
# 数据集类 (简化版本)
# ============================================================================

class NeRF360Dataset(Dataset):
    """Mip_NeRF360数据集类 - 简化版本"""

    def __init__(self, data_path, scene_name="flowers", use_downsample=1, max_images=20):
        """
        初始化数据集

        Args:
            data_path: 数据根路径
            scene_name: 场景名称
            use_downsample: 下采样级别 (1, 2, 4, 8)
            max_images: 最大图像数量
        """
        super().__init__()

        self.data_path = Path(data_path)
        self.scene_name = scene_name
        self.use_downsample = use_downsample
        self.max_images = max_images

        # 构建场景路径
        if "extra_scenes" in str(self.data_path):
            self.scene_path = self.data_path / "360_extra_scenes" / scene_name
        else:
            self.scene_path = self.data_path / "360_v2" / scene_name

        print(f"Loading scene from: {self.scene_path}")

        # 检查路径是否存在
        if not self.scene_path.exists():
            raise ValueError(f"Scene path does not exist: {self.scene_path}")

        # 加载图像
        self.images = self._load_images()

        # 加载点云
        self.points3D = self._load_points3D()

        print(f"Loaded {len(self.images)} images")
        print(f"Loaded {len(self.points3D)} 3D points")

        # 生成相机参数
        self.camera_params = self._generate_camera_params()

    def _load_images(self):
        """加载图像"""
        images = []

        # 确定图像文件夹
        if self.use_downsample == 1:
            img_dir = self.scene_path / "images"
        elif self.use_downsample == 2:
            img_dir = self.scene_path / "images_2"
        elif self.use_downsample == 4:
            img_dir = self.scene_path / "images_4"
        elif self.use_downsample == 8:
            img_dir = self.scene_path / "images_8"
        else:
            img_dir = self.scene_path / "images"

        # 检查图像文件夹是否存在
        if not img_dir.exists():
            print(f"Warning: Image directory {img_dir} not found, using images folder")
            img_dir = self.scene_path / "images"

        # 获取所有图像文件
        img_files = sorted(list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.jpg")) +
                           list(img_dir.glob("*.png")) + list(img_dir.glob("*.PNG")))

        if len(img_files) == 0:
            raise ValueError(f"No images found in {img_dir}")

        # 限制图像数量
        img_files = img_files[:self.max_images]

        # 加载图像
        for i, img_path in enumerate(img_files):
            try:
                # 使用PIL加载图像
                img = Image.open(img_path)

                # 下采样以节省内存
                new_width = img.width // 8
                new_height = img.height // 8
                img = img.resize((new_width, new_height))

                img_array = np.array(img, dtype=np.float32) / 255.0

                # 转换为tensor
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [C, H, W]

                images.append({
                    "path": str(img_path),
                    "name": img_path.name,
                    "tensor": img_tensor,
                    "height": img_tensor.shape[1],
                    "width": img_tensor.shape[2],
                    "index": i
                })
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")

        return images

    def _load_points3D(self):
        """加载3D点云"""
        points = []

        # 尝试加载COLMAP点云文件
        colmap_path = self.scene_path / "sparse" / "0"

        if (colmap_path / "points3D.bin").exists():
            try:
                points3D_dict = ColmapReader.read_points3D_bin(str(colmap_path / "points3D.bin"))

                # 转换为点列表
                for point_id, point_data in points3D_dict.items():
                    # 限制点云数量
                    if len(points) >= 50000:
                        break

                    points.append({
                        "xyz": point_data["xyz"],
                        "rgb": point_data["rgb"] / 255.0,  # 归一化
                        "id": point_id
                    })

                print(f"Loaded {len(points)} points from COLMAP")

            except Exception as e:
                print(f"Warning: Failed to load points3D.bin: {e}")
                print("Generating random points instead")
                points = self._generate_random_points()
        else:
            print("Warning: points3D.bin not found, generating random points")
            points = self._generate_random_points()

        return points

    def _generate_random_points(self, num_points=10000):
        """生成随机点云"""
        points = []

        for i in range(num_points):
            # 随机生成点云 (在单位球内)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            radius = np.random.uniform(0.5, 3.0)

            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)

            xyz = np.array([x, y, z], dtype=np.float32)

            # 随机颜色
            rgb = np.random.rand(3).astype(np.float32)

            points.append({
                "xyz": xyz,
                "rgb": rgb,
                "id": i
            })

        return points

    def _generate_camera_params(self):
        """生成相机参数"""
        camera_params = []

        if len(self.images) == 0:
            return camera_params

        # 使用第一张图像的尺寸
        img_height = self.images[0]["height"]
        img_width = self.images[0]["width"]

        # 生成相机参数
        for i, img_data in enumerate(self.images):
            # 相机内参
            fx = fy = min(img_width, img_height) * 1.2  # 焦距
            cx = img_width / 2.0
            cy = img_height / 2.0

            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)

            # 相机外参 - 围绕场景旋转
            angle = 2 * np.pi * i / len(self.images)
            radius = 3.0

            # 相机位置 (在球面上)
            cam_x = radius * np.cos(angle)
            cam_y = radius * np.sin(angle)
            cam_z = 1.0

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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """获取数据项"""
        image_data = self.images[idx]
        camera_data = self.camera_params[idx]

        return {
            "image": image_data["tensor"],
            "image_path": image_data["path"],
            "image_name": image_data["name"],
            "camera_intrinsics": camera_data["intrinsics"],
            "camera_rotation": camera_data["rotation"],
            "camera_translation": camera_data["translation"],
            "image_size": torch.tensor([image_data["height"], image_data["width"]])
        }

    def get_point_cloud(self):
        """获取点云数据"""
        return self.points3D


# ============================================================================
# 3D高斯模型 (简化版本)
# ============================================================================

class Gaussian3D(nn.Module):
    """3D高斯模型 - 简化和修复版本"""

    def __init__(self, position, color, opacity=0.5, scale=0.1, rotation=None):
        """
        初始化3D高斯

        Args:
            position: 位置 [3]
            color: 颜色 [3] (RGB)
            opacity: 不透明度
            scale: 尺度 [3]
            rotation: 旋转四元数 [4]
        """
        super().__init__()

        # 将numpy数组转换为tensor并移到设备上
        self.position = nn.Parameter(torch.tensor(position, dtype=torch.float32, device=device))
        self.color = nn.Parameter(torch.tensor(color, dtype=torch.float32, device=device))
        self.log_opacity = nn.Parameter(torch.log(torch.tensor(opacity, dtype=torch.float32, device=device)))

        if scale is None:
            scale = [0.1, 0.1, 0.1]
        elif isinstance(scale, (int, float)):
            scale = [float(scale), float(scale), float(scale)]

        # 使用对数尺度以确保为正
        self.log_scale = nn.Parameter(torch.log(torch.tensor(scale, dtype=torch.float32, device=device)))

        if rotation is None:
            rotation = [1.0, 0.0, 0.0, 0.0]  # 单位四元数

        self.rotation = nn.Parameter(torch.tensor(rotation, dtype=torch.float32, device=device))

    @property
    def opacity(self):
        """获取不透明度"""
        return torch.sigmoid(self.log_opacity)

    @property
    def scale(self):
        """获取尺度"""
        return torch.exp(self.log_scale)

    def get_rotation_matrix(self):
        """将四元数转换为旋转矩阵"""
        q = self.rotation
        q = q / torch.norm(q)

        qw, qx, qy, qz = q[0], q[1], q[2], q[3]

        # 四元数转旋转矩阵
        R = torch.zeros((3, 3), device=device)

        R[0, 0] = 1 - 2 * qy * qy - 2 * qz * qz
        R[0, 1] = 2 * qx * qy - 2 * qz * qw
        R[0, 2] = 2 * qx * qz + 2 * qy * qw

        R[1, 0] = 2 * qx * qy + 2 * qz * qw
        R[1, 1] = 1 - 2 * qx * qx - 2 * qz * qz
        R[1, 2] = 2 * qy * qz - 2 * qx * qw

        R[2, 0] = 2 * qx * qz - 2 * qy * qw
        R[2, 1] = 2 * qy * qz + 2 * qx * qw
        R[2, 2] = 1 - 2 * qx * qx - 2 * qy * qy

        return R

    def get_covariance_matrix(self):
        """计算协方差矩阵"""
        R = self.get_rotation_matrix()

        # 尺度矩阵
        S = torch.diag(self.scale)

        # 协方差矩阵 Σ = R S S^T R^T
        covariance = R @ S @ S.T @ R.T

        return covariance

    def clone_with_noise(self, position_noise=0.01, color_noise=0.1):
        """克隆高斯并添加噪声"""
        # 分离梯度
        position = self.position.detach().clone()
        color = self.color.detach().clone()
        log_opacity = self.log_opacity.detach().clone()
        log_scale = self.log_scale.detach().clone()
        rotation = self.rotation.detach().clone()

        # 添加噪声
        position = position + torch.randn_like(position) * position_noise
        color = color + torch.randn_like(color) * color_noise
        color = torch.clamp(color, 0, 1)

        # 创建新的高斯
        new_gaussian = Gaussian3D(
            position.cpu().numpy(),
            color.cpu().numpy(),
            torch.sigmoid(log_opacity).cpu().numpy() * 0.8,  # 降低不透明度
            torch.exp(log_scale).cpu().numpy() * 1.2,  # 增大尺度
            rotation.cpu().numpy()
        )

        return new_gaussian


# ============================================================================
# 高斯场景模型 (修复版本)
# ============================================================================

class GaussianScene(nn.Module):
    """3D高斯场景模型 - 修复版本"""

    def __init__(self, initial_points=None, num_gaussians=5000):
        """
        初始化高斯场景

        Args:
            initial_points: 初始点云数据
            num_gaussians: 高斯数量
        """
        super().__init__()

        self.gaussians = nn.ModuleList()
        self.num_gaussians = num_gaussians

        # 从点云初始化高斯
        if initial_points is not None and len(initial_points) > 0:
            self._initialize_from_points(initial_points)
        else:
            self._initialize_random()

    def _initialize_from_points(self, points):
        """从点云初始化高斯"""
        print(f"Initializing {min(len(points), self.num_gaussians)} Gaussians from point cloud")

        for i, point in enumerate(points[:self.num_gaussians]):
            position = point["xyz"]
            color = point["rgb"]

            # 随机尺度
            scale = np.random.uniform(0.05, 0.2, 3).astype(np.float32)

            # 随机旋转
            rotation = np.random.randn(4).astype(np.float32)
            rotation = rotation / np.linalg.norm(rotation)

            # 随机不透明度
            opacity = np.random.uniform(0.3, 0.8)

            gaussian = Gaussian3D(position, color, opacity, scale, rotation)
            self.gaussians.append(gaussian)

    def _initialize_random(self):
        """随机初始化高斯"""
        print(f"Initializing {self.num_gaussians} random Gaussians")

        for i in range(self.num_gaussians):
            # 随机位置 (在单位球内)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            radius = np.random.uniform(0.5, 3.0)

            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)

            position = np.array([x, y, z], dtype=np.float32)

            # 随机颜色
            color = np.random.rand(3).astype(np.float32)

            # 随机尺度
            scale = np.random.uniform(0.05, 0.2, 3).astype(np.float32)

            # 随机旋转
            rotation = np.random.randn(4).astype(np.float32)
            rotation = rotation / np.linalg.norm(rotation)

            # 随机不透明度
            opacity = np.random.uniform(0.3, 0.8)

            gaussian = Gaussian3D(position, color, opacity, scale, rotation)
            self.gaussians.append(gaussian)

    def forward(self):
        """前向传播 - 返回所有高斯参数"""
        return self.gaussians

    def get_parameters(self):
        """获取所有可学习参数"""
        params = []
        for g in self.gaussians:
            params.extend(g.parameters())
        return params

    def prune_gaussians(self, opacity_threshold=0.01):
        """修剪不透明度低的高斯"""
        indices_to_keep = []

        for i, g in enumerate(self.gaussians):
            if g.opacity > opacity_threshold:
                indices_to_keep.append(i)

        # 创建新的高斯列表
        new_gaussians = nn.ModuleList([self.gaussians[i] for i in indices_to_keep])

        pruned_count = len(self.gaussians) - len(new_gaussians)
        self.gaussians = new_gaussians

        print(f"Pruned {pruned_count} Gaussians, remaining: {len(self.gaussians)}")

    def densify_gaussians(self):
        """基于梯度进行高斯密度控制"""
        # 如果高斯数量太少，克隆一些高斯
        if len(self.gaussians) < self.num_gaussians * 1.2:
            num_to_add = min(100, self.num_gaussians // 10)

            # 随机选择一些高斯进行克隆
            indices = np.random.choice(len(self.gaussians), num_to_add, replace=True)

            for idx in indices:
                gaussian = self.gaussians[idx]
                new_gaussian = gaussian.clone_with_noise()
                self.gaussians.append(new_gaussian)

            print(f"Added {num_to_add} new Gaussians, total: {len(self.gaussians)}")

    def render_image_simple(self, camera_intrinsics, camera_rotation, camera_translation, image_size):
        """
        简化渲染图像 - 使用点渲染而不是高斯溅射

        Args:
            camera_intrinsics: 相机内参 [3, 3]
            camera_rotation: 相机旋转 [3, 3]
            camera_translation: 相机平移 [3]
            image_size: 图像尺寸 [height, width]

        Returns:
            渲染图像 [C, H, W]
        """
        height, width = int(image_size[0]), int(image_size[1])

        # 初始化渲染图像
        rendered_image = torch.zeros((3, height, width), device=device)

        # 获取相机参数
        K = camera_intrinsics
        R = camera_rotation
        t = camera_translation

        # 创建投影矩阵 P = K [R | t]
        RT = torch.cat([R, t.unsqueeze(1)], dim=1)  # [3, 4]
        P = K @ RT  # [3, 4]

        # 限制高斯数量以提高性能
        max_gaussians = min(1000, len(self.gaussians))
        indices = torch.randperm(len(self.gaussians))[:max_gaussians]

        # 渲染每个高斯
        for idx in indices:
            gaussian = self.gaussians[idx]

            # 获取高斯参数
            position = gaussian.position  # [3]
            color = gaussian.color  # [3]
            opacity = gaussian.opacity  # 标量
            scale = gaussian.scale.mean()  # 平均尺度

            # 将3D点投影到2D
            point_homo = torch.cat([position, torch.tensor([1.0], device=device)])  # [4]
            point_cam = P @ point_homo  # [3]

            # 透视除法
            z = point_cam[2]
            if z <= 0.1:  # 点在相机后面或太近
                continue

            u = point_cam[0] / z
            v = point_cam[1] / z

            # 检查是否在图像范围内
            if u < 0 or u >= width or v < 0 or v >= height:
                continue

            # 计算点的半径 (基于尺度)
            radius = max(1, int(scale * 50 / z))

            # 计算绘制范围
            u_min = max(0, int(u - radius))
            u_max = min(width, int(u + radius + 1))
            v_min = max(0, int(v - radius))
            v_max = min(height, int(v + radius + 1))

            if u_min >= u_max or v_min >= v_max:
                continue

            # 在图像上绘制点
            for y in range(v_min, v_max):
                for x in range(u_min, u_max):
                    # 计算距离
                    dist = ((x - u) ** 2 + (y - v) ** 2) ** 0.5

                    if dist <= radius:
                        # 计算权重 (高斯衰减)
                        weight = torch.exp(-dist ** 2 / (2 * (radius / 2) ** 2))
                        alpha = opacity * weight

                        # alpha混合
                        rendered_image[:, y, x] = (
                                alpha * color +
                                (1 - alpha) * rendered_image[:, y, x]
                        )

        return rendered_image

    def render_image_fast(self, camera_intrinsics, camera_rotation, camera_translation, image_size):
        """
        快速渲染图像 - 使用简化方法

        Args:
            camera_intrinsics: 相机内参 [3, 3]
            camera_rotation: 相机旋转 [3, 3]
            camera_translation: 相机平移 [3]
            image_size: 图像尺寸 [height, width]

        Returns:
            渲染图像 [C, H, W]
        """
        height, width = int(image_size[0]), int(image_size[1])

        # 创建空白图像
        image = torch.ones((3, height, width), device=device) * 0.5  # 灰色背景

        # 限制高斯数量
        num_gaussians = min(2000, len(self.gaussians))

        # 收集所有高斯的位置和颜色
        positions = []
        colors = []
        opacities = []
        scales = []

        for i in range(num_gaussians):
            gaussian = self.gaussians[i]
            positions.append(gaussian.position)
            colors.append(gaussian.color)
            opacities.append(gaussian.opacity)
            scales.append(gaussian.scale.mean())

        if len(positions) == 0:
            return image

        # 转换为tensor
        positions = torch.stack(positions)  # [N, 3]
        colors = torch.stack(colors)  # [N, 3]
        opacities = torch.stack(opacities)  # [N]
        scales = torch.stack(scales)  # [N]

        # 投影到2D
        K = camera_intrinsics
        R = camera_rotation
        t = camera_translation

        # 世界坐标到相机坐标
        positions_cam = (R @ positions.T + t.unsqueeze(1)).T  # [N, 3]

        # 深度
        depths = positions_cam[:, 2]

        # 剔除在相机后面的点
        valid_mask = depths > 0.1
        if not valid_mask.any():
            return image

        positions_cam = positions_cam[valid_mask]
        colors = colors[valid_mask]
        opacities = opacities[valid_mask]
        scales = scales[valid_mask]
        depths = depths[valid_mask]

        # 透视投影
        u = K[0, 0] * positions_cam[:, 0] / depths + K[0, 2]
        v = K[1, 1] * positions_cam[:, 1] / depths + K[1, 2]

        # 转换为整数坐标
        u_int = u.round().long()
        v_int = v.round().long()

        # 创建有效掩码
        valid = (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < width)

        if not valid.any():
            return image

        u_int = u_int[valid]
        v_int = v_int[valid]
        colors = colors[valid]
        opacities = opacities[valid]

        # 在图像上绘制点
        for i in range(len(u_int)):
            x, y = u_int[i], v_int[i]
            color = colors[i]
            alpha = opacities[i]

            # alpha混合
            image[:, y, x] = alpha * color + (1 - alpha) * image[:, y, x]

        return image


# ============================================================================
# 训练和优化 (修复版本)
# ============================================================================

class GaussianOptimizer:
    """高斯优化器 - 修复版本"""

    def __init__(self, scene, learning_rate=0.01):
        """
        初始化优化器

        Args:
            scene: GaussianScene对象
            learning_rate: 学习率
        """
        self.scene = scene
        self.learning_rate = learning_rate

        # 获取所有参数
        params = scene.get_parameters()

        # 创建优化器
        self.optimizer = optim.Adam(params, lr=learning_rate)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.95)

        # 损失函数
        self.loss_fn = nn.MSELoss()  # MSE损失

    def train_step(self, batch_data, iteration):
        """
        执行训练步骤

        Args:
            batch_data: 批次数据
            iteration: 当前迭代次数

        Returns:
            损失值
        """
        # 获取数据
        target_image = batch_data["image"].to(device)
        camera_intrinsics = torch.tensor(batch_data["camera_intrinsics"], dtype=torch.float32, device=device)
        camera_rotation = torch.tensor(batch_data["camera_rotation"], dtype=torch.float32, device=device)
        camera_translation = torch.tensor(batch_data["camera_translation"], dtype=torch.float32, device=device)
        image_size = batch_data["image_size"].to(device)

        # 渲染图像
        rendered_image = self.scene.render_image_fast(
            camera_intrinsics, camera_rotation, camera_translation, image_size
        )

        # 调整目标图像尺寸以匹配渲染图像
        target_resized = F.interpolate(
            target_image.unsqueeze(0),
            size=rendered_image.shape[1:],
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # 计算损失
        loss = self.loss_fn(rendered_image, target_resized)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期进行密度控制
        if iteration % 100 == 0 and iteration > 0:
            self.scene.densify_gaussians()

        # 定期修剪
        if iteration % 200 == 0 and iteration > 0:
            self.scene.prune_gaussians(opacity_threshold=0.1)

        return loss.item()

    def update_learning_rate(self):
        """更新学习率"""
        self.scheduler.step()


# ============================================================================
# 主训练函数 (修复版本)
# ============================================================================

def train_gaussian_splatting(dataset, num_iterations=500, save_interval=100):
    """
    训练3D高斯溅射模型

    Args:
        dataset: 数据集
        num_iterations: 迭代次数
        save_interval: 保存间隔
    """
    print("Starting 3D Gaussian Splatting training...")

    # 获取初始点云
    point_cloud = dataset.get_point_cloud()

    # 创建高斯场景
    scene = GaussianScene(initial_points=point_cloud, num_gaussians=2000)
    scene = scene.to(device)

    # 创建优化器
    optimizer = GaussianOptimizer(scene, learning_rate=0.01)

    # 训练循环
    losses = []

    for iteration in tqdm(range(num_iterations), desc="Training"):
        # 随机选择一张图像
        idx = np.random.randint(len(dataset))
        batch_data = dataset[idx]

        # 训练步骤
        loss = optimizer.train_step(batch_data, iteration)
        losses.append(loss)

        # 定期更新学习率
        if iteration % 200 == 0:
            optimizer.update_learning_rate()

        # 打印进度
        if iteration % 50 == 0:
            print(f"Iteration {iteration}, Loss: {loss:.6f}")

        # 定期保存和渲染
        if iteration % save_interval == 0 and iteration > 0:
            # 保存场景
            save_path = f"checkpoints/iteration_{iteration}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(scene.state_dict(), save_path)

            # 渲染示例图像
            render_example(scene, dataset, iteration)

    # 绘制损失曲线
    if len(losses) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('training_loss.png')
        plt.close()

    print("Training completed!")
    return scene


def save_scene(scene, filepath):
    """保存场景"""
    torch.save(scene.state_dict(), filepath)
    print(f"Scene saved to {filepath}")


def load_scene(filepath, num_gaussians=2000):
    """加载场景"""
    # 创建空场景
    scene = GaussianScene(num_gaussians=num_gaussians)
    scene.load_state_dict(torch.load(filepath))
    scene = scene.to(device)

    print(f"Scene loaded from {filepath}")
    return scene


def render_example(scene, dataset, iteration):
    """渲染示例图像"""
    # 选择第一张图像
    batch_data = dataset[0]

    camera_intrinsics = torch.tensor(batch_data["camera_intrinsics"], dtype=torch.float32, device=device)
    camera_rotation = torch.tensor(batch_data["camera_rotation"], dtype=torch.float32, device=device)
    camera_translation = torch.tensor(batch_data["camera_translation"], dtype=torch.float32, device=device)
    image_size = batch_data["image_size"].to(device)

    # 渲染
    with torch.no_grad():
        rendered = scene.render_image_fast(
            camera_intrinsics, camera_rotation, camera_translation, image_size
        )

    # 保存渲染结果
    rendered_np = rendered.cpu().numpy()
    rendered_np = np.clip(rendered_np, 0, 1)
    rendered_np = (rendered_np * 255).astype(np.uint8)

    # 转置为 [H, W, C]
    rendered_np = rendered_np.transpose(1, 2, 0)

    # 保存
    output_dir = "renders"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/render_{iteration:06d}.png"

    # 使用PIL保存
    img = Image.fromarray(rendered_np)
    img.save(output_path)

    print(f"Render saved to {output_path}")

    return output_path


def export_point_cloud(scene, output_path="output.ply"):
    """导出点云"""
    points = []
    colors = []

    with torch.no_grad():
        for g in scene.gaussians:
            pos = g.position.detach().cpu().numpy()
            color = g.color.detach().cpu().numpy()
            opacity = g.opacity.detach().cpu().numpy()

            # 只导出不透明度高的点
            if opacity > 0.1:
                points.append(pos)
                colors.append(color)

    if len(points) == 0:
        print("No points to export")
        return

    points = np.array(points)
    colors = np.array(colors) * 255

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 保存
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud saved to {output_path}")

    # 可视化
    try:
        o3d.visualization.draw_geometries([pcd])
    except:
        print("Visualization not available in headless mode")


# ============================================================================
# 简单测试函数
# ============================================================================

def simple_test():
    """简单测试函数"""
    print("Running simple test...")

    # 创建测试数据集
    class TestDataset:
        def __init__(self):
            self.images = [
                {
                    "tensor": torch.rand(3, 64, 64),
                    "height": 64,
                    "width": 64
                }
            ]
            self.camera_params = [
                {
                    "intrinsics": np.array([[100, 0, 32], [0, 100, 32], [0, 0, 1]], dtype=np.float32),
                    "rotation": np.eye(3, dtype=np.float32),
                    "translation": np.array([0, 0, 3], dtype=np.float32)
                }
            ]

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "image": self.images[idx]["tensor"],
                "camera_intrinsics": self.camera_params[idx]["intrinsics"],
                "camera_rotation": self.camera_params[idx]["rotation"],
                "camera_translation": self.camera_params[idx]["translation"],
                "image_size": torch.tensor([self.images[idx]["height"], self.images[idx]["width"]])
            }

        def get_point_cloud(self):
            # 生成一些测试点
            points = []
            for i in range(100):
                points.append({
                    "xyz": np.random.randn(3).astype(np.float32) * 2.0,
                    "rgb": np.random.rand(3).astype(np.float32),
                    "id": i
                })
            return points

    # 创建测试数据集
    test_dataset = TestDataset()

    # 创建场景
    scene = GaussianScene(initial_points=test_dataset.get_point_cloud(), num_gaussians=500)
    scene = scene.to(device)

    # 渲染测试图像
    batch_data = test_dataset[0]
    camera_intrinsics = torch.tensor(batch_data["camera_intrinsics"], dtype=torch.float32, device=device)
    camera_rotation = torch.tensor(batch_data["camera_rotation"], dtype=torch.float32, device=device)
    camera_translation = torch.tensor(batch_data["camera_translation"], dtype=torch.float32, device=device)
    image_size = batch_data["image_size"].to(device)

    print("Testing rendering...")
    with torch.no_grad():
        rendered = scene.render_image_fast(
            camera_intrinsics, camera_rotation, camera_translation, image_size
        )

    print(f"Rendered image shape: {rendered.shape}")
    print(f"Rendered image range: [{rendered.min():.3f}, {rendered.max():.3f}]")

    # 保存测试渲染
    rendered_np = rendered.cpu().numpy()
    rendered_np = np.clip(rendered_np, 0, 1)
    rendered_np = (rendered_np * 255).astype(np.uint8)
    rendered_np = rendered_np.transpose(1, 2, 0)

    img = Image.fromarray(rendered_np)
    img.save("test_render.png")

    print("Test completed successfully! Check test_render.png")

    return True


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting for NeRF360 Dataset")
    parser.add_argument("--data_path", type=str, default="/home/next_lb/桌面/无人机影像三维重建任务/Mip_NeRF360",
                        help="Path to Mip_NeRF360 dataset")
    parser.add_argument("--scene", type=str, default="flowers",
                        help="Scene name (flowers, threehill, bicycle, etc.)")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of training iterations")
    parser.add_argument("--num_gaussians", type=int, default=2000,
                        help="Number of Gaussians")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--export", action="store_true",
                        help="Export point cloud after training")
    parser.add_argument("--test", action="store_true",
                        help="Run simple test only")

    args = parser.parse_args()

    # 运行测试
    if args.test:
        simple_test()
        return

    # 创建输出目录
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("renders", exist_ok=True)

    # 加载数据集
    print(f"Loading dataset from {args.data_path}")
    try:
        dataset = NeRF360Dataset(
            data_path=args.data_path,
            scene_name=args.scene,
            use_downsample=1,
            max_images=10  # 限制图像数量以加速
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating test dataset...")

        # 创建简单测试数据集
        class SimpleDataset:
            def __init__(self):
                self.images = []
                for i in range(5):
                    img = torch.rand(3, 128, 128)
                    self.images.append({
                        "tensor": img,
                        "height": 128,
                        "width": 128
                    })
                self.points3D = []
                for i in range(1000):
                    self.points3D.append({
                        "xyz": np.random.randn(3).astype(np.float32) * 2.0,
                        "rgb": np.random.rand(3).astype(np.float32),
                        "id": i
                    })

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                # 简单相机参数
                K = np.array([[200, 0, 64], [0, 200, 64], [0, 0, 1]], dtype=np.float32)

                # 相机围绕原点旋转
                angle = 2 * np.pi * idx / len(self)
                R = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ], dtype=np.float32)

                t = R @ np.array([3, 0, 1])

                return {
                    "image": self.images[idx]["tensor"],
                    "camera_intrinsics": K,
                    "camera_rotation": R,
                    "camera_translation": t,
                    "image_size": torch.tensor([self.images[idx]["height"], self.images[idx]["width"]])
                }

            def get_point_cloud(self):
                return self.points3D

        dataset = SimpleDataset()
        print("Using simple test dataset")

    # 加载或创建场景
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        scene = load_scene(args.resume, args.num_gaussians)
    else:
        # 从点云初始化
        point_cloud = dataset.get_point_cloud()
        scene = GaussianScene(
            initial_points=point_cloud,
            num_gaussians=args.num_gaussians
        )
        scene = scene.to(device)

    # 训练
    scene = train_gaussian_splatting(
        dataset,
        num_iterations=args.iterations,
        save_interval=100
    )

    # 导出点云
    if args.export:
        export_point_cloud(scene, f"{args.scene}_reconstruction.ply")

    # 最终保存
    final_checkpoint = f"checkpoints/final_{args.scene}.pt"
    save_scene(scene, final_checkpoint)

    print(f"Training completed! Final model saved to {final_checkpoint}")


# ============================================================================
# 运行
# ============================================================================

if __name__ == "__main__":
    # 显示欢迎信息
    print("=" * 60)
    print("3D Gaussian Splatting Reconstruction - Fixed Version")
    print("=" * 60)

    # 检查参数
    if len(sys.argv) == 1:
        print("\nNo arguments provided, running simple test...")
        print("For full training, run:")
        print(
            "  python gaussian_reconstruction_fixed.py --data_path /path/to/Mip_NeRF360 --scene flowers --iterations 500")
        print("\nOr run a simple test:")
        print("  python gaussian_reconstruction_fixed.py --test\n")

        # 运行简单测试
        simple_test()
    else:
        # 运行主程序
        main()