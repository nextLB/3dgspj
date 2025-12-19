#!/usr/bin/env python3
"""
数据集类 - 加载图像和相机数据
"""
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import imageio
from dataclasses import dataclass


@dataclass
class CameraData:
    """相机数据容器"""
    image_id: int
    image_name: str
    image_path: str
    image: Optional[np.ndarray]
    R: np.ndarray  # 世界到相机
    t: np.ndarray  # 世界到相机
    K: np.ndarray  # 内参矩阵
    width: int
    height: int
    scale: float = 1.0

    def to_dict(self):
        # 确保返回numpy数组而不是tensor
        R = self.R
        t = self.t
        K = self.K

        # 如果是tensor，转换为numpy
        if isinstance(R, torch.Tensor):
            R = R.cpu().numpy()
        if isinstance(t, torch.Tensor):
            t = t.cpu().numpy()
        if isinstance(K, torch.Tensor):
            K = K.cpu().numpy()

        return {
            "image_id": self.image_id,
            "image_name": self.image_name,
            "R": R,
            "t": t,
            "K": K,
            "width": self.width,
            "height": self.height,
            "scale": self.scale
        }

class MipNeRF360Dataset(Dataset):
    """Mip-NeRF 360数据集类"""

    def __init__(self, data_root: str, scene_name: str,
                 split: str = "train", scale: int = 1,
                 load_images: bool = True, max_images: int = -1):
        """
        初始化数据集

        Args:
            data_root: 数据根目录
            scene_name: 场景名称 (如: "flowers", "bicycle")
            split: 数据集划分 ("train", "val", "test")
            scale: 图像缩放比例 (1, 2, 4, 8)
            load_images: 是否加载图像
            max_images: 最大图像数量 (-1表示全部)
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.scene_name = scene_name
        self.split = split
        self.scale = scale
        self.load_images = load_images
        self.max_images = max_images

        # 确定场景路径
        if "360_extra_scenes" in str(self.data_root):
            self.scene_path = self.data_root / "360_extra_scenes" / scene_name
        else:
            self.scene_path = self.data_root / "360_v2" / scene_name

        if not self.scene_path.exists():
            raise ValueError(f"场景路径不存在: {self.scene_path}")

        # 确定图像路径
        if scale == 1:
            self.image_dir = self.scene_path / "images"
        else:
            self.image_dir = self.scene_path / f"images_{scale}"

        if not self.image_dir.exists():
            # 尝试其他可能的路径
            self.image_dir = self.scene_path / "images"
            if not self.image_dir.exists():
                raise ValueError(f"图像目录不存在: {self.image_dir}")

        # 稀疏重建路径
        self.sparse_dir = self.scene_path / "sparse" / "0"

        # poses_bounds文件
        self.poses_bounds_path = self.scene_path / "poses_bounds.npy"

        # 加载数据
        self.cameras = self._load_cameras()

        # 过滤图像
        self.valid_cameras = self._filter_cameras()

        if max_images > 0 and len(self.valid_cameras) > max_images:
            # 均匀采样图像
            indices = np.linspace(0, len(self.valid_cameras) - 1, max_images).astype(int)
            self.valid_cameras = [self.valid_cameras[i] for i in indices]

        print(f"加载了 {len(self.valid_cameras)} 个相机")

    def _load_cameras(self) -> List[CameraData]:
        """加载相机数据"""
        cameras = []

        # 首先尝试从COLMAP文件加载
        if self.sparse_dir.exists():
            from colmap_reader import ColmapReader

            try:
                colmap_cameras, colmap_images, _ = ColmapReader.read_colmap_sparse(self.sparse_dir)
                camera_list = CameraUtils.create_camera_dict(colmap_cameras, colmap_images)

                for i, cam_dict in enumerate(camera_list):
                    image_name = cam_dict["image_name"]
                    image_path = self.image_dir / image_name

                    if not image_path.exists():
                        # 尝试不同的扩展名
                        possible_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                        found = False
                        for ext in possible_extensions:
                            alt_path = self.image_dir / f"{Path(image_name).stem}{ext}"
                            if alt_path.exists():
                                image_path = alt_path
                                found = True
                                break

                        if not found:
                            print(f"警告: 图像文件不存在: {image_path}")
                            continue

                    camera_data = CameraData(
                        image_id=i,
                        image_name=image_name,
                        image_path=str(image_path),
                        image=None,
                        R=cam_dict["R"],
                        t=cam_dict["t"],
                        K=cam_dict["K"],
                        width=cam_dict["width"],
                        height=cam_dict["height"]
                    )

                    cameras.append(camera_data)

                return cameras

            except Exception as e:
                print(f"加载COLMAP数据失败: {e}")

        # 如果COLMAP加载失败，尝试从poses_bounds.npy加载
        if self.poses_bounds_path.exists():
            try:
                return self._load_from_poses_bounds()
            except Exception as e:
                print(f"加载poses_bounds.npy失败: {e}")

        # 如果都没有，使用默认相机
        return self._load_default_cameras()

    def _load_from_poses_bounds(self) -> List[CameraData]:
        """从poses_bounds.npy加载相机数据"""
        from colmap_reader import ColmapReader

        poses, hwf, bounds = ColmapReader.load_poses_bounds_npy(self.poses_bounds_path)

        # 获取图像文件列表
        image_files = sorted([f for f in os.listdir(self.image_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])

        if len(image_files) != len(poses):
            print(f"警告: 图像数量({len(image_files)})与位姿数量({len(poses)})不匹配")
            min_len = min(len(image_files), len(poses))
            image_files = image_files[:min_len]
            poses = poses[:min_len]
            hwf = hwf[:min_len]

        cameras = []
        for i, (image_file, pose, (h, w, f)) in enumerate(zip(image_files, poses, hwf)):
            image_path = self.image_dir / image_file

            # 解析位姿矩阵
            # NeRF格式: [R|t] 其中R是3x3，t是3x1
            R = pose[:3, :3]
            t = pose[:3, 3]

            # 计算内参矩阵
            # hwf包含: [高度, 宽度, 焦距]
            K = np.array([
                [f, 0, w / 2],
                [0, f, h / 2],
                [0, 0, 1]
            ])

            camera_data = CameraData(
                image_id=i,
                image_name=image_file,
                image_path=str(image_path),
                image=None,
                R=R,
                t=t,
                K=K,
                width=int(w),
                height=int(h)
            )

            cameras.append(camera_data)

        return cameras

    def _load_default_cameras(self) -> List[CameraData]:
        """加载默认相机（当没有相机参数文件时）"""
        # 获取图像文件列表
        image_files = sorted([f for f in os.listdir(self.image_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])

        cameras = []
        for i, image_file in enumerate(image_files):
            image_path = self.image_dir / image_file

            # 使用PIL获取图像尺寸
            with Image.open(image_path) as img:
                width, height = img.size

            # 使用默认相机参数
            K = CameraUtils.intrinsic_from_fov(width, height)

            # 使用默认位姿（稍后会归一化）
            R = np.eye(3)
            t = np.array([0, 0, 5])  # 默认位置

            camera_data = CameraData(
                image_id=i,
                image_name=image_file,
                image_path=str(image_path),
                image=None,
                R=R,
                t=t,
                K=K,
                width=width,
                height=height
            )

            cameras.append(camera_data)

        return cameras

    def _filter_cameras(self) -> List[CameraData]:
        """过滤有效的相机"""
        valid_cameras = []

        for cam in self.cameras:
            if not os.path.exists(cam.image_path):
                print(f"警告: 跳过不存在的图像: {cam.image_path}")
                continue

            # 检查相机参数
            if cam.K is None or cam.R is None or cam.t is None:
                print(f"警告: 跳过无效的相机参数: {cam.image_name}")
                continue

            valid_cameras.append(cam)

        return valid_cameras

    def __len__(self) -> int:
        return len(self.valid_cameras)

    def __getitem__(self, idx: int) -> Dict:
        """获取数据项"""
        cam = self.valid_cameras[idx]

        # 加载图像（如果需要）
        if self.load_images and cam.image is None:
            try:
                # 使用imageio加载图像
                image = imageio.imread(cam.image_path)

                # 转换为RGB
                if len(image.shape) == 2:  # 灰度图
                    image = np.stack([image] * 3, axis=-1)
                elif image.shape[2] == 4:  # RGBA
                    image = image[:, :, :3]

                # 调整大小
                if self.scale != 1:
                    new_height = cam.height // self.scale
                    new_width = cam.width // self.scale
                    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

                cam.image = image

            except Exception as e:
                print(f"加载图像失败 {cam.image_path}: {e}")
                # 创建空白图像
                cam.image = np.zeros((cam.height // self.scale, cam.width // self.scale, 3), dtype=np.uint8)

        # 准备数据
        data = {
            "camera": cam.to_dict(),
            "image_path": cam.image_path,
            "image_id": cam.image_id
        }

        if cam.image is not None:
            # 转换为tensor
            image_tensor = torch.from_numpy(cam.image).float() / 255.0
            data["image"] = image_tensor.permute(2, 0, 1)  # HWC -> CHW

        return data

    def get_camera_params(self) -> List[Dict]:
        """获取所有相机参数"""
        return [cam.to_dict() for cam in self.valid_cameras]

    def normalize_scene(self):
        """归一化场景"""
        from camera_utils import CameraUtils

        camera_dicts = [cam.to_dict() for cam in self.valid_cameras]
        normalized_dicts = CameraUtils.normalize_cameras(camera_dicts)

        # 更新相机数据
        for cam, norm_dict in zip(self.valid_cameras, normalized_dicts):
            cam.R = norm_dict["R"]
            cam.t = norm_dict["t"]
            cam.scale = norm_dict.get("scale", 1.0)

    def get_point_cloud(self) -> Optional[np.ndarray]:
        """从COLMAP获取点云"""
        if not self.sparse_dir.exists():
            return None

        try:
            from colmap_reader import ColmapReader
            _, _, points3D = ColmapReader.read_colmap_sparse(self.sparse_dir)

            # 提取点云
            points = []
            colors = []

            for point_id, point_info in points3D.items():
                points.append(point_info["xyz"])
                colors.append(point_info["rgb"])

            if len(points) == 0:
                return None

            points = np.array(points)
            colors = np.array(colors) / 255.0

            # 归一化点云（使用与相机相同的归一化）
            camera_dicts = [cam.to_dict() for cam in self.valid_cameras]
            if camera_dicts:
                from camera_utils import CameraUtils
                normalized_dicts = CameraUtils.normalize_cameras(camera_dicts)
                if normalized_dicts:
                    scale = normalized_dicts[0].get("scale", 1.0)
                    center = normalized_dicts[0].get("center", np.zeros(3))
                    points = (points - center) * scale

            return {"points": points, "colors": colors}

        except Exception as e:
            print(f"获取点云失败: {e}")
            return None