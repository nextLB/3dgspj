import os
import numpy as np
import torch
from PIL import Image
import struct
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import config


@dataclass
class CameraInfo:
    image_id: int
    qvec: np.ndarray  # 四元数 (w, x, y, z)
    tvec: np.ndarray  # 平移向量 (x, y, z)
    camera_id: int
    image_name: str


class SimpleDataset:
    def __init__(self, data_dir: str, image_scale: int = 1, device=torch.device("cpu")):
        self.data_dir = data_dir
        self.image_scale = image_scale
        self.device = device
        self.images_data = []

        # 确定图像文件夹
        if self.image_scale > 1:
            images_dir = os.path.join(self.data_dir, f"images_{self.image_scale}")
            if not os.path.exists(images_dir):
                images_dir = os.path.join(self.data_dir, "images")
        else:
            images_dir = os.path.join(self.data_dir, "images")

        if not os.path.exists(images_dir):
            raise ValueError(f"图像文件夹不存在: {images_dir}")

        print(f"从 {images_dir} 加载图像...")

        # 获取所有图像文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])

        if not image_files:
            raise ValueError(f"在 {images_dir} 中没有找到图像文件")

        # 排序图像文件
        image_files = sorted(image_files)
        print(f"找到 {len(image_files)} 张图像")

        # 读取第一张图像获取尺寸
        first_img_path = os.path.join(images_dir, image_files[0])
        first_img = Image.open(first_img_path)
        self.img_width, self.img_height = first_img.size

        # 根据缩放调整尺寸
        if image_scale > 1:
            self.img_width = self.img_width // image_scale
            self.img_height = self.img_height // image_scale

        print(f"图像尺寸: {self.img_width} x {self.img_height}")

        # 加载所有图像并创建简单的相机位姿
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(images_dir, img_file)
            img = Image.open(img_path)

            # 调整图像大小
            if image_scale > 1:
                new_size = (self.img_width, self.img_height)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # 转换为张量
            img_array = np.array(img).astype(np.float32) / 255.0

            # 如果是灰度图像，转换为RGB
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array, img_array, img_array], axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]  # 丢弃alpha通道

            img_tensor = torch.from_numpy(img_array).float()

            # 创建简单的相机位姿（圆形轨迹）
            num_images = len(image_files)
            radius = 4.0
            angle = 2 * np.pi * i / num_images
            height = 1.5

            # 相机位置
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = height

            # 看向中心点 (0, 0, 0)
            camera_pos = np.array([x, y, z])
            target = np.array([0, 0, 0])

            # 计算相机方向
            forward = target - camera_pos
            forward = forward / np.linalg.norm(forward)

            # 简单的相机矩阵（简化版）
            # 使用lookat矩阵
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            # 构建旋转矩阵
            R = np.eye(3)
            R[0, :] = right
            R[1, :] = up
            R[2, :] = -forward

            # 构建位姿矩阵
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = -R @ camera_pos  # 相机在世界坐标系中的位置

            # 内参矩阵（简化）
            focal_length = max(self.img_width, self.img_height) * 1.2
            cx, cy = self.img_width / 2, self.img_height / 2
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ])

            # 创建CameraInfo（简化）
            cam_info = CameraInfo(
                image_id=i,
                qvec=np.array([1.0, 0.0, 0.0, 0.0]),  # 无旋转
                tvec=-camera_pos,  # 平移
                camera_id=1,
                image_name=img_file
            )

            self.images_data.append({
                'image': img_tensor,
                'pose': pose,
                'intrinsics': K,
                'info': cam_info,
                'image_name': img_file
            })

        print(f"数据集加载完成，共 {len(self.images_data)} 张图像")

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        data = self.images_data[idx]

        return {
            'image': data['image'].to(self.device),
            'pose': torch.from_numpy(data['pose']).float().to(self.device),
            'intrinsics': torch.from_numpy(data['intrinsics']).float().to(self.device),
            'image_name': data['image_name']
        }


# 为了兼容性，保留ColmapDataset名称但使用SimpleDataset实现
ColmapDataset = SimpleDataset