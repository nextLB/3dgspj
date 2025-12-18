import torch
import numpy as np
import os
from PIL import Image
import cv2
import json
from dataclasses import dataclass
from typing import Optional, Tuple
import torch.nn.functional as F


@dataclass
class Camera:
    """相机参数类"""
    R: torch.Tensor  # 3x3旋转矩阵
    T: torch.Tensor  # 3x1平移向量
    fx: float  # 焦距x
    fy: float  # 焦距y
    cx: float  # 主点x
    cy: float  # 主点y
    width: int  # 图像宽度
    height: int  # 图像高度
    image_name: str  # 图像名称
    image_path: Optional[str] = None
    original_image: Optional[torch.Tensor] = None
    bounds: Optional[torch.Tensor] = None  # [近平面, 远平面]

    def to(self, device):
        """将相机参数移到指定设备"""
        self.R = self.R.to(device)
        self.T = self.T.to(device)
        if self.original_image is not None:
            self.original_image = self.original_image.to(device)
        if self.bounds is not None:
            self.bounds = self.bounds.to(device)
        return self

    def get_projection_matrix(self):
        """获取投影矩阵"""
        proj = torch.zeros((3, 4), device=self.R.device, dtype=self.R.dtype)
        proj[0, 0] = self.fx
        proj[1, 1] = self.fy
        proj[0, 2] = self.cx
        proj[1, 2] = self.cy
        proj[2, 2] = 1.0
        return proj

    def get_view_matrix(self):
        """获取视图矩阵"""
        view = torch.eye(4, device=self.R.device, dtype=self.R.dtype)
        view[:3, :3] = self.R.transpose(0, 1)
        view[:3, 3] = -self.R.transpose(0, 1) @ self.T.squeeze()
        return view


class MipNeRF360Dataset:
    """Mip-NeRF 360数据集加载器"""

    def __init__(self, base_path, scene, resolution=1, device='cuda', split='train'):
        """
        初始化数据集

        Args:
            base_path: 数据集根路径 (e.g., ./archive/360_v2)
            scene: 场景名称 (e.g., bicycle, bonsai)
            resolution: 图像分辨率等级 (1, 2, 4, 8)
            device: 设备
            split: 数据集分割 (train/test)
        """
        self.data_path = os.path.join(base_path, scene)
        self.scene = scene
        self.resolution = resolution
        self.device = device
        self.split = split

        print(f"Loading Mip-NeRF 360 dataset: {scene}")
        print(f"Data path: {self.data_path}")

        # 加载相机位姿
        self.poses, self.bounds, self.image_paths = self.load_poses_and_images()

        # 创建相机对象
        self.cameras = self.create_cameras()

        # 设置训练和测试分割
        self.setup_split()

        print(f"Loaded {len(self.cameras)} cameras")
        print(f"Training: {len(self.train_indices)}, Test: {len(self.test_indices)}")

    def load_poses_and_images(self):
        """加载相机位姿和图像路径"""
        # 加载poses_bounds.npy文件
        poses_bounds_path = os.path.join(self.data_path, 'poses_bounds.npy')
        if not os.path.exists(poses_bounds_path):
            raise FileNotFoundError(f"poses_bounds.npy not found at {poses_bounds_path}")

        poses_bounds = np.load(poses_bounds_path)

        # 解析位姿和边界
        # poses_bounds形状: (N, 17)，其中前15个是位姿(3x5)，后2个是边界
        poses = poses_bounds[:, :-2].reshape([-1, 3, 5])  # 形状: (N, 3, 5)
        bounds = poses_bounds[:, -2:]  # 形状: (N, 2)

        # 加载图像路径
        img_dir_name = f"images_{self.resolution}" if self.resolution > 1 else "images"
        img_dir = os.path.join(self.data_path, img_dir_name)

        if not os.path.exists(img_dir):
            # 尝试其他可能的目录名
            img_dir = os.path.join(self.data_path, "images")
            if not os.path.exists(img_dir):
                raise FileNotFoundError(f"Image directory not found: {img_dir}")

        # 获取所有图像文件
        image_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))
        ])

        # 确保图像数量与位姿数量匹配
        if len(image_files) != len(poses):
            print(f"Warning: Mismatch between poses ({len(poses)}) and images ({len(image_files)})")
            # 使用较小的数量
            min_len = min(len(poses), len(image_files))
            poses = poses[:min_len]
            bounds = bounds[:min_len]
            image_files = image_files[:min_len]

        image_paths = [os.path.join(img_dir, f) for f in image_files]

        return poses, bounds, image_paths

    def create_cameras(self):
        """从位姿创建相机对象"""
        cameras = []

        for i, (pose, bounds, img_path) in enumerate(zip(self.poses, self.bounds, self.image_paths)):
            # 解析位姿矩阵
            # pose形状: (3, 5)，其中前3列是旋转矩阵，第4列是平移，第5列是焦距和主点
            R = pose[:, :3]  # 旋转矩阵 (3, 3)
            T = pose[:, 3:4]  # 平移向量 (3, 1)

            # 第5列: [focal_length, cx, cy]
            intrinsics = pose[:, 4]

            # 加载图像获取尺寸
            try:
                with Image.open(img_path) as img:
                    width, height = img.size

                # 根据分辨率缩放图像尺寸
                if self.resolution > 1:
                    width = width // self.resolution
                    height = height // self.resolution

                # 创建相机对象
                camera = Camera(
                    R=torch.tensor(R, dtype=torch.float32),
                    T=torch.tensor(T, dtype=torch.float32),
                    fx=abs(intrinsics[0]) / self.resolution,
                    fy=abs(intrinsics[0]) / self.resolution,  # 假设fx=fy
                    cx=width / 2,
                    cy=height / 2,
                    width=width,
                    height=height,
                    image_name=os.path.basename(img_path),
                    image_path=img_path,
                    bounds=torch.tensor(bounds, dtype=torch.float32)
                )

                # 加载图像
                camera.original_image = self.load_image(img_path, (height, width))

                cameras.append(camera)

            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")

        return cameras

    def load_image(self, image_path, target_size=None):
        """加载并预处理图像"""
        # 加载图像
        image = Image.open(image_path)

        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 调整大小
        if target_size is not None:
            image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)

        # 转换为numpy数组并归一化到[0, 1]
        image_np = np.array(image).astype(np.float32) / 255.0

        # 转换为PyTorch张量并调整维度顺序
        image_tensor = torch.tensor(image_np, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW

        return image_tensor

    def setup_split(self):
        """设置训练和测试分割"""
        n_cameras = len(self.cameras)

        # 使用80%的数据进行训练，20%进行测试
        train_ratio = 0.8
        n_train = int(n_cameras * train_ratio)

        indices = np.arange(n_cameras)
        np.random.seed(42)  # 固定随机种子以确保可重复性
        np.random.shuffle(indices)

        self.train_indices = indices[:n_train]
        self.test_indices = indices[n_train:]

        if self.split == 'train':
            self.current_indices = self.train_indices
        else:
            self.current_indices = self.test_indices

    def __len__(self):
        """返回数据集中相机的数量"""
        return len(self.current_indices)

    def get_camera(self, idx):
        """获取指定索引的相机"""
        actual_idx = self.current_indices[idx]
        camera = self.cameras[actual_idx]
        return camera.to(self.device)

    def get_random_camera(self):
        """随机获取一个训练相机"""
        if self.split != 'train':
            raise ValueError("get_random_camera only available for training split")

        idx = np.random.choice(self.train_indices)
        camera = self.cameras[idx]
        return camera.to(self.device)

    def get_test_camera(self, idx):
        """获取测试相机"""
        if idx >= len(self.test_indices):
            idx = idx % len(self.test_indices)

        actual_idx = self.test_indices[idx]
        camera = self.cameras[actual_idx]
        return camera.to(self.device)

    def get_all_cameras(self):
        """获取所有相机"""
        return [camera.to(self.device) for camera in self.cameras]

    def compute_mean_radius(self):
        """计算场景的平均半径（用于相机轨迹生成）"""
        positions = []
        for camera in self.cameras:
            # 相机在世界坐标系中的位置: -R^T * T
            pos = -camera.R.transpose(0, 1) @ camera.T
            positions.append(pos.squeeze().numpy())

        positions = np.array(positions)
        center = np.mean(positions, axis=0)
        radii = np.linalg.norm(positions - center, axis=1)

        return np.mean(radii), center

    def save_camera_info(self, output_path):
        """保存相机信息到JSON文件"""
        camera_info = []
        for i, camera in enumerate(self.cameras):
            info = {
                'index': i,
                'image_name': camera.image_name,
                'image_path': camera.image_path,
                'R': camera.R.tolist(),
                'T': camera.T.squeeze().tolist(),
                'fx': camera.fx,
                'fy': camera.fy,
                'cx': camera.cx,
                'cy': camera.cy,
                'width': camera.width,
                'height': camera.height,
                'bounds': camera.bounds.tolist() if camera.bounds is not None else None
            }
            camera_info.append(info)

        with open(output_path, 'w') as f:
            json.dump(camera_info, f, indent=2)

        print(f"Camera info saved to {output_path}")


# 辅助函数
def look_at_matrix(eye, center, up=None):
    """生成LookAt矩阵"""
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], device=eye.device)

    forward = center - eye
    forward = forward / torch.norm(forward)

    right = torch.cross(forward, up)
    right = right / torch.norm(right)

    up = torch.cross(right, forward)

    R = torch.stack([right, up, -forward], dim=0)
    return R


def fov_to_focal_length(fov_degrees, image_width):
    """将视场角转换为焦距"""
    fov_rad = fov_degrees * np.pi / 180.0
    focal_length = 0.5 * image_width / np.tan(0.5 * fov_rad)
    return focal_length