import numpy as np
import os
import imageio.v2 as imageio
import torch
from torch.utils.data import Dataset
import cv2


class MipNeRF360Dataset(Dataset):
    """
    加载Mip-NeRF 360数据集的类。
    数据集结构：<data_path>/<scene>/images/，以及poses_bounds.npy文件。
    """

    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.device = torch.device(cfg.device)

        # 构建场景的完整路径
        self.base_path = os.path.join(cfg.data_path, cfg.scene)
        self.image_path = os.path.join(self.base_path, cfg.images)
        self.pose_path = os.path.join(self.base_path, 'poses_bounds.npy')

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载所有图像和对应的相机参数。"""
        # 1. 加载位姿和边界数据
        poses_bounds = np.load(self.pose_path)  # 形状应为 [N, 17]
        # 前15个数字是3x5的相机到世界矩阵（按行优先），最后2个是近、远平面距离
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # 重塑为 [N, 3, 5]
        bounds = poses_bounds[:, -2:]  # [N, 2]

        # 2. 解析内参和位姿
        # 内参矩阵K（3x3）是从poses的最后一列（3个值）和图像尺寸构建的
        self.image_paths = sorted([os.path.join(self.image_path, f) for f in os.listdir(self.image_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        # 使用第一张图像获取尺寸
        sample_img = imageio.imread(self.image_paths[0])
        H, W = sample_img.shape[:2]
        scale_factor = 0.1
        H = int(H * scale_factor)
        W = int(W * scale_factor)

        # 焦距通常由poses提供（fx, fy），但Mip-NeRF格式将其与位姿存储在一起。
        # poses的第三列第0-2行是 (fx, fy, ?)，我们假设图像中心在 (W/2, H/2)
        focal = poses[0, 2, 0]  # 取第一个相机的fx作为焦距估计
        # 相应地缩放内参矩阵
        focal = focal * scale_factor

        # 3. 转换位姿格式：从相机到世界 -> 世界到相机，并转换为4x4矩阵
        num_images = len(self.image_paths)
        self.poses = []
        self.Ks = []

        for i in range(num_images):
            pose_raw = poses[i]  # [3, 5]
            # 提取旋转矩阵（前3列）和平移向量（第4列）
            rotation = pose_raw[:3, :3]  # [3, 3]
            translation = pose_raw[:3, 3]  # [3]
            # 构建相机到世界的4x4矩阵
            c2w = np.eye(4)
            c2w[:3, :3] = rotation
            c2w[:3, 3] = translation
            # 我们需要世界到相机的矩阵 (w2c)，所以求逆
            w2c = np.linalg.inv(c2w)
            self.poses.append(torch.from_numpy(w2c).float())

            # 构建内参矩阵K
            K = torch.eye(3).float()
            K[0, 0] = focal
            K[1, 1] = focal
            K[0, 2] = W / 2.0
            K[1, 2] = H / 2.0
            self.Ks.append(K)

        self.poses = torch.stack(self.poses).to(self.device)  # [N, 4, 4]
        self.Ks = torch.stack(self.Ks).to(self.device)  # [N, 3, 3]
        self.H, self.W = H, W
        self.focal = focal
        self.bounds = torch.from_numpy(bounds).float().to(self.device)  # [N, 2]

        # 4. 加载所有图像（在需要时惰性加载）
        self.all_images = []
        for path in self.image_paths:
            img = imageio.imread(path)
            # 转换为RGB，归一化到[0,1]
            if img.shape[2] == 4:
                img = img[:, :, :3]  # 丢弃alpha通道
            img = torch.from_numpy(img).float() / 255.0
            self.all_images.append(img)
        self.all_images = torch.stack(self.all_images).to(self.device)  # [N, H, W, 3]

        print(f"Loaded {num_images} images for scene '{self.cfg.scene}' with resolution {H}x{W}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """返回单个图像及其对应的相机参数。"""
        return {
            'image': self.all_images[idx].clone(),  # 使用.clone()确保独立副本
            'pose': self.poses[idx].clone(),
            'K': self.Ks[idx].clone(),
            'bounds': self.bounds[idx].clone(),
            'idx': idx
        }

    def get_all_cameras(self):
        """获取所有相机参数，用于推理或评估。"""
        return self.poses, self.Ks, self.H, self.W


def get_dataset(cfg, split='train'):
    return MipNeRF360Dataset(cfg, split)


if __name__ == "__main__":
    from config import get_config

    cfg = get_config()
    dataset = MipNeRF360Dataset(cfg)
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Pose shape: {sample['pose'].shape}")



