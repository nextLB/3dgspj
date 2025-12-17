import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
from pathlib import Path
import cv2


class Camera:
    """相机类"""

    def __init__(self, camera_data):
        self.id = camera_data["id"]
        self.img_name = camera_data["img_name"]
        self.width = camera_data["width"]
        self.height = camera_data["height"]

        # 内参
        self.fx = camera_data["fx"]
        self.fy = camera_data["fy"]
        self.cx = camera_data["cx"]
        self.cy = camera_data["cy"]

        # 畸变参数
        self.k1 = camera_data.get("k1", 0)
        self.k2 = camera_data.get("k2", 0)
        self.p1 = camera_data.get("p1", 0)
        self.p2 = camera_data.get("p2", 0)

        # 外参
        self.position = np.array(camera_data["position"])
        self.rotation = np.array(camera_data["rotation"])

    def get_projection_matrix(self):
        """获取投影矩阵"""
        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

        R = self.rotation
        t = -R @ self.position

        RT = np.hstack([R, t.reshape(-1, 1)])

        return K @ RT


class SceneDataset(Dataset):
    """场景数据集"""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        # 加载相机数据
        with open(self.data_dir / "cameras.json", "r") as f:
            cameras_data = json.load(f)

        self.cameras = []
        self.images = []

        for cam_data in cameras_data:
            camera = Camera(cam_data)
            self.cameras.append(camera)

            # 加载图像
            img_path = self.data_dir / cam_data["img_name"]
            if img_path.exists():
                image = Image.open(img_path)
                image = np.array(image) / 255.0  # 归一化到[0,1]
                self.images.append(image)
            else:
                print(f"警告: 图像 {img_path} 不存在")
                self.images.append(np.zeros((camera.height, camera.width, 3)))

        # 加载点云
        self.pointcloud = self._load_pointcloud()

    def _load_pointcloud(self):
        """加载点云"""
        ply_path = self.data_dir / "pointcloud.ply"
        if ply_path.exists():
            # 使用open3d或plyfile加载点云
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(str(ply_path))
                return np.asarray(pcd.points)
            except:
                print("无法加载点云，使用随机初始化")

        # 如果没有点云，返回空数组
        return np.zeros((0, 3))

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx]).permute(2, 0, 1)  # [C, H, W]
        camera = self.cameras[idx]

        return {
            'image': image,
            'camera': camera,
            'camera_id': camera.id
        }

    def get_all_cameras(self):
        """获取所有相机"""
        return self.cameras

    def get_all_images(self):
        """获取所有图像"""
        return self.images


