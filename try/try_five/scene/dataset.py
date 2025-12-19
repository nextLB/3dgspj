import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json
import random
from torch.utils.data import Dataset
from utils.colmap_utils import read_cameras_binary, read_images_binary
from utils.dataset_utils import getWorld2View2, get_projection_matrix


class SceneDataset(Dataset):
    """
    场景数据集类，支持COLMAP格式
    """

    def __init__(self, source_path, images, resolution, data_device,
                 white_background=False, eval=False):
        super().__init__()

        self.source_path = Path(source_path)
        self.images_folder = images
        self.resolution = resolution
        self.data_device = data_device
        self.white_background = white_background
        self.eval = eval

        # 加载数据
        self.load_data()

        # 设置相机参数
        self.setup_cameras()

        print(f"数据集加载完成，共有 {len(self.image_paths)} 张图像")

    def load_data(self):
        """加载数据"""
        # 图像路径
        images_path = self.source_path / self.images_folder
        self.image_paths = sorted([p for p in images_path.iterdir()
                                   if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']])

        # 加载相机参数
        sparse_path = self.source_path / "sparse" / "0"
        if sparse_path.exists():
            self.cameras = read_cameras_binary(sparse_path / "cameras.bin")
            self.images_data = read_images_binary(sparse_path / "images.bin")

            # 创建图像名到数据的映射
            self.image_name_to_data = {img.name: img for img in self.images_data.values()}
        else:
            self.cameras = None
            self.images_data = None
            self.image_name_to_data = {}

        # 加载图像
        self.images = []
        self.masks = []  # 如果有mask的话

        for img_path in self.image_paths:
            img = Image.open(img_path)

            if self.resolution > 0:
                # 调整分辨率
                width, height = img.size
                scale = self.resolution / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)

            img_array = np.array(img) / 255.0

            if img_array.shape[-1] == 4:  # RGBA
                rgba = img_array
                rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:]) * (1.0 if self.white_background else 0.0)
                mask = rgba[..., 3:]
                self.masks.append(mask)
            else:
                rgb = img_array
                mask = np.ones(rgb.shape[:2])
                self.masks.append(mask)

            self.images.append(rgb)

        self.images = np.stack(self.images)
        self.masks = np.stack(self.masks)

        # 转换为张量
        self.images_tensor = torch.from_numpy(self.images).float().to(self.data_device)
        self.masks_tensor = torch.from_numpy(self.masks).float().to(self.data_device)

    def setup_cameras(self):
        """设置相机参数"""
        self.camera_list = []
        self.camera_extent = 1.0

        if self.cameras is not None:
            # 使用COLMAP相机参数
            for i, img_path in enumerate(self.image_paths):
                img_name = img_path.name

                if img_name in self.image_name_to_data:
                    img_data = self.image_name_to_data[img_name]
                    cam_id = img_data.camera_id
                    cam = self.cameras[cam_id]

                    # 获取当前图像的实际尺寸
                    current_img = self.images[i]
                    height, width = current_img.shape[:2]

                    # 如果COLMAP的相机尺寸与实际图像尺寸不同，调整内参
                    scale_x = width / cam.width
                    scale_y = height / cam.height

                    # 根据相机模型提取内参
                    if cam.model == "SIMPLE_PINHOLE":
                        fx = fy = cam.params[0] * scale_x  # 缩放焦距
                        cx = cam.params[1] * scale_x
                        cy = cam.params[2] * scale_y
                    elif cam.model == "PINHOLE":
                        fx = cam.params[0] * scale_x
                        fy = cam.params[1] * scale_y
                        cx = cam.params[2] * scale_x
                        cy = cam.params[3] * scale_y
                    else:
                        # 简化处理，使用默认值
                        fx = fy = width
                        cx = width / 2
                        cy = height / 2

                    # 计算视野角
                    fovx = 2 * np.arctan(width / (2 * fx))
                    fovy = 2 * np.arctan(height / (2 * fy))

                    # 位姿
                    R = img_data.qvec2rotmat()
                    t = img_data.tvec

                    # 世界到相机的变换矩阵
                    world_view_transform = torch.tensor(getWorld2View2(R, t)).transpose(0, 1).cuda()

                    # 投影矩阵
                    projection_matrix = get_projection_matrix(
                        znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy
                    ).transpose(0, 1).cuda()

                    # 完整的相机到世界的变换矩阵
                    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
                        projection_matrix.unsqueeze(0))).squeeze(0)

                    camera_info = {
                        'image': self.images_tensor[i],
                        'gt_alpha_mask': self.masks_tensor[i],
                        'image_name': img_name,
                        'uid': i,
                        'width': width,
                        'height': height,
                        'FovY': fovy,
                        'FovX': fovx,
                        'world_view_transform': world_view_transform,
                        'projection_matrix': projection_matrix,
                        'full_proj_transform': full_proj_transform,
                        'camera_center': world_view_transform.inverse()[3, :3],
                        'fx': fx,
                        'fy': fy,
                        'cx': cx,
                        'cy': cy
                    }

                    self.camera_list.append(camera_info)

        else:
            # 使用默认相机参数
            for i, img in enumerate(self.images_tensor):
                height, width = img.shape[:2]

                # 默认相机参数 - 根据实际图像尺寸调整
                fovx = 0.857556  # 约49度
                fovy = fovx * height / width

                # 计算默认焦距
                fx = width / (2 * np.tan(fovx / 2))
                fy = height / (2 * np.tan(fovy / 2))

                # 默认位姿（单位矩阵）
                world_view_transform = torch.eye(4).cuda()
                projection_matrix = get_projection_matrix(
                    znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy
                ).transpose(0, 1).cuda()
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
                    projection_matrix.unsqueeze(0))).squeeze(0)

                camera_info = {
                    'image': img,
                    'gt_alpha_mask': self.masks_tensor[i],
                    'image_name': self.image_paths[i].name,
                    'uid': i,
                    'width': width,
                    'height': height,
                    'FovY': fovy,
                    'FovX': fovx,
                    'world_view_transform': world_view_transform,
                    'projection_matrix': projection_matrix,
                    'full_proj_transform': full_proj_transform,
                    'camera_center': torch.tensor([0, 0, 0]).cuda(),
                    'fx': fx,
                    'fy': fy,
                    'cx': width / 2,
                    'cy': height / 2
                }

                self.camera_list.append(camera_info)

        # 计算场景范围
        self.camera_extent = self.compute_extent()
    def compute_extent(self):
        """计算场景范围"""
        if not self.camera_list:
            return 1.0

        # 使用相机中心位置计算范围
        centers = torch.stack([cam['camera_center'] for cam in self.camera_list])
        max_dist = torch.max(torch.norm(centers - centers.mean(dim=0), dim=1))

        return max(1.0, float(max_dist) * 1.1)

    def __len__(self):
        return len(self.camera_list)

    def __getitem__(self, idx):
        return self.camera_list[idx]

    def get_train_cameras(self):
        """获取训练相机"""
        return self.camera_list

    def get_test_cameras(self):
        """获取测试相机"""
        # 这里可以返回不同的相机子集用于测试
        if self.eval and len(self.camera_list) > 10:
            # 取后10%作为测试
            test_size = max(1, len(self.camera_list) // 10)
            return self.camera_list[-test_size:]
        return self.camera_list