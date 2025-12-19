import numpy as np
import torch
import cv2
from PIL import Image
import json
from pathlib import Path


def load_nerf_poses(poses_bounds_path):
    """
    加载NeRF格式的poses_bounds.npy文件
    """
    data = np.load(poses_bounds_path)

    poses = data[:, :-2].reshape([-1, 3, 5])  # 最后两列是边界
    bounds = data[:, -2:]

    # 分离内参和外参
    hwf = poses[:, :, 4]
    poses = poses[:, :, :4]

    # 将OpenGL坐标系转换为COLMAP坐标系
    poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)

    return poses, hwf, bounds


def load_images(image_folder, resolution=-1):
    """
    加载图像文件夹中的所有图像
    """
    image_files = sorted([f for f in Path(image_folder).iterdir()
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']])

    images = []
    for img_path in image_files:
        img = Image.open(img_path)

        if resolution > 0:
            # 调整分辨率
            width, height = img.size
            scale = resolution / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)

        img_array = np.array(img) / 255.0
        if img_array.shape[-1] == 4:  # RGBA
            img_array = img_array[..., :3] * img_array[..., 3:] + (1 - img_array[..., 3:])

        images.append(img_array)

    return np.stack(images)


def create_train_val_split(num_images, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    创建训练集、验证集、测试集划分
    """
    np.random.seed(seed)
    indices = np.arange(num_images)
    np.random.shuffle(indices)

    num_val = int(num_images * val_ratio)
    num_test = int(num_images * test_ratio)
    num_train = num_images - num_val - num_test

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    return train_indices, val_indices, test_indices


def fov2focal(fov, pixels):
    """
    视野角转换为焦距
    """
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal, pixels):
    """
    焦距转换为视野角
    """
    return 2 * np.arctan(pixels / (2 * focal))


def get_projection_matrix(znear, zfar, fovX, fovY):
    """
    获取投影矩阵
    """
    tanHalfFovY = np.tan((fovY / 2))
    tanHalfFovX = np.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def getWorld2View(R, t):
    """
    获取世界坐标系到相机坐标系的变换矩阵
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """
    获取带缩放和平移的世界坐标系到相机坐标系的变换矩阵
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)

    return np.float32(Rt)