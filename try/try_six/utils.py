#!/usr/bin/env python3
"""
工具函数
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(path: str, scale: int = 1) -> np.ndarray:
    """加载图像"""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法加载图像: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if scale != 1:
        h, w = image.shape[:2]
        new_h, new_w = h // scale, w // scale
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


def save_image(image: np.ndarray, path: str):
    """保存图像"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def visualize_cameras(cameras: List[Dict], output_path: str = None):
    """可视化相机位置"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    positions = []
    for cam in cameras:
        # 相机位置: C = -R^T * t
        C = -cam["R"].T @ cam["t"]
        positions.append(C)

        # 相机朝向
        forward = cam["R"][2, :]  # Z轴方向
        ax.quiver(C[0], C[1], C[2],
                  forward[0], forward[1], forward[2],
                  length=0.5, normalize=True, color='r', alpha=0.5)

    positions = np.array(positions)

    # 绘制相机位置
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c='b', marker='o', s=50, alpha=0.8)

    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions')

    # 设置相等的纵横比
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                          positions[:, 1].max() - positions[:, 1].min(),
                          positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0

    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"相机可视化已保存到: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_point_cloud(points: np.ndarray, colors: np.ndarray = None,
                          output_path: str = None):
    """可视化点云"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if colors is None:
        colors = np.ones_like(points)

    # 绘制点云
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors, marker='.', s=1, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud')

    # 设置相等的纵横比
    if len(points) > 0:
        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                              points[:, 1].max() - points[:, 1].min(),
                              points[:, 2].max() - points[:, 2].min()]).max() / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"点云可视化已保存到: {output_path}")
    else:
        plt.show()

    plt.close()


def compare_images(img1: np.ndarray, img2: np.ndarray,
                   title1: str = "Image 1", title2: str = "Image 2",
                   output_path: str = None):
    """比较两张图像"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(img1)
    axes[0].set_title(title1)
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].set_title(title2)
    axes[1].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图像比较已保存到: {output_path}")
    else:
        plt.show()

    plt.close()


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def create_video_from_images(image_dir: str, output_path: str, fps: int = 30):
    """从图像创建视频"""
    import imageio

    image_files = sorted([f for f in os.listdir(image_dir)
                          if f.endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print(f"在 {image_dir} 中没有找到图像")
        return

    # 读取第一张图像获取尺寸
    first_image = imageio.imread(os.path.join(image_dir, image_files[0]))
    height, width = first_image.shape[:2]

    # 创建视频写入器
    writer = imageio.get_writer(output_path, fps=fps)

    for image_file in tqdm(image_files, desc="创建视频"):
        image_path = os.path.join(image_dir, image_file)
        image = imageio.imread(image_path)
        writer.append_data(image)

    writer.close()
    print(f"视频已保存到: {output_path}")


def setup_seed(seed: int = 42):
    """设置随机种子"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """获取设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用CPU")

    return device


def memory_usage():
    """显示内存使用情况"""
    if torch.cuda.is_available():
        print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"GPU内存缓存: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
        print(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")