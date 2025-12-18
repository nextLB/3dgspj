import torch
import numpy as np
import os
import cv2
import json
import math
from PIL import Image
from typing import Optional, Tuple, List, Dict
import matplotlib.pyplot as plt
from datetime import datetime


def save_image(image_tensor: torch.Tensor,
               save_path: str,
               normalize: bool = True,
               value_range: Tuple[float, float] = (0.0, 1.0)):
    """
    保存图像张量到文件

    Args:
        image_tensor: 图像张量 (C, H, W) 或 (H, W, C)
        save_path: 保存路径
        normalize: 是否归一化到[0, 255]
        value_range: 输入值范围
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 转换为numpy数组
    if isinstance(image_tensor, torch.Tensor):
        image_np = image_tensor.detach().cpu().numpy()
    else:
        image_np = image_tensor

    # 调整维度顺序
    if len(image_np.shape) == 3:
        if image_np.shape[0] == 3 or image_np.shape[0] == 1:  # CHW -> HWC
            image_np = image_np.transpose(1, 2, 0)
        elif image_np.shape[2] == 3 or image_np.shape[2] == 1:  # HWC
            pass
        else:
            raise ValueError(f"Unexpected image shape: {image_np.shape}")

    # 归一化到[0, 255]
    if normalize:
        vmin, vmax = value_range
        if vmin != 0 or vmax != 1:
            image_np = (image_np - vmin) / (vmax - vmin)

        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

    # 保存图像
    if image_np.shape[2] == 1:  # 灰度图
        image_np = image_np.squeeze(2)
        Image.fromarray(image_np).save(save_path)
    else:  # RGB图
        # OpenCV使用BGR顺序
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image_bgr)

    print(f"Image saved to {save_path}")


def save_depth_image(depth_tensor: torch.Tensor,
                     save_path: str,
                     normalize: bool = True,
                     colormap: str = 'viridis'):
    """
    保存深度图

    Args:
        depth_tensor: 深度张量 (1, H, W) 或 (H, W)
        save_path: 保存路径
        normalize: 是否归一化
        colormap: 颜色映射
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 转换为numpy数组
    if isinstance(depth_tensor, torch.Tensor):
        depth_np = depth_tensor.detach().cpu().numpy()
    else:
        depth_np = depth_tensor

    # 调整维度
    if len(depth_np.shape) == 3:
        depth_np = depth_np.squeeze(0)

    # 归一化
    if normalize:
        valid_depths = depth_np[depth_np > 0]
        if len(valid_depths) > 0:
            vmin, vmax = valid_depths.min(), valid_depths.max()
        else:
            vmin, vmax = 0, 1

        if vmax > vmin:
            depth_np = (depth_np - vmin) / (vmax - vmin)
        depth_np = np.clip(depth_np, 0, 1)

    # 应用颜色映射
    cmap = plt.get_cmap(colormap)
    depth_colored = (cmap(depth_np)[:, :, :3] * 255).astype(np.uint8)

    # 保存
    depth_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, depth_bgr)

    print(f"Depth image saved to {save_path}")


def load_point_cloud(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从PLY文件加载点云

    Returns:
        points: (N, 3) 点坐标
        colors: (N, 3) RGB颜色 [0, 1]
    """
    try:
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)

        # 提取顶点数据
        vertices = plydata['vertex']

        # 获取坐标
        points = np.stack([
            vertices['x'],
            vertices['y'],
            vertices['z']
        ], axis=1)

        # 获取颜色（如果有）
        if 'red' in vertices.dtype.names:
            colors = np.stack([
                vertices['red'],
                vertices['green'],
                vertices['blue']
            ], axis=1) / 255.0
        elif 'r' in vertices.dtype.names:
            colors = np.stack([
                vertices['r'],
                vertices['g'],
                vertices['b']
            ], axis=1) / 255.0
        else:
            # 使用默认颜色
            colors = np.ones_like(points) * 0.7

        print(f"Loaded {points.shape[0]} points from {ply_path}")
        return points, colors

    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return np.zeros((0, 3)), np.zeros((0, 3))


def create_checkpoint(gaussian_model, optimizer, iteration, args, save_path: str):
    """创建检查点"""
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': gaussian_model.capture_state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path} (iteration {iteration})")


def load_checkpoint(checkpoint_path: str, device='cuda'):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Iteration: {checkpoint.get('iteration', 'unknown')}")
    print(f"Timestamp: {checkpoint.get('timestamp', 'unknown')}")

    return checkpoint


def generate_report(output_dir: str, args, gaussian_model, dataset):
    """生成重建报告"""
    report_path = os.path.join(output_dir, "reconstruction_report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("3D Gaussian Splatting Reconstruction Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Scene: {args.scene}\n")
        f.write(f"Dataset: Mip-NeRF 360\n")
        f.write(f"Output Directory: {output_dir}\n\n")

        f.write("Training Configuration:\n")
        f.write("-" * 40 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("Model Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of Gaussians: {gaussian_model.get_xyz.shape[0]}\n")
        f.write(f"SH Degree: {gaussian_model.active_sh_degree}/{gaussian_model.max_sh_degree}\n")
        f.write(f"Spatial LR Scale: {gaussian_model.spatial_lr_scale}\n\n")

        f.write("Dataset Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Images: {len(dataset.cameras)}\n")
        f.write(f"Training Images: {len(dataset.train_indices)}\n")
        f.write(f"Test Images: {len(dataset.test_indices)}\n")
        f.write(f"Image Resolution: {dataset.cameras[0].width}x{dataset.cameras[0].height}\n\n")

        f.write("Files Generated:\n")
        f.write("-" * 40 + "\n")
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(output_dir, '').count(os.sep)
            indent = '  ' * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = '  ' * (level + 1)
            for file in files:
                if file.endswith(('.pth', '.ply', '.png', '.json', '.txt')):
                    f.write(f"{subindent}{file}\n")

    print(f"Report saved to {report_path}")


def visualize_gaussians(gaussian_model, camera, save_path: str):
    """可视化高斯分布"""
    import matplotlib.pyplot as plt

    xyz = gaussian_model.get_xyz.detach().cpu().numpy()
    scaling = gaussian_model.get_scaling.detach().cpu().numpy()
    colors = gaussian_model.sh_to_rgb(gaussian_model._features_dc[:, :, 0]).detach().cpu().numpy()

    # 创建3D散点图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 根据缩放大小调整点的大小
    sizes = np.mean(scaling, axis=1) * 100
    sizes = np.clip(sizes, 1, 100)

    # 绘制点
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                         c=colors, s=sizes, alpha=0.6, edgecolors='w', linewidth=0.5)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Gaussians ({xyz.shape[0]} points)')

    # 添加相机位置
    if camera is not None:
        cam_pos = (-camera.R.T @ camera.T).squeeze().cpu().numpy()
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2],
                   c='red', s=200, marker='^', label='Camera')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Gaussian visualization saved to {save_path}")


def compute_memory_stats(device='cuda'):
    """计算GPU内存统计"""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(device) / 1e9

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated
        }
    else:
        return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0}


def setup_logging(log_dir: str, experiment_name: str):
    """设置日志记录"""
    import logging

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{experiment_name}.log")

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    return logger


def create_video_from_images(image_dir: str,
                             output_path: str,
                             fps: int = 30,
                             pattern: str = "*.png"):
    """从图像序列创建视频"""
    import glob

    image_files = sorted(glob.glob(os.path.join(image_dir, pattern)))

    if len(image_files) == 0:
        print(f"No images found in {image_dir} with pattern {pattern}")
        return

    # 读取第一张图像获取尺寸
    first_image = cv2.imread(image_files[0])
    height, width = first_image.shape[:2]

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 写入所有图像
    for image_file in image_files:
        image = cv2.imread(image_file)
        video_writer.write(image)

    video_writer.release()
    print(f"Video saved to {output_path} ({len(image_files)} frames)")


def compute_metrics(rendered_images, gt_images, masks=None):
    """计算图像质量指标"""
    metrics = {
        'psnr': [],
        'ssim': [],
        'lpips': []
    }

    # 创建损失函数用于计算指标
    loss_fn = LossFunction(lambda_dssim=0.2)

    for i, (rendered, gt) in enumerate(zip(rendered_images, gt_images)):
        mask = masks[i] if masks is not None else None

        # PSNR
        psnr = loss_fn.compute_psnr(rendered, gt, mask)
        metrics['psnr'].append(psnr)

        # SSIM
        ssim = loss_fn.compute_ssim(rendered, gt, mask)
        metrics['ssim'].append(ssim)

    # 计算平均值
    for key in metrics:
        if len(metrics[key]) > 0:
            metrics[f'{key}_mean'] = np.mean(metrics[key])
            metrics[f'{key}_std'] = np.std(metrics[key])

    return metrics


def tensorboard_logging(writer, metrics, iteration):
    """将指标记录到TensorBoard"""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(key, value, iteration)

    writer.flush()


def export_to_mesh(gaussian_model, output_path: str, resolution: int = 256):
    """将高斯模型转换为网格（简化实现）"""
    # 注意：这只是一个简化实现
    # 完整的网格提取需要更复杂的算法

    xyz = gaussian_model.get_xyz.detach().cpu().numpy()
    scaling = gaussian_model.get_scaling.detach().cpu().numpy()

    # 简单实现：创建点云网格
    # 实际应用中可能需要使用泊松重建或Marching Cubes

    # 保存为PLY点云
    from plyfile import PlyData, PlyElement

    vertices = np.array([(x, y, z) for x, y, z in xyz],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    vertex_element = PlyElement.describe(vertices, 'vertex')
    PlyData([vertex_element]).write(output_path)

    print(f"Mesh (point cloud) exported to {output_path}")
    print("Note: For proper mesh extraction, consider using Poisson reconstruction")


# 数学工具函数
def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """将旋转矩阵转换为四元数"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        S = torch.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return torch.tensor([qw, qx, qy, qz])


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """将四元数转换为旋转矩阵"""
    q = q / torch.norm(q)
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    R = torch.tensor([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]
    ])

    return R


def create_camera_frustum(camera, scale: float = 0.1):
    """创建相机视锥体"""
    # 相机位置
    cam_pos = (-camera.R.T @ camera.T).squeeze()

    # 计算视锥体角点
    w, h = camera.width, camera.height
    fx, fy = camera.fx, camera.fy
    cx, cy = camera.cx, camera.cy

    # 近平面角点
    z_near = 0.1
    top = z_near * (h / 2 - cy) / fy
    bottom = z_near * (-h / 2 - cy) / fy
    right = z_near * (w / 2 - cx) / fx
    left = z_near * (-w / 2 - cx) / fx

    # 近平面角点（相机坐标系）
    near_pts_cam = torch.tensor([
        [left, top, z_near],
        [right, top, z_near],
        [right, bottom, z_near],
        [left, bottom, z_near]
    ]).T

    # 转换到世界坐标系
    near_pts_world = camera.R.T @ near_pts_cam + cam_pos.unsqueeze(1)

    # 远平面角点
    z_far = 5.0
    top_far = z_far * (h / 2 - cy) / fy
    bottom_far = z_far * (-h / 2 - cy) / fy
    right_far = z_far * (w / 2 - cx) / fx
    left_far = z_far * (-w / 2 - cx) / fx

    far_pts_cam = torch.tensor([
        [left_far, top_far, z_far],
        [right_far, top_far, z_far],
        [right_far, bottom_far, z_far],
        [left_far, bottom_far, z_far]
    ]).T

    far_pts_world = camera.R.T @ far_pts_cam + cam_pos.unsqueeze(1)

    # 连接点形成视锥体
    frustum_points = torch.cat([
        cam_pos.unsqueeze(1),  # 相机位置
        near_pts_world,  # 近平面角点
        far_pts_world  # 远平面角点
    ], dim=1)

    # 连接线
    lines = [
        # 从相机到近平面角点
        (0, 1), (0, 2), (0, 3), (0, 4),
        # 近平面边框
        (1, 2), (2, 3), (3, 4), (4, 1),
        # 远平面边框
        (5, 6), (6, 7), (7, 8), (8, 5),
        # 连接近远平面
        (1, 5), (2, 6), (3, 7), (4, 8)
    ]

    return frustum_points, lines



