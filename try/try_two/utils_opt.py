#!/usr/bin/env python3
"""
优化工具函数集 - RTX 3060专用
包含GPU监控、图像处理、数学工具、性能分析等
"""

import os
import sys
import time
import math
import random
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings


# ==================== GPU 监控和优化 ====================

def setup_cuda_optimizations():
    """设置CUDA优化参数"""
    # 启用TF32 (Ampere架构优化)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 启用cudnn自动优化器
    torch.backends.cudnn.benchmark = True

    # 禁用确定性算法以获得更好性能
    torch.backends.cudnn.deterministic = False

    print("✅ CUDA优化设置:")
    print(f"   - TF32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"   - cudNN基准测试: {torch.backends.cudnn.benchmark}")

    # 设置内存分配策略
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        try:
            # 限制GPU内存使用，避免系统卡死
            torch.cuda.set_per_process_memory_fraction(0.8)  # 使用80%显存
            print("   - 显存限制: 80%")
        except:
            pass


def print_gpu_memory(detail: bool = False):
    """打印GPU内存使用情况"""
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return

    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
        total = props.total_memory / 1024 ** 3

        usage_percent = (allocated / total) * 100

        print(f"[显存] 已分配: {allocated:.2f} GB ({usage_percent:.1f}%), "
              f"保留: {reserved:.2f} GB, 总量: {total:.1f} GB")

        if detail:
            # 详细内存统计
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            print(f"      峰值使用: {max_allocated:.2f} GB")

            # 如果有多个GPU
            if torch.cuda.device_count() > 1:
                print(f"      GPU数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                    print(f"      GPU {i}: {name} ({mem:.1f} GB)")

    except Exception as e:
        print(f"❌ 获取GPU内存信息失败: {e}")


def clear_gpu_cache():
    """清除GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🧹 GPU缓存已清除")


def get_gpu_info() -> Dict[str, Any]:
    """获取GPU详细信息"""
    info = {
        'available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': [],
        'current_device': None
    }

    if info['available']:
        info['device_count'] = torch.cuda.device_count()
        info['current_device'] = torch.cuda.current_device()

        for i in range(info['device_count']):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                'id': i,
                'name': props.name,
                'total_memory_gb': props.total_memory / 1024 ** 3,
                'multi_processor_count': props.multi_processor_count,
                'major': props.major,
                'minor': props.minor,
                'is_current': (i == info['current_device'])
            }
            info['devices'].append(device_info)

    return info


def set_seed(seed: int = 42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 为了性能，不强制确定性
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    print(f"🎲 随机种子设置为: {seed}")


# ==================== 图像处理工具 ====================

def load_image_tensor(image_path: str, device: str = 'cuda') -> torch.Tensor:
    """加载图像为tensor"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(device)

            return tensor

    except Exception as e:
        print(f"❌ 加载图像失败 {image_path}: {e}")
        return torch.zeros((3, 256, 256), device=device)


def save_image_tensor(tensor: torch.Tensor, output_path: str,
                      normalize: bool = True):
    """保存tensor为图像"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 分离计算图并转到CPU
    if tensor.requires_grad:
        tensor = tensor.detach()

    tensor = tensor.cpu()

    # 转换为[H, W, C]格式
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)

    # 归一化
    if normalize:
        tensor = torch.clamp(tensor, 0, 1)

    # 转换为numpy
    img_array = tensor.numpy()

    # 转换为0-255范围
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)

    # 保存图像
    Image.fromarray(img_array).save(output_path)
    print(f"💾 图像已保存: {output_path}")


def resize_image_tensor(tensor: torch.Tensor, scale: float = 0.5,
                        mode: str = 'bilinear') -> torch.Tensor:
    """调整tensor图像大小"""
    if scale == 1.0:
        return tensor

    if tensor.dim() == 3:
        C, H, W = tensor.shape
        new_H, new_W = int(H * scale), int(W * scale)

        tensor = tensor.unsqueeze(0)  # 添加批次维度
        tensor = F.interpolate(tensor, size=(new_H, new_W),
                               mode=mode, align_corners=False)
        tensor = tensor.squeeze(0)  # 移除批次维度

    return tensor


def compute_image_gradient(image: torch.Tensor) -> torch.Tensor:
    """计算图像梯度"""
    if image.dim() == 3:
        image = image.unsqueeze(0)  # 添加批次维度

    # Sobel滤波器
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                           dtype=image.dtype, device=image.device)
    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                           dtype=image.dtype, device=image.device)

    # 对每个通道计算梯度
    gradients = []
    for c in range(image.shape[1]):
        grad_x = F.conv2d(image[:, c:c + 1], sobel_x.unsqueeze(0), padding=1)
        grad_y = F.conv2d(image[:, c:c + 1], sobel_y.unsqueeze(0), padding=1)
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        gradients.append(grad_magnitude)

    gradient = torch.cat(gradients, dim=1)

    if image.dim() == 4 and image.shape[0] == 1:
        gradient = gradient.squeeze(0)

    return gradient


def compute_image_stats(image: torch.Tensor) -> Dict[str, float]:
    """计算图像统计信息"""
    stats = {}

    if image.dim() == 3:
        # 通道统计
        for c in range(image.shape[0]):
            channel_data = image[c]
            stats[f'channel_{c}_mean'] = channel_data.mean().item()
            stats[f'channel_{c}_std'] = channel_data.std().item()
            stats[f'channel_{c}_min'] = channel_data.min().item()
            stats[f'channel_{c}_max'] = channel_data.max().item()

        # 整体统计
        stats['overall_mean'] = image.mean().item()
        stats['overall_std'] = image.std().item()
        stats['overall_min'] = image.min().item()
        stats['overall_max'] = image.max().item()

    return stats


# ==================== 数学工具 ====================

def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """归一化点云到单位球内"""
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    # 计算最大距离
    max_distance = np.max(np.linalg.norm(points_centered, axis=1))

    if max_distance > 0:
        points_normalized = points_centered / max_distance
    else:
        points_normalized = points_centered

    return points_normalized, centroid, max_distance


def denormalize_points(points: np.ndarray, centroid: np.ndarray,
                       scale: float) -> np.ndarray:
    """反归一化点云"""
    return points * scale + centroid


def compute_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """计算绕轴旋转的旋转矩阵"""
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    one_minus_cos = 1 - cos_a

    x, y, z = axis
    R = np.array([
        [cos_a + x * x * one_minus_cos, x * y * one_minus_cos - z * sin_a, x * z * one_minus_cos + y * sin_a],
        [y * x * one_minus_cos + z * sin_a, cos_a + y * y * one_minus_cos, y * z * one_minus_cos - x * sin_a],
        [z * x * one_minus_cos - y * sin_a, z * y * one_minus_cos + x * sin_a, cos_a + z * z * one_minus_cos]
    ])

    return R


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """四元数转旋转矩阵"""
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    R = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    ])

    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """旋转矩阵转四元数"""
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def compute_bounding_box(points: np.ndarray) -> Dict[str, np.ndarray]:
    """计算点云的边界框"""
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    center = (min_bound + max_bound) / 2
    extent = max_bound - min_bound

    return {
        'min': min_bound,
        'max': max_bound,
        'center': center,
        'extent': extent
    }


def compute_point_density(points: np.ndarray, voxel_size: float = 0.1) -> float:
    """计算点云密度"""
    if len(points) < 2:
        return 0.0

    # 计算边界框体积
    bbox = compute_bounding_box(points)
    volume = np.prod(bbox['extent'])

    if volume < 1e-12:
        return 0.0

    # 近似密度
    density = len(points) / volume

    return density


# ==================== 性能监控 ====================

class PerformanceTimer:
    """性能计时器"""

    def __init__(self, name: str = "操作"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

        if self.elapsed > 0.1:  # 只显示耗时较长的操作
            print(f"⏱️  {self.name}: {self.elapsed:.3f}秒")

    def get_elapsed(self) -> float:
        """获取经过的时间"""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.elapsed


def benchmark_function(func, *args, repetitions: int = 10, **kwargs):
    """基准测试函数性能"""
    times = []

    # 预热
    for _ in range(3):
        func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 正式测试
    for _ in range(repetitions):
        start_time = time.time()
        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()
        times.append(end_time - start_time)

    # 统计
    times = np.array(times)
    stats = {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'total': np.sum(times),
        'repetitions': repetitions
    }

    print(f"📊 基准测试 {func.__name__}:")
    print(f"   平均: {stats['mean']:.4f} ± {stats['std']:.4f} 秒")
    print(f"   范围: {stats['min']:.4f} - {stats['max']:.4f} 秒")
    print(f"   总计: {stats['total']:.2f} 秒 ({repetitions}次)")

    return stats, result


# ==================== 损失计算 ====================

def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'), device=img1.device)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def compute_lpips(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """计算LPIPS (感知相似性) - 简化版"""
    # 实际应该使用预训练的LPIPS模型
    # 这里使用简化的版本
    diff = img1 - img2

    # 多尺度L1损失
    lpips = 0.0
    scales = [1, 2, 4]

    for scale in scales:
        if scale > 1:
            img1_down = F.avg_pool2d(img1.unsqueeze(0), scale).squeeze(0)
            img2_down = F.avg_pool2d(img2.unsqueeze(0), scale).squeeze(0)
        else:
            img1_down = img1
            img2_down = img2

        lpips += torch.abs(img1_down - img2_down).mean()

    return lpips / len(scales)


def compute_total_variation(image: torch.Tensor) -> torch.Tensor:
    """计算总变分 (平滑性损失)"""
    if image.dim() == 3:
        image = image.unsqueeze(0)

    # 计算水平和垂直梯度
    diff_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    diff_y = image[:, :, 1:, :] - image[:, :, :-1, :]

    tv = torch.mean(diff_x ** 2) + torch.mean(diff_y ** 2)

    return tv


# ==================== 可视化工具 ====================

def create_colormap(name: str = 'viridis') -> LinearSegmentedColormap:
    """创建颜色映射"""
    if name == 'rainbow':
        colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]
    elif name == 'heat':
        colors = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
    elif name == 'coolwarm':
        colors = [(0.23, 0.30, 0.75), (0.87, 0.87, 0.87), (0.70, 0.09, 0.16)]
    else:  # viridis
        colors = [(0.27, 0.00, 0.33), (0.14, 0.45, 0.56),
                  (0.17, 0.73, 0.47), (0.75, 0.86, 0.13)]

    return LinearSegmentedColormap.from_list(name, colors)


def plot_training_curve(stats: Dict[str, List], save_path: Optional[str] = None):
    """绘制训练曲线"""
    if not stats or 'losses' not in stats:
        print("❌ 没有训练统计数据")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 损失曲线
    if 'losses' in stats and stats['losses']:
        axes[0, 0].plot(stats['losses'])
        axes[0, 0].set_title('训练损失')
        axes[0, 0].set_xlabel('迭代')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].grid(True, alpha=0.3)

    # PSNR曲线
    if 'psnrs' in stats and stats['psnrs']:
        axes[0, 1].plot(stats['psnrs'])
        axes[0, 1].set_title('PSNR')
        axes[0, 1].set_xlabel('迭代')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].grid(True, alpha=0.3)

    # 学习率曲线
    if 'learning_rates' in stats and stats['learning_rates']:
        axes[1, 0].plot(stats['learning_rates'])
        axes[1, 0].set_title('学习率')
        axes[1, 0].set_xlabel('迭代')
        axes[1, 0].set_ylabel('学习率')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

    # 时间曲线
    if 'timestamps' in stats and stats['timestamps']:
        axes[1, 1].plot(stats['timestamps'], stats.get('losses', []))
        axes[1, 1].set_title('损失 vs 时间')
        axes[1, 1].set_xlabel('时间 (秒)')
        axes[1, 1].set_ylabel('损失')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📈 训练曲线已保存: {save_path}")

    plt.show()


def plot_point_cloud_3d(points: np.ndarray, colors: Optional[np.ndarray] = None,
                        title: str = "点云", save_path: Optional[str] = None):
    """绘制3D点云"""
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if colors is not None:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                 c=colors, s=1, alpha=0.6, cmap='viridis')
            plt.colorbar(scatter, ax=ax, label='颜色强度')
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                       s=1, alpha=0.6)

        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 设置等比例
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

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 点云图已保存: {save_path}")

        plt.show()

    except Exception as e:
        print(f"❌ 绘制点云失败: {e}")


# ==================== 文件操作 ====================

def create_directory_structure(base_dir: str, scene_name: str) -> Dict[str, str]:
    """创建目录结构"""
    directories = {
        'base': os.path.join(base_dir, scene_name),
        'checkpoints': os.path.join(base_dir, scene_name, 'checkpoints'),
        'renders': os.path.join(base_dir, scene_name, 'renders'),
        'point_clouds': os.path.join(base_dir, scene_name, 'point_clouds'),
        'videos': os.path.join(base_dir, scene_name, 'videos'),
        'logs': os.path.join(base_dir, scene_name, 'logs'),
        'configs': os.path.join(base_dir, scene_name, 'configs')
    }

    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    print(f"📁 目录结构已创建: {directories['base']}")
    return directories


def save_json(data: Dict, path: str, indent: int = 2):
    """保存JSON文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    print(f"💾 JSON已保存: {path}")


def load_json(path: str) -> Dict:
    """加载JSON文件"""
    if not os.path.exists(path):
        print(f"❌ JSON文件不存在: {path}")
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def compute_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """计算文件哈希值"""
    if not os.path.exists(file_path):
        return ""

    hash_func = hashlib.new(algorithm)

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


# ==================== 日志系统 ====================

class Logger:
    """简单的日志系统"""

    def __init__(self, log_dir: str = "./logs", log_file: str = "training.log"):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_file)

        os.makedirs(log_dir, exist_ok=True)

        # 清空或创建日志文件
        with open(self.log_file, 'w') as f:
            f.write(f"=== 训练日志开始于 {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}"

        # 写入文件
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")

        # 打印到控制台
        if level == "ERROR":
            print(f"❌ {message}")
        elif level == "WARNING":
            print(f"⚠️  {message}")
        elif level == "INFO":
            print(f"ℹ️  {message}")
        else:
            print(message)

    def log_metric(self, name: str, value: float, iteration: int):
        """记录指标"""
        self.log(f"迭代 {iteration}: {name} = {value:.6f}", "METRIC")

    def log_config(self, config: Dict):
        """记录配置"""
        self.log("配置参数:", "CONFIG")
        for key, value in config.items():
            self.log(f"  {key}: {value}", "CONFIG")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("🧪 测试工具函数...")

    # 测试GPU监控
    print("\n1. GPU信息:")
    gpu_info = get_gpu_info()
    print(f"   CUDA可用: {gpu_info['available']}")
    print(f"   GPU数量: {gpu_info['device_count']}")

    if gpu_info['available']:
        for device in gpu_info['devices']:
            print(f"   GPU {device['id']}: {device['name']} "
                  f"({device['total_memory_gb']:.1f} GB)")

    # 测试性能计时器
    print("\n2. 性能测试:")
    with PerformanceTimer("测试操作"):
        time.sleep(0.1)

    # 测试数学工具
    print("\n3. 数学工具:")
    points = np.random.randn(100, 3)
    bbox = compute_bounding_box(points)
    print(f"   边界框中心: {bbox['center']}")
    print(f"   边界框范围: {bbox['extent']}")

    # 测试图像处理
    print("\n4. 图像处理:")
    test_image = torch.rand(3, 256, 256)
    stats = compute_image_stats(test_image)
    print(f"   图像均值: {stats.get('overall_mean', 0):.3f}")
    print(f"   图像标准差: {stats.get('overall_std', 0):.3f}")

    # 测试损失计算
    print("\n5. 损失计算:")
    img1 = torch.rand(3, 64, 64)
    img2 = torch.rand(3, 64, 64)
    psnr = compute_psnr(img1, img2)
    print(f"   PSNR: {psnr:.2f} dB")

    print("\n✅ 工具函数测试完成!")











