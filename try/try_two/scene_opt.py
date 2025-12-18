#!/usr/bin/env python3
"""
优化版场景管理类 - RTX 3060专用
管理相机、数据集和高斯模型
"""

import os
import random
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple

# 导入自定义模块
from gaussian_model_opt import OptimizedGaussianModel
from camera_opt import OptimizedCamera
from utils_opt import print_gpu_memory


class OptimizedScene:
    """优化版场景管理类"""

    def __init__(self, dataset: Dict[str, Any], gaussians: OptimizedGaussianModel,
                 output_dir: str, train_test_split: float = 0.9):
        """
        初始化场景

        参数:
            dataset: 数据集字典
            gaussians: 高斯模型
            output_dir: 输出目录
            train_test_split: 训练测试分割比例
        """
        print("🖼️  初始化场景...")

        self.dataset = dataset
        self.gaussians = gaussians
        self.output_dir = output_dir
        self.train_test_split = train_test_split

        # 相机列表
        self.train_cameras: List[OptimizedCamera] = []
        self.test_cameras: List[OptimizedCamera] = []

        # 创建相机
        self._create_cameras()

        # 初始化高斯模型（如果没有已加载的模型）
        if self.gaussians.num_gaussians == 0:
            self._initialize_gaussians()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "renders"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "point_clouds"), exist_ok=True)

        print(f"✅ 场景初始化完成")
        print(f"   训练相机: {len(self.train_cameras)} 个")
        print(f"   测试相机: {len(self.test_cameras)} 个")
        print(f"   高斯点: {self.gaussians.num_gaussians} 个")

    def _create_cameras(self) -> None:
        """从数据集创建相机"""
        print("📷 创建相机...")

        # 从数据集获取数据
        images = self.dataset['images']  # [B, C, H, W]
        world2cam = self.dataset['world2cam']  # [B, 4, 4]
        K = self.dataset['K']  # [B, 3, 3] 或 [3, 3]
        H = self.dataset['H']
        W = self.dataset['W']
        image_files = self.dataset['image_files']

        num_images = len(images)

        # 创建索引列表
        indices = list(range(num_images))

        # 随机打乱
        random.shuffle(indices)

        # 分割训练/测试集
        split_idx = int(num_images * self.train_test_split)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        print(f"   总图像: {num_images}")
        print(f"   训练集: {len(train_indices)} 张")
        print(f"   测试集: {len(test_indices)} 张")

        # 检查K的形状
        if K.dim() == 3:
            # 如果是批量，需要为每个相机提取对应的内参
            K_list = [K[i] for i in range(num_images)]
        else:
            # 如果是单个内参，重复使用
            K_list = [K for _ in range(num_images)]

        # 创建训练相机
        for i, idx in enumerate(train_indices):
            # 确保传递的是二维张量
            camera_world2cam = world2cam[idx] if world2cam.dim() == 3 else world2cam
            camera_K = K_list[idx]
            camera_image = images[idx]

            camera = OptimizedCamera(
                world2cam=camera_world2cam,
                K=camera_K,
                image=camera_image,
                H=H, W=W,
                image_name=image_files[idx],
                uid=i,
                is_train=True
            )
            self.train_cameras.append(camera)

        # 创建测试相机
        for i, idx in enumerate(test_indices):
            # 确保传递的是二维张量
            camera_world2cam = world2cam[idx] if world2cam.dim() == 3 else world2cam
            camera_K = K_list[idx]
            camera_image = images[idx]

            camera = OptimizedCamera(
                world2cam=camera_world2cam,
                K=camera_K,
                image=camera_image,
                H=H, W=W,
                image_name=image_files[idx],
                uid=len(train_indices) + i,
                is_train=False
            )
            self.test_cameras.append(camera)

        print("✅ 相机创建完成")
    def _initialize_gaussians(self) -> None:
        """初始化高斯模型"""
        print("🎯 初始化高斯点云...")

        # 方法1: 尝试从稀疏重建加载
        if self._try_load_sparse_pointcloud():
            return

        # 方法2: 从相机位置生成
        self._generate_from_camera_positions()

    def _try_load_sparse_pointcloud(self) -> bool:
        """尝试从稀疏重建加载点云"""
        scene_dir = self.dataset.get('scene_dir', '')
        scene_name = self.dataset.get('scene_name', '')

        if not scene_dir or not scene_name:
            return False

        # 检查COLMAP稀疏重建文件
        possible_paths = [
            os.path.join(scene_dir, "sparse", "0", "points3D.bin"),
            os.path.join(scene_dir, "sparse", "points3D.ply"),
            os.path.join(scene_dir, "sparse.ply"),
            os.path.join(os.path.dirname(scene_dir), f"{scene_name}_sparse.ply")
        ]

        import open3d as o3d

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    print(f"🔍 尝试从 {path} 加载稀疏点云...")

                    if path.endswith('.bin'):
                        # COLMAP二进制格式（简化处理）
                        # 实际应该使用COLMAP读取器
                        print(f"⚠️  跳过COLMAP二进制格式: {path}")
                        continue
                    elif path.endswith('.ply'):
                        # PLY格式
                        pcd = o3d.io.read_point_cloud(path)
                        points = np.asarray(pcd.points)
                        colors = np.asarray(pcd.colors)

                        if len(points) > 100:  # 有效的点云
                            print(f"✅ 从 {path} 加载 {len(points)} 个点")

                            # 初始化高斯模型
                            self.gaussians.create_from_pcl(points, colors)
                            return True

                except Exception as e:
                    print(f"❌ 加载 {path} 失败: {e}")
                    continue

        return False

    def _generate_from_camera_positions(self, num_points: int = 50000) -> None:
        """从相机位置生成点云"""
        print("🎮 从相机位置生成点云...")

        # 收集相机位置
        camera_positions = []

        for camera in self.train_cameras:
            # 相机在世界坐标系中的位置
            cam_pos = camera.get_position().cpu().numpy()
            camera_positions.append(cam_pos)

        if not camera_positions:
            # 如果没有相机，使用随机点
            points = np.random.randn(num_points, 3).astype(np.float32) * 2.0
            colors = np.random.rand(num_points, 3).astype(np.float32) * 0.8 + 0.2
        else:
            camera_positions = np.array(camera_positions)

            # 计算场景中心
            center = np.mean(camera_positions, axis=0)

            # 计算场景尺度
            distances = np.linalg.norm(camera_positions - center, axis=1)
            scene_radius = np.max(distances) * 1.5

            print(f"   场景中心: {center}")
            print(f"   场景半径: {scene_radius:.2f}")

            # 在相机位置周围生成点
            # 方法1: 在边界盒内均匀采样
            min_bound = np.min(camera_positions, axis=0) - scene_radius * 0.5
            max_bound = np.max(camera_positions, axis=0) + scene_radius * 0.5

            # 生成随机点
            points = np.random.uniform(
                min_bound, max_bound,
                size=(num_points, 3)
            ).astype(np.float32)

            # 给点云添加一些噪声
            points += np.random.randn(num_points, 3).astype(np.float32) * scene_radius * 0.1

            # 生成随机颜色（基于位置）
            normalized_pos = (points - min_bound) / (max_bound - min_bound + 1e-8)
            colors = normalized_pos * 0.5 + 0.3  # 在[0.3, 0.8]范围内

        # 初始化高斯模型
        self.gaussians.create_from_pcl(points, colors, num_points=min(num_points, 50000))

    # ==================== 公共接口 ====================

    def get_random_train_camera(self) -> OptimizedCamera:
        """随机获取一个训练相机"""
        if not self.train_cameras:
            raise ValueError("没有训练相机可用")
        return random.choice(self.train_cameras)

    def get_train_cameras(self) -> List[OptimizedCamera]:
        """获取所有训练相机"""
        return self.train_cameras

    def get_test_cameras(self) -> List[OptimizedCamera]:
        """获取所有测试相机"""
        return self.test_cameras

    def get_camera_by_name(self, name: str) -> Optional[OptimizedCamera]:
        """根据名称获取相机"""
        all_cameras = self.train_cameras + self.test_cameras
        for camera in all_cameras:
            if camera.image_name == name:
                return camera
        return None

    def save_checkpoint(self, checkpoint_dir: str, iteration: int) -> None:
        """保存检查点"""
        print(f"💾 保存检查点 (迭代 {iteration})...")

        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 保存模型状态
        model_path = os.path.join(checkpoint_dir, "model_state.pth")
        self.gaussians.save_checkpoint(model_path)

        # 保存点云（用于可视化）
        ply_path = os.path.join(checkpoint_dir, "point_cloud.ply")
        self.gaussians.save_ply(ply_path)

        # 保存相机信息
        camera_info = {
            'train_cameras': len(self.train_cameras),
            'test_cameras': len(self.test_cameras),
            'iteration': iteration
        }

        import json
        info_path = os.path.join(checkpoint_dir, "scene_info.json")
        with open(info_path, 'w') as f:
            json.dump(camera_info, f, indent=2)

        print(f"✅ 检查点保存完成: {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str) -> bool:
        """加载检查点"""
        if not os.path.exists(checkpoint_dir):
            print(f"❌ 检查点目录不存在: {checkpoint_dir}")
            return False

        # 加载模型状态
        model_path = os.path.join(checkpoint_dir, "model_state.pth")
        if os.path.exists(model_path):
            return self.gaussians.load_checkpoint(model_path)

        return False

    def save_scene_info(self, path: str) -> None:
        """保存场景信息"""
        scene_info = {
            'dataset': {
                'scene_name': self.dataset.get('scene_name', 'unknown'),
                'num_images': len(self.train_cameras) + len(self.test_cameras),
                'image_size': [self.dataset.get('H', 0), self.dataset.get('W', 0)],
                'resolution': self.dataset.get('resolution', 1)
            },
            'gaussians': {
                'num_points': self.gaussians.num_gaussians,
                'sh_degree': self.gaussians.max_sh_degree
            },
            'cameras': {
                'train': len(self.train_cameras),
                'test': len(self.test_cameras)
            }
        }

        import json
        with open(path, 'w') as f:
            json.dump(scene_info, f, indent=2)

        print(f"💾 场景信息已保存: {path}")

    def get_scene_bounds(self) -> Dict[str, np.ndarray]:
        """获取场景边界"""
        if self.gaussians.num_gaussians == 0:
            # 如果没有高斯点，使用相机位置
            all_positions = []
            for camera in self.train_cameras + self.test_cameras:
                all_positions.append(camera.get_position().cpu().numpy())

            if not all_positions:
                return {
                    'min': np.array([-1, -1, -1]),
                    'max': np.array([1, 1, 1]),
                    'center': np.array([0, 0, 0]),
                    'extent': np.array([2, 2, 2])
                }

            all_positions = np.array(all_positions)
            min_bound = np.min(all_positions, axis=0)
            max_bound = np.max(all_positions, axis=0)

        else:
            # 使用高斯点位置
            points = self.gaussians.get_xyz.detach().cpu().numpy()
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

    def print_summary(self) -> None:
        """打印场景摘要"""
        print("\n" + "=" * 50)
        print("📊 场景摘要")
        print("=" * 50)

        # 数据集信息
        print(f"数据集:")
        print(f"  场景名称: {self.dataset.get('scene_name', 'unknown')}")
        print(f"  图像尺寸: {self.dataset.get('W', 0)}x{self.dataset.get('H', 0)}")
        print(f"  分辨率: 1/{self.dataset.get('resolution', 1)}")

        # 相机信息
        print(f"\n相机:")
        print(f"  训练相机: {len(self.train_cameras)} 个")
        print(f"  测试相机: {len(self.test_cameras)} 个")
        print(f"  总计: {len(self.train_cameras) + len(self.test_cameras)} 个")

        # 高斯模型信息
        print(f"\n高斯模型:")
        print(f"  高斯点数: {self.gaussians.num_gaussians:,}")
        print(f"  SH阶数: {self.gaussians.max_sh_degree}")

        # 场景边界
        bounds = self.get_scene_bounds()
        print(f"\n场景边界:")
        print(f"  最小: [{bounds['min'][0]:.2f}, {bounds['min'][1]:.2f}, {bounds['min'][2]:.2f}]")
        print(f"  最大: [{bounds['max'][0]:.2f}, {bounds['max'][1]:.2f}, {bounds['max'][2]:.2f}]")
        print(f"  中心: [{bounds['center'][0]:.2f}, {bounds['center'][1]:.2f}, {bounds['center'][2]:.2f}]")
        print(f"  范围: [{bounds['extent'][0]:.2f}, {bounds['extent'][1]:.2f}, {bounds['extent'][2]:.2f}]")

        # GPU内存
        if torch.cuda.is_available():
            print_gpu_memory()

        print("=" * 50)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("🧪 测试场景类...")

    # 创建测试数据
    test_dataset = {
        'images': torch.randn(10, 3, 256, 256),
        'world2cam': torch.eye(4).unsqueeze(0).repeat(10, 1, 1),
        'K': torch.eye(3).unsqueeze(0).repeat(10, 1, 1),
        'H': 256,
        'W': 256,
        'image_files': [f"test_{i}.jpg" for i in range(10)],
        'scene_name': 'test_scene',
        'resolution': 2
    }

    # 创建高斯模型
    gaussians = OptimizedGaussianModel(sh_degree=0, device='cpu')

    # 创建场景
    scene = OptimizedScene(test_dataset, gaussians, "./test_output")

    # 打印摘要
    scene.print_summary()

    # 测试随机相机获取
    random_cam = scene.get_random_train_camera()
    print(f"\n随机相机: {random_cam.image_name}")

    # 清理
    import shutil

    if os.path.exists("./test_output"):
        shutil.rmtree("./test_output")

    print("✅ 测试完成!")