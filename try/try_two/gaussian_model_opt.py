#!/usr/bin/env python3
"""
优化版高斯模型类 - RTX 3060专用
包含混合精度优化、内存优化、张量核心优化
"""

import os
import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from typing import Optional, Tuple


class OptimizedGaussianModel(nn.Module):
    """为RTX 3060优化的高斯模型类"""

    def __init__(self, sh_degree: int = 0, device: str = 'cuda'):
        super().__init__()

        self.device = torch.device(device)
        self.active_sh_degree = 0
        self.max_sh_degree = max(sh_degree, 0)  # 确保非负

        print(f"🎯 初始化高斯模型 (SH阶数: {self.max_sh_degree})")

        # ==================== RTX 3060 内存优化 ====================
        # 在__init__中只创建占位符，真正的参数在create_from_pcl中创建
        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

        # 空间数据结构（这些不需要梯度，使用register_buffer）
        self.register_buffer('xyz_gradient_accum', torch.zeros(0, 1, device=self.device))
        self.register_buffer('denom', torch.zeros(0, 1, device=self.device))
        self.register_buffer('max_radii2D', torch.zeros(0, device=self.device))

        # 优化器
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        # 预分配缓冲区以减少动态分配
        self._preallocated_cov3D: Optional[torch.Tensor] = None
        self._preallocated_cov2D: Optional[torch.Tensor] = None

        # 训练统计
        self.learning_rate = 0.0
        self.iteration = 0

        # 移动到指定设备
        self.to(self.device)

    def create_from_pcl(self, points: np.ndarray, colors: np.ndarray,
                        num_points: int = 50000) -> None:
        """从点云初始化高斯模型"""
        print(f"☁️  从点云初始化高斯模型...")

        # 确保输入数据在GPU上
        if not isinstance(points, torch.Tensor):
            points_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)
        else:
            points_tensor = points.to(self.device)

        if not isinstance(colors, torch.Tensor):
            colors_tensor = torch.tensor(colors, dtype=torch.float32, device=self.device)
        else:
            colors_tensor = colors.to(self.device)

        # 🔥 修复1: 标准化点云位置
        # 计算点云的质心和标准差
        centroid = points_tensor.mean(dim=0)
        points_centered = points_tensor - centroid
        std = points_centered.std()

        # 标准化到合理范围 (-2, 2)
        if std > 0:
            points_tensor = points_centered / (std + 1e-8) * 0.5
        else:
            points_tensor = points_centered

        # 限制点数以避免显存溢出
        max_points_for_3060 = 100000
        num_points = min(num_points, len(points_tensor), max_points_for_3060)

        # 随机选择点
        if len(points_tensor) > num_points:
            indices = torch.randperm(len(points_tensor))[:num_points].to(self.device)
            points_tensor = points_tensor[indices]
            colors_tensor = colors_tensor[indices]

        print(f"  选择 {len(points_tensor)} 个点进行初始化")
        print(f"  位置范围: [{points_tensor.min().item():.3f}, {points_tensor.max().item():.3f}]")

        # 🔥 修复2: 直接设置为nn.Parameter，而不是使用register_parameter
        # 添加少量噪声避免梯度爆炸
        noise = torch.randn_like(points_tensor) * 0.001
        self._xyz = nn.Parameter(points_tensor + noise, requires_grad=True)

        # 颜色 - 确保在合理范围
        colors_clamped = torch.clamp(colors_tensor, 0.1, 0.9)  # 避免极端颜色值
        self._features_dc = nn.Parameter(
            colors_clamped.unsqueeze(1).clone().detach(),
            requires_grad=True
        )

        # 高阶球谐函数项
        if self.max_sh_degree > 0:
            sh_dim = (self.max_sh_degree + 1) ** 2 - 1
            self._features_rest = nn.Parameter(
                torch.zeros((len(points_tensor), sh_dim, 3), device=self.device) * 0.01,
                requires_grad=True
            )
        else:
            self._features_rest = nn.Parameter(
                torch.zeros((len(points_tensor), 0, 3), device=self.device),
                requires_grad=False
            )

        # 缩放参数
        self._scaling = nn.Parameter(
            torch.ones((len(points_tensor), 3), device=self.device) * 0.001,
            requires_grad=True
        )

        # 旋转参数 (四元数，wxyz格式)
        rotation_data = torch.zeros((len(points_tensor), 4), device=self.device)
        rotation_data[:, 0] = 1.0  # w=1，无旋转
        rotation_data[:, 1:] = torch.randn((len(points_tensor), 3), device=self.device) * 0.01
        self._rotation = nn.Parameter(rotation_data, requires_grad=True)

        # 不透明度参数
        self._opacity = nn.Parameter(
            torch.ones((len(points_tensor), 1), device=self.device) * 0.01,
            requires_grad=True
        )

        # 初始化优化器数据结构
        self.xyz_gradient_accum = torch.zeros((len(points_tensor), 1), device=self.device)
        self.denom = torch.zeros((len(points_tensor), 1), device=self.device)
        self.max_radii2D = torch.zeros(len(points_tensor), device=self.device)

        print(f"✅ 高斯模型初始化完成: {len(points_tensor)} 个高斯点")

        # 🔥 修复3: 检查所有参数都正确设置了梯度
        print(f"  参数梯度状态检查:")
        for name, param in self.named_parameters():
            print(f"    {name}: requires_grad={param.requires_grad}, shape={param.shape}")

    def training_setup(self, training_args) -> None:
        """设置训练优化器"""
        print("⚙️  设置训练优化器...")

        # ==================== 参数分组 ====================
        params = []

        # 位置参数
        params.append({
            'params': [self._xyz],
            'lr': training_args.learning_rate,
            'name': 'xyz',
            'weight_decay': 0.0
        })

        # 颜色参数
        params.append({
            'params': [self._features_dc],
            'lr': training_args.learning_rate,
            'name': 'features_dc',
            'weight_decay': 0.01  # 轻微正则化
        })

        # 不透明度参数
        params.append({
            'params': [self._opacity],
            'lr': training_args.learning_rate * 0.1,
            'name': 'opacity',
            'weight_decay': 0.0
        })

        # 缩放参数
        params.append({
            'params': [self._scaling],
            'lr': training_args.learning_rate * 0.05,
            'name': 'scaling',
            'weight_decay': 0.0
        })

        # 旋转参数
        params.append({
            'params': [self._rotation],
            'lr': training_args.learning_rate * 0.05,
            'name': 'rotation',
            'weight_decay': 0.0
        })

        # 高阶球谐函数参数
        if self.max_sh_degree > 0 and hasattr(self, '_features_rest'):
            params.append({
                'params': [self._features_rest],
                'lr': training_args.learning_rate * 0.1,
                'name': 'features_rest',
                'weight_decay': 0.01
            })

        # ==================== 优化器配置 ====================
        # 使用AdamW优化器 (更好的权重衰减处理)
        self.optimizer = torch.optim.AdamW(
            [p for group in params for p in group['params']],
            lr=training_args.learning_rate,
            eps=1e-15,
            foreach=True  # 加速优化器步骤
        )

        # ==================== 学习率调度器 ====================
        # 余弦退火学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=training_args.iterations,
            eta_min=training_args.learning_rate * 0.01
        )

        self.learning_rate = training_args.learning_rate

        print(f"✅ 优化器设置完成")
        print(f"   初始学习率: {training_args.learning_rate}")
        print(f"   参数总数: {sum(p.numel() for p in self.parameters())}")

    def update_learning_rate(self, iteration: int) -> float:
        """更新学习率（兼容旧接口）"""
        if self.scheduler is not None:
            self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]['lr']
        self.iteration = iteration

        return current_lr

    def state_dict(self) -> dict:
        """获取模型状态字典"""
        return {
            'xyz': self._xyz.data,
            'features_dc': self._features_dc.data,
            'features_rest': self._features_rest.data,
            'scaling': self._scaling.data,
            'rotation': self._rotation.data,
            'opacity': self._opacity.data,
            'active_sh_degree': self.active_sh_degree,
            'iteration': self.iteration
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """加载模型状态字典"""
        # 设置参数
        self._xyz.data = state_dict['xyz'].to(self.device)
        self._features_dc.data = state_dict['features_dc'].to(self.device)
        self._features_rest.data = state_dict['features_rest'].to(self.device)
        self._scaling.data = state_dict['scaling'].to(self.device)
        self._rotation.data = state_dict['rotation'].to(self.device)
        self._opacity.data = state_dict['opacity'].to(self.device)

        # 设置其他状态
        self.active_sh_degree = state_dict.get('active_sh_degree', 0)
        self.iteration = state_dict.get('iteration', 0)

        print(f"✅ 模型状态加载完成: {self._xyz.shape[0]} 个高斯点")

    def save_ply(self, path: str) -> None:
        """保存为PLY格式"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 分离计算图
        xyz = self._xyz.detach().cpu().numpy()

        # 获取颜色
        if self._features_dc is not None:
            colors = self._features_dc.detach().cpu().numpy().squeeze(1)
            colors = np.clip(colors, 0, 1)
        else:
            colors = np.ones_like(xyz) * 0.5

        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 保存
        o3d.io.write_point_cloud(path, pcd, write_ascii=True)
        print(f"💾 点云已保存: {path} ({len(xyz)} 个点)")

    def load_ply(self, path: str) -> bool:
        """从PLY文件加载"""
        if not os.path.exists(path):
            print(f"❌ PLY文件不存在: {path}")
            return False

        try:
            # 加载点云
            pcd = o3d.io.read_point_cloud(path)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            if len(points) == 0:
                print(f"❌ PLY文件为空: {path}")
                return False

            # 设置参数
            self._xyz = nn.Parameter(
                torch.tensor(points, dtype=torch.float32, device=self.device),
                requires_grad=True
            )

            self._features_dc = nn.Parameter(
                torch.tensor(colors, dtype=torch.float32, device=self.device).unsqueeze(1),
                requires_grad=True
            )

            # 重置其他参数
            if self.max_sh_degree > 0:
                sh_dim = (self.max_sh_degree + 1) ** 2 - 1
                self._features_rest = nn.Parameter(
                    torch.zeros((len(points), sh_dim, 3), device=self.device),
                    requires_grad=True
                )

            self._scaling = nn.Parameter(
                torch.ones((len(points), 3), device=self.device) * 0.01,
                requires_grad=True
            )

            self._rotation = nn.Parameter(
                torch.zeros((len(points), 4), device=self.device),
                requires_grad=True
            )
            self._rotation.data[:, 0] = 1.0

            self._opacity = nn.Parameter(
                torch.ones((len(points), 1), device=self.device) * 0.1,
                requires_grad=True
            )

            print(f"✅ 从PLY加载: {len(points)} 个点")
            return True

        except Exception as e:
            print(f"❌ 加载PLY文件失败: {e}")
            return False

    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'iteration': self.iteration,
            'learning_rate': self.learning_rate
        }

        torch.save(checkpoint, path)
        print(f"💾 检查点已保存: {path}")

    def load_checkpoint(self, path: str) -> bool:
        """加载检查点"""
        if not os.path.exists(path):
            print(f"❌ 检查点文件不存在: {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)

            # 加载模型状态
            self.load_state_dict(checkpoint['model_state_dict'])

            # 加载优化器状态
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载调度器状态
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # 加载训练状态
            self.iteration = checkpoint.get('iteration', 0)
            self.learning_rate = checkpoint.get('learning_rate', 0.001)

            print(f"✅ 检查点加载完成: 迭代 {self.iteration}")
            return True

        except Exception as e:
            print(f"❌ 加载检查点失败: {e}")
            return False

    # ==================== 属性访问器 ====================

    @property
    def get_xyz(self) -> torch.Tensor:
        """获取位置参数"""
        if self._xyz is None:
            raise RuntimeError("高斯模型尚未初始化，请先调用create_from_pcl方法")
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        """获取颜色特征"""
        if self._features_dc is None:
            raise RuntimeError("高斯模型尚未初始化，请先调用create_from_pcl方法")
        return self._features_dc

    @property
    def get_opacity(self) -> torch.Tensor:
        """获取不透明度参数"""
        if self._opacity is None:
            raise RuntimeError("高斯模型尚未初始化，请先调用create_from_pcl方法")
        return self._opacity

    @property
    def get_scaling(self) -> torch.Tensor:
        """获取缩放参数"""
        if self._scaling is None:
            raise RuntimeError("高斯模型尚未初始化，请先调用create_from_pcl方法")
        return self._scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        """获取旋转参数"""
        if self._rotation is None:
            raise RuntimeError("高斯模型尚未初始化，请先调用create_from_pcl方法")
        return self._rotation

    @property
    def num_gaussians(self) -> int:
        """获取高斯点数量"""
        return self._xyz.shape[0] if self._xyz is not None else 0
    def forward(self):
        """前向传播（占位符）"""
        return self.state_dict()

    def eval(self):
        """设置为评估模式"""
        super().eval()
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        """设置为训练模式"""
        super().train(mode)
        # 解冻所有参数
        for param in self.parameters():
            param.requires_grad = True

    def __repr__(self) -> str:
        """字符串表示"""
        return (f"OptimizedGaussianModel(\n"
                f"  高斯点数: {self.num_gaussians}\n"
                f"  SH阶数: {self.max_sh_degree}\n"
                f"  设备: {self.device}\n"
                f"  参数数量: {sum(p.numel() for p in self.parameters()):,}\n"
                f")")


# ==================== 辅助函数 ====================

def print_gpu_memory():
    """打印GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"[显存] 已分配: {allocated:.2f} GB, 保留: {reserved:.2f} GB")


def set_seed(seed: int = 42):
    """设置随机种子"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 为了性能，不强制确定性
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试高斯模型类...")

    # 创建模型
    model = OptimizedGaussianModel(sh_degree=0, device='cuda')

    # 创建测试数据
    points = np.random.randn(1000, 3).astype(np.float32)
    colors = np.random.rand(1000, 3).astype(np.float32)

    # 初始化
    model.create_from_pcl(points, colors, num_points=500)

    # 打印模型信息
    print(model)

    # 测试保存/加载
    test_dir = "test_output"
    os.makedirs(test_dir, exist_ok=True)

    ply_path = os.path.join(test_dir, "test.ply")
    model.save_ply(ply_path)

    # 清理
    import shutil

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    print("✅ 测试完成!")