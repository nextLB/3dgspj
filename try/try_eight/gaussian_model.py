import torch
import torch.nn as nn
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import config


@dataclass
class GaussianParams:
    """高斯溅射的参数"""
    xyz: torch.Tensor  # 位置 (N, 3)
    features_dc: torch.Tensor  # 球谐系数 (N, 3)
    features_rest: torch.Tensor  # 高阶球谐系数 (N, 45)
    scaling: torch.Tensor  # 缩放 (N, 3)
    rotation: torch.Tensor  # 旋转四元数 (N, 4)
    opacity: torch.Tensor  # 不透明度 (N, 1)


class GaussianModel:
    def __init__(self, sh_degree: int = 0):
        self.sh_degree = sh_degree
        self.max_sh_degree = sh_degree

        # 高斯参数
        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

        # 优化器
        self.xyz_optimizer = None
        self.feature_optimizer = None
        self.scaling_optimizer = None
        self.rotation_optimizer = None
        self.opacity_optimizer = None

        # 其他参数
        self.active_sh_degree = 0
        self.spatial_lr_scale = 1.0

    def create_from_pcd(self, pcd: np.ndarray, colors: Optional[np.ndarray] = None):
        """从点云创建高斯模型"""
        num_points = len(pcd)

        # 位置
        self._xyz = nn.Parameter(torch.from_numpy(pcd).float().to(config.config_dict['device']))

        # 颜色特征（球谐系数）
        if colors is None:
            colors = np.random.rand(num_points, 3)

        # 直流分量（基础颜色）
        self._features_dc = nn.Parameter(
            torch.from_numpy(colors).float().to(config.config_dict['device']).unsqueeze(1)
        )

        # 高阶球谐分量（初始为0）- 简化版，如果sh_degree为0则不需要
        num_sh_coeffs = max(0, (self.max_sh_degree + 1) ** 2 - 1)
        if num_sh_coeffs > 0:
            self._features_rest = nn.Parameter(
                torch.zeros((num_points, 1, num_sh_coeffs, 3)).float().to(config.config_dict['device'])
            )
        else:
            # 创建空张量
            self._features_rest = nn.Parameter(
                torch.zeros((num_points, 1, 0, 3)).float().to(config.config_dict['device'])
            )

        # 缩放（初始化为对数尺度）
        self._scaling = nn.Parameter(
            torch.log(torch.ones((num_points, 3), device=config.config_dict['device']) * 0.01)
        )

        # 旋转（四元数，初始化为无旋转）
        self._rotation = nn.Parameter(
            torch.cat([
                torch.ones((num_points, 1), device=config.config_dict['device']),  # w
                torch.zeros((num_points, 3), device=config.config_dict['device'])  # x, y, z
            ], dim=1)
        )

        # 不透明度（经过sigmoid变换）
        self._opacity = nn.Parameter(
            torch.logit(torch.ones((num_points, 1), device=config.config_dict['device']) * 0.1)
        )

    def get_params(self) -> GaussianParams:
        """获取所有参数"""
        return GaussianParams(
            xyz=self._xyz,
            features_dc=self._features_dc,
            features_rest=self._features_rest,
            scaling=self._scaling,
            rotation=self._rotation,
            opacity=self._opacity
        )

    def get_scaling(self):
        """获取实际的缩放值（指数变换确保为正）"""
        return torch.exp(self._scaling)

    def get_rotation(self):
        """获取归一化的四元数"""
        return nn.functional.normalize(self._rotation, dim=1)

    def get_opacity(self):
        """获取实际的不透明度（sigmoid变换）"""
        return torch.sigmoid(self._opacity)

    def get_features(self):
        """获取球谐特征"""
        if self._features_rest.shape[2] > 0:
            features = torch.cat([self._features_dc, self._features_rest], dim=2)
        else:
            features = self._features_dc
        return features.permute(0, 2, 1).contiguous()

    def setup_optimizers(self):
        """设置优化器"""
        self.xyz_optimizer = torch.optim.Adam(
            [self._xyz],
            lr=config.config_dict['position_lr_init'] * self.spatial_lr_scale
        )

        self.feature_optimizer = torch.optim.Adam(
            [self._features_dc, self._features_rest],
            lr=config.config_dict['feature_lr']
        )

        self.opacity_optimizer = torch.optim.Adam(
            [self._opacity],
            lr=config.config_dict['opacity_lr']
        )

        self.scaling_optimizer = torch.optim.Adam(
            [self._scaling],
            lr=config.config_dict['scaling_lr']
        )

        self.rotation_optimizer = torch.optim.Adam(
            [self._rotation],
            lr=config.config_dict['rotation_lr']
        )

    def update_learning_rate(self, iteration):
        """更新学习率"""
        for param_group in self.xyz_optimizer.param_groups:
            param_group['lr'] = self.get_position_lr(iteration)

    def get_position_lr(self, iteration):
        """获取位置学习率"""
        lr = config.config_dict['position_lr_init']
        lr = lr * (config.config_dict['position_lr_final'] / config.config_dict['position_lr_init']) ** min(
            iteration / config.config_dict['position_lr_max_steps'], 1.0
        )
        return lr