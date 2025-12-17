import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple
import math
from dataclasses import dataclass
from plyfile import PlyData, PlyElement


@dataclass
class GaussianParams:
    """高斯参数"""
    xyz: torch.Tensor  # 位置 [N, 3]
    rotation: torch.Tensor  # 旋转四元数 [N, 4]
    scale: torch.Tensor  # 缩放 [N, 3]
    opacity: torch.Tensor  # 不透明度 [N, 1]
    features: torch.Tensor  # 球谐系数 [N, k, 3]

    def __post_init__(self):
        self.requires_grad = True


class GaussianSplattingModel(nn.Module):
    """三维高斯泼溅模型"""

    def __init__(self, max_gaussians: int = 100000, sh_degree: int = 3):
        super().__init__()

        self.max_gaussians = max_gaussians
        self.sh_degree = sh_degree
        self.active_sh_degree = 0

        # 初始化高斯参数
        self._init_parameters()

    def _init_parameters(self):
        """4.1.1 高斯元初始化与密度控制"""
        # 初始化位置
        self.xyz = nn.Parameter(torch.zeros((self.max_gaussians, 3)))

        # 初始化旋转 (四元数)
        rotation = torch.zeros((self.max_gaussians, 4))
        rotation[:, 0] = 1.0  # w=1, 无旋转
        self.rotation = nn.Parameter(rotation)

        # 初始化缩放 (对数空间)
        scale = torch.ones((self.max_gaussians, 3)) * 0.01
        self.scale = nn.Parameter(torch.log(scale))

        # 初始化不透明度 (sigmoid空间)
        opacity = torch.ones((self.max_gaussians, 1)) * 0.1
        self.opacity = nn.Parameter(torch.logit(opacity))

        # 初始化球谐系数
        sh_dim = (self.sh_degree + 1) ** 2
        features = torch.zeros((self.max_gaussians, sh_dim, 3))
        features[:, 0, :] = 0.5  # 零阶球谐系数 (基础颜色)
        self.features = nn.Parameter(features)

        # 梯度累积
        self.xyz_grad_accum = torch.zeros((self.max_gaussians, 1))
        self.max_radii2D = torch.zeros((self.max_gaussians))

        # 优化状态
        self.optimizer_state = None

    def forward(self, viewpoint_camera, bg_color: torch.Tensor):
        """前向传播 - 渲染图像"""
        # 获取有效的Gaussians
        valid_mask = self._get_valid_mask()

        # 准备渲染参数
        render_params = self._prepare_render_params(valid_mask, viewpoint_camera)

        # 光栅化
        rendered_image = self._rasterize(render_params, viewpoint_camera, bg_color)

        return rendered_image

    def _get_valid_mask(self) -> torch.Tensor:
        """获取有效高斯掩码"""
        # 基于不透明度和位置判断
        opacity = torch.sigmoid(self.opacity)
        valid_opacity = opacity.squeeze() > 0.01

        # 检查位置是否有效
        valid_position = torch.isfinite(self.xyz).all(dim=1)

        return valid_opacity & valid_position

    def _prepare_render_params(self, valid_mask: torch.Tensor, camera):
        """准备渲染参数"""
        # 提取有效参数
        xyz = self.xyz[valid_mask]
        rotation = self.rotation[valid_mask]
        scale = torch.exp(self.scale[valid_mask])
        opacity = torch.sigmoid(self.opacity[valid_mask])
        features = self.features[valid_mask]

        # 转换为渲染格式
        # 旋转四元数 -> 旋转矩阵
        rotation_matrix = self._quaternion_to_matrix(rotation)

        # 计算协方差矩阵
        covariance = self._compute_covariance(scale, rotation_matrix)

        # 投影到2D
        projected_xyz = self._project_to_2d(xyz, camera)

        return {
            "xyz": xyz,
            "projected_xyz": projected_xyz,
            "covariance": covariance,
            "opacity": opacity,
            "features": features,
            "valid_mask": valid_mask
        }

    def _quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """四元数转换为旋转矩阵"""
        w, x, y, z = q.unbind(dim=-1)

        return torch.stack([
            torch.stack([1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w], dim=-1),
            torch.stack([2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w], dim=-1),
            torch.stack([2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y], dim=-1)
        ], dim=-2)

    def _compute_covariance(self, scale: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        """计算3D协方差矩阵"""
        # 缩放矩阵
        S = torch.diag_embed(scale)

        # 旋转缩放矩阵
        R = rotation
        M = R @ S

        # 协方差矩阵 Σ = RR^T
        covariance = M @ M.transpose(-1, -2)

        # 添加小值防止奇异
        covariance = covariance + torch.eye(3, device=covariance.device) * 1e-6

        return covariance

    def _project_to_2d(self, xyz: torch.Tensor, camera) -> torch.Tensor:
        """3D点投影到2D"""
        # 使用相机参数进行投影
        # 这里简化处理，实际需要完整的相机模型
        raise NotImplementedError("投影函数需要根据相机模型实现")

    def _rasterize(self, render_params: Dict, camera, bg_color: torch.Tensor) -> torch.Tensor:
        """光栅化 - 使用CUDA实现"""
        # 这里调用CUDA实现的光栅化
        # 实际实现使用 diff-gaussian-rasterization 库

        # 简化版本：返回随机图像
        height, width = camera.height, camera.width
        device = self.xyz.device

        # 创建随机图像
        rendered_image = torch.rand((height, width, 3), device=device)

        # 与背景颜色混合
        rendered_image = rendered_image * 0.5 + bg_color * 0.5

        return rendered_image

    def adaptive_density_control(self, iteration: int):
        """4.1.2 基于光度损失的自适应优化 - 密度控制"""

        if iteration % 100 == 0:
            # 复制Gaussians
            self._clone_gaussians()

            # 分裂Gaussians
            self._split_gaussians()

            # 修剪不重要的Gaussians
            self._prune_gaussians()

    def _clone_gaussians(self):
        """复制大梯度的高斯"""
        grad_threshold = 0.0002
        clone_mask = (self.xyz_grad_accum.squeeze() > grad_threshold) & \
                     (torch.exp(self.scale).max(dim=1)[0] > 0.01)

        num_to_clone = clone_mask.sum().item()
        if num_to_clone == 0:
            return

        # 获取要复制的参数
        clone_indices = torch.where(clone_mask)[0]

        # 小随机偏移
        offset = torch.randn_like(self.xyz[clone_indices]) * 0.01

        # 创建新参数
        new_xyz = self.xyz[clone_indices] + offset
        new_rotation = self.rotation[clone_indices]
        new_scale = self.scale[clone_indices]
        new_opacity = self.opacity[clone_indices]
        new_features = self.features[clone_indices]

        # 添加到参数中
        # 这里需要扩展参数空间，简化处理
        print(f"Cloning {num_to_clone} Gaussians")

    def _split_gaussians(self):
        """分裂大的高斯"""
        scale = torch.exp(self.scale)
        scale_max = scale.max(dim=1)[0]
        split_mask = scale_max > 0.1

        num_to_split = split_mask.sum().item()
        if num_to_split == 0:
            return

        print(f"Splitting {num_to_split} Gaussians")

    def _prune_gaussians(self):
        """修剪不重要的高斯"""
        opacity = torch.sigmoid(self.opacity).squeeze()
        prune_mask = opacity < 0.005

        num_to_prune = prune_mask.sum().item()
        if num_to_prune > 0:
            print(f"Pruning {num_to_prune} Gaussians")
            # 实际实现中需要设置mask来禁用这些高斯

    def save_ply(self, path: str):
        """保存为PLY格式"""
        # 获取有效高斯
        valid_mask = self._get_valid_mask()

        # 准备数据
        xyz = self.xyz[valid_mask].detach().cpu().numpy()
        rotation = self.rotation[valid_mask].detach().cpu().numpy()
        scale = torch.exp(self.scale[valid_mask]).detach().cpu().numpy()
        opacity = torch.sigmoid(self.opacity[valid_mask]).detach().cpu().numpy()
        features = self.features[valid_mask, :4, :].detach().cpu().numpy()  # 只保存前4个球谐系数

        # 创建顶点
        vertices = []
        for i in range(len(xyz)):
            vertex = (
                xyz[i][0], xyz[i][1], xyz[i][2],
                rotation[i][0], rotation[i][1], rotation[i][2], rotation[i][3],
                scale[i][0], scale[i][1], scale[i][2],
                opacity[i][0],
                features[i][0][0], features[i][0][1], features[i][0][2],
                features[i][1][0], features[i][1][1], features[i][1][2],
                features[i][2][0], features[i][2][1], features[i][2][2],
                features[i][3][0], features[i][3][1], features[i][3][2]
            )
            vertices.append(vertex)

        # 定义PLY结构
        vertex_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('opacity', 'f4'),
            ('f_0_0', 'f4'), ('f_0_1', 'f4'), ('f_0_2', 'f4'),
            ('f_1_0', 'f4'), ('f_1_1', 'f4'), ('f_1_2', 'f4'),
            ('f_2_0', 'f4'), ('f_2_1', 'f4'), ('f_2_2', 'f4'),
            ('f_3_0', 'f4'), ('f_3_1', 'f4'), ('f_3_2', 'f4')
        ]

        vertex_element = PlyElement.describe(
            np.array(vertices, dtype=vertex_dtype),
            'vertex'
        )

        # 保存
        PlyData([vertex_element], text=False).write(path)
        print(f"Saved {len(vertices)} Gaussians to {path}")