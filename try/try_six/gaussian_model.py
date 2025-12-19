#!/usr/bin/env python3
"""
3D高斯模型 - 表示可微分的3D高斯分布
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math


class Gaussian3D(nn.Module):
    """3D高斯分布表示"""

    def __init__(self, num_points: int, device: torch.device = torch.device("cuda")):
        super().__init__()

        self.num_points = num_points
        self.device = device

        # 位置 (x, y, z)
        self.positions = nn.Parameter(torch.randn(num_points, 3, device=device) * 0.01)

        # 缩放 (sx, sy, sz)，使用对数尺度以确保正数
        self.scales = nn.Parameter(torch.ones(num_points, 3, device=device) * 0.01)

        # 旋转 (四元数: qw, qx, qy, qz)
        quats = torch.randn(num_points, 4, device=device)
        quats = quats / torch.norm(quats, dim=1, keepdim=True)
        self.rotations = nn.Parameter(quats)

        # 不透明度 (使用sigmoid激活确保在[0, 1]范围内)
        self.opacities = nn.Parameter(torch.ones(num_points, 1, device=device) * 0.1)

        # 颜色 (使用球谐函数系数)
        # 这里简化：使用RGB颜色
        self.colors = nn.Parameter(torch.rand(num_points, 3, device=device))

        # 激活函数
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.color_activation = torch.sigmoid

        # 梯度累积
        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=device)
        self.denom = torch.zeros((self.num_points, 1), device=device)

        # 优化器设置
        self.spatial_lr_scale = 1.0
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.005
        self.position_lr = 0.00016

        # 最大缩放限制
        self.max_scaling = 0.05
        self.min_scaling = 0.001

    def forward(self):
        """前向传播，返回激活后的参数"""
        scales = self.scale_activation(self.scales)
        scales = torch.clamp(scales, min=self.min_scaling, max=self.max_scaling)

        opacities = self.opacity_activation(self.opacities)
        colors = self.color_activation(self.colors)

        # 归一化四元数
        rotations = F.normalize(self.rotations, dim=1)

        return {
            "positions": self.positions,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
            "colors": colors
        }

    def get_covariance(self):
        """计算协方差矩阵"""
        params = self.forward()

        # 从四元数计算旋转矩阵
        q = params["rotations"]  # (N, 4)
        R = self.quaternion_to_matrix(q)  # (N, 3, 3)

        # 缩放矩阵
        S = torch.diag_embed(params["scales"])  # (N, 3, 3)

        # 协方差矩阵: R S S^T R^T = R S^2 R^T
        S_squared = torch.diag_embed(params["scales"] ** 2)
        covariance = R @ S_squared @ R.transpose(1, 2)

        return covariance

    def quaternion_to_matrix(self, q):
        """四元数转旋转矩阵"""
        # 归一化
        q = F.normalize(q, dim=1)

        qw, qx, qy, qz = q.unbind(dim=1)

        # 计算旋转矩阵
        R = torch.stack([
            1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
            2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx),
            2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)
        ], dim=1).reshape(-1, 3, 3)

        return R

    def prune_points(self, prune_mask):
        """修剪点"""
        keep_mask = ~prune_mask
        self.num_points = keep_mask.sum().item()

        # 更新所有参数
        self.positions = nn.Parameter(self.positions[keep_mask])
        self.scales = nn.Parameter(self.scales[keep_mask])
        self.rotations = nn.Parameter(self.rotations[keep_mask])
        self.opacities = nn.Parameter(self.opacities[keep_mask])
        self.colors = nn.Parameter(self.colors[keep_mask])

        # 重置梯度累积
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep_mask]
        self.denom = self.denom[keep_mask]

    def densify_points(self, positions, colors, scales=None, rotations=None, opacities=None):
        """增加点密度"""
        num_new = positions.shape[0]

        # 扩展参数
        self.positions = nn.Parameter(torch.cat([self.positions, positions], dim=0))
        self.colors = nn.Parameter(torch.cat([self.colors, colors], dim=0))

        if scales is not None:
            self.scales = nn.Parameter(torch.cat([self.scales, scales], dim=0))
        else:
            self.scales = nn.Parameter(
                torch.cat([self.scales, torch.ones(num_new, 3, device=self.device) * 0.01], dim=0))

        if rotations is not None:
            self.rotations = nn.Parameter(torch.cat([self.rotations, rotations], dim=0))
        else:
            quats = torch.randn(num_new, 4, device=self.device)
            quats = quats / torch.norm(quats, dim=1, keepdim=True)
            self.rotations = nn.Parameter(torch.cat([self.rotations, quats], dim=0))

        if opacities is not None:
            self.opacities = nn.Parameter(torch.cat([self.opacities, opacities], dim=0))
        else:
            self.opacities = nn.Parameter(
                torch.cat([self.opacities, torch.ones(num_new, 1, device=self.device) * 0.1], dim=0))

        # 更新梯度累积
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros(num_new, 1, device=self.device)],
                                            dim=0)
        self.denom = torch.cat([self.denom, torch.zeros(num_new, 1, device=self.device)], dim=0)

        self.num_points += num_new

    def reset_opacities(self):
        """重置不透明度"""
        self.opacities = nn.Parameter(torch.ones(self.num_points, 1, device=self.device) * 0.1)

    def save_ply(self, path):
        """保存为PLY文件"""
        params = self.forward()

        positions = params["positions"].detach().cpu().numpy()
        colors = (params["colors"] * 255).clamp(0, 255).byte().detach().cpu().numpy()
        opacities = params["opacities"].detach().cpu().numpy()
        scales = params["scales"].detach().cpu().numpy()
        rotations = params["rotations"].detach().cpu().numpy()

        # 创建PLY文件
        with open(path, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {self.num_points}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property float nx\n')
            f.write('property float ny\n')
            f.write('property float nz\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('property uchar alpha\n')
            f.write('property float scale_0\n')
            f.write('property float scale_1\n')
            f.write('property float scale_2\n')
            f.write('property float rot_0\n')
            f.write('property float rot_1\n')
            f.write('property float rot_2\n')
            f.write('property float rot_3\n')
            f.write('end_header\n')

            for i in range(self.num_points):
                x, y, z = positions[i]
                r, g, b = colors[i]
                alpha = int(opacities[i][0] * 255)
                scale_x, scale_y, scale_z = scales[i]
                qw, qx, qy, qz = rotations[i]

                f.write(f'{x} {y} {z} 0 0 0 {r} {g} {b} {alpha} {scale_x} {scale_y} {scale_z} {qw} {qx} {qy} {qz}\n')

    @classmethod
    def from_point_cloud(cls, points: np.ndarray, colors: np.ndarray = None,
                         device: torch.device = torch.device("cuda")):
        """从点云初始化"""
        num_points = points.shape[0]
        gaussian = cls(num_points, device)

        # 设置位置
        gaussian.positions = nn.Parameter(torch.from_numpy(points).float().to(device))

        # 设置颜色
        if colors is not None:
            gaussian.colors = nn.Parameter(torch.from_numpy(colors).float().to(device))

        return gaussian