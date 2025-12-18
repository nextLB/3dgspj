import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import math
from plyfile import PlyData, PlyElement
import os


def inverse_sigmoid(x):
    """Sigmoid的逆函数"""
    return torch.log(x / (1 - x))


def build_rotation(r):
    """从四元数构建旋转矩阵"""
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=q.device, dtype=q.dtype)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def build_scaling_rotation(s, r):
    """构建缩放旋转矩阵"""
    L = torch.zeros((s.size(0), 3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    return R @ L


class GaussianModel(nn.Module):
    """3D高斯模型"""

    def __init__(self, sh_degree: int = 3):
        super().__init__()

        self.sh_degree = sh_degree
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        # 初始化高斯属性
        self._xyz = nn.Parameter(torch.empty(0))
        self._features_dc = nn.Parameter(torch.empty(0))
        self._features_rest = nn.Parameter(torch.empty(0))
        self._scaling = nn.Parameter(torch.empty(0))
        self._rotation = nn.Parameter(torch.empty(0))
        self._opacity = nn.Parameter(torch.empty(0))

        # 优化器设置
        self.optimizer = None
        self.spatial_lr_scale = 1.0

        # 密度控制相关
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.percent_dense = 0.0

        # Mip滤波相关
        self.use_mip = True
        self.mip_levels = 3

    @property
    def get_xyz(self):
        """获取位置参数"""
        return self._xyz

    @property
    def get_features(self):
        """获取特征参数"""
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_scaling(self):
        """获取缩放参数"""
        scaling = torch.exp(self._scaling)
        return scaling

    @property
    def get_rotation(self):
        """获取旋转参数"""
        rotation = torch.nn.functional.normalize(self._rotation)
        return rotation

    @property
    def get_opacity(self):
        """获取不透明度参数"""
        opacity = torch.sigmoid(self._opacity)
        return opacity

    def capture_state_dict(self):
        """捕获模型状态字典"""
        return {
            '_xyz': self._xyz.data.clone(),
            '_features_dc': self._features_dc.data.clone(),
            '_features_rest': self._features_rest.data.clone(),
            '_scaling': self._scaling.data.clone(),
            '_rotation': self._rotation.data.clone(),
            '_opacity': self._opacity.data.clone(),
            'active_sh_degree': self.active_sh_degree,
            'max_sh_degree': self.max_sh_degree
        }

    def load_state_dict(self, state_dict):
        """加载模型状态字典"""
        self._xyz = nn.Parameter(state_dict['_xyz'])
        self._features_dc = nn.Parameter(state_dict['_features_dc'])
        self._features_rest = nn.Parameter(state_dict['_features_rest'])
        self._scaling = nn.Parameter(state_dict['_scaling'])
        self._rotation = nn.Parameter(state_dict['_rotation'])
        self._opacity = nn.Parameter(state_dict['_opacity'])
        self.active_sh_degree = state_dict.get('active_sh_degree', 0)
        self.max_sh_degree = state_dict.get('max_sh_degree', self.sh_degree)

    def create_from_pcd(self, pcd, colors, spatial_lr_scale=1.0):
        """从点云初始化高斯模型"""

        self.spatial_lr_scale = spatial_lr_scale

        # 转换为张量
        if isinstance(pcd, np.ndarray):
            pcd = torch.tensor(pcd, dtype=torch.float32)
        if isinstance(colors, np.ndarray):
            colors = torch.tensor(colors, dtype=torch.float32)

        # 确保颜色在[0, 1]范围内
        colors = torch.clamp(colors, 0.0, 1.0)

        # 初始化位置
        self._xyz = nn.Parameter(pcd.clone().requires_grad_(True))

        # 初始化颜色特征（球谐函数）
        fused_color = self.rgb_to_sh(colors)

        # DC分量（0阶球谐）
        features_dc = torch.zeros((pcd.shape[0], 3, 1), dtype=torch.float32)
        features_dc[:, :, 0] = fused_color

        # 高阶球谐分量（初始化为0）
        extra_f_names = ["f_rest_" + str(idx) for idx in range((self.max_sh_degree + 1) ** 2 - 1)]
        features_extra = torch.zeros((pcd.shape[0], 3, len(extra_f_names)), dtype=torch.float32)

        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features_extra.transpose(1, 2).contiguous().requires_grad_(True))

        # 初始化缩放（对数尺度）
        dists = torch.clamp_min(distCUDA2(pcd), 0.0000001)
        scales = torch.log(torch.sqrt(dists))[..., None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)

        self._scaling = nn.Parameter(scales.requires_grad_(True))

        # 初始化旋转（四元数）
        rots = torch.zeros((pcd.shape[0], 4), dtype=torch.float32)
        rots[:, 0] = 1.0  # 实部为1，虚部为0 -> 无旋转

        self._rotation = nn.Parameter(rots.requires_grad_(True))

        # 初始化不透明度
        opacities = inverse_sigmoid(0.1 * torch.ones((pcd.shape[0], 1), dtype=torch.float32))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        # 初始化密度控制变量
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=pcd.device)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=pcd.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=pcd.device)

        print(f"Created Gaussian model with {pcd.shape[0]} points")

    def training_setup(self, args):
        """设置训练参数"""
        self.percent_dense = args.percent_dense

        # 学习率设置
        l = [
            {'params': [self._xyz], 'lr': args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # 学习率调度器
        self.xyz_scheduler_args = {
            'max_steps': args.position_lr_max_steps,
            'lr_init': args.position_lr_init,
            'lr_final': args.position_lr_final
        }

    def update_learning_rate(self, iteration):
        """更新学习率"""
        # 位置学习率衰减
        if iteration < self.xyz_scheduler_args['max_steps']:
            t = iteration / self.xyz_scheduler_args['max_steps']
            lr = self.xyz_scheduler_args['lr_init'] * (1 - t) + self.xyz_scheduler_args['lr_final'] * t
        else:
            lr = self.xyz_scheduler_args['lr_final']

        # 更新优化器中的学习率
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = lr * self.spatial_lr_scale
                break

    def construct_list_of_attributes(self):
        """构建PLY文件属性列表"""
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        # 颜色
        l.append('f_dc_0')
        l.append('f_dc_1')
        l.append('f_dc_2')

        # 高阶球谐
        for i in range((self.max_sh_degree + 1) ** 2 - 1):
            l.append(f'f_rest_{i}')

        l.append('opacity')
        l.append('scale_0')
        l.append('scale_1')
        l.append('scale_2')
        l.append('rot_0')
        l.append('rot_1')
        l.append('rot_2')
        l.append('rot_3')

        return l

    def save_ply(self, path):
        """保存为PLY文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        opacities = self._opacity.detach().cpu().numpy()

        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        print(f"Saved {xyz.shape[0]} Gaussians to {path}")

    def densify_and_prune(self, percent_dense, visibility_filter, radii, iteration):
        """密度控制和剪枝"""
        # 获取当前参数
        xyz = self._xyz
        scaling = self._scaling
        rotation = self._rotation
        opacity = self._opacity
        features_dc = self._features_dc
        features_rest = self._features_rest

        # 重置梯度累积
        self.xyz_gradient_accum = self.xyz_gradient_accum * 0.0
        self.denom = self.denom * 0.0

        # 更新最大半径
        self.max_radii2D[visibility_filter] = torch.max(
            self.max_radii2D[visibility_filter],
            radii[visibility_filter]
        )

        # 计算需要密化的位置
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # 选择梯度大的点进行分裂
        split_mask = (torch.max(torch.exp(scaling), dim=1).values > percent_dense * self.max_radii2D) & \
                     (grads.squeeze() > 0.0002)

        # 选择不透明度低的点进行剪枝
        prune_mask = (self.get_opacity < 0.005).squeeze()

        # 执行分裂
        if split_mask.sum() > 0:
            self.split_gaussians(split_mask)

        # 执行剪枝
        if prune_mask.sum() > 0:
            self.prune_gaussians(prune_mask)

    def split_gaussians(self, split_mask):
        """分裂高斯"""
        n_split = split_mask.sum().item()
        if n_split == 0:
            return

        print(f"Splitting {n_split} Gaussians")

        # 获取需要分裂的高斯参数
        xyz_split = self._xyz[split_mask]
        scaling_split = self._scaling[split_mask]
        rotation_split = self._rotation[split_mask]
        opacity_split = self._opacity[split_mask]
        features_dc_split = self._features_dc[split_mask]
        features_rest_split = self._features_rest[split_mask]

        # 创建新高斯（分裂为两个）
        N = xyz_split.shape[0]

        # 新位置：稍微偏移
        std = torch.exp(scaling_split) * 0.2
        offset = torch.randn((N, 3), device=xyz_split.device) * std
        xyz_new = torch.cat([xyz_split + offset, xyz_split - offset], dim=0)

        # 新缩放：稍微缩小
        scaling_new = torch.cat([scaling_split - 0.1, scaling_split - 0.1], dim=0)

        # 复制其他参数
        rotation_new = rotation_split.repeat(2, 1)
        opacity_new = opacity_split.repeat(2, 1)
        features_dc_new = features_dc_split.repeat(2, 1, 1)
        features_rest_new = features_rest_split.repeat(2, 1, 1)

        # 更新模型参数
        self._xyz = nn.Parameter(torch.cat([self._xyz, xyz_new], dim=0).requires_grad_(True))
        self._scaling = nn.Parameter(torch.cat([self._scaling, scaling_new], dim=0).requires_grad_(True))
        self._rotation = nn.Parameter(torch.cat([self._rotation, rotation_new], dim=0).requires_grad_(True))
        self._opacity = nn.Parameter(torch.cat([self._opacity, opacity_new], dim=0).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, features_dc_new], dim=0).requires_grad_(True))
        self._features_rest = nn.Parameter(
            torch.cat([self._features_rest, features_rest_new], dim=0).requires_grad_(True))

        # 更新密度控制变量
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(2 * N, device=self.max_radii2D.device)], dim=0)
        self.xyz_gradient_accum = torch.cat([
            self.xyz_gradient_accum,
            torch.zeros((2 * N, 1), device=self.xyz_gradient_accum.device)
        ], dim=0)
        self.denom = torch.cat([
            self.denom,
            torch.zeros((2 * N, 1), device=self.denom.device)
        ], dim=0)

    def prune_gaussians(self, prune_mask):
        """剪枝高斯"""
        n_prune = prune_mask.sum().item()
        if n_prune == 0:
            return

        print(f"Pruning {n_prune} Gaussians")

        # 保留不需要剪枝的高斯
        keep_mask = ~prune_mask

        self._xyz = nn.Parameter(self._xyz[keep_mask].requires_grad_(True))
        self._scaling = nn.Parameter(self._scaling[keep_mask].requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation[keep_mask].requires_grad_(True))
        self._opacity = nn.Parameter(self._opacity[keep_mask].requires_grad_(True))
        self._features_dc = nn.Parameter(self._features_dc[keep_mask].requires_grad_(True))
        self._features_rest = nn.Parameter(self._features_rest[keep_mask].requires_grad_(True))

        # 更新密度控制变量
        self.max_radii2D = self.max_radii2D[keep_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep_mask]
        self.denom = self.denom[keep_mask]

    def reset_opacity(self):
        """重置不透明度"""
        opacities = self._opacity.data
        mean_opacity = torch.sigmoid(opacities).mean().item()

        # 如果平均不透明度太低，重置
        if mean_opacity < 0.01:
            new_opacities = inverse_sigmoid(
                torch.min(torch.sigmoid(opacities) * 1.2, torch.ones_like(opacities) * 0.01))
            self._opacity.data = new_opacities

    def rgb_to_sh(self, rgb):
        """将RGB颜色转换为球谐函数表示"""
        C0 = 0.28209479177387814  # sqrt(1/(4*pi))
        return rgb / C0

    def sh_to_rgb(self, sh):
        """将球谐函数表示转换为RGB颜色"""
        C0 = 0.28209479177387814  # sqrt(1/(4*pi))
        return sh * C0

    def get_covariance(self, scaling_modifier=1.0):
        """计算协方差矩阵"""
        scaling = self.get_scaling * scaling_modifier
        rotation = self.get_rotation

        # 构建缩放旋转矩阵
        L = build_scaling_rotation(scaling, rotation)

        # 协方差矩阵 = L @ L^T
        covariance = L @ L.transpose(1, 2)

        return covariance

    def apply_mip_filtering(self, cov_3d, pixel_size):
        """应用Mip滤波"""
        if not self.use_mip:
            return cov_3d

        # 添加像素大小相关的不确定性
        pixel_cov = torch.eye(3, device=cov_3d.device).unsqueeze(0) * (pixel_size ** 2)
        cov_3d_filtered = cov_3d + pixel_cov

        return cov_3d_filtered

    def forward(self):
        """前向传播（返回所有参数）"""
        return {
            'xyz': self.get_xyz,
            'features_dc': self._features_dc,
            'features_rest': self._features_rest,
            'scaling': self.get_scaling,
            'rotation': self.get_rotation,
            'opacity': self.get_opacity,
            'covariance': self.get_covariance()
        }


# 辅助函数
def distCUDA2(points):
    """计算点到其最近邻居的距离平方（优化版本，避免内存溢出）"""
    n_points = points.shape[0]

    if n_points < 2:
        return torch.ones(n_points, device=points.device) * 0.1

    # 方法1：使用KDTree或BallTree（需要scipy）
    try:
        from scipy.spatial import KDTree
        # 将点云转移到CPU进行KDTree查询（节省GPU内存）
        points_cpu = points.cpu().numpy()
        tree = KDTree(points_cpu)

        # 查询每个点的最近邻居（排除自身）
        distances, _ = tree.query(points_cpu, k=2)  # k=2: 第一个是自身，第二个是最近邻居
        distances = distances[:, 1]  # 获取最近邻居的距离

        # 转换为torch张量并返回GPU
        min_dist = torch.tensor(distances, device=points.device, dtype=points.dtype) ** 2
        return min_dist
    except ImportError:
        # 方法2：分批计算（避免O(n²)内存）
        print("Warning: scipy not available, using batch computation")

        batch_size = 1000  # 分批大小，根据GPU内存调整
        min_dist = torch.zeros(n_points, device=points.device)

        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            batch_points = points[i:end_idx]

            # 计算这批点与所有点的距离
            diff = batch_points.unsqueeze(1) - points.unsqueeze(0)  # (batch, n, 3)
            dist = torch.sum(diff ** 2, dim=-1)  # (batch, n)

            # 将对角线设置为大值（排除自身）
            dist[:, i:end_idx] = 1e10

            # 找到最小距离
            batch_min_dist, _ = torch.min(dist, dim=1)
            min_dist[i:end_idx] = batch_min_dist

        return min_dist


def strip_lowerdiag(L):
    """提取下三角部分"""
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]

    return uncertainty


def strip_symmetric(sym):
    """提取对称矩阵的独立元素"""
    return strip_lowerdiag(sym)