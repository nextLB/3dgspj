import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import config


def build_rotation(q: torch.Tensor) -> torch.Tensor:
    """从四元数构建旋转矩阵（简化版）"""
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=q.device)

    r, i, j, k = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R[:, 0, 0] = 1 - 2 * (j * j + k * k)
    R[:, 0, 1] = 2 * (i * j - k * r)
    R[:, 0, 2] = 2 * (i * k + j * r)
    R[:, 1, 0] = 2 * (i * j + k * r)
    R[:, 1, 1] = 1 - 2 * (i * i + k * k)
    R[:, 1, 2] = 2 * (j * k - i * r)
    R[:, 2, 0] = 2 * (i * k - j * r)
    R[:, 2, 1] = 2 * (j * k + i * r)
    R[:, 2, 2] = 1 - 2 * (i * i + j * j)

    return R


def project_gaussian_to_camera(xyz: torch.Tensor, pose: torch.Tensor, intrinsics: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """将3D高斯投影到相机平面（简化版）"""
    # 世界坐标系到相机坐标系的变换
    R = pose[:3, :3]
    t = pose[:3, 3]

    # 变换到相机坐标系
    xyz_homo = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    xyz_cam = (R @ xyz_homo[:, :3].T + t[:, None]).T

    # 深度
    depths = xyz_cam[:, 2]

    # 投影到图像平面
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # 避免除以0
    eps = 1e-8
    z_inv = 1.0 / (xyz_cam[:, 2] + eps)

    u = fx * xyz_cam[:, 0] * z_inv + cx
    v = fy * xyz_cam[:, 1] * z_inv + cy

    uv = torch.stack([u, v], dim=1)

    return uv, depths


def simple_render_gaussians(uv: torch.Tensor, colors: torch.Tensor, opacity: torch.Tensor,
                            depths: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """简化版高斯渲染"""
    H, W = image_size
    device = uv.device

    # 创建输出图像
    rendered = torch.zeros((H, W, 3), device=device)
    weight_sum = torch.zeros((H, W), device=device)

    # 按深度排序（从远到近）
    sorted_indices = torch.argsort(depths, descending=True)

    for idx in sorted_indices:
        # 获取当前高斯的参数
        u, v = uv[idx]
        color = colors[idx]
        op = opacity[idx].item()

        # 检查是否在图像范围内
        if u < 0 or u >= W or v < 0 or v >= H:
            continue

        # 计算高斯核的影响范围（简化）
        radius = 5  # 固定半径

        # 计算像素网格
        u_min = max(0, int(u - radius))
        u_max = min(W, int(u + radius) + 1)
        v_min = max(0, int(v - radius))
        v_max = min(H, int(v + radius) + 1)

        if u_min >= u_max or v_min >= v_max:
            continue

        # 创建像素坐标网格
        u_grid, v_grid = torch.meshgrid(
            torch.arange(u_min, u_max, device=device).float(),
            torch.arange(v_min, v_max, device=device).float(),
            indexing='ij'
        )

        # 计算距离
        du = u_grid - u
        dv = v_grid - v
        distance_sq = du * du + dv * dv

        # 计算高斯权重
        sigma = radius / 3.0  # 标准差
        weight = torch.exp(-distance_sq / (2 * sigma * sigma))

        # 应用不透明度
        weight = weight * op

        # 更新渲染（alpha混合）
        for i in range(u_max - u_min):
            for j in range(v_max - v_min):
                x, y = u_min + i, v_min + j
                w = weight[i, j].item()

                if w > 0.01:  # 阈值
                    alpha = min(w, 1.0)
                    # 简单的alpha混合
                    rendered[x, y] = rendered[x, y] * (1 - alpha) + color * alpha
                    weight_sum[x, y] += alpha

    # 添加白色背景
    if config.config_dict['white_background']:
        for i in range(H):
            for j in range(W):
                if weight_sum[i, j] < 0.5:  # 如果权重太小，使用白色背景
                    rendered[i, j] = rendered[i, j] + (1 - weight_sum[i, j]) * 1.0

    return rendered


class GaussianRenderer:
    def __init__(self):
        pass

    def render(self, gaussian_model, camera_pose, intrinsics, image_size):
        """简化渲染"""
        params = gaussian_model.get_params()

        # 获取参数
        xyz = params.xyz
        opacity = gaussian_model.get_opacity()
        sh_features = gaussian_model.get_features()

        # 投影到相机
        uv, depths = project_gaussian_to_camera(xyz, camera_pose, intrinsics)

        # 计算颜色（简化版，只使用球谐直流分量）
        if sh_features.shape[1] > 0:
            colors = torch.sigmoid(sh_features[:, 0, :])  # 取第一个球谐系数
        else:
            # 如果没有球谐特征，使用默认颜色
            colors = torch.sigmoid(params.features_dc.squeeze(1))

        # 简单渲染
        rendered = simple_render_gaussians(uv, colors, opacity, depths, image_size)

        return rendered, uv, depths


def compute_loss(rendered, target, lambda_dssim=0.2):
    """计算损失函数（简化版）"""
    # L1损失
    l1_loss = F.l1_loss(rendered, target)

    # 如果lambda_dssim为0，只使用L1损失
    if lambda_dssim <= 0:
        return l1_loss

    # 简化版SSIM损失
    try:
        # 计算均值
        mu_x = torch.mean(rendered, dim=[0, 1, 2])
        mu_y = torch.mean(target, dim=[0, 1, 2])

        # 计算方差和协方差
        sigma_x = torch.var(rendered, unbiased=False)
        sigma_y = torch.var(target, unbiased=False)

        # 简化协方差计算
        sigma_xy = torch.mean((rendered - mu_x) * (target - mu_y))

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_value = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                     ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

        ssim_loss = 1 - ssim_value
        loss = (1 - lambda_dssim) * l1_loss + lambda_dssim * ssim_loss
    except:
        # 如果SSIM计算失败，只使用L1损失
        loss = l1_loss

    return loss