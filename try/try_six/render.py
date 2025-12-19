#!/usr/bin/env python3
"""
渲染器 - 3D高斯溅射的可微分渲染
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math


class GaussianRenderer(nn.Module):
    """3D高斯溅射渲染器"""

    def __init__(self, device: torch.device = torch.device("cuda")):
        super().__init__()
        self.device = device

    def project_gaussians(self, gaussian_params: Dict, camera: Dict) -> Dict:
        # 提取参数
        positions = gaussian_params["positions"]  # (N, 3)
        scales = gaussian_params["scales"]  # (N, 3)
        rotations = gaussian_params["rotations"]  # (N, 4)
        opacities = gaussian_params["opacities"]  # (N, 1)

        # 相机参数 - 统一转换为PyTorch张量并移到GPU
        R = camera["R"]
        t = camera["t"]
        K = camera["K"]

        # 如果传入的是numpy数组，则转换为tensor
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).float().to(self.device)
            t = torch.from_numpy(t).float().to(self.device)
            K = torch.from_numpy(K).float().to(self.device)
        else:
            # 已经是tensor，确保在正确设备上
            R = R.float().to(self.device)
            t = t.float().to(self.device)
            K = K.float().to(self.device)

        # 确保所有张量都是二维的（去掉可能的批处理维度）
        # R应该是(3, 3)，如果不是则去掉多余的维度
        if R.dim() > 2:
            R = R.squeeze(0)
        if t.dim() > 1:
            t = t.squeeze(0)
        if K.dim() > 2:
            K = K.squeeze(0)

        # 确保positions是二维的(N, 3)
        if positions.dim() > 2:
            positions = positions.squeeze(0)

        # 世界坐标 -> 相机坐标
        # 使用更安全的矩阵乘法，避免.T警告
        positions_cam = torch.matmul(positions, R.T) + t.unsqueeze(0)  # (N, 3)

        # 剔除相机后面的点
        valid_mask = positions_cam[:, 2] > 0.1  # 深度为正

        # 确保valid_mask是一维的
        if valid_mask.dim() > 1:
            valid_mask = valid_mask.view(-1)

        if not valid_mask.any():
            return None

        # 应用掩码 - 现在所有张量都是二维的，掩码是一维的
        positions_cam = positions_cam[valid_mask]
        scales = scales[valid_mask]
        rotations = rotations[valid_mask]
        opacities = opacities[valid_mask]
        colors = gaussian_params["colors"][valid_mask]

        # 相机坐标 -> 归一化坐标
        positions_norm = positions_cam / positions_cam[:, 2:3]  # (N, 3)

        # 归一化坐标 -> 像素坐标
        positions_pixel = torch.matmul(positions_norm, K.T)  # (N, 3)
        uv = positions_pixel[:, :2]  # (N, 2)
        depths = positions_cam[:, 2]  # (N,)

        # 计算投影后的2D协方差
        cov2d = self.compute_covariance_2d(
            positions_cam, scales, rotations, K
        )

        return {
            "uv": uv,
            "depths": depths,
            "cov2d": cov2d,
            "opacities": opacities,
            "colors": colors,
            "valid_mask": valid_mask
        }

    def compute_covariance_2d(self, positions_cam: torch.Tensor,
                              scales: torch.Tensor, rotations: torch.Tensor,
                              K: torch.Tensor) -> torch.Tensor:
        """计算2D协方差矩阵"""
        N = positions_cam.shape[0]

        # 确保K是二维的(3, 3)
        if K.dim() > 2:
            K = K.squeeze(0)

        # 计算3D协方差矩阵
        # 从四元数计算旋转矩阵
        q = rotations  # (N, 4)
        R = self.quaternion_to_matrix(q)  # (N, 3, 3)

        # 缩放矩阵的平方
        S_squared = torch.diag_embed(scales ** 2)  # (N, 3, 3)

        # 3D协方差: R S^2 R^T
        cov3d = torch.bmm(torch.bmm(R, S_squared), R.transpose(1, 2))  # (N, 3, 3)

        # 投影矩阵: J = ∂(uv)/∂(xyz_cam)
        x, y, z = positions_cam.unbind(dim=1)
        z_inv = 1.0 / z

        # 雅可比矩阵
        J = torch.zeros(N, 2, 3, device=self.device)
        J[:, 0, 0] = K[0, 0] * z_inv
        J[:, 0, 2] = -K[0, 0] * x * z_inv * z_inv
        J[:, 1, 1] = K[1, 1] * z_inv
        J[:, 1, 2] = -K[1, 1] * y * z_inv * z_inv

        # 投影协方差: Σ' = J Σ J^T
        cov2d = torch.bmm(torch.bmm(J, cov3d), J.transpose(1, 2))  # (N, 2, 2)

        # 添加低通滤波避免数值问题
        cov2d += torch.eye(2, device=self.device).unsqueeze(0) * 0.3

        return cov2d

    def quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
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

    def render(self, gaussian_model, camera: Dict) -> torch.Tensor:
        """渲染图像"""
        # 获取高斯参数
        gaussian_params = gaussian_model.forward()
        # 相机参数 - 统一转换为PyTorch张量并移到GPU
        R = camera["R"]
        t = camera["t"]
        K = camera["K"]

        # 如果传入的是numpy数组，则转换为tensor
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).float().to(self.device)
            t = torch.from_numpy(t).float().to(self.device)
            K = torch.from_numpy(K).float().to(self.device)
        else:
            # 已经是tensor，确保在正确设备上
            R = R.float().to(self.device)
            t = t.float().to(self.device)
            K = K.float().to(self.device)


        # 投影高斯到2D
        proj_data = self.project_gaussians(gaussian_params, camera)
        if proj_data is None:
            # 返回黑色图像
            height, width = int(camera["height"]), int(camera["width"])
            return torch.zeros(3, height, width, device=self.device)

        # 提取投影数据
        uv = proj_data["uv"]  # (N, 2)
        depths = proj_data["depths"]  # (N,)
        cov2d = proj_data["cov2d"]  # (N, 2, 2)
        opacities = proj_data["opacities"]  # (N, 1)
        colors = proj_data["colors"]  # (N, 3)

        # 图像尺寸
        height, width = int(camera["height"]), int(camera["width"])

        # 对高斯进行排序（按深度）
        sorted_indices = torch.argsort(depths, descending=True)  # 从远到近
        uv = uv[sorted_indices]
        cov2d = cov2d[sorted_indices]
        opacities = opacities[sorted_indices]
        colors = colors[sorted_indices]

        # 创建图像网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_coords.float(), y_coords.float()], dim=-1)  # (H, W, 2)
        pixel_coords_flat = pixel_coords.reshape(-1, 2)  # (H*W, 2)

        # 初始化渲染图像
        rendered = torch.zeros(height * width, 3, device=self.device)
        alpha = torch.zeros(height * width, 1, device=self.device)

        # 批次处理高斯（避免内存溢出）
        batch_size = min(1024, uv.shape[0])
        num_batches = (uv.shape[0] + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, uv.shape[0])

            batch_uv = uv[start:end]  # (B, 2)
            batch_cov2d = cov2d[start:end]  # (B, 2, 2)
            batch_opacities = opacities[start:end]  # (B, 1)
            batch_colors = colors[start:end]  # (B, 3)

            # 计算每个高斯对每个像素的影响
            # 使用简化方法：计算高斯在像素中心的概率密度

            # 计算像素到高斯中心的距离
            diff = pixel_coords_flat.unsqueeze(1) - batch_uv.unsqueeze(0)  # (P, B, 2)

            # 计算马氏距离: (x-μ)^T Σ^{-1} (x-μ)
            # 由于Σ是对称正定矩阵，使用Cholesky分解求逆
            try:
                L = torch.linalg.cholesky(batch_cov2d)  # (B, 2, 2)
                Linv = torch.inverse(L)  # (B, 2, 2)
                Sigma_inv = Linv.transpose(1, 2) @ Linv  # (B, 2, 2)

                # 计算二次型
                diff_expanded = diff.unsqueeze(-1)  # (P, B, 2, 1)
                mahalanobis = (diff_expanded.transpose(2, 3) @ Sigma_inv.unsqueeze(0) @ diff_expanded).squeeze(
                    -1).squeeze(-1)  # (P, B)

                # 计算高斯权重
                weights = torch.exp(-0.5 * mahalanobis)  # (P, B)

                # 归一化权重
                weights = weights / (2 * math.pi * torch.sqrt(torch.det(batch_cov2d)).unsqueeze(0) + 1e-8)

                # 应用不透明度
                contributions = weights.unsqueeze(-1) * batch_opacities.unsqueeze(0)  # (P, B, 1)

                # 累积颜色（alpha混合）
                for i in range(batch_colors.shape[0]):
                    color = batch_colors[i]  # (3,)
                    contrib = contributions[:, i]  # (P, 1)

                    # 透明度
                    transmittance = 1 - alpha

                    # 当前层的alpha
                    alpha_i = contrib

                    # 更新颜色
                    rendered += transmittance * alpha_i * color

                    # 更新alpha
                    alpha += transmittance * alpha_i

            except RuntimeError:
                # Cholesky分解失败，跳过这个批次
                continue

        # 重塑为图像
        rendered = rendered.reshape(height, width, 3).permute(2, 0, 1)  # (3, H, W)

        # 添加背景颜色（白色）
        alpha_img = alpha.reshape(height, width, 1).permute(2, 0, 1)  # (1, H, W)
        background = torch.ones(3, height, width, device=self.device)
        rendered = rendered + (1 - alpha_img) * background

        return rendered.clamp(0, 1)


class SimpleRenderer:
    """简化渲染器（用于调试）"""

    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device

    def render(self, gaussian_model, camera: Dict) -> torch.Tensor:
        """简化渲染：仅使用点云投影"""
        # 获取高斯参数
        gaussian_params = gaussian_model.forward()

        # 提取位置和颜色
        positions = gaussian_params["positions"]  # (N, 3)
        colors = gaussian_params["colors"]  # (N, 3)
        opacities = gaussian_params["opacities"]  # (N, 1)

        # 相机参数
        R = torch.from_numpy(camera["R"]).float().to(self.device)  # (3, 3)
        t = torch.from_numpy(camera["t"]).float().to(self.device)  # (3,)
        K = torch.from_numpy(camera["K"]).float().to(self.device)  # (3, 3)

        # 世界坐标 -> 相机坐标
        positions_cam = (R @ positions.T).T + t

        # 剔除相机后面的点
        valid_mask = positions_cam[:, 2] > 0.1
        if not valid_mask.any():
            height, width = int(camera["height"]), int(camera["width"])
            return torch.zeros(3, height, width, device=self.device)

        positions_cam = positions_cam[valid_mask]
        colors = colors[valid_mask]
        opacities = opacities[valid_mask]

        # 投影到图像平面
        positions_norm = positions_cam / positions_cam[:, 2:3]
        positions_pixel = (K @ positions_norm.T).T
        uv = positions_pixel[:, :2].round().long()

        # 图像尺寸
        height, width = int(camera["height"]), int(camera["width"])

        # 创建图像
        image = torch.zeros(3, height, width, device=self.device)

        # 将点绘制到图像上
        for i in range(uv.shape[0]):
            x, y = uv[i]
            if 0 <= x < width and 0 <= y < height:
                # 简单混合
                alpha = opacities[i].item()
                image[:, y, x] = (1 - alpha) * image[:, y, x] + alpha * colors[i]

        return image.clamp(0, 1)