import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GaussianRenderer:
    """
    3D高斯的可微分光栅化渲染器。
    实现基于Tile的光栅化和alpha混合。
    """

    def __init__(self, tile_size: int = 16, background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.tile_size = tile_size
        self.background_color = torch.tensor(background_color, dtype=torch.float32)

    def project_gaussians(self, xyz: torch.Tensor, scale: torch.Tensor, rotation: torch.Tensor,
                          K: torch.Tensor, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将3D高斯投影到2D图像平面。
        返回：
            mean2d: 投影后的2D均值 [N, 2]
            cov2d: 投影后的2D协方差 [N, 2, 2]
            depth: 高斯深度（用于排序）[N]
        """
        device = xyz.device

        # 1. 将高斯变换到相机坐标系
        R = pose[:3, :3]  # 旋转矩阵 [3, 3]
        t = pose[:3, 3]  # 平移向量 [3]

        # 世界坐标系到相机坐标系
        xyz_cam = torch.matmul(R, xyz.T).T + t.unsqueeze(0)  # [N, 3]

        # 深度（用于排序）
        depth = xyz_cam[:, 2]  # Z轴深度

        # 2. 投影到图像平面（透视投影）
        xyz_proj = torch.matmul(K, xyz_cam.T).T  # [N, 3]
        uv = xyz_proj[:, :2] / xyz_proj[:, 2:]  # 归一化设备坐标 [N, 2]

        # 3. 计算投影后的2D协方差（简化版本，基于3D协方差的投影）
        # 构建3D协方差矩阵（假设各向异性缩放和旋转）
        # 注意：这是简化版，完整实现需计算精确的投影协方差
        scale_matrix = torch.diag_embed(scale)  # [N, 3, 3]
        # 将四元数转换为旋转矩阵（简化：假设无旋转或使用给定旋转）
        # 此处使用单位矩阵作为旋转的近似
        rot_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(xyz.shape[0], 1, 1)

        # 3D协方差：S * R * R^T * S^T
        cov3d = torch.matmul(scale_matrix, torch.matmul(rot_matrix, scale_matrix.transpose(1, 2)))

        # 投影到2D（雅可比矩阵近似）
        focal_x = K[0, 0]
        focal_y = K[1, 1]
        z = xyz_cam[:, 2]

        # 投影雅可比矩阵：J = [[f_x / z, 0, -f_x * x / z^2],
        #                     [0, f_y / z, -f_y * y / z^2]]
        # 简化：仅考虑主对角线项
        J = torch.zeros(xyz.shape[0], 2, 3, device=device)
        J[:, 0, 0] = focal_x / z
        J[:, 1, 1] = focal_y / z

        # 2D协方差：J * cov3d * J^T
        cov2d = torch.matmul(J, torch.matmul(cov3d, J.transpose(1, 2)))  # [N, 2, 2]

        # 添加小值以确保正定性
        cov2d = cov2d + torch.eye(2, device=device).unsqueeze(0) * 1e-6

        return uv, cov2d, depth

    def compute_alpha_from_covariance(self, cov2d: torch.Tensor, uv: torch.Tensor,
                                      pixel_uv: torch.Tensor) -> torch.Tensor:
        """
        根据2D高斯协方差计算像素位置的不透明度贡献。
        使用Mip滤波技术：考虑像素的积分区域。
        """
        # 像素坐标与高斯中心的偏移
        diff = pixel_uv - uv.unsqueeze(1)  # [N, P, 2]，P是像素数

        # 计算马氏距离：d^T * cov^{-1} * d
        # 求协方差矩阵的逆
        try:
            cov_inv = torch.linalg.inv(cov2d)  # [N, 2, 2]
        except:
            # 如果求逆失败，使用伪逆
            cov_inv = torch.linalg.pinv(cov2d)

        # 扩展维度以便广播
        cov_inv = cov_inv.unsqueeze(1)  # [N, 1, 2, 2]
        diff = diff.unsqueeze(-1)  # [N, P, 2, 1]

        # 计算二次型：diff^T * cov_inv * diff
        mahalanobis = torch.matmul(diff.transpose(2, 3), torch.matmul(cov_inv, diff))  # [N, P, 1, 1]
        mahalanobis = mahalanobis.squeeze(-1).squeeze(-1)  # [N, P]

        # 高斯函数：exp(-0.5 * d^2)
        alpha = torch.exp(-0.5 * mahalanobis)

        # Mip滤波：考虑像素的有限面积（盒式滤波器）
        # 近似：将像素视为半径为0.5的圆盘
        pixel_radius = 0.5
        # 调整alpha以考虑像素积分
        alpha = alpha * (2 * np.pi * torch.det(cov2d).sqrt().unsqueeze(1))
        alpha = torch.clamp(alpha, 0, 1)

        return alpha

    def rasterize(self, uv: torch.Tensor, cov2d: torch.Tensor, depth: torch.Tensor,
                  opacity: torch.Tensor, colors: torch.Tensor,
                  H: int, W: int, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        光栅化：将高斯渲染到图像。
        返回：
            rendered: 渲染图像 [H, W, 3]
            alpha_map: alpha通道 [H, W, 1]
        """
        device = uv.device

        # 1. 按深度对高斯排序（从远到近）
        sorted_indices = torch.argsort(depth, descending=False)  # 从近到远渲染
        uv = uv[sorted_indices]
        cov2d = cov2d[sorted_indices]
        depth = depth[sorted_indices]
        opacity = opacity[sorted_indices]
        colors = colors[sorted_indices]

        # 2. 创建输出图像
        rendered = torch.ones(H, W, 3, device=device) * self.background_color.to(device)
        alpha_map = torch.zeros(H, W, 1, device=device)

        # 3. 基于Tile的光栅化（简化版：逐个高斯处理）
        num_gaussians = uv.shape[0]

        # 生成像素网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        pixel_uv = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)  # [H*W, 2]

        # 4. 对每个高斯计算贡献（实际实现中应使用更高效的方法）
        # 这里简化：仅处理前1000个高斯以避免内存问题
        max_gaussians = min(1000, num_gaussians)

        for i in range(0, max_gaussians, 100):
            end_idx = min(i + 100, max_gaussians)
            batch_uv = uv[i:end_idx]
            batch_cov = cov2d[i:end_idx]
            batch_opacity = opacity[i:end_idx].unsqueeze(1)  # [B, 1]
            batch_colors = colors[i:end_idx]  # [B, 3]

            # 计算alpha值
            batch_alpha = self.compute_alpha_from_covariance(batch_cov, batch_uv, pixel_uv.unsqueeze(0))  # [B, H*W]

            # 应用不透明度
            batch_alpha = batch_alpha * torch.sigmoid(batch_opacity)
            batch_alpha = batch_alpha.reshape(-1, H, W)  # [B, H, W]

            # alpha混合（从后往前）
            for j in range(batch_alpha.shape[0]):
                alpha = batch_alpha[j].unsqueeze(-1)  # [H, W, 1]
                color = batch_colors[j].view(1, 1, 3)  # [1, 1, 3]

                # 过度简化：直接加权平均
                rendered = rendered * (1 - alpha) + color * alpha
                alpha_map = alpha_map * (1 - alpha) + alpha

        return rendered, alpha_map

    def render(self, gaussian_model, K: torch.Tensor, pose: torch.Tensor,
               H: int, W: int) -> torch.Tensor:
        """
        完整渲染流程：投影 + 光栅化。
        """
        # 获取高斯参数
        xyz = gaussian_model.get_xyz
        scale = gaussian_model.get_scaling
        rotation = gaussian_model.get_rotation
        opacity = gaussian_model.get_opacity
        features = gaussian_model.get_features

        # 仅使用0阶球谐系数（基础颜色）
        colors = features[:, 0, :]  # [N, 3]
        colors = torch.sigmoid(colors)  # 应用激活函数

        # 投影到2D
        uv, cov2d, depth = self.project_gaussians(xyz, scale, rotation, K, pose)

        # 光栅化
        rendered, alpha_map = self.rasterize(uv, cov2d, depth, opacity, colors, H, W, K)

        return rendered


if __name__ == "__main__":
    # 测试渲染器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = GaussianRenderer()

    # 创建模拟高斯
    num_gaussians = 100
    xyz = torch.randn(num_gaussians, 3, device=device)
    scale = torch.ones(num_gaussians, 3, device=device) * 0.1
    rotation = torch.randn(num_gaussians, 4, device=device)
    rotation = F.normalize(rotation, dim=-1)

    # 模拟相机参数
    K = torch.eye(3, device=device)
    K[0, 0] = K[1, 1] = 500
    K[0, 2] = 320
    K[1, 2] = 240

    pose = torch.eye(4, device=device)


    # 创建简易高斯模型
    class SimpleGaussianModel:
        def __init__(self):
            self._xyz = xyz
            self._scaling = torch.log(scale)
            self._rotation = rotation
            self._opacity = torch.logit(torch.ones(num_gaussians, 1, device=device) * 0.5)
            self._features_dc = torch.randn(num_gaussians, 1, 3, device=device)
            self._features_rest = torch.zeros(num_gaussians, 15, 3, device=device)

        @property
        def get_xyz(self): return self._xyz

        @property
        def get_scaling(self): return torch.exp(self._scaling)

        @property
        def get_rotation(self): return F.normalize(self._rotation, dim=-1)

        @property
        def get_opacity(self): return torch.sigmoid(self._opacity)

        @property
        def get_features(self):
            return torch.cat([self._features_dc, self._features_rest], dim=1)


    model = SimpleGaussianModel()

    # 渲染
    H, W = 480, 640
    image = renderer.render(model, K, pose, H, W)
    print(f"Rendered image shape: {image.shape}")