import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple
import warnings


class GaussianRasterizer(nn.Module):
    """3D高斯光栅化渲染器"""

    def __init__(self,
                 image_height: int,
                 image_width: int,
                 near_plane: float = 0.01,
                 far_plane: float = 100.0,
                 bg_color: Optional[torch.Tensor] = None,
                 use_mip: bool = True,
                 mip_levels: int = 3):
        super().__init__()

        self.image_height = image_height
        self.image_width = image_width
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.use_mip = use_mip
        self.mip_levels = mip_levels

        # 背景颜色
        if bg_color is None:
            self.bg_color = torch.tensor([0.0, 0.0, 0.0])
        else:
            self.bg_color = bg_color

        # 创建像素坐标网格
        self.register_buffer('pixel_coords', self.create_pixel_coords())

        # Mip滤波核
        if self.use_mip:
            self.mip_kernels = self.create_mip_kernels()

    def create_pixel_coords(self):
        """创建像素坐标网格"""
        xs = torch.linspace(0, self.image_width - 1, self.image_width)
        ys = torch.linspace(0, self.image_height - 1, self.image_height)

        x, y = torch.meshgrid(xs, ys, indexing='xy')
        coords = torch.stack([x, y], dim=-1).float()  # (H, W, 2)

        return coords

    def create_mip_kernels(self):
        """创建Mip滤波核"""
        kernels = []

        # 创建不同尺度的滤波核
        for i in range(self.mip_levels):
            size = 2 * i + 1  # 1x1, 3x3, 5x5, ...
            kernel = torch.ones((1, 1, size, size), dtype=torch.float32)
            kernel = kernel / kernel.sum()
            kernels.append(kernel)

        return kernels

    def project_gaussians(self,
                          xyz: torch.Tensor,
                          cov_3d: torch.Tensor,
                          camera_R: torch.Tensor,
                          camera_T: torch.Tensor,
                          camera_fx: float,
                          camera_fy: float,
                          camera_cx: float,
                          camera_cy: float):
        """将3D高斯投影到2D图像平面"""

        device = xyz.device
        n_gaussians = xyz.shape[0]

        # 相机参数
        fx = camera_fx
        fy = camera_fy
        cx = camera_cx
        cy = camera_cy

        # 将点变换到相机坐标系
        # camera_R: (3, 3), camera_T: (3, 1)
        xyz_cam = camera_R @ xyz.T + camera_T  # (3, n)
        xyz_cam = xyz_cam.T  # (n, 3)

        # 检查点在相机前方
        z = xyz_cam[:, 2]
        valid_mask = z > self.near_plane

        # 投影到图像平面
        x = (xyz_cam[:, 0] / xyz_cam[:, 2]) * fx + cx
        y = (xyz_cam[:, 1] / xyz_cam[:, 2]) * fy + cy

        # 深度值（用于排序）
        depths = z

        # 将3D协方差投影到2D
        # 计算投影雅可比矩阵
        J = torch.zeros((n_gaussians, 2, 3), device=device)
        J[:, 0, 0] = fx / z
        J[:, 0, 2] = -fx * xyz_cam[:, 0] / (z * z)
        J[:, 1, 1] = fy / z
        J[:, 1, 2] = -fy * xyz_cam[:, 1] / (z * z)

        # 旋转矩阵从相机坐标系到图像坐标系
        W = torch.eye(3, device=device).unsqueeze(0).repeat(n_gaussians, 1, 1)
        W[:, :3, :3] = camera_R.unsqueeze(0).repeat(n_gaussians, 1, 1)

        # 完整投影矩阵
        P = torch.zeros((n_gaussians, 3, 4), device=device)
        P[:, 0, 0] = fx
        P[:, 1, 1] = fy
        P[:, 0, 2] = cx
        P[:, 1, 2] = cy
        P[:, 2, 2] = 1.0

        # 计算投影变换后的协方差
        # 方法1：使用近似公式（更稳定）
        cov_2d_approx = torch.zeros((n_gaussians, 2, 2), device=device)

        for i in range(n_gaussians):
            if valid_mask[i]:
                # 提取3D协方差
                cov_3d_i = cov_3d[i]

                # 计算投影后的协方差：J @ cov_3d @ J^T
                cov_2d = J[i] @ cov_3d_i @ J[i].T

                # 添加小值确保正定性
                cov_2d = cov_2d + torch.eye(2, device=device) * 1e-6

                cov_2d_approx[i] = cov_2d

        # 计算2D高斯参数
        mu_2d = torch.stack([x, y], dim=-1)  # (n, 2)

        # 计算逆协方差（精度矩阵）
        try:
            cov_2d_inv = torch.linalg.inv(cov_2d_approx)
        except:
            # 如果求逆失败，使用伪逆
            cov_2d_inv = torch.linalg.pinv(cov_2d_approx)

        # 计算高斯半径（用于剔除）
        eigenvalues = torch.linalg.eigvalsh(cov_2d_approx)
        radii = torch.sqrt(torch.max(eigenvalues, dim=-1).values) * 3.0  # 3-sigma半径

        return {
            'mu_2d': mu_2d,
            'cov_2d': cov_2d_approx,
            'cov_2d_inv': cov_2d_inv,
            'depths': depths,
            'valid_mask': valid_mask,
            'radii': radii,
            'xyz_cam': xyz_cam
        }

    def apply_mip_filtering_2d(self, cov_2d, pixel_size=1.0):
        """应用2D Mip滤波"""
        if not self.use_mip:
            return cov_2d

        # 添加像素大小相关的不确定性
        pixel_cov = torch.eye(2, device=cov_2d.device).unsqueeze(0) * (pixel_size ** 2)
        cov_2d_filtered = cov_2d + pixel_cov

        return cov_2d_filtered

    def compute_gaussian_weights(self,
                                 mu_2d: torch.Tensor,
                                 cov_2d_inv: torch.Tensor,
                                 pixel_coords: torch.Tensor):
        """计算高斯权重"""
        n_gaussians = mu_2d.shape[0]
        n_pixels = pixel_coords.shape[0]

        # 扩展维度以便广播计算
        # mu_2d: (n_gaussians, 1, 2)
        # pixel_coords: (1, n_pixels, 2)
        mu_expanded = mu_2d.unsqueeze(1)  # (n, 1, 2)
        pixels_expanded = pixel_coords.unsqueeze(0)  # (1, m, 2)

        # 计算差值
        diff = pixels_expanded - mu_expanded  # (n, m, 2)

        # 计算马氏距离: (x - μ)^T Σ^{-1} (x - μ)
        # diff: (n, m, 2), cov_2d_inv: (n, 2, 2)
        # 首先计算 diff @ cov_2d_inv
        diff_reshaped = diff.view(-1, 1, 2)  # (n*m, 1, 2)
        cov_inv_reshaped = cov_2d_inv.repeat_interleave(n_pixels, dim=0)  # (n*m, 2, 2)

        temp = torch.bmm(diff_reshaped, cov_inv_reshaped)  # (n*m, 1, 2)

        # 计算二次型
        mahalanobis = torch.bmm(temp, diff_reshaped.transpose(1, 2))  # (n*m, 1, 1)
        mahalanobis = mahalanobis.view(n_gaussians, n_pixels)  # (n, m)

        # 计算高斯权重: exp(-0.5 * d^2)
        weights = torch.exp(-0.5 * mahalanobis)

        return weights

    def rasterize(self,
                  xyz: torch.Tensor,
                  features: torch.Tensor,
                  opacity: torch.Tensor,
                  cov_3d: torch.Tensor,
                  camera_R: torch.Tensor,
                  camera_T: torch.Tensor,
                  camera_fx: float,
                  camera_fy: float,
                  camera_cx: float,
                  camera_cy: float,
                  sh_degree: int = 0,
                  viewdir: Optional[torch.Tensor] = None):
        """光栅化3D高斯"""

        device = xyz.device
        n_gaussians = xyz.shape[0]

        # 1. 投影高斯到2D
        proj_result = self.project_gaussians(
            xyz, cov_3d, camera_R, camera_T,
            camera_fx, camera_fy, camera_cx, camera_cy
        )

        mu_2d = proj_result['mu_2d']
        cov_2d = proj_result['cov_2d']
        cov_2d_inv = proj_result['cov_2d_inv']
        depths = proj_result['depths']
        valid_mask = proj_result['valid_mask']
        radii = proj_result['radii']

        # 只处理有效的高斯
        if not valid_mask.any():
            # 返回空白图像
            blank_image = torch.ones((3, self.image_height, self.image_width),
                                     device=device) * self.bg_color.view(3, 1, 1)
            return {
                'render': blank_image,
                'depth': torch.zeros((1, self.image_height, self.image_width), device=device),
                'alpha': torch.zeros((1, self.image_height, self.image_width), device=device),
                'visibility_filter': torch.zeros(n_gaussians, device=device, dtype=torch.bool),
                'radii': radii
            }

        valid_idx = torch.where(valid_mask)[0]
        mu_2d = mu_2d[valid_idx]
        cov_2d = cov_2d[valid_idx]
        cov_2d_inv = cov_2d_inv[valid_idx]
        depths_valid = depths[valid_idx]
        radii_valid = radii[valid_idx]
        features_valid = features[valid_idx]
        opacity_valid = opacity[valid_idx]

        n_valid = mu_2d.shape[0]

        # 2. 应用Mip滤波
        if self.use_mip:
            cov_2d = self.apply_mip_filtering_2d(cov_2d)
            # 重新计算逆协方差
            try:
                cov_2d_inv = torch.linalg.inv(cov_2d)
            except:
                cov_2d_inv = torch.linalg.pinv(cov_2d)

        # 3. 基于深度的排序（从远到近）
        sorted_indices = torch.argsort(depths_valid, descending=True)

        mu_2d = mu_2d[sorted_indices]
        cov_2d_inv = cov_2d_inv[sorted_indices]
        depths_valid = depths_valid[sorted_indices]
        radii_valid = radii_valid[sorted_indices]
        features_valid = features_valid[sorted_indices]
        opacity_valid = opacity_valid[sorted_indices]

        # 4. 创建像素网格
        pixel_coords = self.pixel_coords.to(device)
        pixel_coords_flat = pixel_coords.view(-1, 2)  # (H*W, 2)

        # 5. 为每个高斯计算影响的像素
        # 使用半径进行粗略剔除
        image_radii = torch.zeros((self.image_height, self.image_width), device=device)

        # 简单实现：遍历高斯（实际实现应使用空间加速结构）
        # 这里使用简化版本，实际应用可能需要优化

        # 6. 计算每个像素的颜色（简化实现）
        # 在实际实现中，这里应该使用并行计算每个像素受哪些高斯影响

        # 简化实现：直接渲染到图像
        rendered_image = torch.ones((self.image_height, self.image_width, 3),
                                    device=device) * self.bg_color
        rendered_depth = torch.zeros((self.image_height, self.image_width, 1), device=device)
        rendered_alpha = torch.zeros((self.image_height, self.image_width, 1), device=device)

        # 可见性过滤器（记录哪些高斯对最终图像有贡献）
        visibility_filter = torch.zeros(n_gaussians, device=device, dtype=torch.bool)

        # 标记可见的高斯
        visibility_filter[valid_idx[sorted_indices]] = True

        # 简化实现：使用点精灵渲染（实际应使用高斯溅射）
        # 将2D位置四舍五入到最近的像素
        pixel_positions = torch.round(mu_2d).long()

        # 确保在图像边界内
        valid_pixels = (pixel_positions[:, 0] >= 0) & (pixel_positions[:, 0] < self.image_width) & \
                       (pixel_positions[:, 1] >= 0) & (pixel_positions[:, 1] < self.image_height)

        pixel_positions = pixel_positions[valid_pixels]
        features_pixels = features_valid[valid_pixels]
        opacity_pixels = opacity_valid[valid_pixels]
        depths_pixels = depths_valid[valid_pixels]

        # 为每个像素分配颜色（从前到后混合）
        for i in range(pixel_positions.shape[0]):
            x, y = pixel_positions[i]
            color = features_pixels[i, :3]  # RGB颜色
            alpha = opacity_pixels[i]

            # 当前像素的当前颜色和透明度
            current_color = rendered_image[y, x]
            current_alpha = rendered_alpha[y, x, 0]

            # 从后到前混合
            new_alpha = current_alpha + alpha * (1 - current_alpha)
            new_color = (current_color * current_alpha + color * alpha * (1 - current_alpha)) / (new_alpha + 1e-10)

            rendered_image[y, x] = new_color
            rendered_alpha[y, x, 0] = new_alpha

            # 深度（加权平均）
            if alpha > 0.1:  # 只考虑显著贡献
                rendered_depth[y, x, 0] = depths_pixels[i]

        # 调整维度顺序：HWC -> CHW
        rendered_image = rendered_image.permute(2, 0, 1)
        rendered_depth = rendered_depth.permute(2, 0, 1)
        rendered_alpha = rendered_alpha.permute(2, 0, 1)

        return {
            'render': rendered_image,
            'depth': rendered_depth,
            'alpha': rendered_alpha,
            'visibility_filter': visibility_filter,
            'radii': radii
        }

    def forward(self, gaussians, camera):
        """前向渲染"""
        # 提取高斯参数
        xyz = gaussians['xyz']
        features_dc = gaussians['features_dc']
        features_rest = gaussians['features_rest']
        opacity = gaussians['opacity']
        cov_3d = gaussians['covariance']

        # 合并特征
        features = torch.cat([features_dc, features_rest], dim=1)

        # 提取相机参数
        camera_R = camera.R
        camera_T = camera.T
        camera_fx = camera.fx
        camera_fy = camera.fy
        camera_cx = camera.cx
        camera_cy = camera.cy

        # 渲染
        result = self.rasterize(
            xyz=xyz,
            features=features,
            opacity=opacity,
            cov_3d=cov_3d,
            camera_R=camera_R,
            camera_T=camera_T,
            camera_fx=camera_fx,
            camera_fy=camera_fy,
            camera_cx=camera_cx,
            camera_cy=camera_cy,
            sh_degree=self.mip_levels if self.use_mip else 0
        )

        return result


def render(camera, gaussians, args, mip_filter=True, mip_levels=3, bg_color=None):
    """渲染函数"""

    if bg_color is None:
        bg_color = torch.tensor([0.0, 0.0, 0.0], device=camera.R.device)

    # 创建渲染器
    rasterizer = GaussianRasterizer(
        image_height=camera.height,
        image_width=camera.width,
        near_plane=0.01,
        far_plane=100.0,
        bg_color=bg_color,
        use_mip=mip_filter,
        mip_levels=mip_levels
    ).to(camera.R.device)

    # 获取高斯参数
    gaussian_params = gaussians.forward()

    # 渲染
    result = rasterizer(gaussian_params, camera)

    return result


class FastRasterizer(nn.Module):
    """快速光栅化实现（使用近似方法）"""

    def __init__(self, image_height, image_width):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width

    def forward(self, gaussians, camera):
        """快速近似渲染"""
        # 简化实现：使用alpha合成
        device = camera.R.device

        # 创建空白图像
        image = torch.zeros((3, self.image_height, self.image_width), device=device)
        depth = torch.zeros((1, self.image_height, self.image_width), device=device)
        alpha = torch.zeros((1, self.image_height, self.image_width), device=device)

        # 获取高斯参数
        xyz = gaussians['xyz']
        colors = gaussians['features_dc'][:, :, 0]  # 只使用DC分量
        opacity = gaussians['opacity']
        scaling = gaussians['scaling']

        # 将点变换到相机坐标系
        xyz_cam = camera.R @ xyz.T + camera.T
        xyz_cam = xyz_cam.T

        # 投影到图像平面
        z = xyz_cam[:, 2]
        x = (xyz_cam[:, 0] / z) * camera.fx + camera.cx
        y = (xyz_cam[:, 1] / z) * camera.fy + camera.cy

        # 转换为像素坐标
        pixel_x = torch.clamp(torch.round(x).long(), 0, self.image_width - 1)
        pixel_y = torch.clamp(torch.round(y).long(), 0, self.image_height - 1)

        # 基于深度排序
        sorted_indices = torch.argsort(z, descending=True)
        pixel_x = pixel_x[sorted_indices]
        pixel_y = pixel_y[sorted_indices]
        colors = colors[sorted_indices]
        opacity = opacity[sorted_indices]
        z = z[sorted_indices]

        # Alpha合成
        for i in range(pixel_x.shape[0]):
            px, py = pixel_x[i], pixel_y[i]

            current_alpha = alpha[0, py, px]
            current_color = image[:, py, px]

            # 从后到前混合
            new_alpha = current_alpha + opacity[i] * (1 - current_alpha)
            new_color = (current_color * current_alpha + colors[i] * opacity[i] * (1 - current_alpha)) / (
                        new_alpha + 1e-10)

            image[:, py, px] = new_color
            alpha[0, py, px] = new_alpha
            depth[0, py, px] = z[i]

        return {
            'render': image,
            'depth': depth,
            'alpha': alpha,
            'visibility_filter': torch.ones(xyz.shape[0], device=device, dtype=torch.bool),
            'radii': torch.ones(xyz.shape[0], device=device) * 5.0
        }


def fast_render(camera, gaussians, bg_color=None):
    """快速渲染（用于调试）"""
    if bg_color is None:
        bg_color = torch.tensor([0.0, 0.0, 0.0], device=camera.R.device)

    rasterizer = FastRasterizer(camera.height, camera.width).to(camera.R.device)
    gaussian_params = gaussians.forward()
    result = rasterizer(gaussian_params, camera)

    # 添加背景
    alpha = result['alpha']
    result['render'] = result['render'] * alpha + bg_color.view(3, 1, 1) * (1 - alpha)

    return result


class SimpleDifferentiableRenderer:
    """简单的可导渲染器"""

    @staticmethod
    def render(camera, gaussians):
        device = camera.R.device

        # 获取参数
        xyz = gaussians.get_xyz
        colors = gaussians._features_dc[:, :, 0]
        opacity = gaussians.get_opacity.squeeze()

        # 变换到相机坐标系
        xyz_cam = camera.R @ xyz.T + camera.T
        xyz_cam = xyz_cam.T

        # 投影
        z = xyz_cam[:, 2] + 1e-8
        x_proj = (xyz_cam[:, 0] / z) * camera.fx + camera.cx
        y_proj = (xyz_cam[:, 1] / z) * camera.fy + camera.cy

        # 创建图像
        image = torch.zeros((3, camera.height, camera.width), device=device, requires_grad=True)

        # 限制点数以提高速度
        max_points = min(5000, xyz.shape[0])
        indices = torch.randperm(xyz.shape[0])[:max_points]

        x_proj = x_proj[indices]
        y_proj = y_proj[indices]
        colors = colors[indices]
        opacity = opacity[indices]

        # 使用可导的soft分配
        # 创建网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(camera.height, device=device, dtype=torch.float32),
            torch.arange(camera.width, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # 对每个点计算权重
        for i in range(max_points):
            # 计算到该点的距离
            dist_x = (grid_x - x_proj[i]) ** 2
            dist_y = (grid_y - y_proj[i]) ** 2
            distance = dist_x + dist_y

            # 高斯权重
            sigma = 5.0
            weight = torch.exp(-distance / (2 * sigma ** 2))
            weight = weight * opacity[i]

            # 添加到图像
            image += weight.unsqueeze(0) * colors[i].view(3, 1, 1)

        # 裁剪到[0, 1]范围
        image = torch.clamp(image, 0, 1)

        return {
            'render': image,
            'depth': torch.zeros((1, camera.height, camera.width), device=device),
            'alpha': torch.ones((1, camera.height, camera.width), device=device),
            'visibility_filter': torch.ones(xyz.shape[0], device=device, dtype=torch.bool),
            'radii': torch.ones(xyz.shape[0], device=device) * 3.0
        }

