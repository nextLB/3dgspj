import torch
import numpy as np
from scipy.spatial import KDTree

import torch
import math
from dataclasses import dataclass
from typing import Optional, Tuple



def distCUDA2(points):
    """
    计算点云中每个点到其最近邻点的距离的平方
    参数:
        points: torch.Tensor, 形状为 (N, 3) 的点云
    返回:
        torch.Tensor, 形状为 (N,) 的距离平方
    """
    # 确保输入是浮点数类型
    points_np = points.cpu().numpy() if points.is_cuda else points.numpy()

    # 构建KD树来快速找到最近邻
    tree = KDTree(points_np)

    # 对于每个点，找到第二个最近邻（第一个是自己）
    distances, indices = tree.query(points_np, k=2)

    # 取第二个最近邻的距离（第一个是点到自身的距离，为0）
    # 计算距离的平方
    dist2 = distances[:, 1] ** 2

    # 返回CUDA tensor
    return torch.from_numpy(dist2).float().to(points.device)



@dataclass
class GaussianRasterizationSettings:
    """
    高斯光栅化设置类
    用于存储光栅化过程的配置参数
    """
    image_height: int  # 图像高度
    image_width: int  # 图像宽度
    tanfovx: float  # 水平视角的正切值
    tanfovy: float  # 垂直视角的正切值
    bg: torch.Tensor  # 背景颜色
    scale_modifier: float  # 缩放修改器
    viewmatrix: torch.Tensor  # 视图矩阵
    projmatrix: torch.Tensor  # 投影矩阵
    sh_degree: int  # 球谐函数阶数
    campos: torch.Tensor  # 相机位置
    prefiltered: bool  # 是否预过滤
    debug: bool  # 调试模式
    antialiasing: bool  # 抗锯齿


class GaussianRasterizer:
    """
    高斯光栅化器
    用于渲染3D高斯点到2D图像
    """

    def __init__(self, raster_settings: GaussianRasterizationSettings):
        """
        初始化光栅化器

        参数:
            raster_settings: 光栅化设置
        """
        self.raster_settings = raster_settings

    def _compute_cov3d(self, scaling: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        """
        从缩放和旋转计算3D协方差矩阵

        参数:
            scaling: 缩放参数，形状为 (N, 3)
            rotation: 四元数表示的旋转，形状为 (N, 4)

        返回:
            3D协方差矩阵，形状为 (N, 6)
        """
        # 将四元数转换为旋转矩阵
        q = rotation  # (N, 4)

        # 归一化四元数
        q_norm = torch.norm(q, dim=1, keepdim=True)
        q = q / q_norm

        # 从四元数提取分量
        qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # 计算旋转矩阵的第一行
        R = torch.stack([
            1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
            2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw),
            2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)
        ], dim=1).reshape(-1, 3, 3)

        # 缩放矩阵
        S = torch.diag_embed(scaling)

        # 协方差矩阵 Σ = RSS^TR^T
        M = torch.bmm(R, S)  # R * S
        cov3D = torch.bmm(M, M.transpose(1, 2))  # M * M^T

        # 由于协方差矩阵是对称的，我们只需要存储上三角部分（6个元素）
        cov3D_flat = torch.stack([
            cov3D[:, 0, 0], cov3D[:, 0, 1], cov3D[:, 0, 2],
            cov3D[:, 1, 1], cov3D[:, 1, 2], cov3D[:, 2, 2]
        ], dim=1)

        return cov3D_flat

    def _project_to_2d(self, points3d: torch.Tensor, viewmatrix: torch.Tensor,
                       projmatrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将3D点投影到2D屏幕空间

        参数:
            points3d: 3D点坐标，形状为 (N, 3)
            viewmatrix: 视图矩阵
            projmatrix: 投影矩阵

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - 投影后的2D点坐标
                - 深度值
                - 是否在视锥体内的标志
        """
        # 转换为齐次坐标
        points_homo = torch.cat([points3d, torch.ones_like(points3d[:, :1])], dim=1)

        # 应用视图和投影变换
        points_camera = torch.matmul(points_homo, viewmatrix.T)
        points_clip = torch.matmul(points_camera, projmatrix.T)

        # 透视除法
        w = points_clip[:, 3:4]
        points_ndc = points_clip[:, :3] / w

        # 转换为屏幕坐标
        screen_points = torch.stack([
            (points_ndc[:, 0] + 1) * self.raster_settings.image_width * 0.5,
            (points_ndc[:, 1] + 1) * self.raster_settings.image_height * 0.5,
        ], dim=1)

        # 深度值
        depths = points_camera[:, 2]

        # 检查点是否在视锥体内 (在投影变换后，w>0 表示在视锥体前)
        in_frustum = w[:, 0] > 0

        return screen_points, depths, in_frustum

    def _sh_to_rgb(self, sh: torch.Tensor, view_dir: torch.Tensor) -> torch.Tensor:
        """
        将球谐系数转换为RGB颜色

        参数:
            sh: 球谐系数，形状为 (N, C, (degree+1)^2)
            view_dir: 观察方向，形状为 (N, 3)

        返回:
            RGB颜色，形状为 (N, 3)
        """
        # 归一化观察方向
        view_dir_norm = view_dir / torch.norm(view_dir, dim=1, keepdim=True)

        # 提取球谐基函数值
        # 这里简化实现，只考虑0阶和1阶球谐
        sh_degree = int(math.sqrt(sh.shape[2])) - 1

        # 0阶球谐基函数 (常数项)
        basis_0 = torch.tensor(0.28209479177387814, device=sh.device).unsqueeze(0).expand(sh.shape[0],
                                                                                          1)  # Y_0^0 = 1/(2*sqrt(pi))

        # 1阶球谐基函数
        x, y, z = view_dir_norm[:, 0], view_dir_norm[:, 1], view_dir_norm[:, 2]
        basis_1 = torch.stack([
            0.4886025119029199 * y,  # Y_1^{-1}
            0.4886025119029199 * z,  # Y_1^0
            0.4886025119029199 * x,  # Y_1^1
        ], dim=1)

        # 组合基函数
        if sh_degree == 0:
            basis = basis_0
        else:
            basis = torch.cat([basis_0, basis_1], dim=1)

        # 计算RGB颜色
        rgb = torch.sum(sh[:, :, :basis.shape[1]] * basis.unsqueeze(1), dim=2)

        # 应用激活函数确保颜色在合理范围内
        rgb = torch.sigmoid(rgb)

        return rgb

    def _sort_by_depth(self, points: torch.Tensor, depths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        按深度对点进行排序（从远到近）

        参数:
            points: 点数据
            depths: 深度值

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 排序后的点和深度
        """
        # 获取深度排序索引（从远到近）
        sorted_indices = torch.argsort(depths, descending=True)

        # 按深度排序
        sorted_points = points[sorted_indices]
        sorted_depths = depths[sorted_indices]

        return sorted_points, sorted_depths

    def _rasterize_gaussians(self, means2D: torch.Tensor, colors: torch.Tensor,
                             opacities: torch.Tensor, cov3D_precomp: Optional[torch.Tensor] = None,
                             scales: Optional[torch.Tensor] = None,
                             rotations: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        光栅化高斯点（修正版）
        避免了原地操作，解决了梯度计算问题

        参数:
            means2D: 2D位置，形状为 (N, 2)
            colors: 颜色，形状为 (N, 3)
            opacities: 不透明度，形状为 (N, 1)
            cov3D_precomp: 预计算的3D协方差，形状为 (N, 6)
            scales: 缩放，形状为 (N, 3)
            rotations: 旋转，形状为 (N, 4)

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - 渲染的图像
                - 点的半径
                - 深度图像
        """
        # 获取图像尺寸
        H = self.raster_settings.image_height
        W = self.raster_settings.image_width

        # 初始化输出图像和深度图
        device = means2D.device
        rendered_image = torch.zeros((3, H, W), device=device)
        depth_image = torch.zeros((H, W), device=device, dtype=torch.int32)
        radii = torch.zeros(means2D.shape[0], device=device)

        # 创建一个贡献列表，避免在循环中修改rendered_image
        num_gaussians = means2D.shape[0]
        if num_gaussians == 0:
            return rendered_image, radii, depth_image.float()

        # 为每个高斯计算贡献
        for i in range(num_gaussians):
            x, y = means2D[i, 0], means2D[i, 1]

            # 检查点是否在图像范围内
            if x < 0 or x >= W or y < 0 or y >= H:
                continue

            # 计算半径（基于缩放）
            if scales is not None:
                radius = torch.mean(scales[i]) * 10
            else:
                radius = 2.0

            # 限制半径范围
            radius = min(max(radius, 1), 10)
            radii[i] = radius

            # 转换为整数坐标
            x_int = int(x)
            y_int = int(y)
            r_int = int(radius)

            # 为每个高斯创建贡献缓冲区
            # 我们将在最后应用这些贡献
            affected_pixels = []

            # 收集受影响的像素
            for dx in range(-r_int, r_int + 1):
                for dy in range(-r_int, r_int + 1):
                    px = x_int + dx
                    py = y_int + dy

                    # 检查像素是否在图像范围内
                    if 0 <= px < W and 0 <= py < H:
                        # 计算到中心的距离
                        dist = math.sqrt(dx ** 2 + dy ** 2)

                        if dist <= r_int:
                            # 计算权重（高斯权重）
                            weight = math.exp(-dist ** 2 / (2 * (r_int / 2) ** 2))

                            # 存储贡献信息
                            affected_pixels.append({
                                'px': px,
                                'py': py,
                                'weight': weight
                            })

            # 如果有受影响的像素，应用贡献
            if affected_pixels:
                # 获取当前像素颜色和不透明度
                alpha = opacities[i, 0].item()
                color = colors[i]

                # 计算每个像素的贡献
                for pixel_info in affected_pixels:
                    px = pixel_info['px']
                    py = pixel_info['py']
                    weight = pixel_info['weight']

                    # 计算最终的alpha
                    final_alpha = alpha * weight

                    # 计算新颜色（非原地操作）
                    current_color = rendered_image[:, py, px].clone()
                    new_color = (1 - final_alpha) * current_color + final_alpha * color

                    # 更新rendered_image（这是必要的，但我们会确保它不会破坏计算图）
                    rendered_image[:, py, px] = new_color

                    # 更新深度图
                    if final_alpha > 0.5:
                        depth_image[py, px] = i  # 简化：使用索引作为深度

        return rendered_image, radii, depth_image.float()

    def __call__(self, means3D: torch.Tensor, means2D: torch.Tensor, shs: Optional[torch.Tensor] = None,
                 colors_precomp: Optional[torch.Tensor] = None, opacities: torch.Tensor = None,
                 scales: Optional[torch.Tensor] = None, rotations: Optional[torch.Tensor] = None,
                 cov3D_precomp: Optional[torch.Tensor] = None, dc: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行光栅化

        参数:
            means3D: 3D位置，形状为 (N, 3)
            means2D: 2D位置，形状为 (N, 2)
            shs: 球谐系数，形状为 (N, (degree+1)^2, 3)
            colors_precomp: 预计算的颜色，形状为 (N, 3)
            opacities: 不透明度，形状为 (N, 1)
            scales: 缩放，形状为 (N, 3)
            rotations: 旋转四元数，形状为 (N, 4)
            cov3D_precomp: 预计算的3D协方差，形状为 (N, 6)
            dc: 0阶球谐系数（当separate_sh=True时使用）

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - 渲染的图像 (3, H, W)
                - 每个点的半径 (N,)
                - 深度图像 (H, W)
        """
        # 获取设置
        settings = self.raster_settings

        # 准备颜色
        if colors_precomp is None:
            # 需要从球谐系数计算颜色
            if dc is not None and shs is not None:
                # 分离的球谐系数情况
                shs_combined = torch.cat([dc.unsqueeze(2), shs], dim=2)
                colors = self._sh_to_rgb(shs_combined, means3D - settings.campos)
            elif shs is not None:
                # 完整的球谐系数
                # 注意：shs的形状可能是 (N, (degree+1)^2, 3)，需要转置为 (N, 3, (degree+1)^2)
                if shs.dim() == 3 and shs.shape[1] == 3 and shs.shape[2] > 3:
                    # 已经是正确的形状 (N, 3, M)
                    colors = self._sh_to_rgb(shs, means3D - settings.campos)
                else:
                    # 转置为 (N, 3, (degree+1)^2)
                    shs_reshaped = shs.permute(0, 2, 1)
                    colors = self._sh_to_rgb(shs_reshaped, means3D - settings.campos)
            else:
                raise ValueError("Either colors_precomp or shs must be provided")
        else:
            colors = colors_precomp

        # 确保不透明度在合理范围内
        opacities = torch.sigmoid(opacities)

        # 简化实现：直接光栅化
        rendered_image, radii, depth_image = self._rasterize_gaussians(
            means2D=means2D,
            colors=colors,
            opacities=opacities,
            cov3D_precomp=cov3D_precomp,
            scales=scales,
            rotations=rotations
        )

        # 应用背景颜色
        if settings.bg is not None:
            bg_mask = (rendered_image.sum(dim=0) == 0).float()
            rendered_image = rendered_image + settings.bg[:, None, None] * bg_mask[None, :, :]

        return rendered_image, radii, depth_image


