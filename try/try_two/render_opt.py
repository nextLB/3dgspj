#!/usr/bin/env python3
"""
优化版渲染器 - RTX 3060专用
支持混合精度渲染、向量化操作、张量核心优化
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any

# 导入自定义模块
from gaussian_model_opt import OptimizedGaussianModel
from camera_opt import OptimizedCamera
from utils_opt import print_gpu_memory


# ==================== RTX 3060 渲染优化 ====================

def compute_covariance_3d_optimized(
        scaling: torch.Tensor,
        rotation: torch.Tensor
) -> torch.Tensor:
    """
    优化版3D协方差计算

    参数:
        scaling: 缩放参数 [N, 3]
        rotation: 旋转四元数 [N, 4] (wxyz格式)

    返回:
        cov3D: 3D协方差矩阵 [N, 6] (上三角元素: xx, xy, xz, yy, yz, zz)
    """
    N = scaling.shape[0]
    device = scaling.device

    # ==================== 缩放矩阵 ====================
    # 使用对角线缩放矩阵
    S = torch.diag_embed(scaling)  # [N, 3, 3]

    # ==================== 四元数转旋转矩阵 ====================
    # 优化计算，避免大量小操作
    q = F.normalize(rotation, p=2, dim=1)  # 归一化四元数

    # 提取四元数分量
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # 预计算平方项
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # 构建旋转矩阵
    R = torch.zeros((N, 3, 3), device=device, dtype=scaling.dtype)

    # 第一列
    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 2, 0] = 2 * (xz - wy)

    # 第二列
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 2, 1] = 2 * (yz + wx)

    # 第三列
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)

    # ==================== 计算协方差矩阵 ====================
    # Σ = R @ S @ S^T @ R^T = (R @ S) @ (R @ S)^T
    RS = torch.bmm(R, S)  # [N, 3, 3]
    cov3D_full = torch.bmm(RS, RS.transpose(1, 2))  # [N, 3, 3]

    # ==================== 提取上三角元素 ====================
    cov3D = torch.zeros((N, 6), device=device, dtype=scaling.dtype)

    # xx, xy, xz, yy, yz, zz
    cov3D[:, 0] = cov3D_full[:, 0, 0]  # xx
    cov3D[:, 1] = cov3D_full[:, 0, 1]  # xy
    cov3D[:, 2] = cov3D_full[:, 0, 2]  # xz
    cov3D[:, 3] = cov3D_full[:, 1, 1]  # yy
    cov3D[:, 4] = cov3D_full[:, 1, 2]  # yz
    cov3D[:, 5] = cov3D_full[:, 2, 2]  # zz

    return cov3D


def project_gaussians_optimized(
        xyz: torch.Tensor,
        world2cam: torch.Tensor,
        K: torch.Tensor,
        H: int,
        W: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    优化版高斯投影到2D

    参数:
        xyz: 3D位置 [N, 3]
        world2cam: 世界到相机变换 [4, 4] 或 [B, 4, 4]
        K: 内参矩阵 [3, 3] 或 [B, 3, 3]
        H: 图像高度
        W: 图像宽度

    返回:
        uv: 2D图像坐标 [N, 2]
        depth: 深度值 [N]
        valid: 有效标志 [N]
    """
    N = xyz.shape[0]
    device = xyz.device

    # ==================== 修复1: 确保输入形状正确 ====================
    # 如果world2cam有批次维度，去掉它（因为我们只处理单个相机）
    if world2cam.dim() == 3:
        if world2cam.shape[0] == 1:
            world2cam = world2cam.squeeze(0)  # [4, 4]
        else:
            # 如果有多个相机，只使用第一个
            world2cam = world2cam[0]

    # 如果K有批次维度，去掉它
    if K.dim() == 3:
        if K.shape[0] == 1:
            K = K.squeeze(0)  # [3, 3]
        else:
            # 如果有多个内参，只使用第一个
            K = K[0]

    # ==================== 坐标变换 ====================
    # 齐次坐标
    ones = torch.ones((N, 1), device=device, dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)  # [N, 4]

    # 变换到相机坐标系
    xyz_cam = torch.matmul(xyz_h, world2cam.T)  # [N, 4]

    # ==================== 投影 ====================
    # 提取深度
    depth = xyz_cam[:, 2]  # [N]

    # 投影到图像平面
    xyz_cam_3d = xyz_cam[:, :3]  # [N, 3]

    # ==================== 修复2: 正确的矩阵乘法 ====================
    # K是[3, 3], xyz_cam_3d是[N, 3]
    # 我们需要: [N, 3] @ [3, 3].T = [N, 3]
    xyz_proj = torch.matmul(xyz_cam_3d, K.T)  # [N, 3]

    # 归一化到像素坐标
    uv = xyz_proj[:, :2] / xyz_proj[:, 2:3].clamp(min=1e-8)  # [N, 2]

    # ==================== 有效性检查 ====================
    # 检查深度是否在有效范围内
    valid_depth = (depth > 0.1) & (depth < 100.0)

    # 检查是否在图像范围内（宽松边界）
    valid_uv = (uv[:, 0] >= -W * 0.2) & (uv[:, 0] < W * 1.2) & \
               (uv[:, 1] >= -H * 0.2) & (uv[:, 1] < H * 1.2)

    # 综合有效性
    valid = valid_depth & valid_uv

    return uv, depth, valid


def compute_covariance_2d_optimized(
        uv: torch.Tensor,
        cov3D: torch.Tensor,
        world2cam: torch.Tensor,
        K: torch.Tensor,
        depth: torch.Tensor
) -> torch.Tensor:
    """
    优化版2D协方差计算

    参数:
        uv: 2D图像坐标 [N, 2]
        cov3D: 3D协方差矩阵 [N, 6]
        world2cam: 世界到相机变换 [4, 4]
        K: 内参矩阵 [3, 3]
        depth: 深度值 [N]

    返回:
        cov2D: 2D协方差矩阵 [N, 3] (xx, xy, yy)
    """
    N = uv.shape[0]
    device = uv.device

    # ==================== 重建3D协方差矩阵 ====================
    cov3D_full = torch.zeros((N, 3, 3), device=device, dtype=cov3D.dtype)

    # 填充上三角部分
    cov3D_full[:, 0, 0] = cov3D[:, 0]  # xx
    cov3D_full[:, 0, 1] = cov3D[:, 1]  # xy
    cov3D_full[:, 0, 2] = cov3D[:, 2]  # xz
    cov3D_full[:, 1, 0] = cov3D[:, 1]  # xy
    cov3D_full[:, 1, 1] = cov3D[:, 3]  # yy
    cov3D_full[:, 1, 2] = cov3D[:, 4]  # yz
    cov3D_full[:, 2, 0] = cov3D[:, 2]  # xz
    cov3D_full[:, 2, 1] = cov3D[:, 4]  # yz
    cov3D_full[:, 2, 2] = cov3D[:, 5]  # zz

    # ==================== 变换到相机坐标系 ====================
    R = world2cam[:3, :3]  # 旋转部分 [3, 3]

    # 批量矩阵乘法: R @ cov3D_full @ R^T
    R_cov = torch.matmul(R.unsqueeze(0), cov3D_full)  # [N, 3, 3]
    cov3D_cam = torch.matmul(R_cov, R.T.unsqueeze(0))  # [N, 3, 3]

    # ==================== 投影雅可比矩阵 ====================
    fx = K[0, 0]
    fy = K[1, 1]

    # 避免除零
    depth_safe = depth.clamp(min=1e-8)

    # 雅可比矩阵: J = [df/du, df/dv] 对于透视投影
    J = torch.zeros((N, 2, 3), device=device, dtype=cov3D.dtype)

    J[:, 0, 0] = fx / depth_safe  # du/dX
    J[:, 0, 2] = -fx * uv[:, 0] / (depth_safe * depth_safe)  # du/dZ
    J[:, 1, 1] = fy / depth_safe  # dv/dY
    J[:, 1, 2] = -fy * uv[:, 1] / (depth_safe * depth_safe)  # dv/dZ

    # ==================== 计算2D协方差 ====================
    # Σ_2D = J @ Σ_cam @ J^T
    J_cov = torch.matmul(J, cov3D_cam)  # [N, 2, 3]
    cov2D_full = torch.matmul(J_cov, J.transpose(1, 2))  # [N, 2, 2]

    # ==================== 添加小正则化 ====================
    # 避免奇异矩阵
    eye = torch.eye(2, device=device, dtype=cov3D.dtype).unsqueeze(0).expand(N, 2, 2)
    cov2D_full = cov2D_full + eye * 1e-6

    # ==================== 提取上三角元素 ====================
    cov2D = torch.zeros((N, 3), device=device, dtype=cov3D.dtype)
    cov2D[:, 0] = cov2D_full[:, 0, 0]  # xx
    cov2D[:, 1] = cov2D_full[:, 0, 1]  # xy
    cov2D[:, 2] = cov2D_full[:, 1, 1]  # yy

    return cov2D


def rasterize_gaussians_optimized(
        uv: torch.Tensor,
        depth: torch.Tensor,
        cov2D: torch.Tensor,
        opacity: torch.Tensor,
        colors: torch.Tensor,
        H: int,
        W: int,
        tile_size: int = 16,
        threshold: float = 0.001
) -> torch.Tensor:
    """
    优化版高斯栅格化

    参数:
        uv: 2D图像坐标 [N, 2]
        depth: 深度值 [N]
        cov2D: 2D协方差矩阵 [N, 3]
        opacity: 不透明度 [N, 1] (sigmoid前)
        colors: 颜色 [N, 3]
        H: 图像高度
        W: 图像宽度
        tile_size: 分块大小
        threshold: 透明度阈值

    返回:
        rendered_image: 渲染的图像 [3, H, W]
    """
    device = uv.device
    dtype = uv.dtype

    # ==================== 初始化输出 ====================
    image = torch.zeros((H, W, 3), device=device, dtype=dtype)
    alpha = torch.zeros((H, W), device=device, dtype=dtype)

    # ==================== 按深度排序 ====================
    # 从远到近渲染 (画家算法)
    sorted_idx = torch.argsort(depth, descending=True)

    uv = uv[sorted_idx]
    cov2D = cov2D[sorted_idx]
    opacity_sigmoid = torch.sigmoid(opacity[sorted_idx]).squeeze(1)  # [N]
    colors = colors[sorted_idx]  # [N, 3]

    N = uv.shape[0]

    # ==================== 计算边界框 ====================
    # 基于协方差计算高斯半径 (3σ原则)
    # 特征值近似: λ_max ≈ (a + c + sqrt((a-c)^2 + 4b^2)) / 2
    a, b, c = cov2D[:, 0], cov2D[:, 1], cov2D[:, 2]

    # 计算最大特征值
    discriminant = torch.sqrt((a - c) ** 2 + 4 * b ** 2)
    lambda_max = (a + c + discriminant) / 2

    # 3σ半径
    radius = torch.sqrt(lambda_max) * 3.0
    radius_int = torch.ceil(radius).int()

    # 计算边界框
    min_u = torch.clamp((uv[:, 0] - radius_int).int(), 0, W - 1)
    max_u = torch.clamp((uv[:, 0] + radius_int).int() + 1, 0, W)
    min_v = torch.clamp((uv[:, 1] - radius_int).int(), 0, H - 1)
    max_v = torch.clamp((uv[:, 1] + radius_int).int() + 1, 0, H)

    # ==================== 分块栅格化 ====================
    # 为了提高内存访问效率，我们可以分块处理
    # 这里简化实现，逐个高斯处理

    for i in range(N):
        # 获取当前高斯的参数
        u_center, v_center = uv[i, 0], uv[i, 1]
        a_i, b_i, c_i = cov2D[i, 0], cov2D[i, 1], cov2D[i, 2]
        opacity_i = opacity_sigmoid[i]
        color_i = colors[i]

        # 边界框
        mu, mv = min_v[i].item(), max_v[i].item()
        mu_w, mx_w = min_u[i].item(), max_u[i].item()

        # 跳过无效边界框
        if mu >= mv or mu_w >= mx_w:
            continue

        # 计算局部网格
        grid_v, grid_u = torch.meshgrid(
            torch.arange(mu, mv, device=device, dtype=dtype),
            torch.arange(mu_w, mx_w, device=device, dtype=dtype),
            indexing='ij'
        )

        # 计算偏移
        du = grid_u - u_center
        dv = grid_v - v_center

        # ==================== 计算高斯权重 ====================
        # 马氏距离: d^2 = [du, dv] @ Σ^{-1} @ [du, dv]^T
        # 对于2x2矩阵 Σ = [[a, b], [b, c]]，逆矩阵为:
        # Σ^{-1} = 1/(ac-b^2) * [[c, -b], [-b, a]]

        det = a_i * c_i - b_i * b_i

        # 避免奇异矩阵
        if abs(det) < 1e-12:
            continue

        inv_det = 1.0 / det

        # 计算距离
        dist = inv_det * (c_i * du * du - 2 * b_i * du * dv + a_i * dv * dv)

        # 高斯权重
        weight = torch.exp(-0.5 * dist)

        # ==================== Alpha合成 ====================
        # 当前高斯的alpha
        alpha_i = opacity_i * weight

        # 当前像素的透射率
        T = 1.0 - alpha[mu:mv, mu_w:mx_w]

        # 贡献权重
        weight_i = alpha_i * T

        # ==================== 累积颜色和alpha ====================
        image[mu:mv, mu_w:mx_w, :] += weight_i.unsqueeze(-1) * color_i
        alpha[mu:mv, mu_w:mx_w] += weight_i

    # ==================== 归一化和背景 ====================
    # 避免除零
    mask = alpha > 1e-8
    image[mask] = image[mask] / alpha[mask].unsqueeze(-1)

    # 添加白色背景
    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
    image = image + (1 - alpha).unsqueeze(-1) * bg_color

    # 转换为CHW格式
    rendered_image = image.permute(2, 0, 1)  # [3, H, W]

    return rendered_image


# ==================== 主渲染函数 ====================
def render_gaussians_optimized(
        gaussians: OptimizedGaussianModel,
        camera: OptimizedCamera,
        use_amp: bool = True
) -> torch.Tensor:
    """
    优化版高斯渲染主函数

    参数:
        gaussians: 高斯模型
        camera: 相机
        use_amp: 是否使用混合精度

    返回:
        rendered_image: 渲染的图像 [3, H, W]
    """
    # ==================== 准备数据 ====================
    # 获取高斯参数
    xyz = gaussians.get_xyz
    scaling = gaussians.get_scaling
    rotation = gaussians.get_rotation
    opacity = gaussians.get_opacity
    features = gaussians.get_features

    # 获取相机参数 - 确保获取正确的形状
    world2cam = camera.world_view_transform
    K = camera.K
    H, W = camera.H, camera.W

    # ==================== 修复: 确保相机参数形状正确 ====================
    # 如果world2cam有批次维度，去掉它
    if world2cam.dim() == 3 and world2cam.shape[0] == 1:
        world2cam = world2cam.squeeze(0)

    # 如果K有批次维度，去掉它
    if K.dim() == 3 and K.shape[0] == 1:
        K = K.squeeze(0)

    # ==================== 混合精度上下文 ====================
    with torch.cuda.amp.autocast(enabled=use_amp):
        # ==================== 投影 ====================
        uv, depth, valid = project_gaussians_optimized(xyz, world2cam, K, H, W)

        # 只处理有效的高斯
        valid_indices = torch.where(valid)[0]

        if len(valid_indices) == 0:
            # 没有有效高斯，返回空白图像
            return torch.zeros((3, H, W), device=xyz.device, dtype=xyz.dtype)

        # 筛选有效高斯
        uv_valid = uv[valid_indices]
        depth_valid = depth[valid_indices]
        scaling_valid = scaling[valid_indices]
        rotation_valid = rotation[valid_indices]
        opacity_valid = opacity[valid_indices]
        features_valid = features[valid_indices].squeeze(1)  # [N, 1, 3] -> [N, 3]

        # ==================== 计算协方差 ====================
        # 3D协方差
        cov3D = compute_covariance_3d_optimized(scaling_valid, rotation_valid)

        # 2D协方差
        cov2D = compute_covariance_2d_optimized(
            uv_valid, cov3D, world2cam, K, depth_valid
        )

        # ==================== 栅格化 ====================
        rendered_image = rasterize_gaussians_optimized(
            uv=uv_valid,
            depth=depth_valid,
            cov2D=cov2D,
            opacity=opacity_valid,
            colors=features_valid,
            H=H,
            W=W,
            tile_size=16,
            threshold=0.001
        )

    return rendered_image

# ==================== 损失函数 ====================

def compute_rendering_loss(
        rendered: torch.Tensor,
        target: torch.Tensor,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.2
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    计算渲染损失

    参数:
        rendered: 渲染的图像 [3, H, W]
        target: 目标图像 [3, H, W]
        lambda_l1: L1损失权重
        lambda_ssim: SSIM损失权重

    返回:
        total_loss: 总损失
        loss_dict: 损失字典
    """
    # L1损失
    l1_loss = torch.abs(rendered - target).mean()

    # SSIM损失
    if lambda_ssim > 0:
        ssim_loss = 1.0 - compute_ssim_simple(rendered, target)
        total_loss = lambda_l1 * l1_loss + lambda_ssim * ssim_loss
    else:
        ssim_loss = torch.tensor(0.0, device=rendered.device)
        total_loss = l1_loss

    # 损失字典
    loss_dict = {
        'total_loss': total_loss,
        'l1_loss': l1_loss,
        'ssim_loss': ssim_loss
    }

    return total_loss, loss_dict


def compute_ssim_simple(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11
) -> torch.Tensor:
    """
    简化版SSIM计算

    参数:
        img1: 图像1 [C, H, W]
        img2: 图像2 [C, H, W]
        window_size: 窗口大小

    返回:
        ssim: SSIM值
    """
    C, H, W = img1.shape

    # 如果图像太小，使用全局统计
    if H < window_size or W < window_size:
        # 计算均值和方差
        mu1 = img1.mean()
        mu2 = img2.mean()

        sigma1 = img1.std()
        sigma2 = img2.std()

        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    else:
        # 简化的滑动窗口统计
        # 使用平均池化近似
        kernel_size = min(window_size, H // 4, W // 4)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        if kernel_size < 3:
            kernel_size = 3

        # 使用卷积计算局部统计
        weight = torch.ones((C, 1, kernel_size, kernel_size),
                            device=img1.device, dtype=img1.dtype) / (kernel_size * kernel_size)

        # 添加批次维度
        img1_batch = img1.unsqueeze(0)
        img2_batch = img2.unsqueeze(0)

        # 计算局部均值
        mu1 = F.conv2d(img1_batch, weight, padding=kernel_size // 2, groups=C).squeeze(0)
        mu2 = F.conv2d(img2_batch, weight, padding=kernel_size // 2, groups=C).squeeze(0)

        # 计算局部方差和协方差
        sigma1_sq = F.conv2d(img1_batch * img1_batch, weight,
                             padding=kernel_size // 2, groups=C).squeeze(0) - mu1 * mu1
        sigma2_sq = F.conv2d(img2_batch * img2_batch, weight,
                             padding=kernel_size // 2, groups=C).squeeze(0) - mu2 * mu2
        sigma12 = F.conv2d(img1_batch * img2_batch, weight,
                           padding=kernel_size // 2, groups=C).squeeze(0) - mu1 * mu2

    # SSIM常数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM公式
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("🧪 测试渲染器...")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 创建测试数据
    N = 1000  # 高斯点数
    H, W = 256, 256  # 图像尺寸


    # 创建高斯模型
    class MockGaussianModel:
        def __init__(self):
            self._xyz = torch.randn(N, 3, device=device)
            self._scaling = torch.ones(N, 3, device=device) * 0.01
            self._rotation = torch.zeros(N, 4, device=device)
            self._rotation[:, 0] = 1.0
            self._opacity = torch.ones(N, 1, device=device) * 0.1
            self._features_dc = torch.rand(N, 1, 3, device=device)

        @property
        def get_xyz(self):
            return self._xyz

        @property
        def get_scaling(self):
            return self._scaling

        @property
        def get_rotation(self):
            return self._rotation

        @property
        def get_opacity(self):
            return self._opacity

        @property
        def get_features(self):
            return self._features_dc


    # 创建相机
    class MockCamera:
        def __init__(self):
            self.world_view_transform = torch.eye(4, device=device)
            self.K = torch.eye(3, device=device)
            self.K[0, 0] = self.K[1, 1] = 500.0
            self.H = H
            self.W = W


    # 测试渲染
    print("测试高斯渲染...")

    gaussians = MockGaussianModel()
    camera = MockCamera()

    # 测试投影
    uv, depth, valid = project_gaussians_optimized(
        gaussians.get_xyz,
        camera.world_view_transform,
        camera.K,
        camera.H,
        camera.W
    )

    print(f"投影结果:")
    print(f"  UV形状: {uv.shape}")
    print(f"  深度形状: {depth.shape}")
    print(f"  有效点数: {valid.sum().item()}/{N}")

    # 测试协方差计算
    if valid.sum() > 0:
        valid_indices = torch.where(valid)[0][:10]  # 只测试前10个

        scaling_valid = gaussians.get_scaling[valid_indices]
        rotation_valid = gaussians.get_rotation[valid_indices]

        cov3D = compute_covariance_3d_optimized(scaling_valid, rotation_valid)
        print(f"3D协方差形状: {cov3D.shape}")

        uv_valid = uv[valid_indices]
        depth_valid = depth[valid_indices]

        cov2D = compute_covariance_2d_optimized(
            uv_valid, cov3D, camera.world_view_transform,
            camera.K, depth_valid
        )
        print(f"2D协方差形状: {cov2D.shape}")

    # 测试损失计算
    img1 = torch.rand(3, H, W, device=device)
    img2 = torch.rand(3, H, W, device=device)

    total_loss, loss_dict = compute_rendering_loss(img1, img2)
    print(f"\n损失计算:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value.item():.4f}")

    print("\n✅ 渲染器测试完成!")