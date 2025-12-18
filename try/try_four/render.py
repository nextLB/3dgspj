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
        # 注意：使用 matmul 而不是 @，确保没有原地操作
        xyz_cam = torch.matmul(R, xyz.T).T + t.unsqueeze(0)  # [N, 3]

        # 深度（用于排序）
        depth = xyz_cam[:, 2]  # Z轴深度

        # 2. 投影到图像平面（透视投影）
        xyz_proj = torch.matmul(K, xyz_cam.T).T  # [N, 3]
        uv = xyz_proj[:, :2] / xyz_proj[:, 2:]  # 归一化设备坐标 [N, 2]

        # 3. 计算投影后的2D协方差
        # 构建3D协方差矩阵
        scale_matrix = torch.diag_embed(scale)  # [N, 3, 3]
        rot_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(xyz.shape[0], 1, 1)

        # 3D协方差：S * R * R^T * S^T
        cov3d = torch.matmul(scale_matrix, torch.matmul(rot_matrix, scale_matrix.transpose(1, 2)))

        # 投影到2D（雅可比矩阵近似）
        focal_x = K[0, 0]
        focal_y = K[1, 1]
        z = xyz_cam[:, 2]

        # 投影雅可比矩阵
        J = torch.zeros(xyz.shape[0], 2, 3, device=device)
        J[:, 0, 0] = focal_x / z
        J[:, 1, 1] = focal_y / z

        # 2D协方差：J * cov3d * J^T
        cov2d = torch.matmul(J, torch.matmul(cov3d, J.transpose(1, 2)))  # [N, 2, 2]

        # 添加小值以确保正定性（不使用原地操作）
        identity = torch.eye(2, device=device).unsqueeze(0) * 1e-6
        cov2d = cov2d + identity

        return uv, cov2d, depth

    def compute_alpha_from_covariance(self, cov2d: torch.Tensor, uv: torch.Tensor,
                                      pixel_uv: torch.Tensor) -> torch.Tensor:
        """
        根据2D高斯协方差计算像素位置的不透明度贡献。
        优化版本，避免内存爆炸。
        """
        # 像素坐标与高斯中心的偏移 [N, P, 2]
        diff = pixel_uv - uv.unsqueeze(1)

        # 逐元素计算，避免大矩阵运算
        N, P = diff.shape[:2]

        # 使用逐像素计算，避免内存爆炸
        alphas = []

        # 分批处理像素，避免一次性计算所有像素
        pixel_batch_size = 1000  # 每批处理的像素数

        for i in range(0, P, pixel_batch_size):
            end_idx = min(i + pixel_batch_size, P)
            batch_diff = diff[:, i:end_idx, :]  # [N, batch_size, 2]

            # 对每个高斯分别处理
            batch_alphas = []
            for j in range(cov2d.shape[0]):
                # 提取单个高斯的协方差
                cov = cov2d[j].unsqueeze(0).unsqueeze(0)  # [1, 1, 2, 2]
                gauss_diff = batch_diff[j].unsqueeze(0)  # [1, batch_size, 2]

                # 计算马氏距离
                try:
                    cov_inv = torch.linalg.inv(cov)
                except:
                    cov_inv = torch.linalg.pinv(cov)

                # 计算二次型: (x-μ)^T Σ^{-1} (x-μ)
                diff_reshaped = gauss_diff.unsqueeze(-1)  # [1, batch_size, 2, 1]
                mahalanobis = torch.matmul(
                    diff_reshaped.transpose(2, 3),
                    torch.matmul(cov_inv, diff_reshaped)
                ).squeeze(-1).squeeze(-1)  # [1, batch_size]

                # 高斯函数
                alpha = torch.exp(-0.5 * mahalanobis)
                batch_alphas.append(alpha)

            # 堆叠所有高斯
            batch_alpha = torch.stack(batch_alphas, dim=0)  # [N, batch_size]
            alphas.append(batch_alpha)

        # 合并所有批次
        alpha = torch.cat(alphas, dim=1)  # [N, P]

        # 应用Mip滤波
        det = torch.det(cov2d).sqrt().unsqueeze(1)  # [N, 1]
        alpha = alpha * (2 * np.pi * det)
        alpha = torch.clamp(alpha, 0, 1)

        return alpha

    def compute_local_alpha(self, cov2d, uv, pixel_uv):
        """计算局部区域的alpha，更高效版本"""
        diff = pixel_uv - uv.unsqueeze(1)  # [1, P, 2]

        # 计算协方差矩阵的逆
        try:
            cov_inv = torch.linalg.inv(cov2d)  # [1, 2, 2]
        except:
            cov_inv = torch.linalg.pinv(cov2d)

        # 使用向量化但内存友好的方式计算马氏距离
        # diff^T @ cov_inv @ diff
        diff_reshaped = diff.unsqueeze(-1)  # [1, P, 2, 1]
        cov_inv_expanded = cov_inv.unsqueeze(1)  # [1, 1, 2, 2]

        # 分两步计算，避免大张量
        temp = torch.matmul(cov_inv_expanded, diff_reshaped)  # [1, P, 2, 1]
        mahalanobis = torch.matmul(diff_reshaped.transpose(2, 3), temp)  # [1, P, 1, 1]
        mahalanobis = mahalanobis.squeeze(-1).squeeze(-1)  # [1, P]

        # 高斯函数
        alpha = torch.exp(-0.5 * mahalanobis)

        # 考虑像素积分
        det = torch.det(cov2d).sqrt().unsqueeze(1)  # [1, 1]
        alpha = alpha * (2 * np.pi * det)

        return torch.clamp(alpha, 0, 1)

    def rasterize(self, uv: torch.Tensor, cov2d: torch.Tensor, depth: torch.Tensor,
                  opacity: torch.Tensor, colors: torch.Tensor,
                  H: int, W: int, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = uv.device

        # 1. 按深度对高斯排序（从近到远）
        sorted_indices = torch.argsort(depth, descending=False)
        uv = uv[sorted_indices]
        cov2d = cov2d[sorted_indices]
        opacity = opacity[sorted_indices]
        colors = colors[sorted_indices]

        # 2. 创建输出图像
        rendered = torch.ones(H, W, 3, device=device) * self.background_color.to(device)
        alpha_map = torch.zeros(H, W, 1, device=device)

        # 3. 显著减少处理的高斯数量
        num_gaussians = uv.shape[0]
        max_gaussians = min(50, num_gaussians)  # 只处理前50个高斯（测试用）

        # 4. 逐高斯处理，避免批量计算
        for i in range(max_gaussians):
            # 只计算该高斯对周围像素的影响（3σ原则）
            sigma = torch.sqrt(torch.max(cov2d[i].diag()))
            radius_pixels = int(3 * sigma.item()) + 1

            # 计算高斯在图像上的边界框
            u, v = uv[i]
            x_min = max(0, int(u.item() - radius_pixels))
            x_max = min(W, int(u.item() + radius_pixels) + 1)
            y_min = max(0, int(v.item() - radius_pixels))
            y_max = min(H, int(v.item() + radius_pixels) + 1)

            if x_min >= x_max or y_min >= y_max:
                continue

            # 创建局部像素网格
            y_coords, x_coords = torch.meshgrid(
                torch.arange(y_min, y_max, device=device, dtype=torch.float32),
                torch.arange(x_min, x_max, device=device, dtype=torch.float32),
                indexing='ij'
            )
            local_pixel_uv = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)

            # 计算局部alpha
            local_alpha = self.compute_local_alpha(
                cov2d[i].unsqueeze(0),  # [1, 2, 2]
                uv[i].unsqueeze(0),  # [1, 2]
                local_pixel_uv.unsqueeze(0)  # [1, num_local_pixels, 2]
            )

            # 重塑为局部图像大小
            local_alpha = local_alpha.reshape(1, y_max - y_min, x_max - x_min)

            # 应用不透明度
            local_alpha = local_alpha * torch.sigmoid(opacity[i])

            # alpha混合 - 不使用原地操作
            for c in range(3):
                # 提取局部区域
                rendered_patch = rendered[y_min:y_max, x_min:x_max, c].clone()
                # 计算新值
                new_patch = rendered_patch * (1 - local_alpha[0]) + colors[i, c] * local_alpha[0]
                # 赋值（不使用原地操作）
                rendered[y_min:y_max, x_min:x_max, c] = new_patch

            # 更新alpha_map - 不使用原地操作
            alpha_map_patch = alpha_map[y_min:y_max, x_min:x_max, 0].clone()
            new_alpha_patch = alpha_map_patch * (1 - local_alpha[0]) + local_alpha[0]
            alpha_map[y_min:y_max, x_min:x_max, 0] = new_alpha_patch

        return rendered, alpha_map

    def render(self, gaussian_model, K: torch.Tensor, pose: torch.Tensor,
               H: int, W: int) -> torch.Tensor:
        """
        完整渲染流程：投影 + 光栅化。
        """
        # 确保K不需要梯度
        K = K.detach().requires_grad_(False)

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