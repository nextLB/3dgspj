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
        device = xyz.device

        # 1. 将高斯变换到相机坐标系
        R = pose[:3, :3]  # 旋转矩阵 [3, 3]
        t = pose[:3, 3]  # 平移向量 [3]

        # 世界坐标系到相机坐标系
        xyz_cam = (R @ xyz.T).T + t.unsqueeze(0)  # [N, 3]

        # 深度（用于排序）
        depth = xyz_cam[:, 2]

        # 2. 投影到图像平面
        xyz_proj = (K @ xyz_cam.T).T  # [N, 3]
        uv = xyz_proj[:, :2] / xyz_proj[:, 2:]  # 归一化设备坐标 [N, 2]

        # 3. 计算投影后的2D协方差
        # 将四元数转换为旋转矩阵
        rot_matrix = self.quaternion_to_matrix(rotation)  # [N, 3, 3]

        # 构建3D协方差矩阵
        scale_matrix = torch.diag_embed(scale)  # [N, 3, 3]

        # 3D协方差：R * S * S^T * R^T
        cov3d = rot_matrix @ scale_matrix @ scale_matrix.transpose(1, 2) @ rot_matrix.transpose(1, 2)

        # 投影到2D
        focal_x = K[0, 0]
        focal_y = K[1, 1]
        z = xyz_cam[:, 2]

        # 投影雅可比矩阵
        J = torch.zeros(xyz.shape[0], 2, 3, device=device)
        J[:, 0, 0] = focal_x / z
        J[:, 0, 2] = -focal_x * xyz_cam[:, 0] / (z * z)
        J[:, 1, 1] = focal_y / z
        J[:, 1, 2] = -focal_y * xyz_cam[:, 1] / (z * z)

        # 2D协方差：J * cov3d * J^T
        cov2d = J @ cov3d @ J.transpose(1, 2)  # [N, 2, 2]

        # 添加小值以确保正定性
        identity = torch.eye(2, device=device).unsqueeze(0) * 1e-4
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

        # 3. 处理所有高斯
        num_gaussians = uv.shape[0]

        # 4. 简化：只处理在图像范围内的高斯
        for i in range(num_gaussians):
            u, v = uv[i]

            # 跳过完全在图像外的高斯
            if u < -50 or u > W + 50 or v < -50 or v > H + 50:
                continue

            # 定义高斯的影响范围（基于协方差）
            # 简单起见，使用固定半径
            radius = 10  # 像素半径

            x_min = max(0, int(u.item() - radius))
            x_max = min(W, int(u.item() + radius) + 1)
            y_min = max(0, int(v.item() - radius))
            y_max = min(H, int(v.item() + radius) + 1)

            if x_min >= x_max or y_min >= y_max:
                continue

            # 创建局部像素网格
            y_coords, x_coords = torch.meshgrid(
                torch.arange(y_min, y_max, device=device, dtype=torch.float32),
                torch.arange(x_min, x_max, device=device, dtype=torch.float32),
                indexing='ij'
            )

            # 修复：正确创建像素坐标
            pixel_uv = torch.stack([x_coords.reshape(-1), y_coords.reshape(-1)], dim=1)  # [P, 2]

            # 计算像素与高斯中心的偏移
            diff = pixel_uv - uv[i].unsqueeze(0)  # [P, 2]

            # 获取当前高斯的协方差
            cov = cov2d[i]  # [2, 2]

            # 计算协方差的逆
            try:
                cov_inv = torch.linalg.inv(cov.unsqueeze(0)).squeeze(0)  # [2, 2]
            except:
                # 如果协方差矩阵不可逆，跳过这个高斯
                continue

            # 计算马氏距离: (x-μ)^T Σ^{-1} (x-μ)
            # diff: [P, 2], cov_inv: [2, 2]
            diff_t = diff.unsqueeze(1)  # [P, 1, 2]
            diff_expanded = diff.unsqueeze(-1)  # [P, 2, 1]

            # 矩阵乘法：(P,1,2) @ (2,2) @ (P,2,1) -> (P,1,1)
            mahalanobis = torch.matmul(
                torch.matmul(diff_t, cov_inv),
                diff_expanded
            ).squeeze(-1).squeeze(-1)  # [P]

            # 高斯函数计算alpha
            alpha = torch.exp(-0.5 * mahalanobis)  # [P]

            # 考虑像素积分（简化）
            det = torch.det(cov).sqrt()
            alpha = alpha * (2 * np.pi * det)
            alpha = torch.clamp(alpha, 0, 1)

            # 应用不透明度
            alpha = alpha * torch.sigmoid(opacity[i])

            # 重塑alpha为局部图像大小
            local_h = y_max - y_min
            local_w = x_max - x_min
            alpha_reshaped = alpha.reshape(local_h, local_w)

            # alpha混合（修复：避免原地操作）
            for c in range(3):
                rendered_patch = rendered[y_min:y_max, x_min:x_max, c]
                new_patch = rendered_patch * (1 - alpha_reshaped) + colors[i, c] * alpha_reshaped
                rendered[y_min:y_max, x_min:x_max, c] = new_patch

            # 更新alpha_map
            alpha_map_patch = alpha_map[y_min:y_max, x_min:x_max, 0]
            new_alpha_patch = alpha_map_patch * (1 - alpha_reshaped) + alpha_reshaped
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


    def quaternion_to_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """将四元数转换为3x3旋转矩阵"""
        q = quat / quat.norm(dim=-1, keepdim=True)
        w, x, y, z = q.unbind(dim=-1)

        return torch.stack([
            1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
            2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w,
            2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y
        ], dim=-1).reshape(-1, 3, 3)



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