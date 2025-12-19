import torch
import torch.nn.functional as F
import numpy as np


class SimpleRenderer:
    """
    简化渲染器：将3D高斯投影到2D图像
    """

    def __init__(self, device='cuda'):
        self.device = device

    def render(self, viewpoint_camera, gaussians, random_background=False):
        """
        简化渲染：将3D高斯投影到2D图像
        """
        height = viewpoint_camera['height']
        width = viewpoint_camera['width']

        # 设置背景
        if random_background:
            background = torch.rand(3, device=self.device)
        else:
            background = torch.zeros(3, device=self.device)

        # 创建空白图像
        image = torch.ones((height, width, 3), device=self.device) * background

        # 获取高斯参数
        xyz = gaussians._xyz  # 位置 [N, 3]
        colors = torch.sigmoid(gaussians._features_dc.squeeze(1))  # 颜色 [N, 3]
        opacity = torch.sigmoid(gaussians._opacity).squeeze(1)  # 不透明度 [N]

        # 相机参数
        world_view_transform = viewpoint_camera['world_view_transform']  # [4, 4]
        projection_matrix = viewpoint_camera['projection_matrix']  # [4, 4]

        # 将点从世界坐标转换到相机坐标
        # 添加齐次坐标
        xyz_homo = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)  # [N, 4]

        # 世界到相机变换
        points_camera = (world_view_transform @ xyz_homo.T).T  # [N, 4]

        # 相机到裁剪空间
        points_clip = (projection_matrix @ points_camera.T).T  # [N, 4]

        # 透视除法得到NDC坐标
        points_ndc = points_clip[:, :3] / points_clip[:, 3:]  # [N, 3]

        # 检查哪些点在视锥体内
        in_frustum = (
                (points_ndc[:, 0].abs() <= 1.0) &
                (points_ndc[:, 1].abs() <= 1.0) &
                (points_clip[:, 2] > 0)  # 深度为正
        )

        if not in_frustum.any():
            # 没有点在视锥体内，返回背景
            return {
                "render": image,
                "viewspace_points": xyz,
                "visibility_filter": in_frustum,
                "radii": torch.ones_like(opacity)
            }

        # 只处理视锥体内的点
        valid_xyz = xyz[in_frustum]
        valid_colors = colors[in_frustum]
        valid_opacity = opacity[in_frustum]
        valid_points_ndc = points_ndc[in_frustum]

        # 将NDC坐标转换为像素坐标
        # NDC: x在[-1, 1]，y在[-1, 1]
        # 像素: x在[0, width]，y在[0, height]
        points_pixel = torch.zeros_like(valid_points_ndc[:, :2])
        points_pixel[:, 0] = (valid_points_ndc[:, 0] + 1) * width / 2  # x坐标
        points_pixel[:, 1] = (1 - valid_points_ndc[:, 1]) * height / 2  # y坐标（图像y轴向下）

        # 将点四舍五入到最近的像素
        points_pixel_int = points_pixel.round().long()

        # 确保不越界
        valid_mask = (
                (points_pixel_int[:, 0] >= 0) &
                (points_pixel_int[:, 0] < width) &
                (points_pixel_int[:, 1] >= 0) &
                (points_pixel_int[:, 1] < height)
        )

        if valid_mask.any():
            # 只处理有效的像素位置
            final_pixels = points_pixel_int[valid_mask]
            final_colors = valid_colors[valid_mask]
            final_opacity = valid_opacity[valid_mask]

            # 简单的alpha合成（不考虑深度排序）
            for i, (x, y) in enumerate(final_pixels):
                # 当前像素颜色
                current_color = image[y, x]
                # 新颜色（带透明度）
                new_color = final_colors[i]
                alpha = final_opacity[i]
                # Alpha合成
                image[y, x] = current_color * (1 - alpha) + new_color * alpha

        return {
            "render": image,
            "viewspace_points": xyz,
            "visibility_filter": in_frustum,
            "radii": torch.ones_like(opacity)
        }

    def render_depth(self, viewpoint_camera, gaussians):
        """
        渲染深度图
        """
        height = viewpoint_camera['height']
        width = viewpoint_camera['width']

        # 创建深度图
        depth_image = torch.zeros((height, width), device=self.device)

        # 获取高斯位置
        xyz = gaussians._xyz

        # 相机参数
        world_view_transform = viewpoint_camera['world_view_transform']

        # 将点从世界坐标转换到相机坐标
        xyz_homo = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
        points_camera = (world_view_transform @ xyz_homo.T).T

        # 深度（相机空间的z坐标）
        depths = points_camera[:, 2]

        # 归一化深度用于可视化
        if len(depths) > 0:
            min_depth = depths.min()
            max_depth = depths.max()
            if max_depth > min_depth:
                depths_normalized = (depths - min_depth) / (max_depth - min_depth)
            else:
                depths_normalized = torch.zeros_like(depths)
        else:
            depths_normalized = torch.zeros_like(depths)

        # 投影到图像（简化版）
        projection_matrix = viewpoint_camera['projection_matrix']
        points_clip = (projection_matrix @ points_camera.T).T
        points_ndc = points_clip[:, :3] / points_clip[:, 3:]

        # 转换到像素坐标
        points_pixel = torch.zeros_like(points_ndc[:, :2])
        points_pixel[:, 0] = (points_ndc[:, 0] + 1) * width / 2
        points_pixel[:, 1] = (1 - points_ndc[:, 1]) * height / 2

        points_pixel_int = points_pixel.round().long()

        # 将深度值赋给对应的像素
        valid_mask = (
                (points_pixel_int[:, 0] >= 0) &
                (points_pixel_int[:, 0] < width) &
                (points_pixel_int[:, 1] >= 0) &
                (points_pixel_int[:, 1] < height)
        )

        if valid_mask.any():
            valid_pixels = points_pixel_int[valid_mask]
            valid_depths = depths_normalized[valid_mask]

            for i, (x, y) in enumerate(valid_pixels):
                depth_image[y, x] = valid_depths[i]

        return depth_image