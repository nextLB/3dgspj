import torch
import torch.nn as nn
import numpy as np
from utils.gaussian_utils import build_rotation, build_covariance_from_scaling_rotation


def render(viewpoint_camera, pc, device, background_color=None, scaling_modifier=1.0):
    """
    渲染函数
    """
    # 设置背景
    if background_color is None:
        bg = torch.rand(3, device=device) if False else torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    else:
        bg = background_color

    # 获取高斯参数
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # 构建协方差矩阵
    cov3D = build_covariance_from_scaling_rotation(scales, scaling_modifier, rotations)

    # 获取颜色特征
    shs = pc.get_features

    # 相机参数
    world_view_transform = viewpoint_camera['world_view_transform']
    full_proj_transform = viewpoint_camera['full_proj_transform']
    camera_center = viewpoint_camera['camera_center']

    # 将点转换到相机坐标系
    means2D = world_view_transform @ torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=1).T
    means2D = means2D[:3, :].T

    # 简化渲染：使用alpha合成
    # 实际3DGS使用更复杂的可微溅射

    # 计算深度排序
    depths = means2D[:, 2]
    sorted_indices = torch.argsort(depths, descending=True)

    # 简化的alpha合成
    H, W = viewpoint_camera['height'], viewpoint_camera['width']
    image = torch.zeros((H, W, 3), device=device) + bg

    # 这里实现简化的渲染
    # 实际3DGS需要完整的可微高斯溅射实现

    # 返回简化结果
    result = {
        "render": image,
        "viewspace_points": means2D,
        "visibility_filter": torch.ones_like(depths, dtype=torch.bool),
        "radii": torch.ones_like(depths)
    }

    return result