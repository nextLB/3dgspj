#!/usr/bin/env python3
"""
简化的3D高斯溅射训练脚本
避免复杂的维度转换问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import json
import open3d as o3d
import matplotlib.pyplot as plt

from utils.gaussian_utils import GaussianModel
from simple_renderer import SimpleRenderer


def simple_train_gaussian_splatting(
        dataset,
        model_path,
        iterations=5000,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
        percent_dense=0.01,
        densification_interval=100,
        opacity_reset_interval=3000,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_grad_threshold=0.0002,
        sh_degree=3,
        random_background=False
):
    """
    简化的3D高斯溅射训练
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    # 创建输出目录
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    # 获取训练相机
    train_cameras = dataset.get_train_cameras()
    test_cameras = dataset.get_test_cameras()

    print(f"训练相机数量: {len(train_cameras)}")
    print(f"测试相机数量: {len(test_cameras)}")

    # 初始化高斯模型
    gaussians = GaussianModel(sh_degree=sh_degree)

    # 从SfM点云初始化或创建随机点云
    sparse_path = dataset.source_path / "sparse" / "0"
    if sparse_path.exists():
        print("从COLMAP点云初始化...")
        points3D_path = sparse_path / "points3D.bin"
        if points3D_path.exists():
            from utils.colmap_utils import read_points3d_binary
            points3d = read_points3d_binary(points3D_path)

            # 创建点云
            points = []
            colors = []

            for point_id, point in points3d.items():
                points.append(point.xyz)
                colors.append(point.rgb / 255.0)

            # 转换为张量
            points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
            colors_tensor = torch.tensor(colors, dtype=torch.float32, device=device)

            print(f"从COLMAP加载了 {len(points)} 个点")

            # 直接初始化高斯参数
            num_points = len(points)

            gaussians._xyz = nn.Parameter(points_tensor.requires_grad_(True))
            gaussians._features_dc = nn.Parameter(colors_tensor.unsqueeze(1).requires_grad_(True))
            gaussians._features_rest = nn.Parameter(
                torch.zeros((num_points, 3, (sh_degree + 1) ** 2 - 1), device=device).requires_grad_(True)
            )
            gaussians._scaling = nn.Parameter(
                torch.log(torch.ones((num_points, 3), device=device) * 0.01).requires_grad_(True)
            )
            gaussians._rotation = nn.Parameter(
                torch.zeros((num_points, 4), device=device).requires_grad_(True)
            )
            gaussians._rotation.data[:, 0] = 1.0
            gaussians._opacity = nn.Parameter(
                torch.logit(0.1 * torch.ones((num_points, 1), device=device)).requires_grad_(True)
            )

        else:
            print("未找到points3D.bin，使用随机初始化")
            create_random_gaussians(gaussians, dataset.camera_extent, device, sh_degree)
    else:
        print("未找到稀疏重建，使用随机初始化")
        create_random_gaussians(gaussians, dataset.camera_extent, device, sh_degree)

    gaussians.active_sh_degree = sh_degree

    # 初始化渲染器
    renderer = SimpleRenderer(device=device)

    # 设置优化器
    training_args = type('Args', (), {
        'position_lr_init': position_lr_init,
        'position_lr_final': position_lr_final,
        'position_lr_delay_mult': position_lr_delay_mult,
        'position_lr_max_steps': position_lr_max_steps,
        'feature_lr': feature_lr,
        'opacity_lr': opacity_lr,
        'scaling_lr': scaling_lr,
        'rotation_lr': rotation_lr,
        'percent_dense': percent_dense,
        'densification_interval': densification_interval,
        'opacity_reset_interval': opacity_reset_interval,
        'densify_from_iter': densify_from_iter,
        'densify_until_iter': densify_until_iter,
        'densify_grad_threshold': densify_grad_threshold
    })()

    setup_optimizer(gaussians, training_args, dataset.camera_extent)

    # 简化的训练循环
    print("开始训练...")

    iteration = 0
    progress_bar = tqdm(range(iterations), desc="Training")

    # 记录损失
    losses = []

    while iteration < iterations:
        # 随机选择一个相机
        idx = iteration % len(train_cameras)
        viewpoint_cam = train_cameras[idx]

        # 获取真实图像
        gt_image = viewpoint_cam["image"].to(device)

        try:
            # 渲染
            render_result = renderer.render(viewpoint_cam, gaussians, random_background)
            image = render_result["render"]

            # 简单的L1损失
            loss = F.l1_loss(image, gt_image)

            # 反向传播
            loss.backward()

            # 优化
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            # 记录损失
            losses.append(loss.item())

            # 更新进度条
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Iter': iteration
                })

        except Exception as e:
            print(f"迭代 {iteration} 失败: {e}")
            # 跳过这次迭代，继续
            pass

        iteration += 1
        progress_bar.update(1)

        # 定期保存
        if iteration % 1000 == 0 or iteration == iterations:
            # 保存点云
            save_gaussians_as_ply(gaussians, model_path / f"point_cloud_iteration_{iteration}.ply")

            # 保存损失曲线
            if losses:
                plt.figure(figsize=(10, 5))
                plt.plot(losses)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Training Loss')
                plt.savefig(model_path / f"loss_iteration_{iteration}.png")
                plt.close()

            # 测试渲染
            if iteration % 2000 == 0 and test_cameras:
                try:
                    test_cam = test_cameras[0]
                    with torch.no_grad():
                        test_render = renderer.render(test_cam, gaussians, random_background)
                        test_image = test_render["render"].cpu().numpy()

                        # 保存测试图像
                        plt.imsave(
                            model_path / f"test_iteration_{iteration}.png",
                            np.clip(test_image, 0, 1)
                        )
                except Exception as e:
                    print(f"测试渲染失败: {e}")

    progress_bar.close()

    # 保存最终模型
    save_gaussians_as_ply(gaussians, model_path / "final_point_cloud.ply")

    # 保存最终损失曲线
    if losses:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(model_path / "final_loss.png")
        plt.close()

    print("训练完成!")
    return gaussians


def create_random_gaussians(gaussians, camera_extent, device, sh_degree=3):
    """创建随机高斯"""
    num_points = 10000  # 减少点数以节省显存
    print(f"创建 {num_points} 个随机高斯点")

    # 随机位置
    points = torch.randn(num_points, 3, device=device) * camera_extent * 0.5

    # 随机颜色
    colors = torch.rand(num_points, 3, device=device)

    # 初始化参数
    gaussians._xyz = nn.Parameter(points.requires_grad_(True))
    gaussians._features_dc = nn.Parameter(colors.unsqueeze(1).requires_grad_(True))
    gaussians._features_rest = nn.Parameter(
        torch.zeros((num_points, 3, (sh_degree + 1) ** 2 - 1), device=device).requires_grad_(True)
    )
    gaussians._scaling = nn.Parameter(
        torch.log(torch.ones((num_points, 3), device=device) * 0.01).requires_grad_(True)
    )
    gaussians._rotation = nn.Parameter(
        torch.zeros((num_points, 4), device=device).requires_grad_(True)
    )
    gaussians._rotation.data[:, 0] = 1.0
    gaussians._opacity = nn.Parameter(
        torch.logit(0.1 * torch.ones((num_points, 1), device=device)).requires_grad_(True)
    )


def setup_optimizer(gaussians, training_args, spatial_lr_scale=1):
    """设置优化器"""
    gaussians.spatial_lr_scale = spatial_lr_scale
    gaussians.percent_dense = training_args.percent_dense

    # 初始化梯度累积
    num_points = gaussians._xyz.shape[0]
    gaussians.xyz_gradient_accum = torch.zeros((num_points, 1), device="cuda")
    gaussians.denom = torch.zeros((num_points, 1), device="cuda")

    # 设置优化器参数
    l = [
        {'params': [gaussians._xyz], 'lr': training_args.position_lr_init * spatial_lr_scale, "name": "xyz"},
        {'params': [gaussians._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
        {'params': [gaussians._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
        {'params': [gaussians._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        {'params': [gaussians._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        {'params': [gaussians._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
    ]

    gaussians.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    gaussians.xyz_scheduler_args = {
        'init_lr': training_args.position_lr_init * spatial_lr_scale,
        'final_lr': training_args.position_lr_final * spatial_lr_scale,
        'delay_mult': training_args.position_lr_delay_mult,
        'max_steps': training_args.position_lr_max_steps
    }


def save_gaussians_as_ply(gaussians, path):
    """保存高斯为PLY文件"""
    import plyfile

    xyz = gaussians._xyz.detach().cpu().numpy()
    features_dc = gaussians._features_dc.detach().cpu().numpy().squeeze(1)
    scaling = gaussians._scaling.detach().cpu().numpy()
    rotation = gaussians._rotation.detach().cpu().numpy()
    opacity = torch.sigmoid(gaussians._opacity).detach().cpu().numpy()

    # 创建顶点
    vertices = np.zeros(len(xyz), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('scale_x', 'f4'), ('scale_y', 'f4'), ('scale_z', 'f4'),
        ('rot_w', 'f4'), ('rot_x', 'f4'), ('rot_y', 'f4'), ('rot_z', 'f4'),
        ('opacity', 'f4')
    ])

    # 填充数据
    vertices['x'] = xyz[:, 0].astype('f4')
    vertices['y'] = xyz[:, 1].astype('f4')
    vertices['z'] = xyz[:, 2].astype('f4')

    # 颜色 (0-255)
    colors = np.clip(features_dc, 0, 1) * 255
    vertices['red'] = colors[:, 0].astype('u1')
    vertices['green'] = colors[:, 1].astype('u1')
    vertices['blue'] = colors[:, 2].astype('u1')

    # 缩放
    scale = np.exp(scaling)
    vertices['scale_x'] = scale[:, 0].astype('f4')
    vertices['scale_y'] = scale[:, 1].astype('f4')
    vertices['scale_z'] = scale[:, 2].astype('f4')

    # 旋转（四元数）
    rotation_norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    rotation_normalized = rotation / rotation_norm
    vertices['rot_w'] = rotation_normalized[:, 0].astype('f4')
    vertices['rot_x'] = rotation_normalized[:, 1].astype('f4')
    vertices['rot_y'] = rotation_normalized[:, 2].astype('f4')
    vertices['rot_z'] = rotation_normalized[:, 3].astype('f4')

    # 不透明度
    vertices['opacity'] = opacity[:, 0].astype('f4')

    # 创建PLY元素
    vertex_element = plyfile.PlyElement.describe(vertices, 'vertex')

    # 写入文件
    plyfile.PlyData([vertex_element], text=False).write(str(path))
    print(f"保存点云到: {path}")


if __name__ == "__main__":
    # 测试代码
    from scene.dataset import SceneDataset

    # 创建数据集
    dataset = SceneDataset(
        source_path="/home/next_lb/桌面/无人机影像三维重建任务/Mip_NeRF360/360_extra_scenes/flowers",
        images="images_4",
        resolution=512,
        data_device="cuda",
        white_background=False,
        eval=False
    )

    # 训练
    simple_train_gaussian_splatting(
        dataset=dataset,
        model_path="./output/flowers_simple",
        iterations=2000,  # 先尝试少量迭代
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001
    )


