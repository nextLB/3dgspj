import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import json
import open3d as o3d

from utils.gaussian_utils import GaussianModel
from gaussian_renderer import render


def train_gaussian_splatting(
        dataset,
        model_path,
        iterations=30000,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
        percent_dense=0.01,
        lambda_dssim=0.2,
        densification_interval=100,
        opacity_reset_interval=3000,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_grad_threshold=0.0002,
        sh_degree=3,
        random_background=False
):
    """
    训练3D高斯溅射模型
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

    # 从SfM点云初始化
    sparse_path = dataset.source_path / "sparse" / "0"
    if sparse_path.exists():
        print("从COLMAP点云初始化...")
        points3D_path = sparse_path / "points3D.bin"
        if points3D_path.exists():
            from utils.colmap_utils import read_points3d_binary
            points3d = read_points3d_binary(points3D_path)

            # 创建点云
            pcd = o3d.geometry.PointCloud()
            points = []
            colors = []

            for point_id, point in points3d.items():
                points.append(point.xyz)
                colors.append(point.rgb / 255.0)

            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # 从点云创建高斯
            gaussians.create_from_pcd(pcd, spatial_lr_scale=dataset.camera_extent)
        else:
            print("未找到points3D.bin，使用随机初始化")
            # 创建随机点云
            num_points = 10000
            points = np.random.randn(num_points, 3) * dataset.camera_extent * 0.5
            colors = np.random.rand(num_points, 3)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            gaussians.create_from_pcd(pcd, spatial_lr_scale=dataset.camera_extent)
    else:
        print("未找到稀疏重建，使用随机初始化")
        # 创建随机点云
        num_points = 10000
        points = np.random.randn(num_points, 3) * dataset.camera_extent * 0.5
        colors = np.random.rand(num_points, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        gaussians.create_from_pcd(pcd, spatial_lr_scale=dataset.camera_extent)

    # 设置训练参数
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

    gaussians.training_setup(training_args)

    # 训练循环
    print("开始训练...")

    iteration = 0
    progress_bar = tqdm(range(iterations), desc="Training")

    while iteration < iterations:
        # 随机选择一个相机
        viewpoint_cam = np.random.choice(train_cameras)

        # 渲染
        render_pkg = render(viewpoint_cam, gaussians, device, random_background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # 计算损失
        gt_image = viewpoint_cam["image"].to(device)

        # L1损失
        Ll1 = torch.abs(image - gt_image).mean()

        # SSIM损失
        ssim_loss = 1.0 - ssim(image, gt_image)

        # 总损失
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * ssim_loss

        # 反向传播
        loss.backward()

        # 优化
        with torch.no_grad():
            # 更新参数
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            # 更新学习率
            gaussians.update_learning_rate(iteration)

            # 密集化
            if iteration % densification_interval == 0 and iteration >= densify_from_iter and iteration <= densify_until_iter:
                # 保存梯度
                gaussians.xyz_gradient_accum[visibility_filter] += torch.norm(
                    viewspace_point_tensor.grad[visibility_filter, :2], dim=-1, keepdim=True)
                gaussians.denom[visibility_filter] += 1

                # 执行密集化
                if iteration > densify_from_iter:
                    size_threshold = 20 if iteration > opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        densify_grad_threshold,
                        0.005,
                        dataset.camera_extent,
                        size_threshold
                    )

            # 重置不透明度
            if iteration % opacity_reset_interval == 0 and iteration >= densify_from_iter:
                gaussians.reset_opacity()

        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'L1': f'{Ll1.item():.4f}',
            'SSIM': f'{ssim_loss.item():.4f}'
        })
        progress_bar.update(1)

        iteration += 1

        # 定期保存
        if iteration % 1000 == 0 or iteration == iterations:
            # 保存模型
            gaussians.save_ply(model_path / f"point_cloud_iteration_{iteration}.ply")

            # 测试渲染
            if iteration % 5000 == 0 and test_cameras:
                test_cam = test_cameras[0]
                with torch.no_grad():
                    render_pkg = render(test_cam, gaussians, device, random_background)
                    test_image = render_pkg["render"].cpu().numpy()

                    # 保存测试图像
                    import matplotlib.pyplot as plt
                    plt.imsave(model_path / f"test_iteration_{iteration}.png",
                               np.clip(test_image, 0, 1))

    progress_bar.close()

    # 保存最终模型
    gaussians.save_ply(model_path / "final_point_cloud.ply")

    print("训练完成!")
    return gaussians


def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算SSIM损失
    """
    # 简化版SSIM计算
    # 实际实现需要更完整的SSIM计算
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1.permute(2, 0, 1).unsqueeze(0), window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2.permute(2, 0, 1).unsqueeze(0), window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1.permute(2, 0, 1).unsqueeze(0).pow(2), window_size, stride=1,
                             padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2.permute(2, 0, 1).unsqueeze(0).pow(2), window_size, stride=1,
                             padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d((img1.permute(2, 0, 1).unsqueeze(0) * img2.permute(2, 0, 1).unsqueeze(0)), window_size,
                           stride=1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)