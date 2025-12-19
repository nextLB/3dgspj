import os
import torch
import numpy as np
import random
from argparse import ArgumentParser
from pathlib import Path
import json
from tqdm import tqdm
import time

from dataset import MipNeRF360Dataset
from scene import Scene, GaussianModel
from pytorch_renderer import render_pytorch
from utils import training_report, save_checkpoint


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    # 准备输出目录
    model_path = Path(opt.model_path)
    model_path.mkdir(exist_ok=True, parents=True)

    # 初始化3D高斯模型
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=opt.start_checkpoint)

    # 设置优化器
    gaussians.training_setup(opt)

    # 训练循环
    iteration = 0
    if opt.start_checkpoint:
        iteration = scene.loaded_iter + 1

    progress_bar = tqdm(range(iteration, opt.iterations), desc="Training progress")

    for iteration in progress_bar:
        # 采样随机相机
        viewpoint_cam = scene.getTrainCameras().pop(random.randint(0, len(scene.getTrainCameras()) - 1))

        # 使用PyTorch渲染器
        render_pkg = render_pytorch(
            viewpoint_camera=viewpoint_cam,
            gaussians=gaussians,
            pipe=pipe,
            bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device=viewpoint_cam.original_image.device)
        )

        image, viewspace_point_tensor, visibility_filter = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"]

        # 计算损失
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # 反向传播
        loss.backward()

        # 优化步骤
        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            # 自适应密度控制
            if iteration % opt.densification_interval == 0 and iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

                if iteration % opt.densify_grad_threshold == 0:
                    gaussians.densify_and_prune(
                        max_grad=opt.densify_grad_threshold,
                        min_opacity=opt.min_opacity,
                        extent=opt.extent,
                        max_screen_size=opt.max_screen_size
                    )

        # 记录和报告
        progress_bar.set_postfix({"Loss": loss.item()})

        # 定期保存
        if iteration in saving_iterations:
            save_checkpoint(scene, iteration, model_path)

        # 定期测试
        if iteration in testing_iterations:
            testing(dataset, gaussians, pipe, iteration, model_path)

    # 最终保存
    save_checkpoint(scene, iteration, model_path)
    print("\nTraining complete.")


def testing(dataset, gaussians, pipe, iteration, model_path):
    torch.cuda.empty_cache()

    # 创建测试目录
    test_path = model_path / f"test_{iteration:07d}"
    test_path.mkdir(exist_ok=True)

    # 渲染测试视图
    for idx, viewpoint in enumerate(tqdm(dataset.getTestCameras(), desc="Testing")):
        render_pkg = render_pytorch(
            viewpoint_camera=viewpoint,
            gaussians=gaussians,
            pipe=pipe,
            bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device=viewpoint.original_image.device)
        )
        image = render_pkg["render"]

        # 保存图像
        from torchvision.utils import save_image
        save_image(image, test_path / f"{idx:05d}.png")

    # 保存点云
    scene_path = model_path / f"point_cloud_{iteration:07d}.ply"
    gaussians.save_ply(str(scene_path))


def l1_loss(pred, target):
    return torch.abs(pred - target).mean()


def ssim(img1, img2, window_size=11, size_average=True):
    """简化的SSIM实现，基于PyTorch"""
    from torch.nn.functional import conv2d

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 创建高斯核
    def create_window(window_size, channel):
        _1D_window = torch.exp(torch.arange(window_size).float() - window_size // 2).pow(2).div(-2.0)
        _1D_window = _1D_window / _1D_window.sum()
        _2D_window = _1D_window[:, None] * _1D_window[None, :]
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)



def main():
    # 参数设置
    parser = ArgumentParser(description="3D Gaussian Splatting Training (PyTorch Version)")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to Mip_NeRF360 dataset folder")
    parser.add_argument("--model_path", type=str, default="./output",
                        help="Path to save trained model")
    parser.add_argument("--scene", type=str, default="flowers",
                        help="Scene name (flowers, threehill, bicycle, etc.)")
    parser.add_argument("--resolution", type=int, default=2,
                        help="Image resolution scale (1, 2, 4, 8)")
    parser.add_argument("--iterations", type=int, default=10000,
                        help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--densify_until_iter", type=int, default=7000)
    parser.add_argument("--densify_from_iter", type=int, default=500)
    parser.add_argument("--densification_interval", type=int, default=100)
    parser.add_argument("--opacity_reset_interval", type=int, default=3000)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--position_lr_init", type=float, default=0.00016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.005)
    parser.add_argument("--rotation_lr", type=float, default=0.001)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--min_opacity", type=float, default=0.005)
    parser.add_argument("--extent", type=float, default=0.5)
    parser.add_argument("--max_screen_size", type=float, default=1.0)
    parser.add_argument("--start_checkpoint", type=str, default=None)

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 管道设置（渲染参数）
    pipe = {
        'convert_SHs_python': True,
        'compute_cov3D_python': True,
        'debug': False
    }

    # 创建数据集
    print("Loading dataset...")
    dataset = MipNeRF360Dataset(
        source_path=args.source_path,
        scene=args.scene,
        resolution=args.resolution,
        device=device
    )

    # 设置测试和保存的迭代次数
    testing_iterations = [1000, 3000, 7000, 10000]
    saving_iterations = [3000, 7000, 10000]

    # 开始训练
    print(f"Starting training for scene: {args.scene}")
    print(f"Training iterations: {args.iterations}")
    training(dataset, args, pipe, testing_iterations, saving_iterations)

if __name__ == "__main__":
    main()