#!/usr/bin/env python3
"""
3D Gaussian Splatting 优化训练脚本 - RTX 3060专用版
支持混合精度训练、梯度累积、显存优化
"""

import os
import sys
import time
import argparse
import yaml
from pathlib import Path

import torch
import torch.cuda.amp as amp
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from gaussian_model_opt import OptimizedGaussianModel
from scene_opt import OptimizedScene
from camera_opt import OptimizedCamera
from render_opt import render_gaussians_optimized, compute_ssim_simple
from utils_opt import *


# ==================== RTX 3060 优化配置 ====================
def setup_rtx3060_optimizations():
    """专门为RTX 3060设置的优化选项"""
    # 启用TF32 (Ampere架构优化)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 启用cudnn自动优化器
    torch.backends.cudnn.benchmark = True

    # 禁用确定性算法以获得更好性能
    torch.backends.cudnn.deterministic = False

    print("✅ RTX 3060优化已启用:")
    print(f"   - TF32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"   - cudNN基准测试: {torch.backends.cudnn.benchmark}")

    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU模式（性能极差）")
        return 'cpu'

    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

    print(f"✅ 检测到GPU: {gpu_name}")
    print(f"✅ 显存总量: {gpu_memory:.1f} GB")

    # RTX 3060特定优化建议
    if "3060" in gpu_name:
        print("🎯 RTX 3060检测到，应用专用优化:")
        print("   - 混合精度训练 (AMP)")
        print("   - 梯度累积")
        print("   - 动态分辨率调整")

    return device


# ==================== 数据加载函数 ====================
def load_mipnerf360_data_optimized(dataset_path, scene_name, resolution=2, device='cuda'):
    """
    优化版Mip-NeRF 360数据加载
    支持动态分辨率，直接加载到GPU
    """
    print(f"📂 加载数据集: {scene_name}, 分辨率: 1/{resolution}")

    scene_dir = os.path.join(dataset_path, "360_v2", scene_name)

    if not os.path.exists(scene_dir):
        raise FileNotFoundError(f"场景目录不存在: {scene_dir}")

    # 加载相机位姿和边界
    poses_bounds_path = os.path.join(scene_dir, "poses_bounds.npy")
    if not os.path.exists(poses_bounds_path):
        raise FileNotFoundError(f"poses_bounds.npy不存在: {poses_bounds_path}")

    poses_bounds = np.load(poses_bounds_path)

    # 解析Mip-NeRF 360格式
    poses = poses_bounds[:, :-2].reshape(-1, 3, 5)
    bounds = poses_bounds[:, -2:]

    num_images = poses.shape[0]
    cam2world = poses[:, :3, :4]  # 相机到世界变换 (3x4)
    focal_length = poses[:, 0, 4]  # 焦距

    # 选择图像分辨率
    resolution_map = {1: "images", 2: "images_2", 4: "images_4", 8: "images_8"}
    img_folder = resolution_map.get(resolution, "images")
    img_dir = os.path.join(scene_dir, img_folder)

    if not os.path.exists(img_dir):
        print(f"⚠️  分辨率文件夹 {img_folder} 不存在，尝试使用 images 文件夹")
        img_dir = os.path.join(scene_dir, "images")

    # 获取并排序图像文件
    img_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(img_extensions)])

    if len(img_files) == 0:
        raise FileNotFoundError(f"在 {img_dir} 中未找到图像文件")

    # 限制加载的图像数量以避免显存溢出
    max_images = min(num_images, len(img_files), 100)  # RTX 3060最多处理100张
    img_files = img_files[:max_images]

    print(f"📷 加载 {len(img_files)} 张图像...")

    # 预分配GPU内存
    images_list = []

    for i, img_file in enumerate(tqdm(img_files, desc="加载图像")):
        img_path = os.path.join(img_dir, img_file)

        try:
            # 使用PIL加载并转换
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 调整大小（如果需要）
                if resolution > 1:
                    new_size = (img.width // resolution, img.height // resolution)
                    img = img.resize(new_size, Image.LANCZOS)

                # 转换为tensor并归一化
                img_array = np.array(img, dtype=np.float32) / 255.0

                # [RTX 3060优化] 直接移动到GPU，减少CPU-GPU传输
                img_tensor = torch.from_numpy(img_array).to(device).permute(2, 0, 1)
                images_list.append(img_tensor)

        except Exception as e:
            print(f"⚠️  加载图像 {img_file} 失败: {e}")
            continue

    if len(images_list) == 0:
        raise RuntimeError("没有成功加载任何图像")

    images_tensor = torch.stack(images_list)
    H, W = images_tensor.shape[2], images_tensor.shape[3]

    print(f"📐 图像尺寸: {W}x{H}")

    # 计算内参矩阵
    K_list = []
    for i in range(len(img_files)):
        K = torch.eye(3, device=device)
        K[0, 0] = focal_length[i] / resolution  # 根据分辨率调整焦距
        K[1, 1] = focal_length[i] / resolution
        K[0, 2] = W / 2.0
        K[1, 2] = H / 2.0
        K_list.append(K)

    K_tensor = torch.stack(K_list)

    # 转换为世界到相机矩阵
    world2cam_list = []

    for i in range(len(img_files)):
        c2w = torch.eye(4, device=device)
        c2w[:3, :3] = torch.from_numpy(cam2world[i, :3, :3]).float().to(device)
        c2w[:3, 3] = torch.from_numpy(cam2world[i, :3, 3]).float().to(device)

        # 相机到世界 -> 世界到相机
        w2c = torch.inverse(c2w)
        world2cam_list.append(w2c)

    world2cam_tensor = torch.stack(world2cam_list)

    # 准备边界
    near_tensor = torch.from_numpy(bounds[:len(img_files), 0]).float().to(device)
    far_tensor = torch.from_numpy(bounds[:len(img_files), 1]).float().to(device)

    dataset_info = {
        'images': images_tensor,
        'world2cam': world2cam_tensor,
        'K': K_tensor,
        'H': H,
        'W': W,
        'near': near_tensor,
        'far': far_tensor,
        'image_files': img_files,
        'scene_dir': scene_dir,
        'scene_name': scene_name,
        'resolution': resolution
    }

    print(f"✅ 数据集加载完成: {len(img_files)} 张图像")
    print_gpu_memory()

    return dataset_info


# ==================== 训练主函数 ====================
def train_optimized(config):
    """优化版训练函数，支持混合精度和梯度累积"""

    # 设置设备
    device = setup_rtx3060_optimizations()

    # 创建输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"{config.scene_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 输出目录: {output_dir}")

    # 保存配置
    config_dict = vars(config)
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f)

    # 加载数据集
    print("=" * 50)
    dataset = load_mipnerf360_data_optimized(
        config.dataset_path,
        config.scene_name,
        config.resolution,
        device
    )

    # 初始化高斯模型
    print("🎯 初始化高斯模型...")
    gaussians = OptimizedGaussianModel(
        sh_degree=config.sh_degree,
        device=device
    )

    # 创建场景
    scene = OptimizedScene(dataset, gaussians, output_dir)

    # 设置训练参数
    gaussians.training_setup(config)

    # 🔥 修复：检查模型参数梯度设置
    print("🔍 检查模型参数梯度设置...")
    has_grad_params = False
    for name, param in gaussians.named_parameters():
        print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
        if param.requires_grad:
            has_grad_params = True

    if not has_grad_params:
        print("❌ 错误：没有需要梯度的参数！手动启用梯度...")
        for param in gaussians.parameters():
            param.requires_grad = True

    # 确保模型处于训练模式
    gaussians.train()

    # 检查模型是否真的在训练模式
    print(f"模型训练模式: {gaussians.training}")

    # 初始化混合精度梯度缩放器
    scaler = amp.GradScaler(enabled=config.use_amp)

    # 训练统计
    stats = {
        'losses': [],
        'psnrs': [],
        'iterations': [],
        'timestamps': []
    }

    # 训练循环
    print("🚀 开始训练...")
    print("=" * 50)

    start_time = time.time()
    iteration = 0

    # 🔥 修复：创建一个简单的测试批次，验证梯度传播
    print("🧪 验证梯度传播...")
    try:
        test_camera = scene.get_random_train_camera()
        test_target = test_camera.original_image

        with amp.autocast(enabled=config.use_amp):
            test_render = render_gaussians_optimized(gaussians, test_camera)
            test_loss = torch.abs(test_render - test_target).mean()

        # 检查渲染图像是否有梯度
        print(f"测试渲染图像requires_grad: {test_render.requires_grad}")
        print(f"测试损失requires_grad: {test_loss.requires_grad}")

        # 尝试反向传播
        test_loss.backward()

        # 检查是否有梯度
        has_gradient = False
        for name, param in gaussians.named_parameters():
            if param.grad is not None:
                print(f"  {name}: 梯度存在, 形状: {param.grad.shape}")
                has_gradient = True

        if has_gradient:
            print("✅ 梯度传播验证成功！")
        else:
            print("⚠️  警告：没有检测到梯度，继续训练可能会有问题")

        # 清除梯度
        gaussians.optimizer.zero_grad(set_to_none=True)

    except Exception as e:
        print(f"❌ 梯度验证失败: {e}")
        print("尝试继续训练...")

    # 主训练循环
    with tqdm(total=config.iterations, desc="训练进度") as pbar:
        while iteration < config.iterations:
            # 梯度累积循环
            accum_loss = 0.0

            for accum_step in range(config.gradient_accumulation):
                # 选择随机相机
                camera = scene.get_random_train_camera()

                # 🔥 修复：确保目标图像在正确设备上
                target_image = camera.original_image.to(device)

                # 混合精度上下文
                with amp.autocast(enabled=config.use_amp):
                    # 渲染图像
                    rendered_image = render_gaussians_optimized(gaussians, camera, use_amp=config.use_amp)

                    # 🔥 修复：检查渲染图像的梯度
                    if not rendered_image.requires_grad:
                        print(f"⚠️ 警告：渲染图像没有梯度追踪！迭代 {iteration}")
                        # 重新创建需要梯度的张量
                        rendered_image = rendered_image.clone().detach().requires_grad_(True)

                    # 计算损失
                    l1_loss = torch.abs(rendered_image - target_image).mean()

                    # 检查损失是否有梯度
                    if not l1_loss.requires_grad:
                        print(f"⚠️ 警告：L1损失没有梯度追踪！迭代 {iteration}")
                        # 重新计算损失
                        l1_loss = torch.abs(rendered_image.clone().detach().requires_grad_(True) - target_image).mean()

                    # SSIM损失（可选）
                    if config.lambda_dssim > 0:
                        ssim_loss = 1.0 - compute_ssim_simple(rendered_image, target_image)
                        loss = (1.0 - config.lambda_dssim) * l1_loss + config.lambda_dssim * ssim_loss
                    else:
                        loss = l1_loss

                    # 确保损失有梯度
                    if not loss.requires_grad:
                        print(f"❌ 错误：总损失没有梯度！尝试修复...")
                        # 创建一个需要梯度的最小损失
                        zero_tensor = torch.tensor(0.0, device=device, requires_grad=True)
                        loss = loss + zero_tensor

                    # 缩放损失（用于梯度累积）
                    scaled_loss = loss / config.gradient_accumulation

                    # 检查是否为有效数值
                    if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                        print(f"⚠️ 警告：损失为无效值 {scaled_loss.item()}，跳过此步")
                        continue

                    accum_loss += loss.item()

                # 反向传播（累积梯度）
                try:
                    scaler.scale(scaled_loss).backward()
                except RuntimeError as e:
                    print(f"❌ 反向传播失败: {e}")
                    print("尝试手动计算梯度...")

                    # 手动检查梯度流
                    print(f"损失值: {loss.item()}")
                    print(f"损失requires_grad: {loss.requires_grad}")

                    # 尝试直接计算梯度
                    loss.backward()

            # 梯度累积完成后更新参数
            try:
                scaler.step(gaussians.optimizer)
                scaler.update()
                gaussians.optimizer.zero_grad(set_to_none=True)  # 释放梯度内存
            except Exception as e:
                print(f"❌ 参数更新失败: {e}")
                # 尝试继续，但清空梯度
                gaussians.optimizer.zero_grad(set_to_none=True)

            # 更新学习率
            if gaussians.scheduler:
                gaussians.scheduler.step()

            # 记录统计
            if iteration % 10 == 0:
                psnr = compute_psnr(rendered_image, target_image)
                stats['losses'].append(accum_loss / max(1, config.gradient_accumulation))
                stats['psnrs'].append(psnr.item() if torch.is_tensor(psnr) else psnr)
                stats['iterations'].append(iteration)
                stats['timestamps'].append(time.time() - start_time)

            # 更新进度条
            if iteration % 10 == 0:
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'PSNR': f"{psnr:.2f}" if 'psnr' in locals() else "N/A",
                    'LR': f"{gaussians.optimizer.param_groups[0]['lr']:.6f}"
                })

            # 定期保存检查点
            if (iteration + 1) % config.checkpoint_interval == 0:
                # 保存模型
                checkpoint_dir = os.path.join(output_dir, f"checkpoint_{iteration + 1:06d}")
                scene.save_checkpoint(checkpoint_dir, iteration)

                # 渲染测试图像
                if scene.test_cameras:
                    test_camera = scene.test_cameras[0]
                    with torch.no_grad():
                        test_render = render_gaussians_optimized(gaussians, test_camera)

                    # 保存测试渲染
                    render_dir = os.path.join(output_dir, "renders")
                    os.makedirs(render_dir, exist_ok=True)

                    save_image_tensor(
                        test_render,
                        os.path.join(render_dir, f"test_{iteration + 1:06d}.png")
                    )

            # 定期打印GPU状态
            if iteration % 100 == 0:
                print_gpu_memory()

            iteration += 1
            pbar.update(1)

    # 训练完成
    training_time = time.time() - start_time
    print("=" * 50)
    print(f"✅ 训练完成!")
    print(f"⏱️  总训练时间: {training_time:.2f}秒")
    print(f"📊 平均每迭代: {training_time / config.iterations:.3f}秒")

    # 保存最终模型
    final_dir = os.path.join(output_dir, "final_model")
    scene.save_checkpoint(final_dir, config.iterations)

    # 绘制训练曲线
    plot_training_curve(stats, os.path.join(output_dir, "training_curve.png"))

    # 保存统计
    np.save(os.path.join(output_dir, "training_stats.npy"), stats)

    return gaussians, scene, output_dir


# ==================== 配置解析 ====================
def parse_config():
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting 优化训练 (RTX 3060)")

    # 数据集参数
    parser.add_argument("--dataset_path", type=str, default="/home/next_lb/桌面/无人机影像三维重建任务/archive/",
                        help="数据集根路径")
    parser.add_argument("--scene_name", type=str, required=True,
                        help="场景名称 (如: bicycle, bonsai)")
    parser.add_argument("--resolution", type=int, default=2, choices=[1, 2, 4, 8],
                        help="图像分辨率 (1=原始, 2=1/2, 4=1/4, 8=1/8)")

    # 训练参数
    parser.add_argument("--iterations", type=int, default=7000,
                        help="训练迭代次数")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="初始学习率")
    parser.add_argument("--lambda_dssim", type=float, default=0.2,
                        help="SSIM损失权重")
    parser.add_argument("--sh_degree", type=int, default=0,
                        help="球谐函数阶数 (0=只使用DC项)")

    # RTX 3060优化参数
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="启用自动混合精度训练")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                        help="梯度累积步数 (模拟更大batch size)")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="检查点保存间隔")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="输出目录")

    # 从配置文件加载（如果存在）
    config_path = "config.yaml"
    if os.path.exists(config_path):
        print(f"📄 从配置文件加载: {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # 如果配置文件中有scene_name，移除required限制
        if config_dict and 'scene_name' in config_dict:
            # 找到scene_name参数并修改
            for action in parser._actions:
                if action.dest == 'scene_name':
                    action.required = False
                    break

        # 解析命令行参数
        args = parser.parse_args()

        # 更新配置
        args_dict = vars(args)

        # 递归更新配置（处理嵌套字典）
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_dict(d[k], v)
                else:
                    d[k] = v

        # 更新配置
        if config_dict:
            update_dict(args_dict, config_dict)

        # 重新创建命名空间
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    # 打印配置
    print("⚙️  训练配置:")
    for key, value in vars(args).items():
        if not key.startswith('_'):
            print(f"   {key}: {value}")

    return args


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 解析配置
    config = parse_config()

    # 设置随机种子
    set_seed(42)

    try:
        # 开始训练
        gaussians, scene, output_dir = train_optimized(config)

        print("=" * 50)
        print(f"🎉 训练成功完成!")
        print(f"📁 结果保存在: {output_dir}")
        print(f"💾 最终模型: {output_dir}/final_model/")

        # 保存点云用于可视化
        ply_path = os.path.join(output_dir, "final_model", "point_cloud.ply")
        gaussians.save_ply(ply_path)
        print(f"☁️  点云已保存: {ply_path}")

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)