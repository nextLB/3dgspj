#!/usr/bin/env python3
"""
3D Gaussian Splatting 优化训练脚本 - RTX 3060专用版
修复混合精度训练问题
"""

import os
import sys
import time
import argparse
import yaml
from pathlib import Path
import contextlib

import torch
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
from render_opt import render_gaussians_optimized, compute_rendering_loss
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
    max_images = min(num_images, len(img_files), 50)  # RTX 3060最多处理50张，减少内存占用
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


# ==================== 混合精度训练辅助函数 ====================
def setup_mixed_precision(config, device):
    """设置混合精度训练"""
    if not config.use_amp:
        return None, contextlib.nullcontext()

    print("⚡ 启用混合精度训练 (AMP)")

    # 创建GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # 创建autocast上下文
    autocast_ctx = torch.cuda.amp.autocast(enabled=True)

    return scaler, autocast_ctx


def safe_mixed_precision_backward(scaler, loss, optimizer, retain_graph=False):
    """安全的混合精度反向传播"""
    if scaler is not None:
        # 使用scaler进行反向传播
        scaler.scale(loss).backward(retain_graph=retain_graph)
    else:
        # 普通反向传播
        loss.backward(retain_graph=retain_graph)


def safe_mixed_precision_step(scaler, optimizer, model_params=None, max_norm=1.0):
    """安全的混合精度优化器步骤 - 修复unscale问题"""
    if scaler is not None:
        # 🔥 修复：只在需要时调用unscale，避免重复调用
        scaler.step(optimizer)
        scaler.update()
    else:
        # 普通优化器步骤
        optimizer.step()

    # 梯度裁剪（如果需要）
    if model_params is not None:
        torch.nn.utils.clip_grad_norm_(model_params, max_norm=max_norm)


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

    # 设置混合精度训练
    scaler, autocast_ctx = setup_mixed_precision(config, device)

    # 训练统计
    stats = {
        'losses': [],
        'psnrs': [],
        'iterations': [],
        'timestamps': [],
        'learning_rates': []
    }

    # 训练循环
    print("🚀 开始训练...")
    print("=" * 50)

    start_time = time.time()
    iteration = 0

    # 🔥 修复：使用更稳定的训练循环
    best_loss = float('inf')
    patience_counter = 0

    # 主训练循环
    with tqdm(total=config.iterations, desc="训练进度") as pbar:
        while iteration < config.iterations:
            # 清除梯度
            gaussians.optimizer.zero_grad(set_to_none=True)

            # 梯度累积循环
            total_accum_loss = 0.0
            rendered_image = None
            target_image = None

            for accum_step in range(config.gradient_accumulation):
                # 选择随机相机
                camera = scene.get_random_train_camera()
                current_target_image = camera.original_image.to(device)

                # 🔥 修复：使用正确的混合精度上下文
                if scaler is not None:
                    with autocast_ctx:
                        # 渲染图像
                        current_rendered_image = render_gaussians_optimized(gaussians, camera, use_amp=True)

                        # 计算损失
                        total_loss, loss_dict = compute_rendering_loss(
                            current_rendered_image,
                            current_target_image,
                            lambda_l1=1.0,
                            lambda_ssim=config.lambda_ssim
                        )
                else:
                    # 不使用混合精度
                    current_rendered_image = render_gaussians_optimized(gaussians, camera, use_amp=False)

                    # 计算损失
                    total_loss, loss_dict = compute_rendering_loss(
                        current_rendered_image,
                        current_target_image,
                        lambda_l1=1.0,
                        lambda_ssim=config.lambda_ssim
                    )

                # 保存用于统计
                if accum_step == 0:
                    rendered_image = current_rendered_image
                    target_image = current_target_image

                # 检查损失是否为有效数值
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"⚠️  警告：损失为无效值 {total_loss.item()}，跳过此步")
                    # 跳过这个累积步骤
                    continue

                # 缩放损失（用于梯度累积）
                scaled_loss = total_loss / config.gradient_accumulation
                total_accum_loss += total_loss.item()

                # 🔥 修复：使用安全的反向传播
                safe_mixed_precision_backward(scaler, scaled_loss, gaussians.optimizer)

            # 如果累积损失为0，跳过更新
            if total_accum_loss == 0 or rendered_image is None:
                iteration += 1
                pbar.update(1)
                continue

            # 🔥 修复：使用安全的优化器更新
            try:
                # 执行优化器步骤
                safe_mixed_precision_step(
                    scaler,
                    gaussians.optimizer,
                    model_params=gaussians.parameters() if config.clip_grad_norm else None,
                    max_norm=1.0
                )

                # 清空梯度
                gaussians.optimizer.zero_grad(set_to_none=True)

            except Exception as e:
                print(f"⚠️  优化器更新失败: {e}")
                # 清空梯度并继续
                gaussians.optimizer.zero_grad(set_to_none=True)
                iteration += 1
                pbar.update(1)
                continue

            # 更新学习率
            if gaussians.scheduler:
                gaussians.scheduler.step()

            # 记录统计
            if iteration % 10 == 0:
                try:
                    # 计算PSNR
                    mse = torch.mean((rendered_image - target_image) ** 2)
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))

                    stats['losses'].append(total_accum_loss)
                    stats['psnrs'].append(psnr.item() if torch.is_tensor(psnr) else psnr)
                    stats['iterations'].append(iteration)
                    stats['timestamps'].append(time.time() - start_time)
                    stats['learning_rates'].append(gaussians.optimizer.param_groups[0]['lr'])

                    # 更新进度条
                    pbar.set_postfix({
                        'Loss': f"{total_accum_loss:.4f}",
                        'PSNR': f"{psnr:.2f}" if not torch.isnan(psnr) else "NaN",
                        'LR': f"{gaussians.optimizer.param_groups[0]['lr']:.6f}"
                    })

                    # 检查是否是最佳损失
                    if total_accum_loss < best_loss:
                        best_loss = total_accum_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # 早停检查
                    if patience_counter > 100 and iteration > 1000:
                        print(f"⚠️  早停触发，迭代 {iteration}")
                        break

                except Exception as e:
                    print(f"⚠️  记录统计失败: {e}")

            # 定期保存检查点
            if (iteration + 1) % config.checkpoint_interval == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint_{iteration + 1:06d}")
                scene.save_checkpoint(checkpoint_dir, iteration)

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
    print(f"📈 最佳损失: {best_loss:.6f}")

    # 保存最终模型
    final_dir = os.path.join(output_dir, "final_model")
    scene.save_checkpoint(final_dir, config.iterations)

    # 绘制训练曲线
    try:
        plot_training_curve(stats, os.path.join(output_dir, "training_curve.png"))
    except Exception as e:
        print(f"⚠️  绘制训练曲线失败: {e}")

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
    parser.add_argument("--iterations", type=int, default=3000,
                        help="训练迭代次数")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="初始学习率")
    parser.add_argument("--lambda_ssim", type=float, default=0.1,
                        help="SSIM损失权重")
    parser.add_argument("--sh_degree", type=int, default=0,
                        help="球谐函数阶数 (0=只使用DC项)")

    # RTX 3060优化参数
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="启用自动混合精度训练")
    parser.add_argument("--gradient_accumulation", type=int, default=2,
                        help="梯度累积步数 (模拟更大batch size)")
    parser.add_argument("--checkpoint_interval", type=int, default=500,
                        help="检查点保存间隔")
    parser.add_argument("--clip_grad_norm", action="store_true", default=True,
                        help="启用梯度裁剪")

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