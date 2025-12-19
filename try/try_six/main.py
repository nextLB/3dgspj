#!/usr/bin/env python3
"""
3D Gaussian Splatting 三维重建 - 主程序
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from tqdm import tqdm

from dataset import MipNeRF360Dataset
from camera_utils import CameraUtils
from gaussian_model import Gaussian3D
from train import GaussianTrainer
from render import GaussianRenderer, SimpleRenderer
from utils import *


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting 三维重建")

    # 数据参数
    parser.add_argument("--data_root", type=str, required=True,
                        help="Mip_NeRF360数据根目录")
    parser.add_argument("--scene_name", type=str, required=True,
                        help="场景名称 (如: flowers, bicycle, room)")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="输出目录")
    parser.add_argument("--iterations", type=int, default=30000,
                        help="训练迭代次数")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批大小")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")

    # 数据参数
    parser.add_argument("--image_scale", type=int, default=1,
                        help="图像缩放比例 (1, 2, 4, 8)")
    parser.add_argument("--max_images", type=int, default=-1,
                        help="最大图像数量 (-1表示全部)")

    # 模型参数
    parser.add_argument("--initial_points", type=int, default=10000,
                        help="初始高斯点数量")
    parser.add_argument("--use_simple_renderer", action="store_true",
                        help="使用简化渲染器（更快但质量较低）")

    # 其他参数
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda 或 cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--visualize", action="store_true",
                        help="可视化相机和点云")
    parser.add_argument("--test_only", action="store_true",
                        help="仅测试，不训练")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    setup_seed(args.seed)

    # 设置设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / args.scene_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {output_dir}")

    # 保存参数
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 加载数据集
    print(f"\n加载场景: {args.scene_name}")

    try:
        dataset = MipNeRF360Dataset(
            data_root=args.data_root,
            scene_name=args.scene_name,
            split="train",
            scale=args.image_scale,
            load_images=True,
            max_images=args.max_images
        )

        # 归一化场景
        dataset.normalize_scene()

        print(f"成功加载 {len(dataset)} 张图像")

    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("请检查数据路径和场景名称是否正确")
        print(f"数据根目录: {args.data_root}")
        print(f"场景名称: {args.scene_name}")

        # 显示可用的场景
        print("\n可能可用的场景:")
        data_root_path = Path(args.data_root)

        # 检查360_extra_scenes
        extra_scenes_path = data_root_path / "360_extra_scenes"
        if extra_scenes_path.exists():
            print("360_extra_scenes 中的场景:")
            for scene in extra_scenes_path.iterdir():
                if scene.is_dir():
                    print(f"  - {scene.name}")

        # 检查360_v2
        v2_scenes_path = data_root_path / "360_v2"
        if v2_scenes_path.exists():
            print("360_v2 中的场景:")
            for scene in v2_scenes_path.iterdir():
                if scene.is_dir():
                    print(f"  - {scene.name}")

        return

    # 可视化（如果启用）
    if args.visualize:
        print("\n可视化相机和点云...")

        # 可视化相机
        cameras = dataset.get_camera_params()
        cam_viz_path = output_dir / "camera_positions.png"
        visualize_cameras(cameras, str(cam_viz_path))

        # 可视化点云
        point_cloud = dataset.get_point_cloud()
        if point_cloud is not None:
            pc_viz_path = output_dir / "point_cloud.png"
            visualize_point_cloud(
                point_cloud["points"],
                point_cloud["colors"],
                str(pc_viz_path)
            )

        # 显示一些示例图像
        print("\n显示示例图像...")
        for i in range(min(3, len(dataset))):
            data = dataset[i]
            if "image" in data:
                image = data["image"].permute(1, 2, 0).cpu().numpy()
                image_path = output_dir / f"example_{i}.png"
                save_image((image * 255).astype(np.uint8), str(image_path))
                print(f"  保存示例图像 {i}: {image_path}")

    # 如果不是仅测试，开始训练
    if not args.test_only:
        print("\n开始训练3D Gaussian Splatting...")

        # 训练配置
        config = {
            "data_root": args.data_root,
            "scene_name": args.scene_name,
            "output_dir": str(output_dir),
            "device": args.device,
            "image_scale": args.image_scale,
            "max_train_images": args.max_images,
            "max_val_images": 5,
            "num_iterations": args.iterations,
            "initial_points": args.initial_points,
            "use_ssim": False,
            "use_simple_renderer": args.use_simple_renderer,
            "num_workers": 0
        }

        # 创建训练器
        trainer = GaussianTrainer(config)

        # 开始训练
        trainer.train()

        print("\n训练完成!")
        print(f"最终模型保存在: {output_dir / 'final_gaussian.ply'}")

    # 测试渲染
    print("\n测试渲染...")

    # 加载训练好的模型（如果存在）
    final_ply = output_dir / "final_gaussian.ply"
    if final_ply.exists() or args.test_only:
        # 这里可以添加渲染测试代码
        print("渲染测试代码待实现...")

        # 示例：渲染一个视图
        if len(dataset) > 0:
            test_data = dataset[0]
            camera = test_data["camera"]
            gt_image = test_data["image"]

            # 创建模型（示例）
            model = Gaussian3D(1000, device)

            # 创建渲染器
            if args.use_simple_renderer:
                renderer = SimpleRenderer(device)
            else:
                renderer = GaussianRenderer(device)

            # 渲染
            with torch.no_grad():
                rendered = renderer.render(model, camera)

            # 保存结果
            rendered_np = rendered.permute(1, 2, 0).cpu().numpy()
            rendered_path = output_dir / "test_render.png"
            save_image((rendered_np * 255).astype(np.uint8), str(rendered_path))
            print(f"测试渲染保存到: {rendered_path}")

    print(f"\n所有输出保存在: {output_dir}")
    print("完成!")


if __name__ == "__main__":
    main()