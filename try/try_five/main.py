# #!/usr/bin/env python3
# """
# 3D Gaussian Splatting 三维重建主程序
# 支持Mip-NeRF360数据集格式
# """
#
# import os
# import argparse
# import torch
# import numpy as np
# from pathlib import Path
# import json
# from tqdm import tqdm
# import imageio
# import cv2
#
# from utils.colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary
# from utils.dataset_utils import load_nerf_poses, create_train_val_split
# from scene.dataset import SceneDataset
# from train import train_gaussian_splatting
#
# def parse_args():
#     parser = argparse.ArgumentParser(description="3D Gaussian Splatting 三维重建")
#
#     parser.add_argument("--source_path", type=str, required=True,
#                         help="数据集路径，如: /path/to/Mip_NeRF360/360_v2/bicycle")
#     parser.add_argument("--model_path", type=str, required=True,
#                         help="输出模型保存路径")
#     parser.add_argument("--images", type=str, default="images",
#                         choices=["images", "images_2", "images_4", "images_8"],
#                         help="使用哪个分辨率的图像")
#     parser.add_argument("--eval", action="store_true",
#                         help="是否使用验证集")
#     parser.add_argument("--test", action="store_true",
#                         help="是否使用测试集")
#     parser.add_argument("--resolution", type=int, default=-1,
#                         help="图像分辨率，-1表示使用原始分辨率")
#     parser.add_argument("--data_device", type=str, default="cuda",
#                         choices=["cuda", "cpu"],
#                         help="数据加载设备")
#     parser.add_argument("--white_background", action="store_true",
#                         help="是否使用白色背景")
#     parser.add_argument("--sh_degree", type=int, default=3,
#                         help="球谐函数的最大阶数")
#
#     # 训练参数
#     parser.add_argument("--iterations", type=int, default=30_000,
#                         help="训练迭代次数")
#     parser.add_argument("--position_lr_init", type=float, default=0.00016)
#     parser.add_argument("--position_lr_final", type=float, default=0.0000016)
#     parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
#     parser.add_argument("--position_lr_max_steps", type=int, default=30_000)
#     parser.add_argument("--feature_lr", type=float, default=0.0025)
#     parser.add_argument("--opacity_lr", type=float, default=0.05)
#     parser.add_argument("--scaling_lr", type=float, default=0.005)
#     parser.add_argument("--rotation_lr", type=float, default=0.001)
#     parser.add_argument("--percent_dense", type=float, default=0.01)
#     parser.add_argument("--lambda_dssim", type=float, default=0.2)
#     parser.add_argument("--densification_interval", type=int, default=100)
#     parser.add_argument("--opacity_reset_interval", type=int, default=3000)
#     parser.add_argument("--densify_from_iter", type=int, default=500)
#     parser.add_argument("--densify_until_iter", type=int, default=15_000)
#     parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)
#     parser.add_argument("--random_background", action="store_true",
#                         help="是否使用随机背景")
#
#     return parser.parse_args()
#
# def load_dataset(source_path, images_folder="images", resolution=-1, white_background=False):
#     """
#     加载Mip-NeRF360格式的数据集
#     """
#     source_path = Path(source_path)
#
#     # 检查数据集结构
#     images_path = source_path / images_folder
#     sparse_path = source_path / "sparse" / "0"
#     poses_bounds_path = source_path / "poses_bounds.npy"
#
#     print(f"数据集路径: {source_path}")
#     print(f"图像路径: {images_path}")
#     print(f"稀疏重建路径: {sparse_path}")
#     print(f"位姿边界文件: {poses_bounds_path}")
#
#     # 读取所有图像
#     image_files = sorted([f for f in images_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']])
#
#     if len(image_files) == 0:
#         raise ValueError(f"在 {images_path} 中没有找到图像文件")
#
#     print(f"找到 {len(image_files)} 张图像")
#
#     # 尝试读取COLMAP格式的相机参数
#     if sparse_path.exists():
#         print("读取COLMAP格式的相机参数...")
#         try:
#             cameras = read_cameras_binary(sparse_path / "cameras.bin")
#             images_data = read_images_binary(sparse_path / "images.bin")
#             points3d = read_points3d_binary(sparse_path / "points3D.bin")
#
#             # 将COLMAP数据转换为需要格式
#             cam_data = {}
#             for cam_id, cam in cameras.items():
#                 cam_data[cam_id] = {
#                     'model': cam.model,
#                     'width': cam.width,
#                     'height': cam.height,
#                     'params': cam.params
#                 }
#
#             return {
#                 'type': 'colmap',
#                 'cameras': cam_data,
#                 'images': images_data,
#                 'points3d': points3d,
#                 'image_files': image_files,
#                 'images_path': images_path
#             }
#         except Exception as e:
#             print(f"读取COLMAP数据失败: {e}")
#
#     # 尝试读取NeRF格式的位姿
#     if poses_bounds_path.exists():
#         print("读取NeRF格式的位姿...")
#         try:
#             poses_bounds = np.load(poses_bounds_path)
#             poses = poses_bounds[:, :-2].reshape([-1, 3, 5])  # 最后两列是边界
#             bounds = poses_bounds[:, -2:]
#
#             # 提取内参和外参
#             hwf = poses[:, :, 4]
#             poses = poses[:, :, :4]
#
#             # 处理位姿，将OpenGL坐标系转换为COLMAP坐标系
#             poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)
#
#             return {
#                 'type': 'nerf',
#                 'poses': poses,
#                 'hwf': hwf,
#                 'bounds': bounds,
#                 'image_files': image_files,
#                 'images_path': images_path
#             }
#         except Exception as e:
#             print(f"读取NeRF位姿失败: {e}")
#
#     # 如果都没有，使用简单的图像读取
#     print("使用默认相机参数...")
#     return {
#         'type': 'simple',
#         'image_files': image_files,
#         'images_path': images_path
#     }
#
# def main():
#     args = parse_args()
#
#     # 设置设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {device}")
#     if torch.cuda.is_available():
#         print(f"GPU: {torch.cuda.get_device_name(0)}")
#         print(f"CUDA可用内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
#
#     # 加载数据集
#     print("加载数据集...")
#     dataset_info = load_dataset(
#         args.source_path,
#         args.images,
#         args.resolution,
#         args.white_background
#     )
#
#     # 创建输出目录
#     model_path = Path(args.model_path)
#     model_path.mkdir(parents=True, exist_ok=True)
#
#     # 保存配置
#     config = vars(args)
#     config['device'] = str(device)
#     with open(model_path / "config.json", "w") as f:
#         json.dump(config, f, indent=2)
#
#     # 创建数据集
#     print("创建数据集对象...")
#     if dataset_info['type'] == 'colmap':
#         # 使用COLMAP数据创建数据集
#         dataset = SceneDataset(
#             source_path=args.source_path,
#             images=args.images,
#             resolution=args.resolution,
#             data_device=args.data_device,
#             white_background=args.white_background,
#             eval=args.eval
#         )
#     else:
#         # 简化版数据集
#         from scene.simple_dataset import SimpleSceneDataset
#         dataset = SimpleSceneDataset(
#             dataset_info=dataset_info,
#             resolution=args.resolution,
#             white_background=args.white_background,
#             device=device
#         )
#
#     print(f"数据集大小: {len(dataset)}")
#
#     # 训练3D高斯溅射模型
#     print("开始训练3D高斯溅射模型...")
#     train_gaussian_splatting(
#         dataset=dataset,
#         model_path=args.model_path,
#         iterations=args.iterations,
#         position_lr_init=args.position_lr_init,
#         position_lr_final=args.position_lr_final,
#         position_lr_delay_mult=args.position_lr_delay_mult,
#         position_lr_max_steps=args.position_lr_max_steps,
#         feature_lr=args.feature_lr,
#         opacity_lr=args.opacity_lr,
#         scaling_lr=args.scaling_lr,
#         rotation_lr=args.rotation_lr,
#         percent_dense=args.percent_dense,
#         lambda_dssim=args.lambda_dssim,
#         densification_interval=args.densification_interval,
#         opacity_reset_interval=args.opacity_reset_interval,
#         densify_from_iter=args.densify_from_iter,
#         densify_until_iter=args.densify_until_iter,
#         densify_grad_threshold=args.densify_grad_threshold,
#         sh_degree=args.sh_degree,
#         random_background=args.random_background
#     )
#
#     print("训练完成!")
#     print(f"模型保存在: {args.model_path}")
#
# if __name__ == "__main__":
#     main()


# !/usr/bin/env python3
"""
3D Gaussian Splatting 三维重建主程序
使用简化版本避免复杂错误
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from scene.dataset import SceneDataset
from simple_train import simple_train_gaussian_splatting


def parse_args():
    parser = argparse.ArgumentParser(description="简化的3D高斯溅射三维重建")

    parser.add_argument("--source_path", type=str, required=True,
                        help="数据集路径，如: /path/to/Mip_NeRF360/360_v2/bicycle")
    parser.add_argument("--model_path", type=str, required=True,
                        help="输出模型保存路径")
    parser.add_argument("--images", type=str, default="images_4",
                        choices=["images", "images_2", "images_4", "images_8"],
                        help="使用哪个分辨率的图像")
    parser.add_argument("--resolution", type=int, default=512,
                        help="图像分辨率，512适合RTX 3060")
    parser.add_argument("--iterations", type=int, default=5000,
                        help="训练迭代次数")
    parser.add_argument("--white_background", action="store_true",
                        help="是否使用白色背景")

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA可用内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 检查数据集
    source_path = Path(args.source_path)
    images_path = source_path / args.images

    if not images_path.exists():
        print(f"错误: 图像路径不存在: {images_path}")
        # 尝试其他分辨率
        for res in ["images_4", "images_2", "images", "images_8"]:
            test_path = source_path / res
            if test_path.exists():
                args.images = res
                images_path = test_path
                print(f"使用图像路径: {images_path}")
                break

    print(f"数据集路径: {source_path}")
    print(f"图像路径: {images_path}")

    # 创建数据集
    print("创建数据集对象...")
    try:
        dataset = SceneDataset(
            source_path=args.source_path,
            images=args.images,
            resolution=args.resolution,
            data_device="cuda",
            white_background=args.white_background,
            eval=False
        )

        print(f"数据集大小: {len(dataset)}")

        # 检查第一张图像
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本图像尺寸: {sample['image'].shape}")
            print(f"相机参数: 宽={sample['width']}, 高={sample['height']}")

    except Exception as e:
        print(f"创建数据集失败: {e}")
        return

    # 创建输出目录
    model_path = Path(args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config = vars(args)
    config['device'] = str(device)
    with open(model_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 训练简化的3D高斯溅射模型
    print(f"开始训练3D高斯溅射模型，迭代 {args.iterations} 次...")

    simple_train_gaussian_splatting(
        dataset=dataset,
        model_path=args.model_path,
        iterations=args.iterations,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=args.iterations,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
        percent_dense=0.01,
        densification_interval=100,
        opacity_reset_interval=1000,
        densify_from_iter=500,
        densify_until_iter=min(4000, args.iterations),
        densify_grad_threshold=0.0002,
        sh_degree=2,  # 降低球谐阶数以节省显存
        random_background=False
    )

    print("训练完成!")
    print(f"模型保存在: {args.model_path}")

    # 可视化结果
    try:
        visualize_results(args.model_path)
    except Exception as e:
        print(f"可视化失败: {e}")


def visualize_results(model_path):
    """可视化结果"""
    import open3d as o3d

    ply_file = Path(model_path) / "final_point_cloud.ply"

    if ply_file.exists():
        print(f"加载点云: {ply_file}")

        # 读取点云
        pcd = o3d.io.read_point_cloud(str(ply_file))

        if len(pcd.points) == 0:
            print("点云为空!")
            return

        print(f"点云包含 {len(pcd.points)} 个点")

        # 创建坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        # 可视化
        print("正在打开可视化窗口...")
        print("按 'q' 或 ESC 退出可视化")

        # 设置点云颜色
        if not pcd.has_colors():
            # 如果没有颜色，添加随机颜色
            pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Gaussian Splatting 重建结果")
        vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)

        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)

        # 运行可视化
        vis.run()
        vis.destroy_window()

        # 保存点云截图
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  # 不显示窗口
        vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)
        vis.poll_events()
        vis.update_renderer()

        # 保存图像
        image_path = Path(model_path) / "point_cloud_screenshot.png"
        vis.capture_screen_image(str(image_path))
        vis.destroy_window()

        print(f"点云截图保存到: {image_path}")

    else:
        print(f"未找到点云文件: {ply_file}")


if __name__ == "__main__":
    main()

