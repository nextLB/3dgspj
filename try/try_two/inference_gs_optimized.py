#!/usr/bin/env python3
"""
3D Gaussian Splatting 优化推理脚本 - RTX 3060专用版
支持点云可视化、网格重建、360度视频渲染
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from gaussian_model_opt import OptimizedGaussianModel
from camera_opt import OptimizedCamera
from render_opt import render_gaussians_optimized
from utils_opt import *

# 导入可视化库
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("⚠️  Open3D未安装，部分可视化功能不可用")
    OPEN3D_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("⚠️  OpenCV未安装，视频创建功能不可用")
    CV2_AVAILABLE = False


# ==================== 模型加载函数 ====================
def load_trained_model_optimized(model_path, device='cuda'):
    """加载训练好的模型"""
    print(f"📂 加载模型: {model_path}")

    # 如果提供的是检查点目录
    if os.path.isdir(model_path):
        # 查找最新的检查点
        checkpoints = []
        for item in os.listdir(model_path):
            item_path = os.path.join(model_path, item)
            if os.path.isdir(item_path):
                if item.startswith('checkpoint_') or item == 'final_model':
                    checkpoints.append(item_path)

        if not checkpoints:
            # 尝试直接加载
            ply_path = os.path.join(model_path, "point_cloud.ply")
            model_state_path = os.path.join(model_path, "model_state.pth")
            if os.path.exists(ply_path) or os.path.exists(model_state_path):
                checkpoints = [model_path]
            else:
                raise FileNotFoundError(f"在 {model_path} 中未找到检查点")

        # 按修改时间排序，取最新的
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        latest_checkpoint = checkpoints[0]
    else:
        latest_checkpoint = model_path

    print(f"🔍 使用检查点: {os.path.basename(latest_checkpoint)}")

    # 加载配置
    config_path = os.path.join(latest_checkpoint, "..", "..", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        sh_degree = config.get('sh_degree', 0)
    else:
        # 尝试从父目录查找
        parent_config = os.path.join(latest_checkpoint, "..", "config.yaml")
        if os.path.exists(parent_config):
            with open(parent_config, 'r') as f:
                config = yaml.safe_load(f)
            sh_degree = config.get('sh_degree', 0)
        else:
            sh_degree = 0

    # 初始化模型
    gaussians = OptimizedGaussianModel(sh_degree=sh_degree, device=device)

    # 尝试加载点云
    ply_path = os.path.join(latest_checkpoint, "point_cloud.ply")
    if os.path.exists(ply_path) and OPEN3D_AVAILABLE:
        print("💾 从PLY文件加载点云...")
        success = gaussians.load_ply(ply_path)
        if success:
            print(f"✅ 加载 {gaussians.get_xyz.shape[0]} 个高斯点")
        else:
            # 尝试加载模型状态
            model_state_path = os.path.join(latest_checkpoint, "model_state.pth")
            if os.path.exists(model_state_path):
                print("💾 从状态文件加载模型...")
                model_state = torch.load(model_state_path, map_location=device)
                gaussians.load_state_dict(model_state)
    else:
        # 加载模型状态
        model_state_path = os.path.join(latest_checkpoint, "model_state.pth")
        if os.path.exists(model_state_path):
            print("💾 从状态文件加载模型...")
            model_state = torch.load(model_state_path, map_location=device)
            gaussians.load_state_dict(model_state)
        else:
            # 尝试查找其他可能的文件
            import glob
            pth_files = glob.glob(os.path.join(latest_checkpoint, "*.pth"))
            if pth_files:
                model_state_path = pth_files[0]
                print(f"💾 从状态文件加载模型: {model_state_path}")
                model_state = torch.load(model_state_path, map_location=device)
                gaussians.load_state_dict(model_state)
            else:
                raise FileNotFoundError(f"未找到模型文件: {ply_path} 或 {model_state_path}")

    print_gpu_memory()
    return gaussians, latest_checkpoint


# ==================== 虚拟相机轨迹生成 ====================
def create_spherical_trajectory(gaussians, num_frames=60, radius_scale=1.5):
    """创建球形相机轨迹"""
    print("🎥 创建相机轨迹...")

    # 计算场景边界
    points = gaussians.get_xyz.cpu().numpy()

    if len(points) == 0:
        center = np.array([0, 0, 0])
        radius = 3.0
    else:
        # 使用更稳定的方法计算场景中心
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        center = (min_bound + max_bound) / 2
        max_extent = np.max(max_bound - min_bound)
        radius = max_extent * radius_scale * 0.5

    print(f"📊 场景中心: {center}, 半径: {radius:.2f}")

    # 创建相机参数
    H, W = 512, 512  # 输出分辨率

    # 内参矩阵
    K = torch.eye(3)
    K[0, 0] = K[1, 1] = 500.0  # 焦距
    K[0, 2] = W / 2.0  # 主点x
    K[1, 2] = H / 2.0  # 主点y

    cameras = []

    for i in tqdm(range(num_frames), desc="生成相机"):
        # 球形坐标
        theta = 2 * np.pi * i / num_frames  # 水平角
        phi = np.pi / 4 + np.pi / 8 * np.sin(2 * np.pi * i / num_frames)  # 俯仰角（动态）

        # 计算相机位置
        x = center[0] + radius * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * np.cos(phi)
        z = center[2] + radius * np.sin(phi) * np.sin(theta)

        camera_pos = np.array([x, y, z])

        # 相机看向中心
        look_at = center.copy()

        # 上方向（动态调整避免翻转）
        up = np.array([0, -1, 0]) if y > center[1] else np.array([0, 1, 0])

        # 计算相机坐标系
        forward = look_at - camera_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)

        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)

        # 构建世界到相机变换
        world2cam = np.eye(4)
        world2cam[:3, 0] = right
        world2cam[:3, 1] = up
        world2cam[:3, 2] = -forward
        world2cam[:3, 3] = -world2cam[:3, :3] @ camera_pos

        # 创建虚拟图像
        dummy_image = torch.zeros((3, H, W), device=gaussians.get_xyz.device)

        # 创建相机对象
        camera = OptimizedCamera(
            world2cam=torch.from_numpy(world2cam).float().to(gaussians.get_xyz.device),
            K=K.to(gaussians.get_xyz.device),
            image=dummy_image,
            H=H, W=W,
            image_name=f"virtual_{i:04d}",
            uid=i
        )

        cameras.append(camera)

    print(f"✅ 生成 {len(cameras)} 个相机位姿")
    return cameras


# ==================== 点云导出 ====================
def export_point_cloud(gaussians, output_path, format='ply'):
    """导出点云"""
    if not OPEN3D_AVAILABLE:
        print("❌ Open3D未安装，无法导出点云")
        return False

    print(f"💾 导出点云到 {format.upper()} 格式...")

    # 获取点云数据
    points = gaussians.get_xyz.detach().cpu().numpy()
    colors = gaussians.get_features.detach().cpu().numpy().squeeze(1)

    # 确保颜色在[0, 1]范围内
    colors = np.clip(colors, 0, 1)

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 保存
    if format.lower() == 'ply':
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
    elif format.lower() == 'obj':
        # 导出为OBJ格式
        with open(output_path, 'w') as f:
            for i in range(len(points)):
                f.write(f"v {points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                        f"{colors[i, 0]:.3f} {colors[i, 1]:.3f} {colors[i, 2]:.3f}\n")
    else:
        print(f"❌ 不支持的格式: {format}")
        return False

    print(f"✅ 点云已保存: {output_path} ({len(points)} 个点)")
    return True


# ==================== 网格重建 ====================
def reconstruct_mesh_from_pointcloud(pointcloud_path, output_mesh_path):
    """从点云重建网格"""
    if not OPEN3D_AVAILABLE:
        print("❌ Open3D未安装，无法重建网格")
        return None

    print("🔄 从点云重建网格...")

    # 加载点云
    pcd = o3d.io.read_point_cloud(pointcloud_path)

    if len(pcd.points) < 100:
        print("❌ 点云点数太少，无法重建网格")
        return None

    # 降采样（如果点数太多）
    if len(pcd.points) > 50000:
        print(f"📉 点云点数太多 ({len(pcd.points)})，进行降采样...")
        pcd = pcd.voxel_down_sample(voxel_size=0.01)

    # 估计法线
    print("📐 估计法线...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    # 泊松重建
    print("🏗️  泊松重建...")
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)

        # 移除低密度顶点
        if len(densities) > 0:
            density_threshold = np.quantile(densities, 0.01)
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

        # 简化网格
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)

        # 保存网格
        o3d.io.write_triangle_mesh(output_mesh_path, mesh)
        print(f"✅ 网格已保存: {output_mesh_path}")

        return mesh

    except Exception as e:
        print(f"❌ 网格重建失败: {e}")
        return None


# ==================== 360度视频渲染 ====================
def render_360_video(gaussians, cameras, output_dir, fps=30):
    """渲染360度视频"""
    print("🎬 渲染360度视频...")

    # 创建输出目录
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frames = []

    # 渲染每一帧
    for i, camera in enumerate(tqdm(cameras, desc="渲染帧")):
        with torch.no_grad():
            try:
                # 渲染图像
                rendered_image = render_gaussians_optimized(gaussians, camera, use_amp=True)

                # 转换为numpy
                img_np = rendered_image.detach().cpu().permute(1, 2, 0).numpy()
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

                # 保存帧
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")

                # 使用PIL保存
                from PIL import Image
                Image.fromarray(img_np).save(frame_path)

                frames.append(img_np)
            except Exception as e:
                print(f"⚠️  渲染帧 {i} 失败: {e}")
                # 创建黑色帧作为占位符
                black_frame = np.zeros((camera.H, camera.W, 3), dtype=np.uint8)
                frames.append(black_frame)

    print(f"✅ 渲染完成: {len(frames)} 帧")

    # 创建视频
    if CV2_AVAILABLE and len(frames) > 0:
        video_path = os.path.join(output_dir, "360_video.mp4")

        # 获取帧尺寸
        height, width = frames[0].shape[:2]

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # 写入帧
        for frame in frames:
            # 转换为BGR格式
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        print(f"🎥 视频已保存: {video_path}")

    return frames


# ==================== 可视化函数 ====================
def visualize_reconstruction_3d(gaussians, cameras=None):
    """3D可视化重建结果"""
    if not OPEN3D_AVAILABLE:
        print("❌ Open3D未安装，无法3D可视化")
        return

    print("👁️  3D可视化...")

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D高斯泼溅重建", width=1024, height=768)

    # 添加点云
    points = gaussians.get_xyz.detach().cpu().numpy()
    colors = gaussians.get_features.detach().cpu().numpy().squeeze(1)
    colors = np.clip(colors, 0, 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis.add_geometry(pcd)

    # 添加相机位姿（如果提供）
    if cameras:
        # 创建相机坐标系可视化
        for i, camera in enumerate(cameras[:5]):  # 只显示前5个
            try:
                # 获取相机位置
                cam_pos = camera.get_position().cpu().numpy()

                # 创建坐标系
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
                coord_frame.translate(cam_pos)

                # 设置颜色
                if i == 0:
                    coord_frame.paint_uniform_color([1, 0, 0])  # 红色
                elif i == len(cameras[:5]) - 1:
                    coord_frame.paint_uniform_color([0, 0, 1])  # 蓝色
                else:
                    coord_frame.paint_uniform_color([0, 1, 0])  # 绿色

                vis.add_geometry(coord_frame)
            except:
                continue

    # 设置视角
    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(0.8)

    # 运行可视化
    print("🖱️  使用鼠标交互:")
    print("  - 左键拖动: 旋转")
    print("  - 右键拖动: 平移")
    print("  - 滚轮: 缩放")
    print("  - 按 'Q' 或 'ESC' 退出")

    vis.run()
    vis.destroy_window()


def visualize_2d_comparison(gaussians, cameras, output_path=None):
    """2D可视化比较"""
    print("📊 2D可视化...")

    if not cameras:
        print("❌ 没有相机数据")
        return

    # 渲染几个视角
    num_views = min(4, len(cameras))

    fig, axes = plt.subplots(2, num_views, figsize=(4 * num_views, 8))

    if num_views == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_views):
        camera = cameras[i]

        with torch.no_grad():
            try:
                # 渲染重建视图
                rendered = render_gaussians_optimized(gaussians, camera)
                rendered_np = rendered.detach().cpu().permute(1, 2, 0).numpy()

                # 显示重建结果
                axes[0, i].imshow(np.clip(rendered_np, 0, 1))
                axes[0, i].set_title(f"视图 {i + 1} - 重建")
                axes[0, i].axis('off')

                # 显示相机位置
                cam_pos = camera.get_position().cpu().numpy()
                axes[1, i].text(0.5, 0.5, f"相机位置:\n[{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]",
                                horizontalalignment='center', verticalalignment='center',
                                transform=axes[1, i].transAxes, fontsize=10)
                axes[1, i].set_title(f"相机 {i + 1}")
                axes[1, i].axis('off')
            except Exception as e:
                print(f"⚠️  渲染视图 {i} 失败: {e}")
                axes[0, i].axis('off')
                axes[1, i].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 可视化图已保存: {output_path}")

    plt.show()


# ==================== 主推理函数 ====================
def inference_optimized(config):
    """优化版推理主函数"""

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 使用设备: {device}")

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU: {gpu_name}")

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 加载模型
    try:
        gaussians, checkpoint_path = load_trained_model_optimized(config.model_path, device)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        # 尝试使用默认路径
        default_path = "./output"
        if os.path.exists(default_path):
            print(f"🔍 尝试使用默认路径: {default_path}")
            try:
                # 查找最新的训练结果
                dirs = [d for d in os.listdir(default_path) if os.path.isdir(os.path.join(default_path, d))]
                if dirs:
                    latest_dir = sorted(dirs)[-1]
                    model_path = os.path.join(default_path, latest_dir)
                    print(f"🔍 使用最新训练结果: {model_path}")
                    gaussians, checkpoint_path = load_trained_model_optimized(model_path, device)
                else:
                    raise FileNotFoundError("未找到训练结果")
            except Exception as e2:
                print(f"❌ 加载默认模型失败: {e2}")
                return None, None
        else:
            return None, None

    # 设置模型为评估模式
    gaussians.eval()

    print("=" * 50)

    # 导出点云
    if config.export_pointcloud:
        print("📦 导出点云...")
        ply_path = os.path.join(config.output_dir, "reconstruction.ply")
        success = export_point_cloud(gaussians, ply_path, format='ply')

        if success:
            print(f"✅ 点云导出成功")
        else:
            print(f"❌ 点云导出失败")

    # 重建网格
    if config.reconstruct_mesh and OPEN3D_AVAILABLE:
        print("🔄 重建网格...")
        ply_path = os.path.join(config.output_dir, "reconstruction.ply")
        mesh_path = os.path.join(config.output_dir, "reconstructed_mesh.ply")

        if os.path.exists(ply_path):
            mesh = reconstruct_mesh_from_pointcloud(ply_path, mesh_path)
            if mesh is not None:
                print(f"✅ 网格重建成功")
        else:
            print(f"⚠️  点云文件不存在，跳过网格重建")

    # 生成相机轨迹
    cameras = None
    if config.render_video or config.visualize_3d or config.visualize_2d:
        print("🎥 生成相机轨迹...")
        try:
            cameras = create_spherical_trajectory(
                gaussians,
                num_frames=min(config.num_frames, 30),  # 减少帧数以加速
                radius_scale=config.radius_scale
            )
        except Exception as e:
            print(f"⚠️  生成相机轨迹失败: {e}")
            cameras = []

    # 渲染360度视频
    if config.render_video and cameras:
        print("🎬 渲染视频...")
        try:
            frames = render_360_video(
                gaussians,
                cameras[:20],  # 只渲染前20帧
                config.output_dir,
                fps=config.fps
            )

            # 显示第一帧
            if frames and config.show_preview and len(frames) > 0:
                plt.figure(figsize=(10, 10))
                plt.imshow(frames[0])
                plt.title("360度视频 - 第一帧")
                plt.axis('off')
                plt.show()
        except Exception as e:
            print(f"❌ 渲染视频失败: {e}")

    # 3D可视化
    if config.visualize_3d and OPEN3D_AVAILABLE:
        try:
            visualize_reconstruction_3d(gaussians, cameras)
        except Exception as e:
            print(f"❌ 3D可视化失败: {e}")

    # 2D可视化
    if config.visualize_2d and cameras:
        try:
            viz_path = os.path.join(config.output_dir, "2d_visualization.png")
            visualize_2d_comparison(gaussians, cameras[:4], viz_path)  # 只可视化前4个视图
        except Exception as e:
            print(f"❌ 2D可视化失败: {e}")

    # 生成报告
    print("=" * 50)
    print("📊 推理报告:")
    print(f"   模型路径: {config.model_path}")
    print(f"   输出目录: {config.output_dir}")
    if hasattr(gaussians, 'get_xyz') and gaussians.get_xyz is not None:
        print(f"   高斯点数量: {gaussians.get_xyz.shape[0]}")

    if cameras:
        print(f"   相机轨迹: {len(cameras)} 个位姿")

    print("✅ 推理完成!")

    return gaussians, cameras


# ==================== 配置解析 ====================
def parse_inference_config():
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting 优化推理 (RTX 3060)")

    # 模型参数
    parser.add_argument("--model_path", type=str, required=False, default="./output",
                        help="训练好的模型路径")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./inference_output",
                        help="输出目录")

    # 点云和网格选项
    parser.add_argument("--export_pointcloud", action="store_true", default=True,
                        help="导出点云")
    parser.add_argument("--reconstruct_mesh", action="store_true", default=False,  # 默认关闭，因为可能失败
                        help="从点云重建网格")

    # 渲染选项
    parser.add_argument("--render_video", action="store_true", default=True,
                        help="渲染360度视频")
    parser.add_argument("--num_frames", type=int, default=30,  # 减少默认帧数
                        help="视频帧数")
    parser.add_argument("--radius_scale", type=float, default=2.0,  # 增加半径以避免视图问题
                        help="相机轨迹半径缩放因子")
    parser.add_argument("--fps", type=int, default=15,  # 降低帧率
                        help="视频帧率")
    parser.add_argument("--show_preview", action="store_true", default=True,
                        help="显示视频预览")

    # 可视化选项
    parser.add_argument("--visualize_3d", action="store_true", default=False,  # 默认关闭，需要交互
                        help="3D可视化")
    parser.add_argument("--visualize_2d", action="store_true", default=True,
                        help="2D可视化")

    # 从配置文件加载（如果存在）
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # 更新默认值
        args_dict = vars(parser.parse_args([]))
        if config_dict and 'inference' in config_dict:
            args_dict.update(config_dict['inference'])

        # 创建命名空间
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 解析配置
    config = parse_inference_config()

    # 打印欢迎信息
    print("=" * 60)
    print("3D Gaussian Splatting - RTX 3060 优化推理")
    print("=" * 60)

    try:
        # 执行推理
        gaussians, cameras = inference_optimized(config)

        if gaussians is not None:
            print("=" * 60)
            print("🎉 推理流程完成!")
            print(f"📁 所有输出文件保存在: {config.output_dir}")

            # 列出生成的文件
            print("\n📋 生成的文件:")
            for root, dirs, files in os.walk(config.output_dir):
                level = root.replace(config.output_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for file in files[:10]:  # 最多显示10个文件
                    print(f'{subindent}{file}')
                if len(files) > 10:
                    print(f'{subindent}... 还有 {len(files) - 10} 个文件')
        else:
            print("❌ 推理失败，未加载到有效模型")

    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)