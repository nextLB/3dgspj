import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json


def visualize_gaussians(gaussians, camera_positions=None, save_path=None, title="3D Gaussian Splatting"):
    """可视化3D高斯点云"""
    try:
        import open3d as o3d
        from mpl_toolkits.mplot3d import Axes3D

        # 获取点云数据
        points = gaussians.get_xyz.detach().cpu().numpy()
        colors = gaussians.get_features[:, :3].detach().cpu().numpy()

        # 创建open3d点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))

        # 添加相机位置
        geometries = [pcd]

        if camera_positions is not None:
            # 创建相机位置点云
            cam_pcd = o3d.geometry.PointCloud()
            cam_positions_np = []
            for cam_pos in camera_positions:
                if isinstance(cam_pos, torch.Tensor):
                    cam_pos = cam_pos.cpu().numpy()
                cam_positions_np.append(cam_pos)

            if cam_positions_np:
                cam_positions_np = np.array(cam_positions_np)
                cam_pcd.points = o3d.utility.Vector3dVector(cam_positions_np)
                cam_pcd.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(cam_positions_np), 1)))  # 红色
                geometries.append(cam_pcd)

        # 可视化
        if save_path:
            # 保存点云
            o3d.io.write_point_cloud(str(save_path), pcd)
            print(f"Point cloud saved to: {save_path}")

            # 创建2D可视化
            fig = plt.figure(figsize=(15, 5))

            # 3D视图
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=colors, s=1, alpha=0.5)
            if camera_positions_np is not None:
                ax1.scatter(camera_positions_np[:, 0], camera_positions_np[:, 1], camera_positions_np[:, 2],
                            c='red', s=20, marker='^', label='Cameras')
                ax1.legend()
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title(f'{title} - 3D View')

            # XY平面视图
            ax2 = fig.add_subplot(132)
            ax2.scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.5)
            if camera_positions_np is not None:
                ax2.scatter(camera_positions_np[:, 0], camera_positions_np[:, 1],
                            c='red', s=20, marker='^', label='Cameras')
                ax2.legend()
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('XY Plane')
            ax2.axis('equal')

            # XZ平面视图
            ax3 = fig.add_subplot(133)
            ax3.scatter(points[:, 0], points[:, 2], c=colors, s=1, alpha=0.5)
            if camera_positions_np is not None:
                ax3.scatter(camera_positions_np[:, 0], camera_positions_np[:, 2],
                            c='red', s=20, marker='^', label='Cameras')
                ax3.legend()
            ax3.set_xlabel('X')
            ax3.set_ylabel('Z')
            ax3.set_title('XZ Plane')
            ax3.axis('equal')

            plt.suptitle(title)
            plt.tight_layout()

            # 保存2D可视化
            vis_path = Path(save_path).with_suffix('.png')
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Visualization saved to: {vis_path}")

            # 在notebook中显示
            if 'ipykernel' in sys.modules:
                plt.show()

        else:
            # 交互式可视化
            o3d.visualization.draw_geometries(geometries, window_name=title)

        return pcd

    except ImportError as e:
        print(f"Open3D not installed: {e}")
        # 使用matplotlib后备方案
        return visualize_gaussians_matplotlib(gaussians, camera_positions, save_path, title)


def visualize_gaussians_matplotlib(gaussians, camera_positions=None, save_path=None, title="3D Gaussian Splatting"):
    """使用matplotlib可视化3D高斯点云（后备方案）"""
    # 获取点云数据
    points = gaussians.get_xyz.detach().cpu().numpy()
    colors = gaussians.get_features[:, :3].detach().cpu().numpy()

    # 创建3D图
    fig = plt.figure(figsize=(15, 5))

    # 3D视图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                c=colors, s=1, alpha=0.5)

    if camera_positions is not None:
        cam_positions_np = []
        for cam_pos in camera_positions:
            if isinstance(cam_pos, torch.Tensor):
                cam_pos = cam_pos.cpu().numpy()
            cam_positions_np.append(cam_pos)

        if cam_positions_np:
            cam_positions_np = np.array(cam_positions_np)
            ax1.scatter(cam_positions_np[:, 0], cam_positions_np[:, 1], cam_positions_np[:, 2],
                        c='red', s=20, marker='^', label='Cameras')
            ax1.legend()

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'{title} - 3D View')

    # XY平面视图
    ax2 = fig.add_subplot(132)
    ax2.scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.5)
    if camera_positions is not None and cam_positions_np is not None:
        ax2.scatter(cam_positions_np[:, 0], cam_positions_np[:, 1],
                    c='red', s=20, marker='^', label='Cameras')
        ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Plane')
    ax2.axis('equal')

    # XZ平面视图
    ax3 = fig.add_subplot(133)
    ax3.scatter(points[:, 0], points[:, 2], c=colors, s=1, alpha=0.5)
    if camera_positions is not None and cam_positions_np is not None:
        ax3.scatter(cam_positions_np[:, 0], cam_positions_np[:, 2],
                    c='red', s=20, marker='^', label='Cameras')
        ax3.legend()
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Plane')
    ax3.axis('equal')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()

    return None


def visualize_training_progress(log_file, save_path=None):
    """可视化训练进度"""
    from utils import plot_training_curve
    return plot_training_curve(log_file, save_path)


def compare_renderings(gt_images, rendered_images, save_dir=None, titles=None):
    """比较真实图像和渲染图像"""
    if titles is None:
        titles = ['Ground Truth', 'Rendered']

    num_images = min(len(gt_images), len(rendered_images), 5)  # 最多比较5张

    if num_images == 0:
        print("No images to compare")
        return

    # 计算PSNR和SSIM
    from utils import compute_psnr, compute_ssim

    psnr_values = []
    ssim_values = []

    for i in range(num_images):
        gt_img = gt_images[i]
        render_img = rendered_images[i]

        # 调整大小以匹配
        if gt_img.shape != render_img.shape:
            render_img = torch.nn.functional.interpolate(
                render_img.unsqueeze(0),
                size=gt_img.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        psnr = compute_psnr(render_img, gt_img)
        ssim = compute_ssim(render_img.unsqueeze(0), gt_img.unsqueeze(0))

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    # 创建对比图
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_images):
        # 真实图像
        gt_img = gt_images[i].detach().cpu().numpy()
        if gt_img.shape[0] == 3:  # CHW格式
            gt_img = gt_img.transpose(1, 2, 0)
        gt_img = np.clip(gt_img, 0, 1)

        axes[i, 0].imshow(gt_img)
        axes[i, 0].set_title(f'{titles[0]} {i + 1}')
        axes[i, 0].axis('off')

        # 渲染图像
        render_img = rendered_images[i].detach().cpu().numpy()
        if render_img.shape[0] == 3:  # CHW格式
            render_img = render_img.transpose(1, 2, 0)
        render_img = np.clip(render_img, 0, 1)

        axes[i, 1].imshow(render_img)
        axes[i, 1].set_title(f'{titles[1]} {i + 1}')
        axes[i, 1].axis('off')

        # 差异图
        diff = np.abs(gt_img - render_img).mean(axis=2)
        im = axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Difference (PSNR: {psnr_values[i]:.2f}, SSIM: {ssim_values[i]:.3f})')
        axes[i, 2].axis('off')

        # 添加颜色条
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_dir / 'comparison.png', dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {save_dir / 'comparison.png'}")

        # 保存指标
        metrics = {
            'psnr': psnr_values,
            'ssim': ssim_values,
            'mean_psnr': np.mean(psnr_values),
            'mean_ssim': np.mean(ssim_values)
        }

        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to: {save_dir / 'metrics.json'}")
        print(f"Mean PSNR: {metrics['mean_psnr']:.2f}, Mean SSIM: {metrics['mean_ssim']:.3f}")

    else:
        plt.show()

    plt.close()

    return psnr_values, ssim_values


def create_trajectory_video(gaussians, cameras, output_path, num_frames=60):
    """创建相机轨迹视频（简化版）"""
    try:
        import cv2

        # 获取相机位置
        cam_positions = []
        for cam in cameras[:10]:  # 只使用前10个相机
            pos = cam.camera_center.detach().cpu().numpy()
            cam_positions.append(pos)

        if not cam_positions:
            print("No camera positions available")
            return False

        cam_positions = np.array(cam_positions)

        # 创建简单的轨迹插值
        from scipy.interpolate import interp1d

        # 参数化轨迹
        t = np.linspace(0, 1, len(cam_positions))
        t_new = np.linspace(0, 1, num_frames)

        # 插值每个坐标
        trajectory = []
        for i in range(3):
            f = interp1d(t, cam_positions[:, i], kind='cubic')
            trajectory.append(f(t_new))

        trajectory = np.column_stack(trajectory)

        # 渲染每一帧（这里简化，实际需要完整的渲染）
        frames = []
        for i, pos in enumerate(trajectory):
            # 创建简单的帧
            frame = np.zeros((512, 512, 3), dtype=np.uint8)

            # 添加文本
            cv2.putText(frame, f'Trajectory Frame {i + 1}/{num_frames}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            frames.append(frame)

        # 保存为视频
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(output_path), fourcc, 10, (width, height))

        for frame in frames:
            video.write(frame)

        video.release()
        print(f"Trajectory video saved: {output_path}")
        return True

    except ImportError:
        print("OpenCV or SciPy not installed, skipping trajectory video")
        return False
    except Exception as e:
        print(f"Error creating trajectory video: {e}")
        return False


def visualize_camera_poses(cameras, save_path=None):
    """可视化相机位姿"""
    try:
        from mpl_toolkits.mplot3d import Axes3D

        # 提取相机位置和方向
        positions = []
        directions = []

        for cam in cameras[:50]:  # 限制数量
            pos = cam.camera_center.detach().cpu().numpy()
            # 从视图矩阵提取方向
            R = cam.world_view_transform[:3, :3].detach().cpu().numpy()
            forward = -R[2, :3]  # 相机的前向方向

            positions.append(pos)
            directions.append(forward)

        positions = np.array(positions)
        directions = np.array(directions)

        # 创建可视化
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制相机位置
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c='red', s=50, label='Camera Positions')

        # 绘制相机方向
        scale = 0.5  # 方向向量的缩放
        for i in range(len(positions)):
            ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                      directions[i, 0] * scale, directions[i, 1] * scale, directions[i, 2] * scale,
                      color='blue', alpha=0.5, arrow_length_ratio=0.1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Poses Visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 设置相等的比例
        max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                              positions[:, 1].max() - positions[:, 1].min(),
                              positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0

        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Camera poses visualization saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    except Exception as e:
        print(f"Error visualizing camera poses: {e}")


def render_views_interactive(gaussians, cameras, output_dir, num_views=5):
    """渲染多个视角并创建交互式可视化"""
    try:
        from pytorch_renderer import render_pytorch

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # 选择一些相机视角
        if len(cameras) > num_views:
            indices = np.linspace(0, len(cameras) - 1, num_views, dtype=int)
            selected_cameras = [cameras[i] for i in indices]
        else:
            selected_cameras = cameras

        # 渲染每个视角
        rendered_images = []
        for i, cam in enumerate(selected_cameras):
            print(f"Rendering view {i + 1}/{len(selected_cameras)}")

            render_pkg = render_pytorch(
                viewpoint_camera=cam,
                gaussians=gaussians,
                pipe={'debug': False},
                bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device=gaussians.device)
            )

            rendered_image = render_pkg["render"]
            rendered_images.append(rendered_image)

            # 保存渲染结果
            save_path = output_dir / f"view_{i:03d}.png"
            from utils import save_render_image
            save_render_image(rendered_image, save_path)

        # 创建对比图
        gt_images = [cam.original_image for cam in selected_cameras]
        compare_renderings(gt_images, rendered_images, output_dir / "comparison")

        print(f"All views rendered and saved to: {output_dir}")

        return rendered_images

    except Exception as e:
        print(f"Error rendering views: {e}")
        return []


if __name__ == "__main__":
    print("Visualization module loaded")