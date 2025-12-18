# inference.py
import os
import argparse
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from config import get_config
from dataset import get_dataset
from gaussian_model import GaussianModel
from render import GaussianRenderer


class GaussianInference:
    """3D高斯重建推理器"""

    def __init__(self, cfg, checkpoint_path=None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(cfg.output_dir, f"inference_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化组件
        print("Initializing Gaussian Model...")
        self.model = GaussianModel(sh_degree=cfg.sh_degree).to(self.device)

        print("Initializing Renderer...")
        self.renderer = GaussianRenderer(
            background_color=(1.0, 1.0, 1.0) if cfg.white_background else (0.0, 0.0, 0.0)
        )

        # 加载检查点
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        else:
            print("Warning: No checkpoint provided. Using untrained model.")

    def load_checkpoint(self, checkpoint_path):
        """加载训练好的模型检查点"""
        print(f"Loading checkpoint from {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 恢复配置（如果存在）
        if 'config' in checkpoint:
            print("Restoring configuration from checkpoint")

        # 加载模型参数
        if 'model_state_dict' in checkpoint:
            # 使用新的状态字典加载方法
            state_dict = checkpoint['model_state_dict']

            # 手动设置每个参数
            self.model._xyz = torch.nn.Parameter(state_dict['xyz'].to(self.device))
            self.model._opacity = torch.nn.Parameter(state_dict['opacity'].to(self.device))
            self.model._scaling = torch.nn.Parameter(state_dict['scaling'].to(self.device))
            self.model._rotation = torch.nn.Parameter(state_dict['rotation'].to(self.device))
            self.model._features_dc = torch.nn.Parameter(state_dict['features_dc'].to(self.device))
            self.model._features_rest = torch.nn.Parameter(state_dict['features_rest'].to(self.device))

            print(f"Loaded {self.model._xyz.shape[0]} Gaussians")
        elif 'model' in checkpoint:
            # 如果保存的是整个模型
            self.model = checkpoint['model'].to(self.device)
            print(f"Loaded complete model with {self.model._xyz.shape[0]} Gaussians")
        else:
            raise ValueError("Invalid checkpoint format")

        # 设置模型为评估模式
        self.model.eval()
        print("Model loaded successfully!")

    def render_from_pose(self, pose, K, H, W, save_path=None):
        """从指定相机位姿渲染图像"""
        with torch.no_grad():
            # 确保数据在正确设备上且不需要梯度
            pose = pose.to(self.device).detach()
            K = K.to(self.device).detach()

            # 渲染图像
            rendered_image = self.renderer.render(self.model, K, pose, H, W)

            # 转换为numpy数组并归一化到[0, 255]
            image_np = rendered_image.cpu().numpy()
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

            # 保存图像
            if save_path:
                imageio.imwrite(save_path, image_np)
                print(f"Image saved to {save_path}")

            return image_np

    def render_dataset_views(self, dataset, save_dir=None, num_views=None):
        """渲染数据集中的所有视角"""
        if save_dir is None:
            save_dir = os.path.join(self.output_dir, "dataset_views")
        os.makedirs(save_dir, exist_ok=True)

        # 获取所有相机参数
        poses, Ks, H, W = dataset.get_all_cameras()

        # 限制渲染数量（如果指定）
        if num_views is not None:
            indices = torch.linspace(0, len(poses) - 1, num_views).int()
        else:
            indices = range(len(poses))

        print(f"Rendering {len(indices)} views...")

        all_images = []
        for i, idx in enumerate(tqdm(indices, desc="Rendering views")):
            save_path = os.path.join(save_dir, f"view_{idx:04d}.png")

            # 渲染当前视角
            image = self.render_from_pose(poses[idx], Ks[idx], H, W, save_path)
            all_images.append(image)

        # 创建视频（如果有多张图像）
        if len(all_images) > 1:
            video_path = os.path.join(self.output_dir, "orbit.mp4")
            self.create_video(all_images, video_path, fps=30)
            print(f"Video saved to {video_path}")

        return all_images

    def render_circular_orbit(self, center, radius, num_frames=120,
                              elevation=30.0, H=512, W=512, focal=500.0):
        """渲染圆形轨迹（环绕物体）"""
        save_dir = os.path.join(self.output_dir, "circular_orbit")
        os.makedirs(save_dir, exist_ok=True)

        print(f"Rendering circular orbit with {num_frames} frames...")

        all_images = []
        for i in tqdm(range(num_frames), desc="Rendering orbit"):
            # 计算相机位置（球坐标）
            theta = 2 * np.pi * i / num_frames
            phi = np.radians(elevation)

            # 球坐标转笛卡尔坐标
            x = center[0] + radius * np.sin(phi) * np.cos(theta)
            y = center[1] + radius * np.sin(phi) * np.sin(theta)
            z = center[2] + radius * np.cos(phi)

            camera_pos = np.array([x, y, z])

            # 计算看向中心的相机矩阵
            pose = self.look_at(camera_pos, np.array(center), up=np.array([0, 0, 1]))

            # 创建内参矩阵
            K = torch.eye(3, device=self.device)
            K[0, 0] = focal
            K[1, 1] = focal
            K[0, 2] = W / 2.0
            K[1, 2] = H / 2.0

            # 渲染
            save_path = os.path.join(save_dir, f"frame_{i:04d}.png")
            image = self.render_from_pose(pose, K, H, W, save_path)
            all_images.append(image)

        # 创建视频
        video_path = os.path.join(self.output_dir, "circular_orbit.mp4")
        self.create_video(all_images, video_path, fps=30)
        print(f"Orbit video saved to {video_path}")

        return all_images

    def look_at(self, eye, target, up):
        """创建看向目标的相机位姿矩阵"""
        # 计算相机坐标系轴
        z_axis = eye - target
        z_axis = z_axis / np.linalg.norm(z_axis)

        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)

        # 构建旋转矩阵
        R = np.eye(4)
        R[:3, 0] = x_axis
        R[:3, 1] = y_axis
        R[:3, 2] = z_axis
        R[:3, 3] = eye

        # 返回世界到相机的变换矩阵
        w2c = np.linalg.inv(R)
        return torch.from_numpy(w2c).float().to(self.device)

    def create_video(self, images, output_path, fps=30):
        """从图像列表创建视频"""
        if not images:
            return

        # 确保所有图像尺寸相同
        height, width = images[0].shape[:2]

        # 创建视频写入器
        with imageio.get_writer(output_path, fps=fps) as writer:
            for image in tqdm(images, desc="Writing video"):
                writer.append_data(image)

    def render_depth_map(self, pose, K, H, W, save_path=None):
        """渲染深度图（实验功能）"""
        with torch.no_grad():
            # 获取高斯参数
            xyz = self.model.get_xyz
            scale = self.model.get_scaling
            rotation = self.model.get_rotation

            # 投影到相机坐标系
            R = pose[:3, :3]
            t = pose[:3, 3]
            xyz_cam = torch.matmul(R, xyz.T).T + t.unsqueeze(0)
            depth = xyz_cam[:, 2]  # Z轴深度

            # 投影到图像平面
            xyz_proj = torch.matmul(K, xyz_cam.T).T
            uv = xyz_proj[:, :2] / xyz_proj[:, 2:]

            # 创建深度图像
            depth_image = torch.ones(H, W, device=self.device) * 1000.0  # 初始化为大值

            # 简单地将每个高斯的深度投影到其影响区域
            # 注意：这是简化版本，完整实现需要考虑高斯覆盖范围
            for i in range(len(xyz)):
                u, v = uv[i]
                x, y = int(u.item()), int(v.item())

                if 0 <= x < W and 0 <= y < H:
                    # 取最近的高斯深度
                    if depth[i] < depth_image[y, x]:
                        depth_image[y, x] = depth[i]

            # 归一化深度图
            valid_depth = depth_image < 1000.0
            if valid_depth.any():
                min_depth = depth_image[valid_depth].min()
                max_depth = depth_image[valid_depth].max()
                depth_image[valid_depth] = (depth_image[valid_depth] - min_depth) / (max_depth - min_depth)

            # 转换为彩色热图
            depth_np = depth_image.cpu().numpy()
            depth_colored = plt.cm.viridis(depth_np)[:, :, :3]  # 使用viridis配色
            depth_colored = (depth_colored * 255).astype(np.uint8)

            # 保存深度图
            if save_path:
                imageio.imwrite(save_path, depth_colored)
                print(f"Depth map saved to {save_path}")

            return depth_colored

    def export_ply(self, save_path):
        """导出高斯模型为PLY文件"""
        ply_path = save_path if save_path.endswith('.ply') else save_path + '.ply'
        self.model.save_ply(ply_path)
        print(f"Gaussian model exported to {ply_path}")

    def compute_statistics(self):
        """计算并打印模型统计信息"""
        print("\n" + "=" * 50)
        print("MODEL STATISTICS")
        print("=" * 50)

        num_gaussians = self.model._xyz.shape[0]
        print(f"Number of Gaussians: {num_gaussians:,}")

        # 位置统计
        xyz = self.model._xyz.cpu().detach().numpy()
        print(f"Position range: X [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
        print(f"                Y [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
        print(f"                Z [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")

        # 不透明度统计
        opacity = torch.sigmoid(self.model._opacity).cpu().detach().numpy()
        print(f"Opacity range: [{opacity.min():.3f}, {opacity.max():.3f}]")
        print(f"Mean opacity: {opacity.mean():.3f}")

        # 缩放统计
        scale = self.model.get_scaling.cpu().detach().numpy()
        print(f"Scale range: [{scale.min():.3f}, {scale.max():.3f}]")
        print(f"Mean scale: {scale.mean():.3f}")

        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting Inference")

    # 必需参数
    parser.add_argument("--checkpoint", type=str, required=False, default='/home/next_lb/桌面/无人机影像三维重建任务/code/try/try_four/output/bicycle/checkpoint_003000.pth',
                        help="Path to trained model checkpoint (.pth file)")

    # 数据参数
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to dataset (for rendering training views)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene name (if using dataset)")

    # 渲染选项
    parser.add_argument("--render_views", action="store_true",
                        help="Render dataset views")
    parser.add_argument("--render_orbit", action="store_true",
                        help="Render circular orbit around object")
    parser.add_argument("--num_frames", type=int, default=120,
                        help="Number of frames for orbit")

    # 导出选项
    parser.add_argument("--export_ply", action="store_true",
                        help="Export model as PLY file")
    parser.add_argument("--compute_stats", action="store_true",
                        help="Compute and print model statistics")

    # 输出选项
    parser.add_argument("--output_dir", type=str, default="./inference_output",
                        help="Directory for inference outputs")

    args = parser.parse_args()

    # 创建配置（使用训练时的默认值）
    cfg = get_config()
    cfg.output_dir = args.output_dir

    # 如果提供了数据路径和场景，使用它们
    if args.data_path:
        cfg.data_path = args.data_path
    if args.scene:
        cfg.scene = args.scene

    # 初始化推理器
    inference = GaussianInference(cfg, checkpoint_path=args.checkpoint)

    # 计算统计信息
    if args.compute_stats:
        inference.compute_statistics()

    # 导出PLY文件
    if args.export_ply:
        ply_path = os.path.join(inference.output_dir, "gaussians")
        inference.export_ply(ply_path)

    # 渲染数据集视角
    if args.render_views and args.data_path and args.scene:
        print("\nLoading dataset for rendering...")
        dataset = get_dataset(cfg, split='train')
        inference.render_dataset_views(dataset, num_views=min(50, len(dataset)))

    # 渲染圆形轨迹
    if args.render_orbit:
        print("\nRendering circular orbit...")
        # 使用高斯位置的质心作为轨道中心
        center = inference.model._xyz.mean(dim=0).cpu().numpy()
        radius = inference.model._xyz.std().item() * 3.0  # 3倍标准差作为半径

        inference.render_circular_orbit(
            center=center,
            radius=radius,
            num_frames=args.num_frames,
            elevation=30.0,
            H=512,
            W=512,
            focal=500.0
        )

    print(f"\nAll outputs saved to: {inference.output_dir}")


if __name__ == "__main__":
    main()