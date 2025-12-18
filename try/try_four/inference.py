import os
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt

from config import get_config
from dataset import get_dataset
from gaussian_model import GaussianModel
from renderer import GaussianRenderer


class Reconstructor:
    def __init__(self, cfg, checkpoint_path=None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # 加载数据集（用于获取相机参数）
        self.dataset = get_dataset(cfg, split='train')

        # 初始化模型和渲染器
        self.model = GaussianModel(sh_degree=cfg.sh_degree).to(self.device)
        self.renderer = GaussianRenderer(
            background_color=(1.0, 1.0, 1.0) if cfg.white_background else (0.0, 0.0, 0.0)
        )

        # 加载检查点
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        else:
            # 尝试加载最终模型
            final_path = os.path.join(cfg.output_dir, cfg.scene, "final_model.pth")
            if os.path.exists(final_path):
                self.load_checkpoint(final_path)
            else:
                raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """加载训练好的模型检查点。"""
        print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型参数
        if 'model_state_dict' in checkpoint:
            # 检查点格式
            state_dict = checkpoint['model_state_dict']
            self.model._xyz = torch.nn.Parameter(state_dict['xyz'])
            self.model._opacity = torch.nn.Parameter(state_dict['opacity'])
            self.model._scaling = torch.nn.Parameter(state_dict['scaling'])
            self.model._rotation = torch.nn.Parameter(state_dict['rotation'])
            self.model._features_dc = torch.nn.Parameter(state_dict['features_dc'])
            self.model._features_rest = torch.nn.Parameter(state_dict['features_rest'])
        elif 'model' in checkpoint:
            # 完整模型格式
            self.model = checkpoint['model'].to(self.device)

        print(f"Loaded model with {self.model._xyz.shape[0]} Gaussians")

    def render_novel_view(self, pose=None, K=None, H=None, W=None):
        """从新视角渲染图像。"""
        if pose is None:
            # 使用第一个相机的位姿
            pose = self.dataset.poses[0]

        if K is None:
            K = self.dataset.Ks[0]

        if H is None or W is None:
            H, W = self.dataset.H, self.dataset.W

        # 渲染
        with torch.no_grad():
            image = self.renderer.render(self.model, K, pose, H, W)

        return image.cpu().numpy()

    def create_point_cloud(self, num_points=None):
        """从高斯模型创建点云。"""
        xyz = self.model.get_xyz.cpu().numpy()
        colors = self.model._features_dc.detach().cpu().numpy().squeeze()
        colors = 1 / (1 + np.exp(-colors))  # Sigmoid反激活

        if num_points is not None and num_points < xyz.shape[0]:
            # 随机下采样
            indices = np.random.choice(xyz.shape[0], num_points, replace=False)
            xyz = xyz[indices]
            colors = colors[indices]

        # 创建Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def create_mesh(self, pcd=None):
        """从点云创建网格（泊松重建）。"""
        if pcd is None:
            pcd = self.create_point_cloud(50000)  # 下采样

        print("Computing normals...")
        pcd.estimate_normals()

        print("Running Poisson reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

        # 过滤低密度区域
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        return mesh

    def render_trajectory(self, output_dir, num_frames=60):
        """渲染相机轨迹视频的帧。"""
        os.makedirs(output_dir, exist_ok=True)

        # 创建圆形相机轨迹
        radius = 3.0  # 轨迹半径
        center = torch.mean(self.model.get_xyz, dim=0).cpu().numpy()

        # 使用数据集中的第一个相机作为参考
        ref_pose = self.dataset.poses[0].cpu().numpy()
        K = self.dataset.Ks[0].cpu().numpy()
        H, W = self.dataset.H, self.dataset.W

        for i in range(num_frames):
            # 计算新相机位姿（绕Y轴旋转）
            angle = 2 * np.pi * i / num_frames

            # 创建旋转矩阵
            R = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])

            # 计算新位置
            t = center + np.array([radius * np.sin(angle), 0, radius * np.cos(angle)])

            # 构建位姿矩阵（世界到相机）
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = -R @ t  # 相机位置

            # 转换为PyTorch张量
            pose_tensor = torch.from_numpy(pose).float().to(self.device)
            K_tensor = torch.from_numpy(K).float().to(self.device)

            # 渲染
            image = self.render_novel_view(pose_tensor, K_tensor, H, W)

            # 保存帧
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            plt.imsave(frame_path, np.clip(image, 0, 1))

            if i % 10 == 0:
                print(f"Rendered frame {i}/{num_frames}")

        print(f"All frames saved to {output_dir}")
        print(
            f"To create video: ffmpeg -framerate 30 -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4")

    def export_reconstruction(self, output_dir):
        """导出完整的3D重建结果。"""
        os.makedirs(output_dir, exist_ok=True)

        print("1. Creating point cloud...")
        pcd = self.create_point_cloud(100000)
        pcd_path = os.path.join(output_dir, "reconstruction.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
        print(f"Point cloud saved to {pcd_path}")

        print("2. Creating mesh...")
        try:
            mesh = self.create_mesh(pcd)
            mesh_path = os.path.join(output_dir, "reconstruction_mesh.ply")
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f"Mesh saved to {mesh_path}")
        except Exception as e:
            print(f"Mesh reconstruction failed: {e}")

        print("3. Rendering novel views...")
        novel_dir = os.path.join(output_dir, "novel_views")
        self.render_trajectory(novel_dir, num_frames=30)

        print("4. Rendering training views...")
        train_dir = os.path.join(output_dir, "training_views")
        os.makedirs(train_dir, exist_ok=True)

        # 渲染所有训练视角
        for i in range(min(10, len(self.dataset))):  # 限制数量
            data = self.dataset[i]
            image = self.render_novel_view(data['pose'], data['K'], self.dataset.H, self.dataset.W)

            view_path = os.path.join(train_dir, f"view_{i:02d}.png")
            plt.imsave(view_path, np.clip(image, 0, 1))

        print(f"Training views saved to {train_dir}")

        print("\nReconstruction export completed!")
        print(f"All results saved to: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="3D Gaussian Splatting Inference")
    parser.add_argument("--config", action="store_true", help="Use config.py settings")
    parser.add_argument("--scene", type=str, default="bicycle", help="Scene name")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--output_dir", type=str, default="./reconstruction", help="Output directory")
    parser.add_argument("--export_all", action="store_true", help="Export all reconstruction results")

    args = parser.parse_args()

    if args.config:
        cfg = get_config()
        if args.scene != "bicycle":
            cfg.scene = args.scene
    else:
        # 创建基本配置
        class Config:
            def __init__(self):
                self.data_path = "./archive/360_v2"
                self.scene = args.scene
                self.images = "images"
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.white_background = True
                self.output_dir = "./output"

        cfg = Config()

    # 设置检查点路径
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.output_dir, cfg.scene, "final_model.pth")

    # 运行重建
    reconstructor = Reconstructor(cfg, checkpoint_path)

    if args.export_all:
        reconstructor.export_reconstruction(args.output_dir)
    else:
        # 交互式选项
        print("\n" + "=" * 50)
        print("3D Gaussian Splatting Reconstruction")
        print("=" * 50)
        print(f"Scene: {cfg.scene}")
        print(f"Number of Gaussians: {reconstructor.model._xyz.shape[0]}")
        print("\nOptions:")
        print("1. Render novel view from first camera")
        print("2. Create and visualize point cloud")
        print("3. Create mesh reconstruction")
        print("4. Render camera trajectory")
        print("5. Export all reconstruction results")

        choice = input("\nEnter your choice (1-5): ")

        if choice == "1":
            image = reconstructor.render_novel_view()
            plt.imshow(image)
            plt.title("Novel View")
            plt.axis('off')
            plt.show()

            # 保存图像
            save_path = os.path.join(args.output_dir, "novel_view.png")
            os.makedirs(args.output_dir, exist_ok=True)
            plt.imsave(save_path, np.clip(image, 0, 1))
            print(f"Image saved to {save_path}")

        elif choice == "2":
            pcd = reconstructor.create_point_cloud(50000)

            # 可视化
            o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud")

            # 保存
            pcd_path = os.path.join(args.output_dir, "point_cloud.ply")
            os.makedirs(args.output_dir, exist_ok=True)
            o3d.io.write_point_cloud(pcd_path, pcd)
            print(f"Point cloud saved to {pcd_path}")

        elif choice == "3":
            try:
                mesh = reconstructor.create_mesh()
                o3d.visualization.draw_geometries([mesh], window_name="3D Mesh")

                mesh_path = os.path.join(args.output_dir, "mesh.ply")
                os.makedirs(args.output_dir, exist_ok=True)
                o3d.io.write_triangle_mesh(mesh_path, mesh)
                print(f"Mesh saved to {mesh_path}")
            except Exception as e:
                print(f"Mesh reconstruction failed: {e}")

        elif choice == "4":
            traj_dir = os.path.join(args.output_dir, "trajectory")
            reconstructor.render_trajectory(traj_dir, num_frames=60)

        elif choice == "5":
            reconstructor.export_reconstruction(args.output_dir)

        else:
            print("Invalid choice. Using default export...")
            reconstructor.export_reconstruction(args.output_dir)


if __name__ == "__main__":
    main()