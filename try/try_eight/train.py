import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib.pyplot as plt
import config
from data_loader import ColmapDataset
from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer, compute_loss


class Trainer:
    def __init__(self, dataset, gaussian_model, renderer):
        self.dataset = dataset
        self.model = gaussian_model
        self.renderer = renderer
        self.device = config.config_dict['device']

        # 训练参数
        self.num_iterations = config.config_dict['num_iterations']
        self.save_interval = config.config_dict['save_interval']

        # 创建输出目录
        self.output_dir = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)

        # 统计数据
        self.losses = []
        self.psnrs = []

    def compute_psnr(self, rendered, target):
        """计算PSNR"""
        mse = torch.mean((rendered - target) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()

    def train(self):
        """训练循环"""
        print(f"开始训练，共{self.num_iterations}次迭代")
        print(f"设备: {self.device}")
        print(f"输出目录: {self.output_dir}")

        # 设置优化器
        self.model.setup_optimizers()

        # 训练循环
        progress_bar = tqdm(range(self.num_iterations), desc="训练进度")

        for iteration in progress_bar:
            # 随机选择一张图像
            idx = np.random.randint(len(self.dataset))
            data = self.dataset[idx]

            # 获取数据
            target_image = data['image']
            camera_pose = data['pose']
            intrinsics = data['intrinsics']

            # 图像尺寸
            H, W, _ = target_image.shape

            # 渲染
            rendered, uv, depths = self.renderer.render(
                self.model, camera_pose, intrinsics, (H, W)
            )

            # 计算损失
            loss = compute_loss(rendered, target_image, config.config_dict['lambda_dssim'])

            # 反向传播
            loss.backward()

            # 更新参数
            self.model.xyz_optimizer.step()
            self.model.feature_optimizer.step()
            self.model.opacity_optimizer.step()
            self.model.scaling_optimizer.step()
            self.model.rotation_optimizer.step()

            # 清零梯度
            self.model.xyz_optimizer.zero_grad()
            self.model.feature_optimizer.zero_grad()
            self.model.opacity_optimizer.zero_grad()
            self.model.scaling_optimizer.zero_grad()
            self.model.rotation_optimizer.zero_grad()

            # 更新学习率
            self.model.update_learning_rate(iteration)

            # 记录损失
            self.losses.append(loss.item())

            # 计算PSNR
            psnr = self.compute_psnr(rendered, target_image)
            self.psnrs.append(psnr)

            # 更新进度条
            progress_bar.set_description(
                f"Iter: {iteration}, Loss: {loss.item():.4f}, PSNR: {psnr:.2f}"
            )

            # 定期保存
            if (iteration % self.save_interval == 0 or
                    iteration == self.num_iterations - 1 or
                    iteration < 10):
                self.save_checkpoint(iteration)

                # 保存渲染结果（前几次迭代也保存，方便查看进展）
                if iteration % (max(1, self.save_interval // 5)) == 0 or iteration < 5:
                    self.save_rendered_image(rendered, target_image, iteration)

        print("训练完成!")
        self.save_final_results()

    def save_checkpoint(self, iteration):
        """保存检查点"""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_{iteration:06d}.pth")

        checkpoint = {
            'iteration': iteration,
            'model_state_dict': {
                'xyz': self.model._xyz.data,
                'features_dc': self.model._features_dc.data,
                'features_rest': self.model._features_rest.data,
                'scaling': self.model._scaling.data,
                'rotation': self.model._rotation.data,
                'opacity': self.model._opacity.data,
            },
            'losses': self.losses,
            'psnrs': self.psnrs,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")

    def save_rendered_image(self, rendered, target, iteration):
        """保存渲染图像"""
        # 确保目录存在
        os.makedirs(os.path.join(self.output_dir, "renders"), exist_ok=True)

        # 转换为numpy并确保值在[0, 1]范围内
        rendered_np = rendered.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # 限制值范围
        rendered_np = np.clip(rendered_np, 0, 1)
        target_np = np.clip(target_np, 0, 1)

        # 保存图像
        from PIL import Image
        rendered_img = Image.fromarray((rendered_np * 255).astype(np.uint8))
        rendered_img.save(os.path.join(self.output_dir, "renders", f"render_{iteration:06d}.png"))

        # 保存对比图（每5次保存一次，避免太多文件）
        if iteration % (self.save_interval // 2) == 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(rendered_np)
            axes[0].set_title(f"Rendered (Iter {iteration})")
            axes[0].axis('off')

            axes[1].imshow(target_np)
            axes[1].set_title("Target")
            axes[1].axis('off')

            # 差异图
            diff = np.abs(rendered_np - target_np)
            axes[2].imshow(diff, cmap='hot')
            axes[2].set_title("Difference")
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"comparison_{iteration:06d}.png"), dpi=150)
            plt.close()

    def save_final_results(self):
        """保存最终结果"""
        # 保存训练曲线
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        axes[0].plot(self.losses)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(True)

        # PSNR曲线
        axes[1].plot(self.psnrs)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("PSNR")
        axes[1].set_title("PSNR")
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_curves.png"), dpi=150)
        plt.close()

        # 保存点云
        self.save_point_cloud()

        # 保存配置
        config_path = os.path.join(self.output_dir, "config.txt")
        with open(config_path, 'w') as f:
            for key, value in config.config_dict.items():
                if key != 'device':  # 跳过设备，因为它不是可序列化的
                    f.write(f"{key}: {value}\n")

        print(f"最终结果已保存到: {self.output_dir}")

    def save_point_cloud(self):
        """保存点云为PLY格式"""
        try:
            import open3d as o3d

            xyz = self.model._xyz.detach().cpu().numpy()
            colors = torch.sigmoid(self.model._features_dc).detach().cpu().numpy().squeeze(1)

            # 限制颜色范围
            colors = np.clip(colors, 0, 1)

            # 创建点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # 保存
            ply_path = os.path.join(self.output_dir, "final_point_cloud.ply")
            o3d.io.write_point_cloud(ply_path, pcd)
            print(f"点云已保存: {ply_path}")

            # 也保存为简单的文本格式
            txt_path = os.path.join(self.output_dir, "point_cloud.txt")
            np.savetxt(txt_path, np.hstack([xyz, colors]),
                       fmt='%.6f', header='x y z r g b')
            print(f"点云文本格式已保存: {txt_path}")

        except Exception as e:
            print(f"保存点云时出错: {e}")
            # 保存为numpy格式
            xyz = self.model._xyz.detach().cpu().numpy()
            colors = torch.sigmoid(self.model._features_dc).detach().cpu().numpy().squeeze(1)

            np.save(os.path.join(self.output_dir, "point_cloud_xyz.npy"), xyz)
            np.save(os.path.join(self.output_dir, "point_cloud_colors.npy"), colors)
            print(f"点云已保存为numpy格式")