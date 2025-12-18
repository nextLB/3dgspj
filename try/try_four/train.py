import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from config import get_config
from dataset import get_dataset
from gaussian_model import GaussianModel
from render import GaussianRenderer


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # 设置输出目录
        os.makedirs(cfg.output_dir, exist_ok=True)
        self.scene_dir = os.path.join(cfg.output_dir, cfg.scene)
        os.makedirs(self.scene_dir, exist_ok=True)

        # 初始化组件
        print("Loading dataset...")
        self.dataset = get_dataset(cfg, split='train')

        print("Initializing Gaussian Model...")
        self.model = GaussianModel(sh_degree=cfg.sh_degree).to(self.device)

        # 从点云初始化（使用所有相机光线的交点近似）
        initial_pcd = self.initialize_from_random_points(5000)
        self.model.create_from_pcd(initial_pcd)
        self.model.setup_optimizer(cfg)

        print("Initializing Renderer...")
        self.renderer = GaussianRenderer(
            background_color=(1.0, 1.0, 1.0) if cfg.white_background else (0.0, 0.0, 0.0)
        )

        # 损失函数
        self.l1_loss = nn.L1Loss(reduction='mean')

        # 训练状态
        self.iteration = 0
        self.best_psnr = 0

    def initialize_from_random_points(self, num_points: int = 5000) -> torch.Tensor:
        """
        生成随机点云作为高斯初始位置。
        基于场景边界在相机视锥内采样。
        """
        device = self.device

        # 使用第一个相机的近远平面
        near = self.dataset.bounds[0, 0].item()
        far = self.dataset.bounds[0, 1].item()

        # 生成随机点：在单位球内采样，然后缩放到场景尺度
        points = torch.randn(num_points, 3, device=device)
        points = F.normalize(points, dim=-1)

        # 在近远平面之间分配深度
        depths = torch.rand(num_points, 1, device=device) * (far - near) + near

        # 转换到世界坐标系（使用第一个相机）
        pose = self.dataset.poses[0]  # w2c矩阵
        K = self.dataset.Ks[0]

        # 反转投影：从NDC到世界坐标
        # 生成随机图像坐标
        H, W = self.dataset.H, self.dataset.W
        u = torch.rand(num_points, device=device) * W
        v = torch.rand(num_points, device=device) * H

        # 创建齐次坐标
        uv_h = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # [N, 3]

        # 计算射线方向（相机坐标系）
        K_inv = torch.linalg.inv(K)
        dir_cam = torch.matmul(K_inv, uv_h.T).T  # [N, 3]
        dir_cam = F.normalize(dir_cam, dim=-1)

        # 转换到世界坐标系
        R = pose[:3, :3].T  # 相机到世界的旋转
        t = pose[:3, 3]
        dir_world = torch.matmul(R, dir_cam.T).T  # [N, 3]

        # 计算世界坐标点
        points_world = dir_world * depths + t

        return points_world

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算光度损失：L1 + SSIM（结构相似性）。"""
        # L1损失
        l1 = self.l1_loss(pred, target)

        # SSIM损失（简化版）
        if self.cfg.lambda_dssim > 0:
            ssim_loss = 1.0 - self.ssim(pred, target)
            loss = (1 - self.cfg.lambda_dssim) * l1 + self.cfg.lambda_dssim * ssim_loss
        else:
            loss = l1

        return loss

    def ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """计算SSIM（结构相似性指数）。"""
        # 简化实现（完整SSIM更复杂）
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.avg_pool2d(img1.permute(2, 0, 1).unsqueeze(0), window_size, stride=1, padding=window_size // 2)
        mu2 = F.avg_pool2d(img2.permute(2, 0, 1).unsqueeze(0), window_size, stride=1, padding=window_size // 2)

        sigma1_sq = F.avg_pool2d(img1.permute(2, 0, 1).unsqueeze(0) ** 2, window_size, stride=1,
                                 padding=window_size // 2) - mu1 ** 2
        sigma2_sq = F.avg_pool2d(img2.permute(2, 0, 1).unsqueeze(0) ** 2, window_size, stride=1,
                                 padding=window_size // 2) - mu2 ** 2
        sigma12 = F.avg_pool2d(img1.permute(2, 0, 1).unsqueeze(0) * img2.permute(2, 0, 1).unsqueeze(0),
                               window_size, stride=1, padding=window_size // 2) - mu1 * mu2

        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
                    (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def train_iteration(self, data):
        """执行一次训练迭代。"""
        self.model.optimizer.zero_grad()

        # 获取数据
        target_image = data['image'].to(self.device)  # [H, W, 3]
        pose = data['pose'].to(self.device)  # [4, 4]
        K = data['K'].to(self.device)  # [3, 3]
        H, W = target_image.shape[:2]

        # 渲染
        pred_image = self.renderer.render(self.model, K, pose, H, W)

        # 计算损失
        loss = self.compute_loss(pred_image, target_image)

        # 反向传播
        loss.backward()

        # 更新模型参数
        self.model.optimizer.step()

        # 更新学习率
        self.model.update_learning_rate(self.iteration)

        # 自适应密度控制
        if self.iteration % self.cfg.opacity_reset_interval == 0:
            scene_extent = torch.max(self.model.get_xyz).item()
            self.model.densify_and_prune(
                grad_threshold=self.cfg.densify_grad_threshold,
                scene_extent=scene_extent,
                iteration=self.iteration,
                densify_until=self.cfg.densify_until_iter,
                percent_dense=self.cfg.percent_dense
            )

        return loss.item(), pred_image.detach()

    def train(self):
        """主训练循环。"""
        print(f"Starting training for scene: {self.cfg.scene}")
        print(f"Number of training images: {len(self.dataset)}")
        print(f"Output directory: {self.scene_dir}")

        # 训练循环
        pbar = tqdm(range(self.cfg.iterations), desc="Training")

        for iteration in pbar:
            self.iteration = iteration

            # 随机选择一张训练图像
            idx = torch.randint(0, len(self.dataset), (1,)).item()
            data = self.dataset[idx]

            # 训练迭代
            loss, pred = self.train_iteration(data)

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss:.6f}',
                'Gaussians': self.model._xyz.shape[0] if hasattr(self.model, '_xyz') else 0
            })

            # 定期保存
            if iteration % self.cfg.save_interval == 0:
                self.save_checkpoint(iteration)
                self.visualize_training(pred, data['image'], iteration)

        # 训练结束
        print("Training completed!")
        self.save_final_model()

    def save_checkpoint(self, iteration):
        """保存模型检查点。"""
        checkpoint_path = os.path.join(self.scene_dir, f"checkpoint_{iteration:06d}.pth")

        checkpoint = {
            'iteration': iteration,
            'model_state_dict': {
                'xyz': self.model._xyz.data,
                'opacity': self.model._opacity.data,
                'scaling': self.model._scaling.data,
                'rotation': self.model._rotation.data,
                'features_dc': self.model._features_dc.data,
                'features_rest': self.model._features_rest.data,
            },
            'optimizer_state_dict': self.model.optimizer.state_dict() if self.model.optimizer else None,
            'config': vars(self.cfg)
        }

        torch.save(checkpoint, checkpoint_path)

        # 同时保存PLY文件
        ply_path = os.path.join(self.scene_dir, f"gaussians_{iteration:06d}.ply")
        self.model.save_ply(ply_path)

    def visualize_training(self, pred, target, iteration):
        """可视化训练结果。"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 预测图像
        pred_np = pred.cpu().numpy()
        axes[0].imshow(pred_np)
        axes[0].set_title(f'Prediction (Iteration {iteration})')
        axes[0].axis('off')

        # 目标图像
        target_np = target.cpu().numpy()
        axes[1].imshow(target_np)
        axes[1].set_title('Target')
        axes[1].axis('off')

        # 差异图像
        diff = np.abs(pred_np - target_np).mean(axis=-1)
        im = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Difference')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])

        plt.tight_layout()
        vis_path = os.path.join(self.scene_dir, f"visualization_{iteration:06d}.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_final_model(self):
        """保存最终模型。"""
        final_path = os.path.join(self.scene_dir, "final_model.pth")
        ply_path = os.path.join(self.scene_dir, "final_gaussians.ply")

        # 保存PyTorch检查点
        torch.save({
            'model': self.model,
            'config': vars(self.cfg)
        }, final_path)

        # 保存PLY文件
        self.model.save_ply(ply_path)

        print(f"Final model saved to {final_path}")
        print(f"Final Gaussians saved to {ply_path}")


def main():
    cfg = get_config()
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()