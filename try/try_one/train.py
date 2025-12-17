
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
from datetime import datetime
import wandb

from gaussian_splatting import GaussianSplattingModel
from dataset import SceneDataset
from losses import PhotometricLoss

class GaussianSplattingTrainer:
    """训练器类"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        # 初始化模型
        self.model = GaussianSplattingModel(
            max_gaussians=config.max_gaussians,
            sh_degree=config.sh_degree
        ).to(self.device)

        # 初始化优化器
        self.optimizer = optim.Adam([
            {'params': self.model.xyz, 'lr': config.position_lr},
            {'params': self.model.features, 'lr': config.feature_lr},
            {'params': self.model.opacity, 'lr': config.opacity_lr},
            {'params': self.model.scale, 'lr': config.scaling_lr},
            {'params': self.model.rotation, 'lr': config.rotation_lr}
        ])

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.lr_milestones,
            gamma=config.lr_gamma
        )

        # 损失函数
        self.criterion = PhotometricLoss()

        # 实验跟踪
        if config.use_wandb:
            wandb.init(project="3d-gaussian-splatting", config=config)

        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, train_loader, val_loader):
        """训练循环"""
        print(f"开始训练，共{self.config.epochs}个epoch")

        for epoch in range(self.config.epochs):
            # 训练阶段
            train_loss = self._train_epoch(train_loader, epoch)

            # 验证阶段
            if epoch % self.config.val_freq == 0:
                val_loss, val_metrics = self._validate(val_loader, epoch)

                # 保存模型
                if epoch % self.config.save_freq == 0:
                    self._save_checkpoint(epoch, val_loss)

                # 记录指标
                self._log_metrics(epoch, train_loss, val_loss, val_metrics)

            # 学习率调整
            self.scheduler.step()

            # 自适应密度控制
            if epoch % self.config.density_control_freq == 0:
                self.model.adaptive_density_control(epoch)

    def _train_epoch(self, loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # 获取数据
            images = batch['image'].to(self.device)
            cameras = batch['camera']
            bg_color = torch.rand(3, device=self.device)  # 随机背景颜色

            # 前向传播
            rendered_images = []
            for camera in cameras:
                rendered = self.model(camera, bg_color)
                rendered_images.append(rendered)

            rendered_images = torch.stack(rendered_images)

            # 计算损失
            loss = self.criterion(rendered_images, images)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 累积损失
            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})

            # 梯度累积
            self._accumulate_gradients()

        return total_loss / num_batches

    def _validate(self, loader, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="验证"):
                images = batch['image'].to(self.device)
                cameras = batch['camera']
                bg_color = torch.tensor([0.5, 0.5, 0.5], device=self.device)

                # 渲染
                rendered_images = []
                for camera in cameras:
                    rendered = self.model(camera, bg_color)
                    rendered_images.append(rendered)

                rendered_images = torch.stack(rendered_images)

                # 计算损失和指标
                loss = self.criterion(rendered_images, images)
                psnr = self._compute_psnr(rendered_images, images)

                total_loss += loss.item()
                total_psnr += psnr
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches

        return avg_loss, {'psnr': avg_psnr}

    def _accumulate_gradients(self):
        """累积梯度用于密度控制"""
        if self.model.xyz.grad is not None:
            self.model.xyz_grad_accum += self.model.xyz.grad.norm(dim=1, keepdim=True).detach()

    def _compute_psnr(self, rendered, target):
        """计算PSNR"""
        mse = torch.mean((rendered - target) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()

    def _log_metrics(self, epoch, train_loss, val_loss, val_metrics):
        """记录指标"""
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Validation PSNR: {val_metrics['psnr']:.2f}")

        if self.config.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_psnr': val_metrics['psnr'],
                'learning_rate': self.scheduler.get_last_lr()[0],
                'num_gaussians': self.model._get_valid_mask().sum().item()
            })

    def _save_checkpoint(self, epoch, val_loss):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # 保存PLY
        ply_path = self.output_dir / f"gaussians_epoch_{epoch}.ply"
        self.model.save_ply(str(ply_path))

        print(f"检查点保存到: {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description="训练三维高斯泼溅模型")

    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")

    # 模型参数
    parser.add_argument("--max_gaussians", type=int, default=100000, help="最大高斯数量")
    parser.add_argument("--sh_degree", type=int, default=3, help="球谐度数")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=1000, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--device", type=str, default="cuda", help="设备")

    # 优化器参数
    parser.add_argument("--position_lr", type=float, default=0.00016, help="位置学习率")
    parser.add_argument("--feature_lr", type=float, default=0.0025, help="特征学习率")
    parser.add_argument("--opacity_lr", type=float, default=0.05, help="不透明度学习率")
    parser.add_argument("--scaling_lr", type=float, default=0.005, help="缩放学习率")
    parser.add_argument("--rotation_lr", type=float, default=0.001, help="旋转学习率")

    # 学习率调度
    parser.add_argument("--lr_milestones", nargs='+', type=int, default=[500, 800], help="学习率里程碑")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="学习率衰减系数")

    # 训练控制
    parser.add_argument("--val_freq", type=int, default=10, help="验证频率")
    parser.add_argument("--save_freq", type=int, default=50, help="保存频率")
    parser.add_argument("--density_control_freq", type=int, default=100, help="密度控制频率")

    # 实验跟踪
    parser.add_argument("--use_wandb", action="store_true", help="使用wandb")

    args = parser.parse_args()

    # 创建训练器
    trainer = GaussianSplattingTrainer(args)

    # 加载数据
    dataset = SceneDataset(args.data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 开始训练
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()