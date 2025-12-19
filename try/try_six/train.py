#!/usr/bin/env python3
"""
训练循环 - 3D高斯溅射的训练
"""
from pathlib import Path
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

from gaussian_model import Gaussian3D
from dataset import MipNeRF360Dataset
from render import GaussianRenderer


class GaussianTrainer:
    """3D高斯溅射训练器"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # 创建输出目录
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # 初始化模型和渲染器
        self.model = None
        self.renderer = None
        self.optimizer = None
        self.scheduler = None

        # 训练状态
        self.iteration = 0
        self.best_psnr = 0.0

    def setup_data(self):
        """设置数据加载器"""
        print("加载数据集...")

        self.train_dataset = MipNeRF360Dataset(
            data_root=self.config["data_root"],
            scene_name=self.config["scene_name"],
            split="train",
            scale=self.config.get("image_scale", 1),
            load_images=True,
            max_images=self.config.get("max_train_images", -1)
        )

        # 归一化场景
        self.train_dataset.normalize_scene()

        # 获取点云初始化
        point_cloud = self.train_dataset.get_point_cloud()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,  # 一次处理一个图像
            shuffle=True,
            num_workers=self.config.get("num_workers", 0),
            pin_memory=True
        )

        # 验证集
        self.val_dataset = MipNeRF360Dataset(
            data_root=self.config["data_root"],
            scene_name=self.config["scene_name"],
            split="train",  # 使用相同的图像进行验证
            scale=self.config.get("image_scale", 1),
            load_images=True,
            max_images=self.config.get("max_val_images", 5)
        )
        self.val_dataset.normalize_scene()

        print(f"训练图像: {len(self.train_dataset)}, 验证图像: {len(self.val_dataset)}")

        return point_cloud

    def setup_model(self, point_cloud: Optional[Dict] = None):
        """设置模型"""
        print("初始化3D高斯模型...")

        # 从配置中获取初始点数
        num_points = self.config.get("initial_points", 5000)  # 默认5000

        if point_cloud is not None:
            # 从点云初始化
            points = point_cloud["points"]
            colors = point_cloud["colors"]
            # 如果点云点数少于配置的点数，使用点云点数
            actual_points = min(len(points), num_points)
            self.model = Gaussian3D.from_point_cloud(points[:actual_points],
                                                     colors[:actual_points],
                                                     self.device)
            print(f"从点云初始化 {actual_points} 个高斯")
        else:
            # 随机初始化
            self.model = Gaussian3D(num_points, self.device)
            print(f"随机初始化 {num_points} 个高斯")

        # 总是使用简化渲染器（避免显存溢出）
        from render import SimpleRenderer
        self.renderer = SimpleRenderer(self.device)
        print("使用简化渲染器（节省显存）")

        # 设置优化器
        self.setup_optimizer()
    def setup_optimizer(self):
        """设置优化器"""
        # 不同参数使用不同的学习率
        params = [
            {"params": [self.model.positions], "lr": self.model.position_lr * self.model.spatial_lr_scale,
             "name": "positions"},
            {"params": [self.model.colors], "lr": 0.025, "name": "colors"},
            {"params": [self.model.opacities], "lr": self.model.opacity_lr, "name": "opacities"},
            {"params": [self.model.scales], "lr": self.model.scaling_lr, "name": "scales"},
            {"params": [self.model.rotations], "lr": self.model.rotation_lr, "name": "rotations"},
        ]

        self.optimizer = optim.Adam(params, lr=0.001, eps=1e-15)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def train_step(self, batch: Dict) -> Dict:
        """单个训练步骤"""
        self.model.train()

        # 获取相机参数 - 确保转换为tensor并移到设备
        camera = batch["camera"].copy()  # 创建副本避免修改原始数据
        gt_image = batch["image"].to(self.device)

        # 确保相机参数是tensor并移到正确设备
        for key in ["R", "t", "K"]:
            if isinstance(camera[key], np.ndarray):
                camera[key] = torch.from_numpy(camera[key]).float().to(self.device)
            else:
                camera[key] = camera[key].float().to(self.device)

        # 渲染图像
        rendered = self.renderer.render(self.model, camera)
        # 计算损失
        loss_dict = self.compute_loss(rendered, gt_image)

        # 反向传播
        total_loss = loss_dict["total"]
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 更新梯度累积
        with torch.no_grad():
            self.model.xyz_gradient_accum += torch.norm(self.model.positions.grad, dim=1, keepdim=True)
            self.model.denom += 1

        # 记录指标
        metrics = {
            "loss": total_loss.item(),
            "color_loss": loss_dict["color"].item(),
            "psnr": self.compute_psnr(rendered, gt_image).item()
        }

        return metrics

    def compute_loss(self, rendered: torch.Tensor, gt_image: torch.Tensor) -> Dict:
        """计算损失函数"""
        # L1颜色损失
        color_loss = torch.abs(rendered - gt_image).mean()

        # SSIM损失（可选）
        if self.config.get("use_ssim", False):
            ssim_loss = 1 - self.ssim(rendered, gt_image)
            color_loss = 0.8 * color_loss + 0.2 * ssim_loss

        # 总损失
        total_loss = color_loss

        return {
            "color": color_loss,
            "total": total_loss
        }

    def compute_psnr(self, rendered: torch.Tensor, gt_image: torch.Tensor) -> torch.Tensor:
        """计算PSNR"""
        mse = torch.mean((rendered - gt_image) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr

    def ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True):
        """计算SSIM"""
        from math import exp

        # 参数
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # 创建高斯窗口
        def gaussian(window_size, sigma):
            gauss = torch.Tensor(
                [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()

        def create_window(window_size, channel):
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window

        channel = img1.size(0)
        window = create_window(window_size, channel).to(img1.device)

        mu1 = F.conv2d(img1.unsqueeze(0), window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2.unsqueeze(0), window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1.unsqueeze(0) * img1.unsqueeze(0), window, padding=window_size // 2,
                             groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size // 2,
                             groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size // 2,
                           groups=channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def densify_and_prune(self):
        """密度控制和修剪"""
        with torch.no_grad():
            # 基于梯度幅度进行密度控制
            grad_norm = torch.norm(self.model.positions.grad, dim=1)
            grad_threshold = 0.0002

            # 需要增加密度的位置
            need_densify = grad_norm > grad_threshold

            if need_densify.any():
                # 克隆需要增加密度的点
                params = self.model.forward()

                positions_to_clone = params["positions"][need_densify]
                colors_to_clone = params["colors"][need_densify]
                scales_to_clone = params["scales"][need_densify] * 0.8  # 稍微缩小

                # 添加随机扰动
                noise = torch.randn_like(positions_to_clone) * 0.01
                new_positions = positions_to_clone + noise

                self.model.densify_points(
                    positions=new_positions,
                    colors=colors_to_clone,
                    scales=scales_to_clone
                )

            # 修剪不透明度低的点
            opacities = params["opacities"].squeeze()
            prune_mask = opacities < 0.01

            if prune_mask.any():
                self.model.prune_points(prune_mask)

            # 重置不透明度（可选）
            if self.iteration % 1000 == 0:
                self.model.reset_opacities()

    def validate(self) -> Dict:
        """验证"""
        self.model.eval()

        val_metrics = {
            "psnr": [],
            "loss": []
        }

        with torch.no_grad():
            for i in range(min(5, len(self.val_dataset))):
                batch = self.val_dataset[i]

                camera = batch["camera"]
                gt_image = batch["image"].to(self.device)

                # 渲染
                rendered = self.renderer.render(self.model, camera)

                # 计算指标
                loss = torch.abs(rendered - gt_image).mean()
                psnr = self.compute_psnr(rendered, gt_image)

                val_metrics["psnr"].append(psnr.item())
                val_metrics["loss"].append(loss.item())

        # 平均指标
        avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}

        return avg_metrics

    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            "iteration": self.iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_psnr": self.best_psnr,
            "config": self.config
        }

        # 保存检查点
        checkpoint_path = self.output_dir / f"checkpoint_{self.iteration:06d}.pth"
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)

        # 保存PLY点云
        ply_path = self.output_dir / f"gaussian_{self.iteration:06d}.ply"
        self.model.save_ply(ply_path)

        # 清理旧的检查点
        if self.iteration % 5000 != 0:  # 只保留每5000次迭代的检查点
            if checkpoint_path.exists():
                checkpoint_path.unlink()

    def train(self):
        """主训练循环"""
        print("开始训练...")

        # 设置数据
        point_cloud = self.setup_data()

        # 设置模型
        self.setup_model(point_cloud)

        # 训练循环
        num_iterations = self.config.get("num_iterations", 30000)

        progress_bar = tqdm(range(num_iterations), desc="训练")

        for iteration in progress_bar:
            self.iteration = iteration

            # 获取一个训练批次
            try:
                batch = next(iter(self.train_loader))
            except StopIteration:
                # 重新创建数据加载器
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True
                )
                batch = next(iter(self.train_loader))

            # 训练步骤
            metrics = self.train_step(batch)

            # 密度控制和修剪
            if iteration % 100 == 0:
                self.densify_and_prune()

            # 更新学习率
            if iteration % 1000 == 0 and self.scheduler:
                self.scheduler.step()

            # 验证和保存
            if iteration % 500 == 0:
                val_metrics = self.validate()
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                # 保存检查点
                is_best = val_metrics["psnr"] > self.best_psnr
                if is_best:
                    self.best_psnr = val_metrics["psnr"]

                self.save_checkpoint(is_best)

            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{metrics.get('loss', 0):.4f}",
                "psnr": f"{metrics.get('psnr', 0):.2f}",
                "val_psnr": f"{metrics.get('val_psnr', 0):.2f}"
            })

        print("训练完成！")

        # 保存最终模型
        final_ply = self.output_dir / "final_gaussian.ply"
        self.model.save_ply(final_ply)
        print(f"最终模型保存到: {final_ply}")

        return self.model


def main(config_path: str = None):
    """主函数"""
    import json
    from pathlib import Path

    # 默认配置
    config = {
        "data_root": "/path/to/Mip_NeRF360",
        "scene_name": "flowers",
        "output_dir": "./outputs/flowers",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "image_scale": 1,
        "max_train_images": -1,
        "max_val_images": 5,
        "num_iterations": 30000,
        "initial_points": 10000,
        "use_ssim": False,
        "num_workers": 0
    }

    # 加载配置文件
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        config.update(user_config)

    # 创建训练器
    trainer = GaussianTrainer(config)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--data_root", type=str, help="数据根目录")
    parser.add_argument("--scene_name", type=str, help="场景名称")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--iterations", type=int, default=30000, help="训练迭代次数")

    args = parser.parse_args()

    # 更新配置
    config = {}
    if args.data_root:
        config["data_root"] = args.data_root
    if args.scene_name:
        config["scene_name"] = args.scene_name
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.iterations:
        config["num_iterations"] = args.iterations

    main(args.config if hasattr(args, 'config') else None)