import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import lpips
import math


class LossFunction(nn.Module):
    """3D高斯泼溅损失函数"""

    def __init__(self, lambda_dssim=0.2, lambda_lpips=0.0, use_lpips=False):
        super().__init__()

        self.lambda_dssim = lambda_dssim
        self.lambda_lpips = lambda_lpips
        self.use_lpips = use_lpips

        # L1损失
        self.l1_loss = nn.L1Loss(reduction='mean')

        # SSIM损失
        self.ssim_loss = SSIMLoss(window_size=11)

        # LPIPS感知损失（可选）
        if use_lpips and lambda_lpips > 0:
            try:
                self.lpips_loss = lpips.LPIPS(net='vgg')
                print("LPIPS loss initialized")
            except:
                print("Warning: LPIPS not available, disabling LPIPS loss")
                self.use_lpips = False
                self.lambda_lpips = 0.0

    def forward(self, rendered_image, gt_image, mask=None):
        """
        计算损失

        Args:
            rendered_image: 渲染图像 (C, H, W)
            gt_image: 真实图像 (C, H, W)
            mask: 可选掩码 (1, H, W)

        Returns:
            总损失
        """
        # 确保图像在[0, 1]范围内
        rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
        gt_image = torch.clamp(gt_image, 0.0, 1.0)

        # 应用掩码（如果有）
        if mask is not None:
            rendered_image = rendered_image * mask
            gt_image = gt_image * mask
            valid_pixels = mask.sum() + 1e-8
        else:
            valid_pixels = rendered_image.numel() / 3  # RGB通道数

        # L1损失
        l1_loss = self.l1_loss(rendered_image, gt_image)

        # SSIM损失
        ssim_loss_value = self.ssim_loss(rendered_image, gt_image)

        # 组合损失
        total_loss = (1 - self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss_value

        # LPIPS感知损失（可选）
        if self.use_lpips and self.lambda_lpips > 0:
            # 调整图像维度以符合LPIPS输入要求
            rendered_lpips = rendered_image.unsqueeze(0)  # (1, C, H, W)
            gt_lpips = gt_image.unsqueeze(0)  # (1, C, H, W)

            lpips_loss_value = self.lpips_loss(rendered_lpips, gt_lpips).mean()
            total_loss = total_loss + self.lambda_lpips * lpips_loss_value

        return total_loss

    def compute_psnr(self, rendered_image, gt_image, mask=None):
        """计算PSNR"""
        if mask is not None:
            rendered_image = rendered_image * mask
            gt_image = gt_image * mask
            mse = F.mse_loss(rendered_image, gt_image, reduction='sum') / (mask.sum() + 1e-8)
        else:
            mse = F.mse_loss(rendered_image, gt_image)

        if mse.item() == 0:
            return float('inf')

        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()

    def compute_ssim(self, rendered_image, gt_image, mask=None):
        """计算SSIM"""
        return self.ssim_loss.compute_ssim(rendered_image, gt_image, mask)


class SSIMLoss(nn.Module):
    """SSIM损失函数"""

    def __init__(self, window_size=11, size_average=True, val_range=1.0):
        super().__init__()

        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # 创建高斯窗口
        self.register_buffer('gaussian_window', self.create_gaussian_window(window_size))

        # 补偿项
        self.C1 = (0.01 * val_range) ** 2
        self.C2 = (0.03 * val_range) ** 2

    def create_gaussian_window(self, window_size, sigma=1.5):
        """创建高斯窗口"""
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])

        window = gauss.unsqueeze(1) * gauss.unsqueeze(0)  # (window_size, window_size)
        window = window / window.sum()
        window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)

        return window

    def ssim(self, img1, img2, mask=None):
        """计算SSIM"""
        _, channels, height, width = img1.size()

        # 扩展窗口到通道数
        window = self.gaussian_window.repeat(channels, 1, 1, 1)

        # 计算均值
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channels)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # 计算方差
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channels) - mu1_mu2

        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        # 应用掩码（如果有）
        if mask is not None:
            ssim_map = ssim_map * mask

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2, mask=None):
        """计算SSIM损失（1 - SSIM）"""
        # 调整维度以符合SSIM输入要求
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        ssim_value = self.ssim(img1, img2, mask)
        return 1.0 - ssim_value

    def compute_ssim(self, img1, img2, mask=None):
        """计算SSIM值（不转换为损失）"""
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        return self.ssim(img1, img2, mask).item()


class AdaptiveLoss(nn.Module):
    """自适应损失函数（根据训练进度调整权重）"""

    def __init__(self, initial_lambda_dssim=0.2, final_lambda_dssim=0.2,
                 warmup_iterations=1000, use_lpips=False):
        super().__init__()

        self.initial_lambda_dssim = initial_lambda_dssim
        self.final_lambda_dssim = final_lambda_dssim
        self.warmup_iterations = warmup_iterations
        self.use_lpips = use_lpips

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()

        if use_lpips:
            try:
                self.lpips_loss = lpips.LPIPS(net='vgg')
            except:
                self.use_lpips = False

    def forward(self, rendered_image, gt_image, iteration):
        """根据训练进度计算自适应损失"""
        # 计算当前lambda值
        if iteration < self.warmup_iterations:
            t = iteration / self.warmup_iterations
            lambda_dssim = self.initial_lambda_dssim * t + self.final_lambda_dssim * (1 - t)
        else:
            lambda_dssim = self.final_lambda_dssim

        # 计算各项损失
        l1_loss = self.l1_loss(rendered_image, gt_image)
        ssim_loss_value = self.ssim_loss(rendered_image, gt_image)

        # 组合损失
        total_loss = (1 - lambda_dssim) * l1_loss + lambda_dssim * ssim_loss_value

        # 可选：LPIPS损失（在训练后期使用）
        if self.use_lpips and iteration > self.warmup_iterations:
            rendered_lpips = rendered_image.unsqueeze(0)
            gt_lpips = gt_image.unsqueeze(0)
            lpips_loss_value = self.lpips_loss(rendered_lpips, gt_lpips).mean()

            # 逐渐增加LPIPS权重
            lpips_weight = min(0.1, 0.01 * (iteration - self.warmup_iterations) / 1000)
            total_loss = total_loss + lpips_weight * lpips_loss_value

        return total_loss, {
            'l1_loss': l1_loss.item(),
            'ssim_loss': ssim_loss_value.item(),
            'lambda_dssim': lambda_dssim,
            'total_loss': total_loss.item()
        }


class RegularizationLoss(nn.Module):
    """正则化损失（用于控制高斯属性）"""

    def __init__(self, lambda_scale=0.01, lambda_opacity=0.01):
        super().__init__()

        self.lambda_scale = lambda_scale
        self.lambda_opacity = lambda_opacity

    def forward(self, scaling, opacity):
        """计算正则化损失"""
        # 缩放正则化（鼓励各向同性）
        scale_reg = torch.mean(torch.abs(scaling[:, 0] - scaling[:, 1])) + \
                    torch.mean(torch.abs(scaling[:, 1] - scaling[:, 2])) + \
                    torch.mean(torch.abs(scaling[:, 2] - scaling[:, 0]))

        # 不透明度正则化（鼓励适中的不透明度）
        opacity_reg = torch.mean((opacity - 0.5) ** 2)

        total_reg = self.lambda_scale * scale_reg + self.lambda_opacity * opacity_reg

        return total_reg, {
            'scale_reg': scale_reg.item(),
            'opacity_reg': opacity_reg.item(),
            'total_reg': total_reg.item()
        }


class DepthConsistencyLoss(nn.Module):
    """深度一致性损失（多视图）"""

    def __init__(self, lambda_depth=0.1):
        super().__init__()
        self.lambda_depth = lambda_depth

    def forward(self, depth_maps, cameras):
        """
        计算深度一致性损失

        Args:
            depth_maps: 深度图列表 [N, 1, H, W]
            cameras: 相机列表
        """
        if len(depth_maps) < 2:
            return torch.tensor(0.0, device=depth_maps[0].device)

        total_loss = 0.0
        n_pairs = 0

        for i in range(len(depth_maps)):
            for j in range(i + 1, len(depth_maps)):
                # 将深度图i投影到相机j
                depth_i = depth_maps[i]
                cam_i = cameras[i]
                cam_j = cameras[j]

                # 这里需要实现深度图重投影
                # 简化实现：暂时返回0

                n_pairs += 1

        if n_pairs > 0:
            total_loss = total_loss / n_pairs

        return self.lambda_depth * total_loss


def compute_total_variation_loss(image):
    """计算总变差损失（用于平滑图像）"""
    diff_h = torch.abs(image[:, :, 1:] - image[:, :, :-1])
    diff_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])

    tv_loss = torch.mean(diff_h) + torch.mean(diff_w)
    return tv_loss


def compute_edge_aware_loss(rendered_image, gt_image):
    """边缘感知损失（在边缘处赋予更大权重）"""
    # 计算GT图像的梯度
    gt_grad_x = torch.abs(gt_image[:, :, 1:] - gt_image[:, :, :-1])
    gt_grad_y = torch.abs(gt_image[:, :, :, 1:] - gt_image[:, :, :, :-1])

    # 创建权重图（边缘处权重更大）
    edge_weight = torch.exp(5.0 * (gt_grad_x.mean(dim=1, keepdim=True) +
                                   gt_grad_y.mean(dim=1, keepdim=True)))

    # 计算加权L1损失
    diff = torch.abs(rendered_image - gt_image)
    weighted_diff = diff * edge_weight

    return weighted_diff.mean()



