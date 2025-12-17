import torch
import torch.nn as nn
import torch.nn.functional as F


class PhotometricLoss(nn.Module):
    """4.1.2 基于光度损失的自适应优化 - 光度损失"""

    def __init__(self, lambda_dssim=0.2):
        super().__init__()
        self.lambda_dssim = lambda_dssim

    def forward(self, pred, target):
        """计算损失

        Args:
            pred: 预测图像 [B, H, W, 3] 或 [B, 3, H, W]
            target: 目标图像 [B, H, W, 3] 或 [B, 3, H, W]
        """
        # 确保通道在最后
        if pred.shape[1] == 3:
            pred = pred.permute(0, 2, 3, 1)
        if target.shape[1] == 3:
            target = target.permute(0, 2, 3, 1)

        # L1损失
        l1_loss = F.l1_loss(pred, target)

        # SSIM损失
        ssim_loss = 1 - self._ssim(pred, target)

        # 组合损失
        total_loss = (1 - self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss

        return total_loss

    def _ssim(self, img1, img2, window_size=11, size_average=True):
        """计算SSIM"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.avg_pool2d(img1.permute(0, 3, 1, 2), window_size, stride=1, padding=window_size // 2)
        mu2 = F.avg_pool2d(img2.permute(0, 3, 1, 2), window_size, stride=1, padding=window_size // 2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1.permute(0, 3, 1, 2).pow(2), window_size, stride=1,
                                 padding=window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2.permute(0, 3, 1, 2).pow(2), window_size, stride=1,
                                 padding=window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(img1.permute(0, 3, 1, 2) * img2.permute(0, 3, 1, 2), window_size, stride=1,
                               padding=window_size // 2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class TotalVariationLoss(nn.Module):
    """总变差损失，用于正则化"""

    def forward(self, x):
        """计算总变差损失"""
        batch_size = x.size(0)
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return (h_tv + w_tv) / batch_size