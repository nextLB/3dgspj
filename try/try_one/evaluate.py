import torch
import numpy as np
from pathlib import Path
import json
import argparse
from PIL import Image
import lpips
from skimage.metrics import structural_similarity as ssim


class Evaluator:
    """5.2 评估指标"""

    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)

    def compute_psnr(self, pred, target):
        """5.2.1 重建精度 - PSNR"""
        mse = np.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))

    def compute_ssim(self, pred, target):
        """5.2.1 重建精度 - SSIM"""
        # 转换为灰度
        if len(pred.shape) == 3:
            pred_gray = np.dot(pred[..., :3], [0.2989, 0.5870, 0.1140])
            target_gray = np.dot(target[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            pred_gray = pred
            target_gray = target

        return ssim(pred_gray, target_gray, data_range=1.0)

    def compute_lpips(self, pred, target):
        """5.2.1 重建精度 - LPIPS"""
        pred_tensor = torch.FloatTensor(pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        target_tensor = torch.FloatTensor(target).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            lpips_value = self.lpips_model(pred_tensor, target_tensor)

        return lpips_value.item()

    def evaluate_scene(self, pred_dir, gt_dir):
        """评估整个场景"""
        pred_files = sorted(list(Path(pred_dir).glob("*.png")))
        gt_files = sorted(list(Path(gt_dir).glob("*.png")))

        metrics = {
            'psnr': [],
            'ssim': [],
            'lpips': []
        }

        for pred_file, gt_file in zip(pred_files[:10], gt_files[:10]):  # 只评估前10张
            pred_img = np.array(Image.open(pred_file)) / 255.0
            gt_img = np.array(Image.open(gt_file)) / 255.0

            # 调整大小一致
            if pred_img.shape != gt_img.shape:
                pred_img = np.array(
                    Image.fromarray((pred_img * 255).astype(np.uint8)).resize((gt_img.shape[1], gt_img.shape[0])))
                pred_img = pred_img / 255.0

            psnr = self.compute_psnr(pred_img, gt_img)
            ssim_val = self.compute_ssim(pred_img, gt_img)
            lpips_val = self.compute_lpips(pred_img, gt_img)

            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim_val)
            metrics['lpips'].append(lpips_val)

            print(f"图像 {pred_file.name}: PSNR={psnr:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")

        # 计算平均值
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        print("\n平均指标:")
        print(f"PSNR: {avg_metrics['psnr']:.2f}")
        print(f"SSIM: {avg_metrics['ssim']:.4f}")
        print(f"LPIPS: {avg_metrics['lpips']:.4f}")

        return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="评估三维重建结果")
    parser.add_argument("--pred_dir", type=str, required=True, help="预测图像目录")
    parser.add_argument("--gt_dir", type=str, required=True, help="真实图像目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备")

    args = parser.parse_args()

    evaluator = Evaluator(args.device)
    metrics = evaluator.evaluate_scene(args.pred_dir, args.gt_dir)

    # 保存结果
    with open(Path(args.pred_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()