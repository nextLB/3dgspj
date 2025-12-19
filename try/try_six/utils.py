import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from datetime import datetime


def save_checkpoint(scene, iteration, model_path):
    """保存检查点"""
    scene.save_checkpoint(iteration, model_path)
    print(f"Checkpoint saved at iteration {iteration}")


def load_checkpoint(checkpoint_path, scene):
    """加载检查点"""
    return scene.load_checkpoint(checkpoint_path)


def training_report(iteration, loss, l1_loss, elapsed_time, log_file=None):
    """训练报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"[{timestamp}] Iteration: {iteration:6d}, Loss: {loss:.6f}, L1 Loss: {l1_loss:.6f}, Time: {elapsed_time:.2f}s"
    print(report)

    if log_file:
        with open(log_file, 'a') as f:
            f.write(report + '\n')


def save_render_image(image_tensor, save_path, normalize=True):
    """保存渲染图像"""
    if isinstance(save_path, str):
        save_path = Path(save_path)

    save_path.parent.mkdir(exist_ok=True, parents=True)

    image = image_tensor.detach().cpu()

    if image.dim() == 4:  # [B, C, H, W]
        image = image[0]

    if image.dim() == 3:  # [C, H, W]
        if normalize:
            # 归一化到[0, 1]
            image_min = image.min()
            image_max = image.max()
            if image_max - image_min > 0:
                image = (image - image_min) / (image_max - image_min)
            else:
                image = torch.zeros_like(image)

        # 转换为numpy并保存
        image_np = image.permute(1, 2, 0).numpy()  # [H, W, C]
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

        Image.fromarray(image_np).save(save_path)
    else:
        print(f"Warning: Unexpected image shape: {image.shape}")

    return save_path


def compute_psnr(pred, target):
    """计算PSNR"""
    if pred.shape != target.shape:
        pred = torch.nn.functional.interpolate(pred.unsqueeze(0), size=target.shape[-2:], mode='bilinear',
                                               align_corners=False).squeeze(0)

    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(pred, target):
    """计算SSIM（简化版）"""
    from torch.nn.functional import conv2d

    # 确保形状匹配
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    if pred.shape != target.shape:
        pred = torch.nn.functional.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 创建高斯窗口
    def create_window(window_size, channel):
        _1D_window = torch.exp(torch.arange(window_size).float() - window_size // 2).pow(2).div(-2.0)
        _1D_window = _1D_window / _1D_window.sum()
        _2D_window = _1D_window[:, None] * _1D_window[None, :]
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    window_size = 11
    channel = pred.size(1)
    window = create_window(window_size, channel).to(pred.device)

    mu1 = conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def create_video_from_images(image_folder, output_path, fps=30):
    """从图像创建视频"""
    try:
        import cv2

        image_folder = Path(image_folder)
        if not image_folder.exists():
            print(f"Image folder not found: {image_folder}")
            return False

        images = sorted(list(image_folder.glob("*.png")))
        if not images:
            print(f"No PNG images found in {image_folder}")
            return False

        # 读取第一张图像获取尺寸
        first_image = cv2.imread(str(images[0]))
        if first_image is None:
            print(f"Failed to read image: {images[0]}")
            return False

        height, width, _ = first_image.shape

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not video.isOpened():
            print(f"Failed to create video writer for {output_path}")
            return False

        for image_path in tqdm(images, desc="Creating video"):
            img = cv2.imread(str(image_path))
            if img is not None:
                video.write(img)

        video.release()
        print(f"Video saved: {output_path}")
        return True

    except ImportError:
        print("OpenCV not installed, skipping video creation")
        return False
    except Exception as e:
        print(f"Error creating video: {e}")
        return False


def plot_training_curve(log_file, output_path=None):
    """绘制训练曲线"""
    if not Path(log_file).exists():
        print(f"Log file not found: {log_file}")
        return

    iterations = []
    losses = []

    with open(log_file, 'r') as f:
        for line in f:
            if "Iteration:" in line:
                parts = line.split(',')
                iter_part = parts[0].split(':')
                if len(iter_part) >= 2:
                    try:
                        iter_num = int(iter_part[1].strip())
                        loss = float(parts[1].split(':')[1].strip())
                        iterations.append(iter_num)
                        losses.append(loss)
                    except:
                        continue

    if not iterations:
        print("No training data found in log file")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 添加移动平均
    if len(losses) > 10:
        window_size = min(50, len(losses) // 10)
        moving_avg = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
        plt.plot(iterations[window_size - 1:], moving_avg, 'r-', linewidth=2,
                 label=f'Moving Average (window={window_size})')
        plt.legend()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Training curve saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def setup_logging(model_path):
    """设置日志"""
    log_dir = Path(model_path)
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file = log_dir / "training_log.txt"
    config_file = log_dir / "config.json"

    return log_file, config_file


def save_config(args, config_file):
    """保存配置"""
    config = vars(args)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_file}")


def get_memory_usage():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
        reserved = torch.cuda.memory_reserved() / 1024 ** 3  # GB
        return allocated, reserved
    return 0, 0


def print_memory_usage():
    """打印GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated, reserved = get_memory_usage()
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")