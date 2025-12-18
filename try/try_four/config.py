import argparse


def get_config():
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting for Mip-NeRF 360")

    # 数据路径
    parser.add_argument("--data_path", type=str, default="/home/next_lb/桌面/无人机影像三维重建任务/archive/360_v2/", help="Path to the 360_v2 dataset root")
    parser.add_argument("--scene", type=str, default="bicycle", help="Scene name (e.g., bicycle, bonsai)")
    parser.add_argument("--images", type=str, default="images", help="Image folder to use (images, images_2, etc.)")

    # 训练参数
    parser.add_argument("--iterations", type=int, default=30000, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (number of rays per iteration)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--position_lr_init", type=float, default=0.00016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--position_lr_max_steps", type=int, default=30_000)

    # 高斯模型参数
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical harmonics degree")
    parser.add_argument("--white_background", type=bool, default=True, help="Use white background")
    parser.add_argument("--lambda_dssim", type=float, default=0.2, help="SSIM loss weight")

    # 自适应密度控制参数
    parser.add_argument("--opacity_reset_interval", type=int, default=3000)
    parser.add_argument("--densify_from_iter", type=int, default=500)
    parser.add_argument("--densify_until_iter", type=int, default=15_000)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    parser.add_argument("--percent_dense", type=float, default=0.01)

    # 设备与输出
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda or cpu")
    parser.add_argument("--save_interval", type=int, default=1000, help="Interval to save model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory for outputs and checkpoints")

    return parser.parse_args()


if __name__ == "__main__":
    cfg = get_config()
    print(cfg)