import torch
import argparse

# 默认配置参数
DATA_DIR = "/home/next_lb/桌面/无人机影像三维重建任务/Mip_NeRF360/360_v2/bicycle"
NUM_ITERATIONS = 3000  # 先设为3000次迭代，用于测试
LEARNING_RATE = 0.001
BATCH_SIZE = 1
SAVE_INTERVAL = 500

# 高斯参数
INITIAL_POINTS = 1000  # 减少点云数量以节省内存
DENSIFICATION_INTERVAL = 100
OPACITY_RESET_INTERVAL = 3000
DENSIFY_FROM_ITER = 500
DENSIFY_UNTIL_ITER = 15000
DENSIFY_GRAD_THRESHOLD = 0.0002

# 渲染参数
IMAGE_SCALE = 2  # 使用1/2分辨率的图像
SH_DEGREE = 0  # 设为0简化球谐函数，加速训练
WHITE_BACKGROUND = True

# 优化器参数
POSITION_LR_INIT = 0.00016
POSITION_LR_FINAL = 0.0000016
POSITION_LR_DELAY_MULT = 0.01
POSITION_LR_MAX_STEPS = 30000
FEATURE_LR = 0.0025
OPACITY_LR = 0.05
SCALING_LR = 0.005
ROTATION_LR = 0.001

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 其他参数
LAMBDA_DSSIM = 0.2
PERCENT_DENSE = 0.01
OPACITY_MIN = 0.005
OPACITY_MAX = 0.99

# 用于更新的配置字典
config_dict = {
    'data_dir': DATA_DIR,
    'num_iterations': NUM_ITERATIONS,
    'learning_rate': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'save_interval': SAVE_INTERVAL,
    'initial_points': INITIAL_POINTS,
    'densification_interval': DENSIFICATION_INTERVAL,
    'opacity_reset_interval': OPACITY_RESET_INTERVAL,
    'densify_from_iter': DENSIFY_FROM_ITER,
    'densify_until_iter': DENSIFY_UNTIL_ITER,
    'densify_grad_threshold': DENSIFY_GRAD_THRESHOLD,
    'image_scale': IMAGE_SCALE,
    'sh_degree': SH_DEGREE,
    'white_background': WHITE_BACKGROUND,
    'position_lr_init': POSITION_LR_INIT,
    'position_lr_final': POSITION_LR_FINAL,
    'position_lr_delay_mult': POSITION_LR_DELAY_MULT,
    'position_lr_max_steps': POSITION_LR_MAX_STEPS,
    'feature_lr': FEATURE_LR,
    'opacity_lr': OPACITY_LR,
    'scaling_lr': SCALING_LR,
    'rotation_lr': ROTATION_LR,
    'device': DEVICE,
    'lambda_dssim': LAMBDA_DSSIM,
    'percent_dense': PERCENT_DENSE,
    'opacity_min': OPACITY_MIN,
    'opacity_max': OPACITY_MAX
}


def update_config_from_args(args):
    """从命令行参数更新配置"""
    if args.data_dir:
        config_dict['data_dir'] = args.data_dir
    if args.image_scale:
        config_dict['image_scale'] = args.image_scale
    if args.iterations:
        config_dict['num_iterations'] = args.iterations

    # 打印配置
    print("配置参数:")
    print(f"  数据目录: {config_dict['data_dir']}")
    print(f"  图像缩放: {config_dict['image_scale']}")
    print(f"  迭代次数: {config_dict['num_iterations']}")
    print(f"  初始点云数: {config_dict['initial_points']}")
    print(f"  球谐阶数: {config_dict['sh_degree']}")
    print(f"  设备: {config_dict['device']}")
    print()