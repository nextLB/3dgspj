import argparse
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 现在导入配置和其他模块
import config
from data_loader import ColmapDataset
from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer
from train import Trainer
from utils import check_environment, prepare_initial_points


def parse_args():
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting 三维重建")
    parser.add_argument("--data_dir", type=str, default=config.DATA_DIR,
                        help="数据目录路径")
    parser.add_argument("--image_scale", type=int, default=config.IMAGE_SCALE,
                        help="图像缩放比例 (1=原始, 2=1/2分辨率, 4=1/4分辨率)")
    parser.add_argument("--iterations", type=int, default=config.NUM_ITERATIONS,
                        help="训练迭代次数")
    parser.add_argument("--check", action="store_true",
                        help="只检查环境，不运行训练")

    return parser.parse_args()


def main():
    # 解析参数
    args = parse_args()

    # 更新配置
    config.update_config_from_args(args)

    # 检查环境
    check_environment()

    if args.check:
        print("环境检查完成，退出程序")
        return

    # 检查数据目录
    if not os.path.exists(config.config_dict['data_dir']):
        print(f"错误: 数据目录不存在: {config.config_dict['data_dir']}")
        print("请确保数据集路径正确")
        return

    # 创建数据集
    try:
        print("\n加载数据集...")
        dataset = ColmapDataset(
            config.config_dict['data_dir'],
            config.config_dict['image_scale'],
            config.config_dict['device']
        )
        print(f"数据集加载成功: {len(dataset)}张图像")
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 准备初始点云
    print("\n准备初始点云...")
    points, colors = prepare_initial_points(dataset)

    # 创建高斯模型
    print("初始化高斯模型...")
    gaussian_model = GaussianModel(config.config_dict['sh_degree'])
    gaussian_model.create_from_pcd(points, colors)

    # 创建渲染器
    renderer = GaussianRenderer()

    # 创建训练器并开始训练
    print("\n开始训练...")
    trainer = Trainer(dataset, gaussian_model, renderer)
    trainer.train()

    print("\n训练完成！")
    print(f"结果保存在: {trainer.output_dir}")


if __name__ == "__main__":
    main()