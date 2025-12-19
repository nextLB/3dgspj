import numpy as np
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import config


def create_test_point_cloud(num_points=1000):
    """创建测试点云（如果没有COLMAP点云）"""
    # 创建一个简单的球体点云
    points = []

    # 球体
    for _ in range(num_points):
        theta = np.random.random() * 2 * np.pi
        phi = np.random.random() * np.pi
        r = np.random.random() * 2.0 + 0.5  # 半径在0.5到2.5之间

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta) * 0.5  # 稍微压扁
        z = r * np.cos(phi)

        points.append([x, y, z])

    points = np.array(points)

    # 随机颜色
    colors = np.random.rand(num_points, 3)

    return points, colors


def visualize_camera_poses(dataset, output_path="camera_poses.png"):
    """可视化相机位姿"""
    try:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 提取相机位置
        positions = []
        for i in range(min(20, len(dataset))):  # 只绘制前20个相机
            data = dataset[i]
            pose = data['pose'].cpu().numpy()

            # 从位姿矩阵提取相机位置
            R = pose[:3, :3]
            t = pose[:3, 3]

            # 相机位置 = -R^T * t
            position = -R.T @ t
            positions.append(position)

        positions = np.array(positions)

        # 绘制相机位置
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o', s=50)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Poses')

        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"相机位姿可视化已保存: {output_path}")
    except:
        print("无法可视化相机位姿")


def check_environment():
    """检查环境配置"""
    print("=" * 50)
    print("环境检查")
    print("=" * 50)

    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA可用")
        print(f"  设备: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        print("✗ CUDA不可用，将使用CPU")

    # 检查PyTorch
    print(f"✓ PyTorch版本: {torch.__version__}")

    # 检查关键库
    try:
        import numpy
        print(f"✓ NumPy版本: {numpy.__version__}")
    except:
        print("✗ NumPy未找到")

    try:
        import PIL
        print(f"✓ PIL可用")
    except:
        print("✗ PIL未找到")

    try:
        import open3d
        print(f"✓ Open3D可用")
    except:
        print("✗ Open3D未找到，点云将保存为numpy格式")

    print("=" * 50)


def prepare_initial_points(dataset):
    """准备初始点云"""
    # 创建测试点云
    print("创建测试点云...")
    points, colors = create_test_point_cloud(config.config_dict['initial_points'])
    print(f"创建点云: {len(points)}个点")

    return points, colors


def load_sample_image(dataset_path, image_scale=2):
    """加载并显示示例图像"""
    try:
        # 确定图像文件夹
        if image_scale > 1:
            images_dir = os.path.join(dataset_path, f"images_{image_scale}")
            if not os.path.exists(images_dir):
                images_dir = os.path.join(dataset_path, "images")
        else:
            images_dir = os.path.join(dataset_path, "images")

        # 获取第一张图像
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            first_img_path = os.path.join(images_dir, image_files[0])
            img = Image.open(first_img_path)
            print(f"示例图像: {first_img_path}")
            print(f"图像尺寸: {img.size}")
            return True
        else:
            print(f"在 {images_dir} 中没有找到图像文件")
            return False
    except Exception as e:
        print(f"加载示例图像时出错: {e}")
        return False