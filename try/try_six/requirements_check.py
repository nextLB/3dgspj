#!/usr/bin/env python3
"""
环境检查脚本 - 确保所有必要的包都已安装
"""
import sys
import importlib.util  # 新增：导入importlib


def check_packages():
    """检查必要的包是否已安装"""
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',  # pip包名
        'PIL',  # Pillow的导入名
        'tqdm',
        'imageio',
        'plyfile',
        'pycolmap',
        'open3d',
        'kornia',
        'matplotlib'
    ]

    missing_packages = []

    for package in required_packages:
        # 使用importlib.util.find_spec检查包是否存在
        spec = importlib.util.find_spec(package.replace('-', '_'))
        if spec is not None:
            print(f"✓ {package} 已安装")
        else:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")

    return missing_packages


def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用，当前设备: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  PyTorch CUDA版本: {torch.cuda.get_device_capability(0)}")
            return True
        else:
            print("✗ CUDA 不可用")
            return False
    except Exception as e:
        print(f"✗ 检查CUDA时出错: {e}")
        return False


def main():
    print("=" * 60)
    print("3D Gaussian Splatting 环境检查")
    print("=" * 60)

    print("\n1. 检查Python版本:")
    print(f"   Python版本: {sys.version}")

    print("\n2. 检查CUDA:")
    cuda_available = check_cuda()

    print("\n3. 检查必要包:")
    missing = check_packages()

    print("\n" + "=" * 60)
    if not missing and cuda_available:
        print("✓ 所有检查通过！环境配置正确。")
        print("您可以运行 main.py 开始三维重建。")
        return True
    else:
        if missing:
            print(f"✗ 缺少以下包: {missing}")
        if not cuda_available:
            print("✗ CUDA不可用，将无法使用GPU加速")
        print("\n请安装缺失的包后再运行。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)