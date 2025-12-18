"""
    三维高斯溅射模型训练的主程序文件
"""

from algorithms_and_models.V1_0.dataset import MipNeRF360Dataset


def main():
    print('3D Gaussian Sputtering Model Training, version 1.0')

    # 构建数据集
    MipNeRF360DatasetInstance = MipNeRF360Dataset()
    MipNeRF360DatasetInstance.load_all_data()







