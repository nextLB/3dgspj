"""
    数据集构建的程序文件
"""

import os
import re
import copy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



BASE_DATASET_PATH = '/home/next_lb/桌面/无人机影像三维重建任务/Mip_NeRF360/'
V2_360 = '360_v2'
CLASS_NAME = 'bicycle'
RESOLUTION_SELECT = 'images'

RE_IMAGE_NAME = r'_DSC(\d+).JPG'




class MipNeRF360Dataset:
    def __init__(self):
        self.name = 'MipNeRF360Dataset'
        self.imageDataRootPath = ''
        self.allSortedImagePath = []
        self.allSortedImageData = []
        self.resolution = 0


        # 加载图像数据
        self.load_images_data()

    # 加载图像数据
    def load_images_data(self):
        self.imageDataRootPath = os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, RESOLUTION_SELECT)
        if RESOLUTION_SELECT == 'images':
            self.resolution = 1
        elif RESOLUTION_SELECT == 'images_2':
            self.resolution = 0.5
        elif RESOLUTION_SELECT == 'images_4':
            self.resolution = 0.25
        elif RESOLUTION_SELECT == 'images_8':
            self.resolution = 0.125

        # 使用正则表达式提取数字并排序
        allImageNames = os.listdir(self.imageDataRootPath)
        sortedImageNames = sorted(allImageNames, key=lambda x: int(re.search(RE_IMAGE_NAME, x).group(1)))

        for name in sortedImageNames:
            readImagePath = os.path.join(self.imageDataRootPath, name)
            self.allSortedImagePath.append(copy.deepcopy(readImagePath))
            self.allSortedImageData.append(Image.open(readImagePath))

        self.show_single_image(self.allSortedImageData[0])

    # 可视化单张图像
    def show_single_image(self, imageData):
        plt.figure(figsize=(10, 8))
        plt.imshow(np.array(imageData))
        plt.show()






