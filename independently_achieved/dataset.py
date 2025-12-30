"""
    数据集构建的程序文件
"""

# ====================================================================================== #
# ====================================================================================== #
# ====================================================================================== #

# 关于Mip NeRF 360数据集的说明
# PINHOLE 针孔模型
# fx    以像素为单位的x轴焦距
#           由物理焦距(mm)除以像元宽度(mm/像素)得到，影响图像在x方向的缩放
# fy    以像素为单位的y轴焦距
#           有物理焦距除以像元高度得到，与fx不同时，感光元件像素可能非正方形
# cx    图像主点的x坐标(像素)
#           通常接近图像中心(width/2=?),代表光轴与成像平面的交点
# cy    图像主点的y坐标
#           同样接近中心


# ====================================================================================== #
# ====================================================================================== #
# ====================================================================================== #




import os
import re
import copy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np






BASE_DATASET_PATH = '/home/next_lb/桌面/无人机影像三维重建任务/Mip_NeRF360/'
EXTRA_SCENES_360 = '360_extra_scenes'
V2_360 = '360_v2'
CLASS_NAME = 'bicycle'
RESOLUTION_SELECT = 'images'
CAMERA_INFO_DIR = 'sparse/0'
POSES_FILE_INFO_NAME = 'poses_bounds.npy'


RE_FLOWERS_IMAGE_NAME = r'_DSC(\d+).JPG'
RE_THEEHILL_IMAGE_NAME = r'_DSC(\d+).JPG'
RE_BICYCLE_IMAGE_NAME = r'_DSC(\d+).JPG'
RE_BONSAI_IMAGE_NAME = r'DSCF(\d+).JPG'
RE_COUNTER_IMAGE_NAME = r'DSCF(\d+).JPG'
RE_GARDEN_IMAGE_NAME = r'DSC(\d+).JPG'
RE_KITCHEN_IMAGE_NAME = r'DSCF(\d+).JPG'
RE_ROOM_IMAGE_NAME = r'DSCF(\d+).JPG'
RE_STUMP_IMAGE_NAME = r'_DSC(\d+).JPG'






class MipNeRF360Dataset:
    def __init__(self, trainLogger):
        self.name = 'MipNeRF360Dataset'
        self.logger = trainLogger
        self.imageDataRootPath = ''
        self.allSortedImagePath = []
        self.allSortedImageData = []
        self.resolution = 0

        self.cameraFilePath = ''
        self.imageFilePath = ''
        self.pointFilePath = ''
        self.poseBoundFilePath = ''
        self.sceneData = None



        # 加载图像数据
        self.load_images_data()

        # 加载相机和点云数据
        self.load_camera_and_point_cloud()






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
        sortedImageNames = sorted(allImageNames, key=lambda x: int(re.search(RE_BICYCLE_IMAGE_NAME, x).group(1)))

        for name in sortedImageNames:
            readImagePath = os.path.join(self.imageDataRootPath, name)
            self.allSortedImagePath.append(copy.deepcopy(readImagePath))
            self.allSortedImageData.append(Image.open(readImagePath))

        # 可视化单张图像
        # self.show_single_image(self.allSortedImageData[0])

    # 可视化单张图像
    def show_single_image(self, imageData):
        plt.figure(figsize=(10, 8))
        plt.imshow(np.array(imageData))
        plt.show()



    # ############################################################################### #
    # ############################################################################### #
    # ############################################################################### #

    # 加载相机和点云数据
    def load_camera_and_point_cloud(self):
        self.cameraFilePath = os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, CAMERA_INFO_DIR, 'cameras.bin')
        self.imageFilePath = os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, CAMERA_INFO_DIR, 'images.bin')
        self.pointFilePath = os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, CAMERA_INFO_DIR, 'points3D.bin')
        self.poseBoundFilePath = os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, 'poses_bounds.npy')

        print(self.cameraFilePath)
        print(self.imageFilePath)
        print(self.pointFilePath)
        print(self.poseBoundFilePath)





