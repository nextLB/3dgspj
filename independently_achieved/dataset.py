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
import pycolmap



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
    def __init__(self):
        self.name = 'MipNeRF360Dataset'
        self.imageDataRootPath = ''
        self.allSortedImagePath = []
        self.allSortedImageData = []
        self.resolution = 0

        self.rawReconstructionInfo = {}


        # 加载图像数据
        self.load_images_data()
        # 加载图像对应的信息
        self.load_images_info()



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

        # self.show_single_image(self.allSortedImageData[0])

    # 可视化单张图像
    def show_single_image(self, imageData):
        plt.figure(figsize=(10, 8))
        plt.imshow(np.array(imageData))
        plt.show()


    # 加载图像对应的信息
    def load_images_info(self):
        self.rawReconstructionInfo = pycolmap.Reconstruction(os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, CAMERA_INFO_DIR))
        camerasInfo = self.rawReconstructionInfo.cameras
        imagesInfo = self.rawReconstructionInfo.images
        pointsInfo = self.rawReconstructionInfo.points3D

        # K = [[fx,  0,  cx],
        #      [ 0,  fy, cy],
        #      [ 0,  0,   1]]
        K = np.array([[camerasInfo[1].params[0], 0, camerasInfo[1].params[2]],
             [0, camerasInfo[1].params[1], camerasInfo[1].params[3]],
             [0, 0, 1]])


        all3DPointsInfo = {}
        for idx, point3d in pointsInfo.items():
            threeDPointIdx = []
            for element in point3d.track.elements:
                threeDPointIdx.append(copy.deepcopy((element.image_id, element.point2D_idx)))
            threeDPointIdx = np.array(threeDPointIdx)
            all3DPointsInfo[f"{idx}"] = {
                "xyz": point3d.xyz,
                "threeDPointIdx": threeDPointIdx
            }
        print(len(all3DPointsInfo))
        print(all3DPointsInfo["1"]["xyz"])
        print(all3DPointsInfo["1"]["threeDPointIdx"])
        print(all3DPointsInfo["1"]["threeDPointIdx"][1])


        all2DTriangulateInfo = {}
        for i in range(len(self.allSortedImagePath)):
            triangulatePoints = []
            for idx, point2d in enumerate(imagesInfo[i+1].points2D):
                if point2d.point3D_id == 18446744073709551615:
                    tempCoordinate = (point2d.xy[0], point2d.xy[1], -1)
                else:
                    tempCoordinate = (point2d.xy[0], point2d.xy[1], point2d.point3D_id)
                triangulatePoints.append(copy.deepcopy(tempCoordinate))
            triangulatePoints = np.array(triangulatePoints)
            all2DTriangulateInfo[f"{i+1}"] = {
                "image_path": self.allSortedImagePath[i],
                "has_pose": imagesInfo[i+1].has_pose,
                "triangulatePoints": triangulatePoints
            }

        firstIndex = all3DPointsInfo["1"]["threeDPointIdx"][1][0]
        secondIndex = all3DPointsInfo["1"]["threeDPointIdx"][1][1]
        print(firstIndex)
        print(secondIndex)
        print(len(all2DTriangulateInfo))
        print(all2DTriangulateInfo[f"{firstIndex}"]["triangulatePoints"])
        print(all2DTriangulateInfo[f"{firstIndex}"]["triangulatePoints"][secondIndex])









