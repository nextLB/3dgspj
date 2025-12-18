"""
    基于Mip_NeRF360无人机影像三维重构数据集构建的程序文件
"""

import os
import pycolmap
from skimage.morphology import reconstruction
from torch.utils.data import Dataset
import numpy as np

MIP_NERF360_ROOT = '/home/next_lb/桌面/无人机影像三维重建任务/Mip_NeRF360/'
DIR_SELECT = '360_v2/bicycle'
RESOLUTION_IMAGE = 'images_8'   # 可以是 images、images_2、images_4、images_8
SPARSE_PATH = 'sparse/0'
POSES_NAME = 'poses_bounds.npy'


class MipNeRF360Dataset(Dataset):
    def __init__(self):
        self.name = 'Mip_NeRF360'

        self.camerasInfo = None
        self.imagesInfo = None
        self.points3D = None

        self.posesBounds = None

    def load_sparse_info(self):
        reconstruction = pycolmap.Reconstruction(os.path.join(MIP_NERF360_ROOT, DIR_SELECT, SPARSE_PATH))
        self.camerasInfo = reconstruction.cameras
        self.imagesInfo = reconstruction.images
        self.points3D = reconstruction.points3D

    def load_poses_info(self):
        self.posesBounds = np.load(os.path.join(MIP_NERF360_ROOT, DIR_SELECT, POSES_NAME))

    def integrate_to_3dgsjs_information(self):
        pass

    def load_all_data(self):
        self.load_sparse_info()
        self.load_poses_info()
        self.integrate_to_3dgsjs_information()



