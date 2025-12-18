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

        self.camerasDict = {}
        self.integrateDataInfo = []



    def load_sparse_info(self):
        reconstruction = pycolmap.Reconstruction(os.path.join(MIP_NERF360_ROOT, DIR_SELECT, SPARSE_PATH))
        self.camerasInfo = reconstruction.cameras
        self.imagesInfo = reconstruction.images
        self.points3D = reconstruction.points3D

    def load_poses_info(self):
        self.posesBounds = np.load(os.path.join(MIP_NERF360_ROOT, DIR_SELECT, POSES_NAME))

    def integrate_to_3dgsjs_information(self):
        """将MipNeRF360数据集转换为3D Gaussian Splatting可用的格式"""
        for cameraID, cameraInfo in self.camerasInfo.items():
            # 解析相机参数
            if cameraInfo.model == 'SIMPLE_PINHOLE':
                fx = fy = cameraInfo.params[0]
                cx, cy = cameraInfo.params[1], cameraInfo.params[2]
            elif cameraInfo.model == 'PINHOLE':
                fx, fy, cx, cy = cameraInfo.params[0], cameraInfo.params[1], cameraInfo.params[2], cameraInfo.params[3]
            else:
                # 对于其他模型，使用近似值
                fx = fy = cameraInfo.params[0]
                cx, cy = cameraInfo.params[1], cameraInfo.params[2]
            self.camerasDict[cameraID] = {
                'width': cameraInfo.width,
                'height': cameraInfo.height,
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy
            }


        tempIntegrateDataInfo = {}
        for idx, (imageID, imageInfo) in enumerate(self.imagesInfo.items()):

            # 获取所有的整合信息
            tempIntegrateDataInfo["image_id"] = imageID
            tempIntegrateDataInfo["image_name"] = imageInfo.name
            tempIntegrateDataInfo["image_path"] = os.path.join(MIP_NERF360_ROOT, DIR_SELECT, RESOLUTION_IMAGE, imageInfo.name)
            tempIntegrateDataInfo["has_pose"] = imageInfo.has_pose
            tempIntegrateDataInfo["width"] = self.camerasDict[imageInfo.camera_id]['width']
            tempIntegrateDataInfo["height"] = self.camerasDict[imageInfo.camera_id]['height']
            tempIntegrateDataInfo["fx"] = float(self.camerasDict[imageInfo.camera_id]['fx'])
            tempIntegrateDataInfo["fy"] = float(self.camerasDict[imageInfo.camera_id]['fy'])
            tempIntegrateDataInfo["cx"] = float(self.camerasDict[imageInfo.camera_id]['cx'])
            tempIntegrateDataInfo["cy"] = float(self.camerasDict[imageInfo.camera_id]['cy'])
            poseBound = self.posesBounds[idx].tolist()
            print(len(poseBound))



            self.integrateDataInfo.append(tempIntegrateDataInfo)
            print(tempIntegrateDataInfo)



    def load_all_data(self):
        self.load_sparse_info()
        self.load_poses_info()
        self.integrate_to_3dgsjs_information()



