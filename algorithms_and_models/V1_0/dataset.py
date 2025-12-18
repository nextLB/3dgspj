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


        for idx, (imageID, imageInfo) in enumerate(self.imagesInfo.items()):
            tempIntegrateDataInfo = {}
            print(imageID)
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
            poseBound = self.posesBounds[idx]

            # 解析pose_bound为c2w矩阵
            # pose_bound通常包含17个值：前15个是3x5矩阵，后2个是边界
            if len(poseBound) == 17:
                # 前15个值重塑为3x5矩阵
                mat = poseBound[:15].reshape(3, 5)

                # 提取旋转矩阵(前3列)和平移向量(第4列)
                R = mat[:, :3]      # 3x3 旋转矩阵
                t = mat[:, 3:4]     # 3x1 平移向量

                # 构建4x4的c2w矩阵
                c2w = np.eye(4)
                c2w[:3, :3] = R
                c2w[:3, 3] = t.flatten()

                # 边界参数
                bounds = poseBound[15:17]
            else:
                c2w = np.eye(4)
                R = c2w[:3, :3]
                bounds = [2.0, 6.0]     # 默认值

            tempIntegrateDataInfo["position"] = c2w[:3, 3].tolist()
            tempIntegrateDataInfo["rotation"] = self.matrix_to_quaternion(R).tolist()   # 旋转(四元数)
            tempIntegrateDataInfo["c2w"] = c2w.flatten().tolist()       # 展平的c2w矩阵
            tempIntegrateDataInfo["normalization"] = bounds.tolist()


            self.integrateDataInfo.append(tempIntegrateDataInfo)

        # 最后的排序整合
        self.integrateDataInfo.sort(key=lambda x: x['image_id'])


    def load_all_data(self):
        self.load_sparse_info()
        self.load_poses_info()
        self.integrate_to_3dgsjs_information()

    def matrix_to_quaternion(self, R):
        """将旋转矩阵转换为四元数"""
        q = np.zeros(4)
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[0] = 0.25 / s
            q[1] = (R[2, 1] - R[1, 2]) * s
            q[2] = (R[0, 2] - R[2, 0]) * s
            q[3] = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                q[0] = (R[2, 1] - R[1, 2]) / s
                q[1] = 0.25 * s
                q[2] = (R[0, 1] + R[1, 0]) / s
                q[3] = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                q[0] = (R[0, 2] - R[2, 0]) / s
                q[1] = (R[0, 1] + R[1, 0]) / s
                q[2] = 0.25 * s
                q[3] = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                q[0] = (R[1, 0] - R[0, 1]) / s
                q[1] = (R[0, 2] + R[2, 0]) / s
                q[2] = (R[1, 2] + R[2, 1]) / s
                q[3] = 0.25 * s

        return q / np.linalg.norm(q)

