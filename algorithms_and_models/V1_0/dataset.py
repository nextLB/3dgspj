"""
    基于Mip_NeRF360无人机影像三维重构数据集构建的程序文件
"""

import json
import os
import pycolmap
from torch.utils.data import Dataset
import numpy as np

MIP_NERF360_ROOT = '/home/next_lb/桌面/无人机影像三维重建任务/Mip_NeRF360/'
DIR_SELECT = '360_v2/bicycle'
RESOLUTION_IMAGE = 'images_8'   # 可以是 images、images_2、images_4、images_8
SPARSE_PATH = 'sparse/0'
POSES_NAME = 'poses_bounds.npy'
SAVE_INTEGRATED_DATA_JSON_PATH = 'integrated_data.json'
SAVE_3D_POINTS_DATA_JSON_PATH = '3d_points.json'




class MipNeRF360Dataset(Dataset):
    def __init__(self):
        self.name = 'Mip_NeRF360'

        self.camerasInfo = None
        self.imagesInfo = None
        self.points3D = None

        self.posesBounds = None

        self.camerasDict = {}
        self.integrateDataInfo = []
        self.points3DInfo = []

        # 加载所有的数据信息
        self.load_all_data()



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
            # 获取所有的整合信息
            tempIntegrateDataInfo["image_id"] = imageID
            tempIntegrateDataInfo["image_name"] = imageInfo.name
            tempIntegrateDataInfo["image_path"] = os.path.join(MIP_NERF360_ROOT, DIR_SELECT, RESOLUTION_IMAGE, imageInfo.name)
            tempIntegrateDataInfo["has_pose"] = imageInfo.has_pose
            tempIntegrateDataInfo["triangulated"] = imageInfo.points2D
            tempIntegrateDataInfo["width"] = self.camerasDict[imageInfo.camera_id]['width']
            tempIntegrateDataInfo["height"] = self.camerasDict[imageInfo.camera_id]['height']
            tempIntegrateDataInfo["fx"] = float(self.camerasDict[imageInfo.camera_id]['fx'])
            tempIntegrateDataInfo["fy"] = float(self.camerasDict[imageInfo.camera_id]['fy'])
            tempIntegrateDataInfo["cx"] = float(self.camerasDict[imageInfo.camera_id]['cx'])
            tempIntegrateDataInfo["cy"] = float(self.camerasDict[imageInfo.camera_id]['cy'])
            poseBound = self.posesBounds[imageID-1]
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

        for point_id, point3d in self.points3D.items():
            tempPoint3DData = {}
            tempPoint3DData["3d_point_id"] = point_id
            tempPoint3DData["3d_position"] = point3d.xyz
            tempPoint3DData["3d_track_len"] = point3d.track

            self.points3DInfo.append(tempPoint3DData)


    def next_handle_all_integrated_data(self):
        pass


    def load_all_data(self):
        self.load_sparse_info()
        self.load_poses_info()
        self.integrate_to_3dgsjs_information()
        self.next_handle_all_integrated_data()
        self.save_integrated_data_to_json()

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

    def __len__(self):
        return len(self.integrateDataInfo)

    def __getitem__(self, item):
        pass


    # 保存整合后的数据到.json文件中
    def save_integrated_data_to_json(self):
        # 创建可序列化的数据结构
        serializable_data = []

        # 遍历每个图像的数据
        for image_data in self.integrateDataInfo:
            serializable_image_data = {}

            # 复制所有简单的键值对
            for key, value in image_data.items():
                if key == 'triangulated' and value is not None:
                    # 处理Point2D对象列表
                    serializable_image_data[key] = []
                    for point2d in value:
                        # 创建点字典
                        point_dict = {}

                        # 处理xy坐标
                        if hasattr(point2d, 'xy'):
                            if isinstance(point2d.xy, np.ndarray):
                                point_dict["xy"] = point2d.xy.tolist()
                            elif hasattr(point2d.xy, '__iter__'):
                                point_dict["xy"] = list(point2d.xy)
                            else:
                                point_dict["xy"] = [float(point2d.xy[0]), float(point2d.xy[1])]

                        # 处理point3D_id - 这里需要特殊处理无符号整数的-1
                        if hasattr(point2d, 'point3D_id'):
                            point3d_id = point2d.point3D_id

                            # 检查是否是numpy的无符号整数类型
                            if isinstance(point3d_id, np.unsignedinteger):
                                # 如果是无符号整数，检查是否为最大值（表示-1）
                                max_uint64 = np.uint64(2 ** 64 - 1)
                                max_uint32 = np.uint32(2 ** 32 - 1)

                                if point3d_id == max_uint64 or point3d_id == max_uint32:
                                    point_dict["point3D_id"] = -1
                                else:
                                    point_dict["point3D_id"] = int(point3d_id)
                            else:
                                # 对于有符号整数，直接转换
                                point_dict["point3D_id"] = int(point3d_id)
                        else:
                            point_dict["point3D_id"] = -1

                        serializable_image_data[key].append(point_dict)

                elif key == 'c2w' and isinstance(value, np.ndarray):
                    # 将numpy数组转换为列表
                    serializable_image_data[key] = value.flatten().tolist()
                elif isinstance(value, np.ndarray):
                    # 其他numpy数组转换为列表
                    serializable_image_data[key] = value.tolist()
                else:
                    # 其他类型直接复制
                    serializable_image_data[key] = value

            serializable_data.append(serializable_image_data)

        filename = os.path.join(MIP_NERF360_ROOT, SAVE_INTEGRATED_DATA_JSON_PATH)
        # 将数据写入JSON文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        # 准备可序列化的数据
        serializable_data = []

        for point3d_data in self.points3DInfo:
            serializable_point = {}

            # 复制所有字段
            for key, value in point3d_data.items():
                if key == '3d_position' and isinstance(value, np.ndarray):
                    # 处理3D位置数组
                    serializable_point[key] = value.tolist()

                elif key == '3d_track_len':
                    # 处理Track对象
                    if hasattr(value, 'elements'):
                        track_elements = []
                        for element in value.elements:
                            # 假设TrackElement有image_id和point2D_idx属性
                            track_element_dict = {}

                            if hasattr(element, 'image_id'):
                                track_element_dict['image_id'] = int(element.image_id)

                            if hasattr(element, 'point2D_idx'):
                                track_element_dict['point2D_idx'] = int(element.point2D_idx)

                            track_elements.append(track_element_dict)

                        serializable_point[key] = {
                            'elements': track_elements,
                            'track_length': len(track_elements)  # 添加track长度
                        }
                    else:
                        # 如果Track对象没有elements属性，尝试其他方式
                        serializable_point[key] = str(value)  # 或者尝试其他序列化方式

                else:
                    # 其他字段直接复制
                    serializable_point[key] = value

            serializable_data.append(serializable_point)

        filename = os.path.join(MIP_NERF360_ROOT, SAVE_3D_POINTS_DATA_JSON_PATH)
        # 保存到JSON文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)



        print(len(self.integrateDataInfo))
        print(len(self.points3DInfo))











