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
from scipy.spatial.transform import Rotation as R





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

        self.sparseDirPath = ''
        self.poseBoundFilePath = ''
        self.sceneData = None
        self.poseData = None



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
        self.sparseDirPath = os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, CAMERA_INFO_DIR)
        self.poseBoundFilePath = os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, 'poses_bounds.npy')

        reconstruction = pycolmap.Reconstruction(self.sparseDirPath)

        self.sceneData = self.parse_colmap_to_gaussian_splatting(reconstruction.cameras, reconstruction.images, reconstruction.points3D)

        npPosesData = np.load(self.poseBoundFilePath)






    def parse_colmap_to_gaussian_splatting(self, cameras, images, points3D):
        """
        将COLMAP重建数据解析为高斯溅射三维重建所需的数据格式

        参数:
            cameras: pycolmap.Reconstruction.cameras (CameraMap)
            images: pycolmap.Reconstruction.images (ImageMap)
            points3D: pycolmap.Reconstruction.points3D (Point3DMap)

        返回:
            dict: 包含相机参数、图像位姿和3D点的字典
        """

        # 1. 解析相机参数
        camera_data = []
        for camera_id, camera in cameras.items():
            try:
                # 根据相机模型提取参数
                if hasattr(camera, 'model_name'):
                    camera_model = camera.model_name()
                elif hasattr(camera, 'model'):
                    camera_model = camera.model
                else:
                    camera_model = "UNKNOWN"

                # 获取相机尺寸
                if hasattr(camera, 'width') and hasattr(camera, 'height'):
                    width = camera.width
                    height = camera.height
                else:
                    width = 0
                    height = 0

                # 解析相机参数
                if hasattr(camera, 'params'):
                    params = camera.params
                else:
                    params = []

                # 根据相机模型设置焦距和主点
                if camera_model == "SIMPLE_PINHOLE":
                    # 参数: f, cx, cy
                    fx = params[0] if len(params) > 0 else 0
                    fy = params[0] if len(params) > 0 else 0
                    cx = params[1] if len(params) > 1 else width / 2
                    cy = params[2] if len(params) > 2 else height / 2
                elif camera_model == "PINHOLE":
                    # 参数: fx, fy, cx, cy
                    fx = params[0] if len(params) > 0 else 0
                    fy = params[1] if len(params) > 1 else fx
                    cx = params[2] if len(params) > 2 else width / 2
                    cy = params[3] if len(params) > 3 else height / 2
                elif camera_model == "SIMPLE_RADIAL":
                    # 参数: f, cx, cy, k
                    fx = params[0] if len(params) > 0 else 0
                    fy = params[0] if len(params) > 0 else 0
                    cx = params[1] if len(params) > 1 else width / 2
                    cy = params[2] if len(params) > 2 else height / 2
                elif camera_model == "RADIAL":
                    # 参数: f, cx, cy, k1, k2
                    fx = params[0] if len(params) > 0 else 0
                    fy = params[0] if len(params) > 0 else 0
                    cx = params[1] if len(params) > 1 else width / 2
                    cy = params[2] if len(params) > 2 else height / 2
                else:
                    # 默认处理：假设为PINHOLE模型
                    fx = params[0] if len(params) > 0 else 0
                    fy = params[1] if len(params) > 1 else fx
                    cx = params[2] if len(params) > 2 else width / 2
                    cy = params[3] if len(params) > 3 else height / 2

                camera_info = {
                    'camera_id': camera_id,
                    'model': camera_model,
                    'width': width,
                    'height': height,
                    'fx': float(fx),
                    'fy': float(fy),
                    'cx': float(cx),
                    'cy': float(cy),
                    'params': [float(p) for p in params]
                }
                camera_data.append(camera_info)
            except Exception as e:
                self.logger.info(f"警告: 相机 {camera_id} 解析出错: {e}")
                continue

        # 2. 解析图像位姿
        image_data = []
        processed_images = 0
        skipped_images = 0

        for image_id, image in images.items():
            try:
                # 检查图像是否有必要属性
                if not hasattr(image, 'qvec') or not hasattr(image, 'tvec'):
                    self.logger.info(f"警告: 图像 {image_id} 缺少必要属性: qvec或tvec")
                    skipped_images += 1
                    continue

                # 从四元数获取旋转矩阵（COLMAP存储的是四元数，不是直接的旋转矩阵）
                # COLMAP的四元数格式：qw, qx, qy, qz
                qvec = image.qvec  # 世界到相机的四元数

                # 将四元数转换为旋转矩阵
                # scipy的R.from_quat期望四元数顺序：qx, qy, qz, qw
                qvec_scipy = [qvec[1], qvec[2], qvec[3], qvec[0]]  # qx, qy, qz, qw
                rotation_matrix = R.from_quat(qvec_scipy).as_matrix()  # 世界到相机的旋转矩阵

                tvec = image.tvec  # 世界到相机的平移向量

                # COLMAP使用世界到相机的变换，但我们需要相机到世界的变换
                # 对于高斯溅射，通常需要相机到世界的变换
                # 计算相机在世界坐标系中的位置
                rotation_c2w = rotation_matrix.T  # 相机到世界的旋转
                camera_position = -rotation_c2w @ tvec

                # 将旋转矩阵转换为四元数（相机到世界）
                quaternion_c2w = R.from_matrix(rotation_c2w).as_quat()  # (qx, qy, qz, qw)

                # COLMAP格式的四元数 (qw, qx, qy, qz)
                quaternion_w2c = [qvec[0], qvec[1], qvec[2], qvec[3]]

                # 获取图像的其他属性
                image_name = getattr(image, 'name', f"image_{image_id}")
                camera_id_value = getattr(image, 'camera_id', 0)
                has_pose = True  # 如果图像在重建中，通常认为它有位姿

                # 尝试获取2D点的数量
                if hasattr(image, 'num_points2D'):
                    num_points2D = image.num_points2D()
                elif hasattr(image, 'points2D'):
                    num_points2D = len(image.points2D)
                else:
                    num_points2D = 0

                image_info = {
                    'image_id': image_id,
                    'image_name': image_name,
                    'camera_id': camera_id_value,
                    'has_pose': has_pose,
                    'num_points2D': num_points2D,

                    # COLMAP原始位姿（世界到相机）
                    'rotation_w2c': rotation_matrix.tolist(),  # 世界到相机的旋转
                    'translation_w2c': tvec.tolist(),  # 世界到相机的平移

                    # 相机到世界的变换（更常用的格式）
                    'rotation_c2w': rotation_c2w.tolist(),  # 相机到世界的旋转
                    'translation_c2w': camera_position.tolist(),  # 相机在世界中的位置

                    # 四元数表示
                    'quaternion_w2c': [float(q) for q in quaternion_w2c],
                    'quaternion_c2w': [float(q) for q in quaternion_c2w],
                }
                image_data.append(image_info)
                processed_images += 1

            except AttributeError as e:
                # 如果属性不存在，打印错误并跳过这个图像
                self.logger.info(f"警告: 图像 {image_id} 缺少必要属性: {e}")
                skipped_images += 1
                continue
            except Exception as e:
                # 其他错误
                self.logger.info(f"警告: 图像 {image_id} 解析出错: {e}")
                skipped_images += 1
                continue

        # 3. 解析3D点云
        points_data = []
        for point_id, point in points3D.items():
            try:
                # 获取点云属性
                xyz = getattr(point, 'xyz', [0, 0, 0])
                color = getattr(point, 'color', (255, 255, 255))
                track_len = getattr(point, 'track_len', 0)
                error = getattr(point, 'error', 0.0)

                point_info = {
                    'point_id': point_id,
                    'xyz': [float(coord) for coord in xyz],
                    'rgb': [int(c) for c in color],
                    'track_length': int(track_len),
                    'error': float(error),
                }
                points_data.append(point_info)
            except Exception as e:
                self.logger.info(f"警告: 点云 {point_id} 解析出错: {e}")
                continue

        # 4. 统计信息
        stats = {
            'num_cameras': len(cameras),
            'num_images': len(images),
            'num_processed_images': processed_images,
            'num_skipped_images': skipped_images,
            'num_points3D': len(points3D),
            'images_with_pose': processed_images,
            'image_resolution': {
                'max_width': max([cam['width'] for cam in camera_data]) if camera_data else 0,
                'max_height': max([cam['height'] for cam in camera_data]) if camera_data else 0,
            }
        }

        # 5. 组织为高斯溅射所需格式
        # 高斯溅射通常需要：相机参数、相机位姿（c2w）、图像列表、点云
        gaussian_data = {
            'cameras': camera_data,
            'images': sorted(image_data, key=lambda x: x['image_id']),
            'points3D': points_data,
            'statistics': stats,
        }

        return gaussian_data




