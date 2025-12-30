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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
import struct







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






@dataclass
class CameraData:
    """存储相机数据的类"""
    image_names: List[str]
    image_paths: List[str]
    # 相机内参
    intrinsics: np.ndarray  # [N, 4] 或 [N, 3, 3]
    # 相机外参（世界到相机）
    extrinsics: np.ndarray  # [N, 4, 4] 或 [N, 3, 4]
    # 图像尺寸
    image_sizes: np.ndarray  # [N, 2] (height, width)
    # 相机类型和参数
    camera_types: List[str]
    # 3D点云（可选）
    points3D: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    # 边界（用于NeRF格式）
    bounds: Optional[np.ndarray] = None
    # 其他元数据
    metadata: Dict[str, Any] = None


def read_colmap_bin_file(filename: str, max_params: int = 100) -> Dict[int, Any]:
    """
    读取COLMAP二进制文件，增加安全限制

    参数:
    -----------
    filename : str
        二进制文件路径
    max_params : int
        最大允许的参数数量，防止异常值
    """
    data = {}

    try:
        with open(filename, "rb") as fid:
            # 读取条目数量
            header = fid.read(8)
            if len(header) < 8:
                print(f"警告: {filename} 文件太小或为空")
                return data

            num_entries = struct.unpack("Q", header)[0]

            print(f"  找到 {num_entries} 个条目")

            for _ in range(num_entries):
                # 读取条目ID
                id_bytes = fid.read(4)
                if len(id_bytes) < 4:
                    print(f"警告: 读取条目ID时文件结束")
                    break

                entry_id = struct.unpack("I", id_bytes)[0]

                # 根据文件类型解析
                if "cameras" in filename.lower():
                    # 相机参数格式: model_id, width, height, params...
                    model_bytes = fid.read(4)
                    if len(model_bytes) < 4:
                        print(f"警告: 读取相机模型ID时文件结束")
                        break
                    model_id = struct.unpack("i", model_bytes)[0]

                    width_bytes = fid.read(8)
                    height_bytes = fid.read(8)
                    if len(width_bytes) < 8 or len(height_bytes) < 8:
                        print(f"警告: 读取图像尺寸时文件结束")
                        break

                    width = struct.unpack("Q", width_bytes)[0]
                    height = struct.unpack("Q", height_bytes)[0]

                    num_params_bytes = fid.read(8)
                    if len(num_params_bytes) < 8:
                        print(f"警告: 读取参数数量时文件结束")
                        break

                    num_params = struct.unpack("Q", num_params_bytes)[0]

                    # 安全检查：限制参数数量
                    if num_params > max_params:
                        print(f"警告: 相机 {entry_id} 参数数量异常 ({num_params})，限制为 {max_params}")
                        num_params = max_params

                    # 读取参数
                    params_bytes = fid.read(8 * num_params)
                    if len(params_bytes) < 8 * num_params:
                        print(f"警告: 读取参数时文件结束")
                        break

                    params = struct.unpack("d" * num_params, params_bytes)

                    data[entry_id] = {
                        "model_id": model_id,
                        "width": width,
                        "height": height,
                        "params": np.array(params)
                    }

                elif "images" in filename.lower():
                    # 图像参数格式: qvec(4), tvec(3), camera_id, name, num_points2D, points2D...
                    qvec_bytes = fid.read(8 * 4)
                    if len(qvec_bytes) < 8 * 4:
                        print(f"警告: 读取四元数时文件结束")
                        break
                    qvec = struct.unpack("dddd", qvec_bytes)

                    tvec_bytes = fid.read(8 * 3)
                    if len(tvec_bytes) < 8 * 3:
                        print(f"警告: 读取平移向量时文件结束")
                        break
                    tvec = struct.unpack("ddd", tvec_bytes)

                    camera_id_bytes = fid.read(4)
                    if len(camera_id_bytes) < 4:
                        print(f"警告: 读取相机ID时文件结束")
                        break
                    camera_id = struct.unpack("I", camera_id_bytes)[0]

                    # 读取图像名（以null结尾的字符串）
                    name_chars = []
                    while True:
                        char = fid.read(1)
                        if not char:
                            print(f"警告: 读取图像名时文件结束")
                            break
                        if char == b'\x00':
                            break
                        name_chars.append(char.decode('utf-8', errors='ignore'))

                    name = ''.join(name_chars)

                    num_points2D_bytes = fid.read(8)
                    if len(num_points2D_bytes) < 8:
                        print(f"警告: 读取2D点数量时文件结束")
                        break
                    num_points2D = struct.unpack("Q", num_points2D_bytes)[0]

                    # 读取2D点（可选）
                    points2D = []
                    for _ in range(num_points2D):
                        x_bytes = fid.read(8)
                        y_bytes = fid.read(8)
                        point3D_id_bytes = fid.read(8)

                        if len(x_bytes) < 8 or len(y_bytes) < 8 or len(point3D_id_bytes) < 8:
                            print(f"警告: 读取2D点时文件结束")
                            break

                        x = struct.unpack("d", x_bytes)[0]
                        y = struct.unpack("d", y_bytes)[0]
                        point3D_id = struct.unpack("q", point3D_id_bytes)[0]
                        points2D.append((x, y, point3D_id))

                    data[entry_id] = {
                        "qvec": np.array(qvec),
                        "tvec": np.array(tvec),
                        "camera_id": camera_id,
                        "name": name,
                        "points2D": np.array(points2D) if points2D else None
                    }

                elif "points3D" in filename.lower():
                    # 3D点格式: xyz(3), rgb(3), error, track_length, track...
                    xyz_bytes = fid.read(8 * 3)
                    if len(xyz_bytes) < 8 * 3:
                        print(f"警告: 读取3D坐标时文件结束")
                        break
                    xyz = struct.unpack("ddd", xyz_bytes)

                    rgb_bytes = fid.read(4 * 3)
                    if len(rgb_bytes) < 4 * 3:
                        print(f"警告: 读取RGB颜色时文件结束")
                        break
                    rgb = struct.unpack("III", rgb_bytes)

                    error_bytes = fid.read(8)
                    if len(error_bytes) < 8:
                        print(f"警告: 读取误差时文件结束")
                        break
                    error = struct.unpack("d", error_bytes)[0]

                    track_length_bytes = fid.read(8)
                    if len(track_length_bytes) < 8:
                        print(f"警告: 读取轨迹长度时文件结束")
                        break
                    track_length = struct.unpack("Q", track_length_bytes)[0]

                    # 读取track（图像ID和2D点ID对）
                    track = []
                    for _ in range(track_length):
                        image_id_bytes = fid.read(4)
                        point2D_idx_bytes = fid.read(4)

                        if len(image_id_bytes) < 4 or len(point2D_idx_bytes) < 4:
                            print(f"警告: 读取track时文件结束")
                            break

                        image_id = struct.unpack("I", image_id_bytes)[0]
                        point2D_idx = struct.unpack("I", point2D_idx_bytes)[0]
                        track.append((image_id, point2D_idx))

                    data[entry_id] = {
                        "xyz": np.array(xyz),
                        "rgb": np.array(rgb),
                        "error": error,
                        "track": track
                    }

    except Exception as e:
        print(f"读取文件 {filename} 时出错: {e}")
        print("可能文件格式不正确或已损坏")

    return data


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """将四元数转换为旋转矩阵"""
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec

    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    ])


def create_extrinsic_matrix(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """创建4x4外参矩阵（世界到相机）"""
    R = qvec2rotmat(qvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T


def create_intrinsic_matrix(params: np.ndarray, width: int, height: int, model_id: int) -> np.ndarray:
    """根据相机模型创建内参矩阵"""
    # 相机模型ID映射
    camera_model_ids = {
        0: "SIMPLE_PINHOLE",  # f, cx, cy
        1: "PINHOLE",  # fx, fy, cx, cy
        2: "SIMPLE_RADIAL",  # f, cx, cy, k
        3: "RADIAL",  # f, cx, cy, k1, k2
        4: "OPENCV",  # fx, fy, cx, cy, k1, k2, p1, p2
        5: "OPENCV_FISHEYE",  # fx, fy, cx, cy, k1, k2, k3, k4
        6: "FULL_OPENCV",  # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    }

    model_name = camera_model_ids.get(model_id, f"UNKNOWN_{model_id}")

    try:
        if model_id == 0:  # SIMPLE_PINHOLE
            if len(params) >= 3:
                fx, cx, cy = params[0], params[1], params[2]
                fy = fx
            else:
                raise ValueError(f"SIMPLE_PINHOLE需要3个参数，得到{len(params)}个")
        elif model_id == 1:  # PINHOLE
            if len(params) >= 4:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                raise ValueError(f"PINHOLE需要4个参数，得到{len(params)}个")
        elif model_id == 2:  # SIMPLE_RADIAL
            if len(params) >= 4:
                f, cx, cy, k = params[0], params[1], params[2], params[3]
                fx = fy = f
            else:
                raise ValueError(f"SIMPLE_RADIAL需要4个参数，得到{len(params)}个")
        elif model_id == 3:  # RADIAL
            if len(params) >= 5:
                f, cx, cy, k1, k2 = params[0], params[1], params[2], params[3], params[4]
                fx = fy = f
            else:
                raise ValueError(f"RADIAL需要5个参数，得到{len(params)}个")
        elif model_id == 4:  # OPENCV
            if len(params) >= 8:
                fx, fy, cx, cy, k1, k2, p1, p2 = params[0], params[1], params[2], params[3], params[4], params[5], \
                params[6], params[7]
            else:
                raise ValueError(f"OPENCV需要8个参数，得到{len(params)}个")
        elif model_id == 5:  # OPENCV_FISHEYE
            if len(params) >= 8:
                fx, fy, cx, cy, k1, k2, k3, k4 = params[0], params[1], params[2], params[3], params[4], params[5], \
                params[6], params[7]
            else:
                raise ValueError(f"OPENCV_FISHEYE需要8个参数，得到{len(params)}个")
        elif model_id == 6:  # FULL_OPENCV
            if len(params) >= 12:
                fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = params[0], params[1], params[2], params[3], params[4], \
                params[5], params[6], params[7], params[8], params[9], params[10], params[11]
            else:
                raise ValueError(f"FULL_OPENCV需要12个参数，得到{len(params)}个")
        else:
            warnings.warn(f"未知相机模型ID: {model_id}，使用PINHOLE近似")
            if len(params) >= 4:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif len(params) >= 3:
                fx, cx, cy = params[0], params[1], params[2]
                fy = fx
            else:
                # 默认值
                fx = fy = 0.5 * max(width, height)
                cx = width / 2.0
                cy = height / 2.0
    except Exception as e:
        print(f"创建内参矩阵时出错 (model_id={model_id}, params={params}): {e}")
        # 使用默认值
        fx = fy = 0.5 * max(width, height)
        cx = width / 2.0
        cy = height / 2.0

    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K, model_name


def parse_poses_bounds(poses_bounds_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """解析poses_bounds.npy文件（NeRF格式）"""
    try:
        data = np.load(poses_bounds_path)

        # poses_bounds格式: [N, 17] 或 [N, 15] 或 [N, 14]
        n_cameras = data.shape[0]

        print(f"   poses_bounds形状: {data.shape}")

        # 根据形状确定格式
        if data.shape[1] == 17:
            # 包含H, W, focal信息
            poses = data[:, :12].reshape(-1, 3, 4)
            bounds = data[:, 12:14]  # near, far
            hwf = data[:, 14:]  # height, width, focal
        elif data.shape[1] == 15:
            # 标准格式
            poses = data[:, :12].reshape(-1, 3, 4)
            bounds = data[:, 12:14]
            hwf = None
        elif data.shape[1] == 14:
            # 只有pose和bounds，没有hwf
            poses = data[:, :12].reshape(-1, 3, 4)
            bounds = data[:, 12:14]
            hwf = None
        else:
            raise ValueError(f"未知poses_bounds格式: {data.shape}")

        # 将3x4矩阵转换为4x4齐次矩阵
        extrinsics = np.zeros((n_cameras, 4, 4))
        extrinsics[:, :3, :4] = poses
        extrinsics[:, 3, 3] = 1.0

        # 如果提供了hwf，创建内参矩阵
        intrinsics = None
        if hwf is not None:
            intrinsics = []
            for i in range(n_cameras):
                h, w, f = hwf[i]
                K = np.eye(3)
                K[0, 0] = f
                K[1, 1] = f
                K[0, 2] = w / 2.0
                K[1, 2] = h / 2.0
                intrinsics.append(K)
            intrinsics = np.array(intrinsics)

        return extrinsics, bounds, intrinsics

    except Exception as e:
        print(f"解析poses_bounds文件时出错: {e}")
        return np.array([]), np.array([]), None










class MipNeRF360Dataset:
    def __init__(self):
        self.name = 'MipNeRF360Dataset'
        self.imageDataRootPath = ''
        self.allSortedImagePath = []
        self.allSortedImageData = []
        self.resolution = 0


        # 加载图像数据
        self.load_images_data()

        # 加载相机与拍摄时的数据等等
        self.cameraPointsData = self.parse_camera_data_for_gaussian_splatting(
            os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, CAMERA_INFO_DIR, "cameras.bin"),
            os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, CAMERA_INFO_DIR, "images.bin"),
            os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, CAMERA_INFO_DIR, "points3D.bin"),
            os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, "poses_bounds.npy")
        )



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


    def parse_camera_data_for_gaussian_splatting(
            self,
            cameras_bin_path: str,
            images_bin_path: str,
            points3D_bin_path: str,
            poses_bounds_npy_path: str,
            image_folder: Optional[str] = None
    ) -> CameraData:
        """
        解析COLMAP和NeRF格式的相机数据，转换为3D Gaussian Splatting可用的格式

        参数:
        -----------
        cameras_bin_path : str
            COLMAP cameras.bin文件路径
        images_bin_path : str
            COLMAP images.bin文件路径
        points3D_bin_path : str
            COLMAP points3D.bin文件路径（可选，但建议提供）
        poses_bounds_npy_path : str
            NeRF格式的poses_bounds.npy文件路径
        image_folder : Optional[str]
            图像文件夹路径，用于构建完整的图像路径

        返回:
        --------
        CameraData : 包含所有解析后的相机数据和点云数据
        """

        print("开始解析相机数据...")

        # 1. 解析COLMAP二进制文件
        print(f"  1. 解析COLMAP cameras.bin: {cameras_bin_path}")
        cameras_data = read_colmap_bin_file(cameras_bin_path)
        print(f"    解析到 {len(cameras_data)} 个相机参数")

        print(f"  2. 解析COLMAP images.bin: {images_bin_path}")
        images_data = read_colmap_bin_file(images_bin_path)
        print(f"    解析到 {len(images_data)} 个图像参数")

        print(f"  3. 解析COLMAP points3D.bin: {points3D_bin_path}")
        points3D_data = read_colmap_bin_file(points3D_bin_path)
        print(f"    解析到 {len(points3D_data)} 个3D点")

        # 2. 解析NeRF格式的poses_bounds
        print(f"  4. 解析NeRF poses_bounds.npy: {poses_bounds_npy_path}")
        nerf_extrinsics, nerf_bounds, nerf_intrinsics = parse_poses_bounds(poses_bounds_npy_path)
        print(f"    解析到 {len(nerf_extrinsics)} 个NeRF位姿")

        # 3. 整理图像数据
        image_names = []
        image_paths = []
        intrinsics_list = []
        extrinsics_list = []
        image_sizes_list = []
        camera_types = []

        # 确保图像按ID排序
        image_ids = sorted(images_data.keys())

        print(f"  5. 整理 {len(image_ids)} 个图像数据")

        for idx, image_id in enumerate(image_ids):
            img_info = images_data[image_id]
            camera_id = img_info["camera_id"]

            # 图像名称和路径
            image_name = img_info["name"]
            image_names.append(image_name)

            if image_folder:
                image_path = str(Path(image_folder) / image_name)
            else:
                image_path = image_name
            image_paths.append(image_path)

            # 相机内参
            if camera_id in cameras_data:
                cam_info = cameras_data[camera_id]
                width = cam_info["width"]
                height = cam_info["height"]
                model_id = cam_info["model_id"]
                params = cam_info["params"]

                K, camera_type = create_intrinsic_matrix(params, width, height, model_id)
                intrinsics_list.append(K)
                image_sizes_list.append([height, width])
                camera_types.append(camera_type)
            else:
                # 如果没有找到相机参数，使用NeRF的内参（如果有）
                if nerf_intrinsics is not None and idx < len(nerf_intrinsics):
                    intrinsics_list.append(nerf_intrinsics[idx])
                    # 估计图像尺寸
                    fx = nerf_intrinsics[idx][0, 0]
                    w = int(2 * nerf_intrinsics[idx][0, 2])
                    h = int(2 * nerf_intrinsics[idx][1, 2])
                    image_sizes_list.append([h, w])
                    camera_types.append("NERF_ESTIMATED")
                else:
                    # 默认值
                    intrinsics_list.append(np.eye(3))
                    image_sizes_list.append([800, 800])  # 默认尺寸
                    camera_types.append("DEFAULT")

            # 相机外参
            # 优先使用COLMAP的外参
            qvec = img_info["qvec"]
            tvec = img_info["tvec"]

            # 检查四元数是否有效
            if np.any(np.isnan(qvec)) or np.any(np.isinf(qvec)):
                print(f"警告: 图像 {image_name} 的四元数包含NaN或Inf")
                qvec = np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数

            extrinsic = create_extrinsic_matrix(qvec, tvec)
            extrinsics_list.append(extrinsic)

        # 4. 转换为numpy数组
        intrinsics = np.array(intrinsics_list)  # [N, 3, 3]
        extrinsics = np.array(extrinsics_list)  # [N, 4, 4]
        image_sizes = np.array(image_sizes_list)  # [N, 2]

        # 5. 提取3D点云数据
        points3D = []
        colors = []

        for point_id in points3D_data:
            point_info = points3D_data[point_id]
            points3D.append(point_info["xyz"])
            colors.append(point_info["rgb"] / 255.0)  # 归一化到[0, 1]

        if points3D:
            points3D_array = np.array(points3D)
            colors_array = np.array(colors)
        else:
            points3D_array = None
            colors_array = None

        # 6. 构建元数据
        metadata = {
            "num_cameras": len(image_names),
            "num_points3D": len(points3D) if points3D else 0,
            "camera_models": list(set(camera_types)),
            "colmap_cameras": len(cameras_data),
            "colmap_images": len(images_data),
            "nerf_cameras": len(nerf_extrinsics),
            "image_extensions": list(set(Path(name).suffix for name in image_names))
        }

        # 7. 创建输出数据结构
        result = CameraData(
            image_names=image_names,
            image_paths=image_paths,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_sizes=image_sizes,
            camera_types=camera_types,
            points3D=points3D_array,
            colors=colors_array,
            bounds=nerf_bounds,
            metadata=metadata
        )

        print(f"解析完成！共 {len(image_names)} 个相机，{len(points3D) if points3D else 0} 个3D点")

        return result







