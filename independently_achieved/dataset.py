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
from dataclasses import dataclass, field
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
    intrinsics: np.ndarray  # [N, 3, 3]
    extrinsics: np.ndarray  # [N, 4, 4]
    image_sizes: np.ndarray  # [N, 2] (height, width)
    camera_types: List[str]
    points_3D: Optional[np.ndarray] = None  # 修改为points_3D
    colors: Optional[np.ndarray] = None
    bounds: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def readColmapBinFile(filename: str) -> Dict[int, Any]:
    """
    读取COLMAP二进制文件（修正版本）
    COLMAP二进制文件使用小端字节序(little-endian)

    参数:
    -----------
    filename : str
        二进制文件路径

    返回:
    --------
    Dict[int, Any]: 解析后的数据字典
    """
    data = {}

    try:
        with open(filename, "rb") as fid:
            # COLMAP使用小端字节序
            num_entries_bytes = fid.read(8)
            if len(num_entries_bytes) < 8:
                print(f"警告: {filename} 文件太小或为空")
                return data

            # 小端字节序：'<Q' 表示无符号long long
            num_entries = struct.unpack("<Q", num_entries_bytes)[0]
            print(f"  找到 {num_entries} 个条目")

            file_type = ""
            if "cameras" in filename.lower():
                file_type = "cameras"
            elif "images" in filename.lower():
                file_type = "images"
            elif "points3d" in filename.lower() or "points" in filename.lower():
                file_type = "points3D"

            for _ in range(num_entries):
                # 读取条目ID (32位无符号整数)
                id_bytes = fid.read(4)
                if len(id_bytes) < 4:
                    print(f"警告: 读取条目ID时文件结束")
                    break
                entry_id = struct.unpack("<I", id_bytes)[0]

                if file_type == "cameras":
                    # 相机格式: model_id(4), width(8), height(8), params
                    model_bytes = fid.read(4)
                    if len(model_bytes) < 4:
                        break
                    model_id = struct.unpack("<i", model_bytes)[0]

                    width_bytes = fid.read(8)
                    height_bytes = fid.read(8)
                    if len(width_bytes) < 8 or len(height_bytes) < 8:
                        break
                    width = struct.unpack("<Q", width_bytes)[0]
                    height = struct.unpack("<Q", height_bytes)[0]

                    num_params_bytes = fid.read(8)
                    if len(num_params_bytes) < 8:
                        break
                    num_params = struct.unpack("<Q", num_params_bytes)[0]

                    # 读取参数 (双精度浮点数)
                    params_bytes = fid.read(8 * num_params)
                    if len(params_bytes) < 8 * num_params:
                        break

                    # '<d' * num_params 表示小端双精度
                    fmt = "<" + "d" * num_params
                    params = struct.unpack(fmt, params_bytes)

                    data[entry_id] = {
                        "model_id": model_id,
                        "width": width,
                        "height": height,
                        "params": np.array(params, dtype=np.float64)
                    }

                elif file_type == "images":
                    # 图像格式: qvec(4*8), tvec(3*8), camera_id(4), name, num_points2D(8), points2D...
                    # 四元数 (4个双精度浮点数)
                    qvec_bytes = fid.read(8 * 4)
                    if len(qvec_bytes) < 8 * 4:
                        break
                    qvec = struct.unpack("<dddd", qvec_bytes)

                    # 平移向量 (3个双精度浮点数)
                    tvec_bytes = fid.read(8 * 3)
                    if len(tvec_bytes) < 8 * 3:
                        break
                    tvec = struct.unpack("<ddd", tvec_bytes)

                    # 相机ID (32位无符号整数)
                    camera_id_bytes = fid.read(4)
                    if len(camera_id_bytes) < 4:
                        break
                    camera_id = struct.unpack("<I", camera_id_bytes)[0]

                    # 图像名称 (以null结尾的字符串)
                    name_chars = []
                    while True:
                        char = fid.read(1)
                        if not char:
                            break
                        if char == b'\x00':
                            break
                        name_chars.append(char.decode('utf-8', errors='ignore'))
                    name = ''.join(name_chars)

                    # 2D点数量 (64位无符号整数)
                    num_points2D_bytes = fid.read(8)
                    if len(num_points2D_bytes) < 8:
                        break
                    num_points2D = struct.unpack("<Q", num_points2D_bytes)[0]

                    # 读取2D点
                    points2D = []
                    for _ in range(num_points2D):
                        point_bytes = fid.read(8 * 2 + 8)  # x(8), y(8), point3D_id(8)
                        if len(point_bytes) < 8 * 3:
                            break
                        x, y, point3D_id = struct.unpack("<ddq", point_bytes)
                        points2D.append((x, y, point3D_id))

                    data[entry_id] = {
                        "qvec": np.array(qvec, dtype=np.float64),
                        "tvec": np.array(tvec, dtype=np.float64),
                        "camera_id": camera_id,
                        "name": name,
                        "points2D": np.array(points2D) if points2D else np.array([])
                    }

                elif file_type == "points3D":
                    # 3D点格式: xyz(3*8), rgb(3*1), error(8), track_length(8), track...
                    # 位置 (3个双精度浮点数)
                    xyz_bytes = fid.read(8 * 3)
                    if len(xyz_bytes) < 8 * 3:
                        break
                    xyz = struct.unpack("<ddd", xyz_bytes)

                    # RGB颜色 (3个无符号字节)
                    rgb_bytes = fid.read(3)
                    if len(rgb_bytes) < 3:
                        break
                    # 使用 unsigned char (B) 读取
                    rgb = struct.unpack("<BBB", rgb_bytes)

                    # 重投影误差 (双精度浮点数)
                    error_bytes = fid.read(8)
                    if len(error_bytes) < 8:
                        break
                    error = struct.unpack("<d", error_bytes)[0]

                    # 轨迹长度 (64位无符号整数)
                    track_length_bytes = fid.read(8)
                    if len(track_length_bytes) < 8:
                        break
                    track_length = struct.unpack("<Q", track_length_bytes)[0]

                    # 读取轨迹 (多个 (image_id, point2D_idx) 对)
                    track = []
                    for _ in range(track_length):
                        track_bytes = fid.read(4 + 4)  # image_id(4), point2D_idx(4)
                        if len(track_bytes) < 8:
                            break
                        image_id, point2D_idx = struct.unpack("<II", track_bytes)
                        track.append((image_id, point2D_idx))

                    data[entry_id] = {
                        "xyz": np.array(xyz, dtype=np.float64),
                        "rgb": np.array(rgb, dtype=np.uint8),
                        "error": error,
                        "track": track,
                        "track_length": track_length
                    }

    except Exception as e:
        print(f"读取文件 {filename} 时出错: {e}")
        import traceback
        traceback.print_exc()

    return data


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """将四元数转换为旋转矩阵"""
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec

    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    ], dtype=np.float64)


def createExtrinsicMatrix(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """创建4x4外参矩阵（世界到相机）"""
    R = qvec2rotmat(qvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T


def createIntrinsicMatrix(params: np.ndarray, width: int, height: int, model_id: int) -> Tuple[np.ndarray, str]:
    """根据相机模型创建内参矩阵"""
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

    # 默认值
    fx = fy = 0.5 * max(width, height)
    cx = width / 2.0
    cy = height / 2.0

    try:
        if model_id == 0:  # SIMPLE_PINHOLE
            if len(params) >= 3:
                fx = params[0]
                cx = params[1]
                cy = params[2]
                fy = fx
        elif model_id == 1:  # PINHOLE
            if len(params) >= 4:
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
        elif model_id == 2:  # SIMPLE_RADIAL
            if len(params) >= 4:
                f = params[0]
                cx = params[1]
                cy = params[2]
                fx = fy = f
        elif model_id == 3:  # RADIAL
            if len(params) >= 5:
                f = params[0]
                cx = params[1]
                cy = params[2]
                fx = fy = f
        elif model_id == 4:  # OPENCV
            if len(params) >= 8:
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
        elif model_id == 5:  # OPENCV_FISHEYE
            if len(params) >= 8:
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
        elif model_id == 6:  # FULL_OPENCV
            if len(params) >= 12:
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
    except Exception as e:
        print(f"创建内参矩阵时出错 (model_id={model_id}): {e}")

    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    return K, model_name


def parsePosesBounds(poses_bounds_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """解析poses_bounds.npy文件（NeRF格式）"""
    try:
        data = np.load(poses_bounds_path)
        n_cameras = data.shape[0]

        print(f"   poses_bounds形状: {data.shape}")

        if data.shape[1] == 17:
            poses = data[:, :12].reshape(-1, 3, 4)
            bounds = data[:, 12:14]
            hwf = data[:, 14:]
        elif data.shape[1] == 15:
            poses = data[:, :12].reshape(-1, 3, 4)
            bounds = data[:, 12:14]
            hwf = None
        elif data.shape[1] == 14:
            poses = data[:, :12].reshape(-1, 3, 4)
            bounds = data[:, 12:14]
            hwf = None
        else:
            raise ValueError(f"未知poses_bounds格式: {data.shape}")

        extrinsics = np.zeros((n_cameras, 4, 4), dtype=np.float64)
        extrinsics[:, :3, :4] = poses
        extrinsics[:, 3, 3] = 1.0

        intrinsics = None
        if hwf is not None:
            intrinsics = []
            for i in range(n_cameras):
                h, w, f = hwf[i]
                K = np.eye(3, dtype=np.float64)
                K[0, 0] = f
                K[1, 1] = f
                K[0, 2] = w / 2.0
                K[1, 2] = h / 2.0
                intrinsics.append(K)
            intrinsics = np.array(intrinsics)

        return extrinsics, bounds, intrinsics

    except Exception as e:
        print(f"解析poses_bounds文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return np.array([]), np.array([]), None


def validateAndNormalizeQuaternion(qvec: np.ndarray) -> np.ndarray:
    """验证并归一化四元数"""
    if np.any(np.isnan(qvec)) or np.any(np.isinf(qvec)):
        print(f"警告: 四元数包含NaN或Inf，使用单位四元数")
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    norm = np.linalg.norm(qvec)
    if norm < 1e-8:
        print(f"警告: 四元数范数为零，使用单位四元数")
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    return qvec / norm


def getImageFolderPath(base_path: str) -> Optional[str]:
    """尝试自动找到图像文件夹路径"""
    base_dir = Path(base_path).parent.parent.parent  # 通常向上3级到数据集根目录
    possible_folders = ["images", "image", "imgs", "rgb"]

    for folder in possible_folders:
        images_dir = base_dir / folder
        if images_dir.exists():
            return str(images_dir)

    # 如果没有找到标准文件夹，尝试查找包含.jpg/.png的文件夹
    for item in base_dir.iterdir():
        if item.is_dir():
            # 检查文件夹中是否有图像文件
            img_files = list(item.glob("*.jpg")) + list(item.glob("*.JPG")) + \
                        list(item.glob("*.png")) + list(item.glob("*.PNG"))
            if img_files:
                return str(item)

    return None







class MipNeRF360Dataset:
    def __init__(self):
        self.name = 'MipNeRF360Dataset'
        self.imageDataRootPath = ''
        self.allSortedImagePath = []
        self.allSortedImageData = []
        self.resolution = 0
        self.verbose = True


        # 加载图像数据
        self.load_images_data()

        # 加载相机与拍摄时的数据等等
        self.cameraPointsData = self.parse_camera_data_for_gaussian_splatting(
            os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, CAMERA_INFO_DIR, "cameras.bin"),
            os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, CAMERA_INFO_DIR, "images.bin"),
            os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, CAMERA_INFO_DIR, "points3D.bin"),
            os.path.join(BASE_DATASET_PATH, V2_360, CLASS_NAME, "poses_bounds.npy"),
            self.imageDataRootPath
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

    def log(self, message: str):
        if self.verbose:
            print(message)

    def parse_camera_data_for_gaussian_splatting(
            self,
            cameras_bin_path: str,
            images_bin_path: str,
            points3D_bin_path: str,
            poses_bounds_npy_path: str,
            image_folder: Optional[str]
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
            COLMAP points3D.bin文件路径
        poses_bounds_npy_path : str
            NeRF格式的poses_bounds.npy文件路径
        image_folder : Optional[str]
            图像文件夹路径，如果为None则尝试自动查找

        返回:
        --------
        CameraData : 包含所有解析后的相机数据和点云数据
        """

        self.log("开始解析相机数据...")

        # 1. 解析COLMAP二进制文件
        self.log(f"  1. 解析COLMAP cameras.bin: {cameras_bin_path}")
        cameras_data = readColmapBinFile(cameras_bin_path)
        self.log(f"    解析到 {len(cameras_data)} 个相机参数")

        if cameras_data:
            for cam_id, cam_info in list(cameras_data.items())[:3]:  # 打印前3个相机信息
                self.log(f"    相机 {cam_id}: {cam_info['width']}x{cam_info['height']}, "
                         f"模型ID: {cam_info['model_id']}, 参数: {cam_info['params'][:4]}...")

        self.log(f"  2. 解析COLMAP images.bin: {images_bin_path}")
        images_data = readColmapBinFile(images_bin_path)
        self.log(f"    解析到 {len(images_data)} 个图像参数")

        self.log(f"  3. 解析COLMAP points3D.bin: {points3D_bin_path}")
        points3D_data = readColmapBinFile(points3D_bin_path)
        self.log(f"    解析到 {len(points3D_data)} 个3D点")

        # 2. 解析NeRF格式的poses_bounds
        self.log(f"  4. 解析NeRF poses_bounds.npy: {poses_bounds_npy_path}")
        nerf_extrinsics, nerf_bounds, nerf_intrinsics = parsePosesBounds(poses_bounds_npy_path)
        self.log(f"    解析到 {len(nerf_extrinsics)} 个NeRF位姿")

        # 3. 如果未提供图像文件夹，尝试自动查找
        if image_folder is None:
            self.log("  5. 尝试自动查找图像文件夹...")
            image_folder = getImageFolderPath(cameras_bin_path)
            if image_folder:
                self.log(f"    找到图像文件夹: {image_folder}")
            else:
                self.log("    警告: 未找到图像文件夹，使用相对路径")

        # 4. 整理图像数据
        image_names = []
        image_paths = []
        intrinsics_list = []
        extrinsics_list = []
        image_sizes_list = []
        camera_types = []

        # 确保图像按ID排序
        image_ids = sorted(images_data.keys())

        self.log(f"  6. 整理 {len(image_ids)} 个图像数据")

        for idx, image_id in enumerate(image_ids):
            img_info = images_data[image_id]
            camera_id = img_info["camera_id"]

            # 图像名称和路径
            image_name = img_info["name"]
            image_names.append(image_name)

            if image_folder:
                # 尝试查找图像文件的完整路径
                image_path = self.findImageFile(image_folder, image_name)
                if image_path is None:
                    self.log(f"    警告: 未找到图像文件 {image_name}，使用相对路径")
                    image_path = image_name
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

                K, camera_type = createIntrinsicMatrix(params, width, height, model_id)
                intrinsics_list.append(K)
                image_sizes_list.append([height, width])
                camera_types.append(camera_type)
            else:
                # 如果没有找到相机参数，使用NeRF的内参（如果有）
                if nerf_intrinsics is not None and idx < len(nerf_intrinsics):
                    intrinsics_list.append(nerf_intrinsics[idx])
                    # 估计图像尺寸
                    K = nerf_intrinsics[idx]
                    # 从内参矩阵获取cx, cy并计算宽高
                    if K.shape == (3, 3):
                        w = int(2 * K[0, 2])
                        h = int(2 * K[1, 2])
                    else:
                        w = h = 800  # 默认值
                    image_sizes_list.append([h, w])
                    camera_types.append("NERF_ESTIMATED")
                else:
                    # 使用第一个找到的相机参数或默认值
                    if cameras_data:
                        first_cam = next(iter(cameras_data.values()))
                        width = first_cam["width"]
                        height = first_cam["height"]
                        K = np.eye(3, dtype=np.float64)
                        K[0, 0] = 0.5 * max(width, height)
                        K[1, 1] = K[0, 0]
                        K[0, 2] = width / 2.0
                        K[1, 2] = height / 2.0
                        camera_types.append("DEFAULT_FROM_FIRST_CAM")
                    else:
                        K = np.eye(3, dtype=np.float64)
                        K[0, 0] = K[1, 1] = 800.0
                        K[0, 2] = K[1, 2] = 400.0
                        image_sizes_list.append([800, 800])
                        camera_types.append("DEFAULT")
                    intrinsics_list.append(K)

            # 相机外参
            qvec = validateAndNormalizeQuaternion(img_info["qvec"])
            tvec = img_info["tvec"]

            extrinsic = createExtrinsicMatrix(qvec, tvec)
            extrinsics_list.append(extrinsic)

        # 5. 转换为numpy数组
        intrinsics = np.array(intrinsics_list, dtype=np.float64)  # [N, 3, 3]
        extrinsics = np.array(extrinsics_list, dtype=np.float64)  # [N, 4, 4]
        image_sizes = np.array(image_sizes_list, dtype=np.int32)  # [N, 2]

        # 6. 提取3D点云数据
        points_3D_list = []
        colors_list = []

        if points3D_data:
            for point_id in points3D_data:
                point_info = points3D_data[point_id]
                points_3D_list.append(point_info["xyz"])
                # RGB是0-255的整数，转换为0-1的浮点数
                colors_list.append(point_info["rgb"] / 255.0)

            if points_3D_list:
                points_3D_array = np.array(points_3D_list, dtype=np.float64)
                colors_array = np.array(colors_list, dtype=np.float32)  # 通常颜色用float32
            else:
                points_3D_array = None
                colors_array = None
        else:
            points_3D_array = None
            colors_array = None
            self.log("    警告: 没有解析到3D点云数据")

        # 7. 验证数据完整性
        self.validateData(intrinsics, extrinsics, image_sizes, points_3D_array, colors_array)

        # 8. 构建元数据
        metadata = {
            "num_cameras": len(image_names),
            "num_points3D": len(points_3D_list) if points_3D_list else 0,
            "camera_models": list(set(camera_types)),
            "colmap_cameras": len(cameras_data),
            "colmap_images": len(images_data),
            "colmap_points3D": len(points3D_data),
            "nerf_cameras": len(nerf_extrinsics),
            "image_extensions": list(set(Path(name).suffix for name in image_names)),
            "image_size_range": {
                "height_min": int(image_sizes[:, 0].min()),
                "height_max": int(image_sizes[:, 0].max()),
                "width_min": int(image_sizes[:, 1].min()),
                "width_max": int(image_sizes[:, 1].max())
            }
        }

        # 9. 创建输出数据结构
        result = CameraData(
            image_names=image_names,
            image_paths=image_paths,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_sizes=image_sizes,
            camera_types=camera_types,
            points_3D=points_3D_array,  # 修改为points_3D
            colors=colors_array,
            bounds=nerf_bounds,
            metadata=metadata
        )

        self.log(f"解析完成！共 {len(image_names)} 个相机，{len(points_3D_list) if points_3D_list else 0} 个3D点")
        self.printDataSummary(result)

        return result

    def findImageFile(self, image_folder: str, image_name: str) -> Optional[str]:
        """查找图像文件，支持多种常见扩展名"""
        base_name = Path(image_name).stem
        possible_extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]

        for ext in possible_extensions:
            image_path = Path(image_folder) / f"{base_name}{ext}"
            if image_path.exists():
                return str(image_path)

        # 如果没找到，尝试使用原始名称
        image_path = Path(image_folder) / image_name
        if image_path.exists():
            return str(image_path)

        return None

    def validateData(self, intrinsics, extrinsics, image_sizes, points_3D, colors):
        """验证数据的完整性和合理性"""
        self.log("  7. 验证数据完整性...")

        # 检查NaN和Inf
        if np.any(np.isnan(intrinsics)) or np.any(np.isinf(intrinsics)):
            self.log("    警告: 内参矩阵包含NaN或Inf值")

        if np.any(np.isnan(extrinsics)) or np.any(np.isinf(extrinsics)):
            self.log("    警告: 外参矩阵包含NaN或Inf值")

        # 检查内参矩阵的对角线元素是否为正数
        for i, K in enumerate(intrinsics):
            if K[0, 0] <= 0 or K[1, 1] <= 0:
                self.log(f"    警告: 相机 {i} 的焦距为负或零: fx={K[0, 0]}, fy={K[1, 1]}")

        # 检查图像尺寸是否合理
        for i, (h, w) in enumerate(image_sizes):
            if h <= 0 or w <= 0:
                self.log(f"    警告: 相机 {i} 的图像尺寸不合理: {h}x{w}")

        if points_3D is not None:
            if np.any(np.isnan(points_3D)) or np.any(np.isinf(points_3D)):
                self.log("    警告: 3D点包含NaN或Inf值")

            # 检查点云范围
            if len(points_3D) > 0:
                bbox_min = points_3D.min(axis=0)
                bbox_max = points_3D.max(axis=0)
                self.log(f"    3D点云边界框: [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}] "
                         f"到 [{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]")

    def printDataSummary(self, camera_data: CameraData):
        """打印数据摘要"""
        self.log("\n" + "=" * 60)
        self.log("数据摘要:")
        self.log(f"  相机数量: {len(camera_data.image_names)}")
        self.log(f"  图像尺寸范围: {camera_data.metadata['image_size_range']['height_min']}x"
                 f"{camera_data.metadata['image_size_range']['width_min']} 到 "
                 f"{camera_data.metadata['image_size_range']['height_max']}x"
                 f"{camera_data.metadata['image_size_range']['width_max']}")
        self.log(f"  3D点数量: {camera_data.metadata['num_points3D']}")
        self.log(f"  相机模型: {', '.join(camera_data.metadata['camera_models'])}")

        if camera_data.points_3D is not None:
            self.log(f"  3D点形状: {camera_data.points_3D.shape}")
            self.log(f"  颜色形状: {camera_data.colors.shape}")

        if camera_data.bounds is not None:
            self.log(f"  边界数量: {len(camera_data.bounds)}")
            self.log(f"  近平面范围: {camera_data.bounds[:, 0].min():.2f} 到 {camera_data.bounds[:, 0].max():.2f}")
            self.log(f"  远平面范围: {camera_data.bounds[:, 1].min():.2f} 到 {camera_data.bounds[:, 1].max():.2f}")

        self.log("=" * 60)





