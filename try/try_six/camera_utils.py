#!/usr/bin/env python3
"""
相机工具函数 - 处理相机参数和坐标变换
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2


class CameraUtils:
    """相机工具类"""

    @staticmethod
    def get_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """获取投影矩阵 P = K [R|t]"""
        Rt = np.hstack([R, t.reshape(-1, 1)])
        return K @ Rt

    @staticmethod
    def intrinsic_from_fov(width: int, height: int, fov_x_deg: float = 60.0) -> np.ndarray:
        """从FOV计算内参矩阵"""
        fov_x = np.radians(fov_x_deg)
        focal_length = width / (2 * np.tan(fov_x / 2))

        K = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ])
        return K

    @staticmethod
    def create_camera_dict(cameras: Dict, images: Dict) -> List[Dict]:
        """从COLMAP数据创建相机字典列表"""
        camera_list = []

        for img_id, img_info in images.items():
            camera_id = img_info["camera_id"]
            camera_info = cameras.get(camera_id)

            if camera_info is None:
                continue

            # 创建相机字典
            camera_dict = {
                "image_id": img_id,
                "image_name": img_info["name"],
                "R": img_info["R"],  # 世界到相机
                "t": img_info["t"],  # 世界到相机
                "K": None,  # 需要从相机参数计算
                "width": camera_info["width"],
                "height": camera_info["height"],
                "model": camera_info["model"],
                "params": camera_info["params"]
            }

            # 根据相机模型计算K矩阵
            if camera_info["model"] == "SIMPLE_PINHOLE":
                f, cx, cy = camera_info["params"]
                camera_dict["K"] = np.array([
                    [f, 0, cx],
                    [0, f, cy],
                    [0, 0, 1]
                ])
            elif camera_info["model"] == "PINHOLE":
                fx, fy, cx, cy = camera_info["params"]
                camera_dict["K"] = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
            elif camera_info["model"] == "SIMPLE_RADIAL":
                f, cx, cy, k = camera_info["params"]
                camera_dict["K"] = np.array([
                    [f, 0, cx],
                    [0, f, cy],
                    [0, 0, 1]
                ])
                camera_dict["distortion"] = np.array([k, 0, 0, 0])  # 径向畸变
            else:
                # 对于其他模型，使用简单的内参估计
                camera_dict["K"] = CameraUtils.intrinsic_from_fov(
                    camera_info["width"], camera_info["height"]
                )

            camera_list.append(camera_dict)

        return camera_list

    @staticmethod
    def undistort_image(image: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
        """校正图像畸变"""
        h, w = image.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 0)
        undistorted = cv2.undistort(image, K, dist_coeffs, None, new_K)
        return undistorted

    @staticmethod
    def world_to_camera(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """世界坐标转相机坐标"""
        return (R @ points.T).T + t

    @staticmethod
    def camera_to_world(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """相机坐标转世界坐标"""
        return (R.T @ (points - t).T).T

    @staticmethod
    def project_points(points_3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """3D点投影到2D图像"""
        # 世界坐标 -> 相机坐标
        points_cam = CameraUtils.world_to_camera(points_3d, R, t)

        # 相机坐标 -> 归一化坐标
        points_norm = points_cam / points_cam[:, 2:3]

        # 归一化坐标 -> 像素坐标
        points_2d = (K @ points_norm.T).T

        return points_2d[:, :2]

    @staticmethod
    def normalize_cameras(camera_list: List[Dict]) -> List[Dict]:
        """归一化相机位姿，使场景中心在原点"""
        # 收集所有相机位置
        positions = []
        for cam in camera_list:
            # 相机位置在世界坐标系中: C = -R^T * t
            C = -cam["R"].T @ cam["t"]
            positions.append(C)

        positions = np.array(positions)

        # 计算中心点
        center = np.mean(positions, axis=0)

        # 计算包围盒大小
        max_distance = np.max(np.linalg.norm(positions - center, axis=1))

        # 归一化尺度
        scale = 1.0 / max_distance if max_distance > 0 else 1.0

        # 应用变换
        normalized_cameras = []
        for cam in camera_list:
            # 平移相机位置
            t_normalized = cam["t"] + cam["R"] @ center

            # 缩放
            t_normalized *= scale

            normalized_cam = cam.copy()
            normalized_cam["t"] = t_normalized
            normalized_cam["scale"] = scale
            normalized_cam["center"] = center

            normalized_cameras.append(normalized_cam)

        return normalized_cameras

    @staticmethod
    def create_view_frustum(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                            near: float = 0.1, far: float = 10.0) -> np.ndarray:
        """创建视锥体顶点"""
        # 相机在世界坐标系中的位置
        C = -R.T @ t

        # 计算视锥体角点（在相机坐标系中）
        h, w = int(K[1, 2] * 2), int(K[0, 2] * 2)
        fx, fy = K[0, 0], K[1, 1]

        # 近平面角点
        x_near = near * (w / 2 - K[0, 2]) / fx
        y_near = near * (h / 2 - K[1, 2]) / fy

        # 远平面角点
        x_far = far * (w / 2 - K[0, 2]) / fx
        y_far = far * (h / 2 - K[1, 2]) / fy

        # 相机坐标系中的角点
        corners_cam = np.array([
            # 近平面
            [-x_near, -y_near, near],
            [x_near, -y_near, near],
            [x_near, y_near, near],
            [-x_near, y_near, near],
            # 远平面
            [-x_far, -y_far, far],
            [x_far, -y_far, far],
            [x_far, y_far, far],
            [-x_far, y_far, far]
        ])

        # 转换到世界坐标系
        corners_world = (R.T @ corners_cam.T).T + C

        return corners_world