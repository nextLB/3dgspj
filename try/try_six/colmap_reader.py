#!/usr/bin/env python3
"""
COLMAP数据读取器 - 读取cameras.bin, images.bin, points3D.bin
"""
import numpy as np
import struct
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle


class ColmapReader:
    """读取COLMAP二进制文件"""

    @staticmethod
    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
        """读取并解包二进制数据"""
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    @staticmethod
    def read_cameras_bin(path):
        """读取cameras.bin文件"""
        cameras = {}
        with open(path, "rb") as fid:
            num_cameras = ColmapReader.read_next_bytes(fid, 8, "Q")[0]

            for _ in range(num_cameras):
                camera_properties = ColmapReader.read_next_bytes(
                    fid, num_bytes=24, format_char_sequence="iiQQ"
                )
                camera_id = camera_properties[0]
                model_id = camera_properties[1]
                width = camera_properties[2]
                height = camera_properties[3]
                num_params = camera_properties[4]

                params = list(ColmapReader.read_next_bytes(
                    fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
                ))

                # 模型ID到模型名称的映射
                model_name = {
                    1: "SIMPLE_PINHOLE",
                    2: "PINHOLE",
                    3: "SIMPLE_RADIAL",
                    4: "RADIAL",
                    5: "OPENCV",
                    6: "OPENCV_FISHEYE",
                    7: "FULL_OPENCV",
                    8: "FOV",
                    9: "SIMPLE_RADIAL_FISHEYE",
                    10: "RADIAL_FISHEYE",
                    11: "THIN_PRISM_FISHEYE"
                }.get(model_id, f"UNKNOWN_{model_id}")

                cameras[camera_id] = {
                    "model": model_name,
                    "width": width,
                    "height": height,
                    "params": np.array(params)
                }

        return cameras

    @staticmethod
    def read_images_bin(path):
        """读取images.bin文件"""
        images = {}
        with open(path, "rb") as fid:
            num_images = ColmapReader.read_next_bytes(fid, 8, "Q")[0]

            for _ in range(num_images):
                image_properties = ColmapReader.read_next_bytes(
                    fid, num_bytes=64, format_char_sequence="idddddddi"
                )
                image_id = image_properties[0]
                qw, qx, qy, qz = image_properties[1:5]  # 四元数 (w, x, y, z)
                tx, ty, tz = image_properties[5:8]  # 平移
                camera_id = image_properties[8]

                # 读取图像名称
                image_name = ""
                current_char = fid.read(1)
                while current_char != b'\x00':
                    image_name += current_char.decode("utf-8")
                    current_char = fid.read(1)

                # 读取特征点数量
                num_points2D = ColmapReader.read_next_bytes(fid, 8, "Q")[0]

                # 读取2D点坐标和对应的3D点ID
                xys = np.zeros((num_points2D, 2))
                point3D_ids = np.full((num_points2D,), -1, dtype=np.int64)

                for i in range(num_points2D):
                    x, y = ColmapReader.read_next_bytes(fid, 16, "dd")
                    point3D_id = ColmapReader.read_next_bytes(fid, 8, "Q")[0]
                    xys[i] = (x, y)
                    point3D_ids[i] = point3D_id

                # 构造旋转矩阵
                R = ColmapReader.qvec2rotmat(np.array([qw, qx, qy, qz]))
                t = np.array([tx, ty, tz])

                images[image_id] = {
                    "name": image_name,
                    "R": R,
                    "t": t,
                    "camera_id": camera_id,
                    "xys": xys,
                    "point3D_ids": point3D_ids
                }

        return images

    @staticmethod
    def read_points3D_bin(path):
        """读取points3D.bin文件"""
        points3D = {}
        with open(path, "rb") as fid:
            num_points = ColmapReader.read_next_bytes(fid, 8, "Q")[0]

            for _ in range(num_points):
                point_properties = ColmapReader.read_next_bytes(
                    fid, num_bytes=32, format_char_sequence="QdddBBB"
                )
                point_id = point_properties[0]
                x, y, z = point_properties[1:4]
                r, g, b = point_properties[4:7]
                error = point_properties[7]

                # 读取轨迹长度
                track_length = ColmapReader.read_next_bytes(fid, 8, "Q")[0]

                # 读取轨迹（图像ID, 2D点索引）
                track = []
                for _ in range(track_length):
                    image_id, point2D_idx = ColmapReader.read_next_bytes(fid, 16, "QQ")
                    track.append((image_id, point2D_idx))

                points3D[point_id] = {
                    "xyz": np.array([x, y, z]),
                    "rgb": np.array([r, g, b]),
                    "error": error,
                    "track": track
                }

        return points3D

    @staticmethod
    def qvec2rotmat(qvec):
        """四元数转换为旋转矩阵"""
        qvec = qvec / np.linalg.norm(qvec)
        w, x, y, z = qvec
        return np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
        ])

    @staticmethod
    def rotmat2qvec(R):
        """旋转矩阵转换为四元数"""
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
        K = np.array([
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
        ]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec = -qvec
        return qvec

    @classmethod
    def read_colmap_sparse(cls, sparse_dir):
        """读取COLMAP稀疏重建结果"""
        sparse_path = Path(sparse_dir)

        cameras = cls.read_cameras_bin(sparse_path / "cameras.bin")
        images = cls.read_images_bin(sparse_path / "images.bin")
        points3D = cls.read_points3D_bin(sparse_path / "points3D.bin")

        return cameras, images, points3D

    @classmethod
    def load_poses_bounds_npy(cls, npy_path):
        """加载poses_bounds.npy文件"""
        data = np.load(npy_path)
        poses = data[:, :-2].reshape([-1, 3, 5])  # 最后两列是bounds
        bounds = data[:, -2:]  # near, far

        # 转换格式：从NeRF格式转换为标准相机位姿
        hwf = poses[:, :, 4]
        poses = poses[:, :, :4]

        return poses, hwf, bounds