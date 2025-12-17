import os
import sys
import argparse
import subprocess
import json
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import shutil


class DataPreprocessor:
    """数据预处理模块"""

    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_colmap(self, images_dir):
        """运行COLMAP进行稀疏重建"""
        print(f"Running COLMAP on {images_dir}")

        # 创建工作目录
        colmap_dir = self.output_dir / "sparse"
        colmap_dir.mkdir(exist_ok=True)

        # 数据库路径
        database_path = colmap_dir / "database.db"

        # 1. 特征提取
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", "OPENCV",
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.domain_size_pooling", "1"
        ]
        subprocess.run(cmd, check=True)

        # 2. 特征匹配
        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(database_path)
        ]
        subprocess.run(cmd, check=True)

        # 3. 稀疏重建
        sparse_dir = colmap_dir / "0"
        sparse_dir.mkdir(exist_ok=True)

        cmd = [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir)
        ]
        subprocess.run(cmd, check=True)

        # 4. 模型转换
        cmd = [
            "colmap", "model_converter",
            "--input_path", str(sparse_dir),
            "--output_path", str(sparse_dir / "sparse.ply"),
            "--output_type", "PLY"
        ]
        subprocess.run(cmd, check=True)

        return sparse_dir

    def convert_to_3dgs_format(self, images_dir, sparse_dir):
        """将COLMAP输出转换为3DGS格式"""
        print("Converting to 3DGS format...")

        # 创建输出目录
        gs_dir = self.output_dir / "3dgs_ready"
        gs_dir.mkdir(exist_ok=True)

        # 加载相机参数
        cameras_file = sparse_dir / "cameras.txt"
        images_file = sparse_dir / "images.txt"

        # 解析相机参数
        cameras = self._parse_cameras(cameras_file)
        images_data = self._parse_images(images_file)

        # 创建相机JSON文件
        camera_json = []

        for img_id, data in images_data.items():
            # 获取图像路径
            img_name = data["name"]
            img_path = images_dir / img_name

            if not img_path.exists():
                continue

            # 读取图像
            img = Image.open(img_path)
            w, h = img.size

            # 获取相机参数
            cam_id = data["camera_id"]
            if cam_id not in cameras:
                continue

            cam = cameras[cam_id]

            # 构建相机条目
            camera_entry = {
                "id": img_id,
                "img_name": img_name,
                "width": w,
                "height": h,
                "position": data["position"].tolist(),
                "rotation": data["rotation"].tolist(),
                "fx": cam["params"][0],
                "fy": cam["params"][1],
                "cx": cam["params"][2],
                "cy": cam["params"][3],
                "k1": cam["params"][4] if len(cam["params"]) > 4 else 0,
                "k2": cam["params"][5] if len(cam["params"]) > 5 else 0,
                "p1": cam["params"][6] if len(cam["params"]) > 6 else 0,
                "p2": cam["params"][7] if len(cam["params"]) > 7 else 0,
            }
            camera_json.append(camera_entry)

            # 复制图像到输出目录
            shutil.copy(img_path, gs_dir / img_name)

        # 保存相机JSON
        with open(gs_dir / "cameras.json", "w") as f:
            json.dump(camera_json, f, indent=2)

        # 复制点云
        pointcloud_file = sparse_dir / "sparse.ply"
        if pointcloud_file.exists():
            shutil.copy(pointcloud_file, gs_dir / "pointcloud.ply")

        return gs_dir

    def _parse_cameras(self, cameras_file):
        """解析COLMAP cameras.txt文件"""
        cameras = {}
        with open(cameras_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }

        return cameras

    def _parse_images(self, images_file):
        """解析COLMAP images.txt文件"""
        images = {}
        with open(images_file, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            if lines[i].startswith("#"):
                i += 1
                continue

            # 图像行
            img_parts = lines[i].strip().split()
            if len(img_parts) < 10:
                i += 1
                continue

            img_id = int(img_parts[0])
            qw, qx, qy, qz = map(float, img_parts[1:5])
            tx, ty, tz = map(float, img_parts[5:8])
            cam_id = int(img_parts[8])
            img_name = img_parts[9]

            # 转换为旋转矩阵
            rotation = self._quaternion_to_rotation_matrix(qw, qx, qy, qz)
            position = np.array([tx, ty, tz])

            images[img_id] = {
                "name": img_name,
                "camera_id": cam_id,
                "rotation": rotation,
                "position": position
            }

            i += 2  # 跳过点行

        return images

    def _quaternion_to_rotation_matrix(self, qw, qx, qy, qz):
        """四元数转换为旋转矩阵"""
        q = np.array([qw, qx, qy, qz])
        q = q / np.linalg.norm(q)

        R = np.array([
            [1 - 2 * q[2] ** 2 - 2 * q[3] ** 2, 2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[1] * q[3] + 2 * q[0] * q[2]],
            [2 * q[1] * q[2] + 2 * q[0] * q[3], 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2, 2 * q[2] * q[3] - 2 * q[0] * q[1]],
            [2 * q[1] * q[3] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1], 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2]
        ])

        return R


def main():
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument("--data_dir", type=str, required=True, help="输入数据目录")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="输出目录")
    parser.add_argument("--skip_colmap", action="store_true", help="跳过COLMAP步骤")

    args = parser.parse_args()

    # 初始化预处理器
    preprocessor = DataPreprocessor(args.data_dir, args.output_dir)

    # 假设图像在data_dir/images目录下
    images_dir = Path(args.data_dir) / "images"

    if not images_dir.exists():
        print(f"错误: {images_dir} 不存在")
        return

    # 运行COLMAP
    if not args.skip_colmap:
        sparse_dir = preprocessor.run_colmap(images_dir)
    else:
        sparse_dir = Path(args.output_dir) / "sparse" / "0"

    # 转换为3DGS格式
    gs_dir = preprocessor.convert_to_3dgs_format(images_dir, sparse_dir)

    print(f"预处理完成! 数据已保存到: {gs_dir}")


if __name__ == "__main__":
    main()