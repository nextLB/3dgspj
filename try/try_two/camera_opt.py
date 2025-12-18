#!/usr/bin/env python3
"""
优化版相机类 - RTX 3060专用
处理相机参数、投影变换、坐标系转换
"""

import torch
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """相机内参"""
    fx: float = 500.0  # 焦距 x
    fy: float = 500.0  # 焦距 y
    cx: float = 256.0  # 主点 x
    cy: float = 256.0  # 主点 y
    width: int = 512  # 图像宽度
    height: int = 512  # 图像高度
    near: float = 0.01  # 近平面
    far: float = 100.0  # 远平面

    def to_tensor(self, device: str = 'cuda') -> torch.Tensor:
        """转换为内参矩阵tensor"""
        K = torch.eye(3, device=device)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K

    def get_fov(self) -> Tuple[float, float]:
        """计算视野角度 (弧度)"""
        fov_x = 2 * math.atan(self.width / (2 * self.fx))
        fov_y = 2 * math.atan(self.height / (2 * self.fy))
        return fov_x, fov_y

    def clone(self) -> 'CameraIntrinsics':
        """克隆内参"""
        return CameraIntrinsics(
            fx=self.fx, fy=self.fy,
            cx=self.cx, cy=self.cy,
            width=self.width, height=self.height,
            near=self.near, far=self.far
        )


class OptimizedCamera:
    """优化版相机类"""

    def __init__(self,
                 world2cam: torch.Tensor,
                 K: torch.Tensor,
                 image: torch.Tensor,
                 H: int,
                 W: int,
                 image_name: str = "",
                 uid: int = 0,
                 is_train: bool = True,
                 device: str = 'cuda'):
        """
        初始化相机

        参数:
            world2cam: 世界到相机的变换矩阵 [4, 4] 或 [B, 4, 4]
            K: 内参矩阵 [3, 3] 或 [B, 3, 3]
            image: 原始图像 [C, H, W] 或 [B, C, H, W]
            H: 图像高度
            W: 图像宽度
            image_name: 图像名称
            uid: 相机ID
            is_train: 是否为训练相机
            device: 设备
        """
        self.device = torch.device(device)
        self.uid = uid
        self.image_name = image_name
        self.is_train = is_train
        self.H = H
        self.W = W

        # ==================== 修复: 确保矩阵形状正确 ====================
        # 变换矩阵 - 确保是二维 [4, 4]
        if not isinstance(world2cam, torch.Tensor):
            world2cam = torch.tensor(world2cam, dtype=torch.float32, device=self.device)
        else:
            world2cam = world2cam.to(self.device)

        # 移除批次维度
        if world2cam.dim() == 3:
            if world2cam.shape[0] == 1:
                world2cam = world2cam.squeeze(0)  # [4, 4]
            else:
                raise ValueError(f"world2cam应有形状[4,4]或[1,4,4]，但得到{world2cam.shape}")
        elif world2cam.dim() != 2:
            raise ValueError(f"world2cam应有形状[4,4]，但得到{world2cam.shape}")

        # 检查形状
        if world2cam.shape != (4, 4):
            raise ValueError(f"world2cam应有形状[4,4]，但得到{world2cam.shape}")

        self.world2cam = world2cam  # [4, 4]

        # ==================== 内参矩阵 - 确保是二维 [3, 3] ====================
        if not isinstance(K, torch.Tensor):
            K = torch.tensor(K, dtype=torch.float32, device=self.device)
        else:
            K = K.to(self.device)

        # 移除批次维度
        if K.dim() == 3:
            if K.shape[0] == 1:
                K = K.squeeze(0)  # [3, 3]
            else:
                raise ValueError(f"K应有形状[3,3]或[1,3,3]，但得到{K.shape}")
        elif K.dim() != 2:
            raise ValueError(f"K应有形状[3,3]，但得到{K.shape}")

        # 检查形状
        if K.shape != (3, 3):
            raise ValueError(f"K应有形状[3,3]，但得到{K.shape}")

        self.K = K  # [3, 3]

        # ==================== 图像数据 ====================
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32, device=self.device)
        else:
            image = image.to(self.device)

        # 图像可以是三维 [C, H, W] 或二维 [H, W]
        if image.dim() == 2:
            image = image.unsqueeze(0)  # [1, H, W]
        elif image.dim() == 3:
            pass  # 已经是 [C, H, W]
        else:
            raise ValueError(f"image应有形状[C,H,W]或[H,W]，但得到{image.shape}")

        self.original_image = image  # [C, H, W]

        # ==================== 计算相机参数 ====================
        self._compute_camera_parameters()

        # ==================== 预计算变换矩阵 ====================
        self._precompute_transforms()

        print(f"📷 相机 {uid} 初始化: {image_name} ({W}x{H})")

    def _compute_camera_parameters(self):
        """计算相机参数"""
        # 使用第一个相机（如果是批量）
        # 现在已经是二维，直接使用
        K0 = self.K  # [3, 3]
        w2c0 = self.world2cam  # [4, 4]

        # 内参
        self.fx = K0[0, 0].item()
        self.fy = K0[1, 1].item()
        self.cx = K0[0, 2].item()
        self.cy = K0[1, 2].item()

        # 计算视野角度
        self.FoVx = 2 * math.atan(self.W / (2 * self.fx))
        self.FoVy = 2 * math.atan(self.H / (2 * self.fy))

        # 相机位置 (世界坐标系)
        self.position = self._compute_camera_position(w2c0)

        # 相机方向
        self.R = w2c0[:3, :3].T  # 旋转矩阵 (相机到世界)
        self.T = w2c0[:3, 3]  # 平移向量

        # 前向向量
        self.forward = self.R[:, 2].cpu().numpy()

        # 上方向向量
        self.up = self.R[:, 1].cpu().numpy()

    def _compute_camera_position(self, w2c: torch.Tensor) -> np.ndarray:
        """计算相机在世界坐标系中的位置"""
        # 相机在世界坐标系中的位置: -R^T @ T
        R = w2c[:3, :3].T  # 转置得到相机到世界的旋转
        T = w2c[:3, 3]

        # P = -R^T @ T
        position = -torch.matmul(R.T, T).cpu().numpy()
        return position

    def _precompute_transforms(self):
        """预计算变换矩阵"""
        # 使用第一个相机（如果是批量）
        w2c0 = self.world2cam[0] if self.world2cam.dim() == 3 else self.world2cam

        # 世界视图变换矩阵 (世界到相机)
        self.world_view_transform = w2c0

        # 投影矩阵
        self.projection_matrix = self._compute_projection_matrix()

        # 完整变换矩阵
        self.full_proj_transform = torch.matmul(
            self.world_view_transform,
            self.projection_matrix
        )

        # 相机中心 (相机坐标系原点在世界坐标系中的位置)
        self.camera_center = torch.matmul(
            torch.inverse(self.world_view_transform),
            torch.tensor([0, 0, 0, 1], device=self.device, dtype=torch.float32)
        )[:3]

    def _compute_projection_matrix(self) -> torch.Tensor:
        """计算投影矩阵"""
        znear = 0.01
        zfar = 100.0

        tan_half_fovx = math.tan(self.FoVx * 0.5)
        tan_half_fovy = math.tan(self.FoVy * 0.5)

        # 透视投影矩阵
        P = torch.zeros((4, 4), device=self.device, dtype=torch.float32)

        P[0, 0] = 1.0 / (tan_half_fovx * self.W / 2)
        P[1, 1] = 1.0 / (tan_half_fovy * self.H / 2)
        P[2, 2] = -(zfar + znear) / (zfar - znear)
        P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
        P[3, 2] = -1.0

        return P

    # ==================== 公共方法 ====================

    def get_position(self) -> torch.Tensor:
        """获取相机位置 (世界坐标系)"""
        return torch.tensor(self.position, device=self.device, dtype=torch.float32)

    def get_view_direction(self) -> torch.Tensor:
        """获取观察方向"""
        return torch.tensor(self.forward, device=self.device, dtype=torch.float32)

    def get_up_vector(self) -> torch.Tensor:
        """获取上方向"""
        return torch.tensor(self.up, device=self.device, dtype=torch.float32)

    def get_intrinsics(self) -> CameraIntrinsics:
        """获取相机内参"""
        return CameraIntrinsics(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=self.W,
            height=self.H,
            near=0.01,
            far=100.0
        )

    def project_points(self, points_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将3D点投影到2D图像平面

        参数:
            points_3d: 3D点 [N, 3] 或 [B, N, 3]

        返回:
            points_2d: 2D点 [N, 2] 或 [B, N, 2]
            depths: 深度值 [N] 或 [B, N]
        """
        # 添加齐次坐标
        N = points_3d.shape[-2] if points_3d.dim() == 3 else points_3d.shape[0]

        if points_3d.dim() == 2:
            points_3d_h = torch.cat([
                points_3d,
                torch.ones((N, 1), device=self.device, dtype=points_3d.dtype)
            ], dim=1)  # [N, 4]

            # 变换到相机坐标系
            points_cam = torch.matmul(points_3d_h, self.world_view_transform.T)  # [N, 4]

            # 投影
            points_proj = torch.matmul(points_cam[:, :3], self.K[0].T)  # [N, 3]

        else:  # 批量处理
            B = points_3d.shape[0]
            points_3d_h = torch.cat([
                points_3d,
                torch.ones((B, N, 1), device=self.device, dtype=points_3d.dtype)
            ], dim=2)  # [B, N, 4]

            # 变换到相机坐标系
            points_cam = torch.matmul(points_3d_h, self.world2cam.transpose(1, 2))  # [B, N, 4]

            # 投影
            points_proj = torch.matmul(points_cam[:, :, :3], self.K.transpose(1, 2))  # [B, N, 3]

        # 归一化到像素坐标
        points_2d = points_proj[..., :2] / points_proj[..., 2:3].clamp(min=1e-8)

        # 深度值
        depths = points_cam[..., 2]

        return points_2d, depths

    def unproject_points(self, points_2d: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """
        将2D点反投影到3D空间

        参数:
            points_2d: 2D点 [N, 2] 或 [B, N, 2]
            depths: 深度值 [N] 或 [B, N]

        返回:
            points_3d: 3D点 [N, 3] 或 [B, N, 3]
        """
        if points_2d.dim() == 2:
            N = points_2d.shape[0]

            # 归一化坐标
            x_normalized = (points_2d[:, 0] - self.cx) / self.fx
            y_normalized = (points_2d[:, 1] - self.cy) / self.fy

            # 相机坐标系中的点
            points_cam = torch.stack([
                x_normalized * depths,
                y_normalized * depths,
                depths
            ], dim=1)  # [N, 3]

            # 变换到世界坐标系
            R_inv = self.world_view_transform[:3, :3].T
            T_inv = -torch.matmul(R_inv, self.world_view_transform[:3, 3])

            points_world = torch.matmul(points_cam, R_inv.T) + T_inv

        else:  # 批量处理
            B, N = points_2d.shape[:2]

            # 归一化坐标
            x_normalized = (points_2d[..., 0] - self.cx) / self.fx
            y_normalized = (points_2d[..., 1] - self.cy) / self.fy

            # 相机坐标系中的点
            points_cam = torch.stack([
                x_normalized * depths,
                y_normalized * depths,
                depths
            ], dim=-1)  # [B, N, 3]

            # 变换到世界坐标系
            R_inv = self.world2cam[:, :3, :3].transpose(1, 2)  # [B, 3, 3]
            T_inv = -torch.bmm(R_inv, self.world2cam[:, :3, 3:4])  # [B, 3, 1]

            points_world = torch.bmm(points_cam, R_inv.transpose(1, 2)) + T_inv.transpose(1, 2)

        return points_world

    def is_point_visible(self, point_3d: torch.Tensor,
                         margin: float = 0.1) -> bool:
        """
        检查3D点是否在相机视野内

        参数:
            point_3d: 3D点 [3]
            margin: 边界裕量 (百分比)

        返回:
            visible: 是否可见
        """
        # 投影到2D
        point_2d, depth = self.project_points(point_3d.unsqueeze(0))
        point_2d = point_2d.squeeze(0)
        depth = depth.squeeze(0)

        # 检查条件
        if depth <= 0:  # 在相机后面
            return False

        # 检查是否在图像范围内 (考虑裕量)
        margin_w = self.W * margin
        margin_h = self.H * margin

        visible = (point_2d[0] >= -margin_w) and (point_2d[0] < self.W + margin_w) and \
                  (point_2d[1] >= -margin_h) and (point_2d[1] < self.H + margin_h)

        return visible

    def get_frustum(self, depth_min: float = 0.1, depth_max: float = 10.0) -> torch.Tensor:
        """
        获取相机视锥体顶点

        参数:
            depth_min: 最小深度
            depth_max: 最大深度

        返回:
            vertices: 视锥体顶点 [8, 3] (世界坐标系)
        """
        # 近平面和远平面的角点 (相机坐标系)
        corners_camera = torch.tensor([
            # 近平面
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            # 远平面
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ], device=self.device, dtype=torch.float32)

        # 缩放深度
        corners_camera[:, 2] = (corners_camera[:, 2] + 1) / 2  # [-1, 1] -> [0, 1]
        corners_camera[:, 2] = corners_camera[:, 2] * (depth_max - depth_min) + depth_min

        # 缩放x,y到图像平面
        tan_half_fovx = math.tan(self.FoVx * 0.5)
        tan_half_fovy = math.tan(self.FoVy * 0.5)

        corners_camera[:, 0] *= corners_camera[:, 2] * tan_half_fovx
        corners_camera[:, 1] *= corners_camera[:, 2] * tan_half_fovy

        # 变换到世界坐标系
        R_inv = self.world_view_transform[:3, :3].T
        T_inv = -torch.matmul(R_inv, self.world_view_transform[:3, 3])

        vertices = torch.matmul(corners_camera, R_inv.T) + T_inv

        return vertices

    def clone(self) -> 'OptimizedCamera':
        """克隆相机"""
        return OptimizedCamera(
            world2cam=self.world2cam.clone(),
            K=self.K.clone(),
            image=self.original_image.clone(),
            H=self.H,
            W=self.W,
            image_name=self.image_name,
            uid=self.uid,
            is_train=self.is_train,
            device=self.device
        )

    def to(self, device: str) -> 'OptimizedCamera':
        """移动到指定设备"""
        if device == str(self.device):
            return self

        return OptimizedCamera(
            world2cam=self.world2cam.to(device),
            K=self.K.to(device),
            image=self.original_image.to(device),
            H=self.H,
            W=self.W,
            image_name=self.image_name,
            uid=self.uid,
            is_train=self.is_train,
            device=device
        )

    def get_info(self) -> Dict[str, Any]:
        """获取相机信息"""
        return {
            'uid': self.uid,
            'image_name': self.image_name,
            'is_train': self.is_train,
            'image_size': (self.W, self.H),
            'position': self.position.tolist(),
            'forward': self.forward.tolist(),
            'focal_length': (self.fx, self.fy),
            'principal_point': (self.cx, self.cy),
            'fov': (math.degrees(self.FoVx), math.degrees(self.FoVy))
        }

    def __repr__(self) -> str:
        """字符串表示"""
        info = self.get_info()
        return (f"OptimizedCamera(\n"
                f"  ID: {info['uid']}, 名称: {info['image_name']}\n"
                f"  尺寸: {info['image_size'][0]}x{info['image_size'][1]}\n"
                f"  位置: [{info['position'][0]:.2f}, {info['position'][1]:.2f}, {info['position'][2]:.2f}]\n"
                f"  焦距: ({info['focal_length'][0]:.1f}, {info['focal_length'][1]:.1f})\n"
                f"  视野: ({info['fov'][0]:.1f}°, {info['fov'][1]:.1f}°)\n"
                f")")


# ==================== 相机工具函数 ====================

def create_camera_from_pose(pose: np.ndarray,
                            intrinsics: CameraIntrinsics,
                            image: Optional[torch.Tensor] = None,
                            image_name: str = "",
                            uid: int = 0,
                            device: str = 'cuda') -> OptimizedCamera:
    """
    从位姿创建相机

    参数:
        pose: 相机位姿 [4, 4] (相机到世界)
        intrinsics: 相机内参
        image: 图像数据
        image_name: 图像名称
        uid: 相机ID
        device: 设备

    返回:
        camera: 相机对象
    """
    if image is None:
        image = torch.zeros((3, intrinsics.height, intrinsics.width),
                            device=device, dtype=torch.float32)

    # 相机到世界 -> 世界到相机
    world2cam = np.linalg.inv(pose)

    return OptimizedCamera(
        world2cam=torch.tensor(world2cam, dtype=torch.float32, device=device),
        K=intrinsics.to_tensor(device),
        image=image,
        H=intrinsics.height,
        W=intrinsics.width,
        image_name=image_name,
        uid=uid,
        device=device
    )


def create_virtual_camera(position: np.ndarray,
                          look_at: np.ndarray,
                          up: np.ndarray = np.array([0, -1, 0]),
                          intrinsics: Optional[CameraIntrinsics] = None,
                          image_name: str = "virtual",
                          uid: int = 0,
                          device: str = 'cuda') -> OptimizedCamera:
    """
    创建虚拟相机

    参数:
        position: 相机位置 [3]
        look_at: 观察点 [3]
        up: 上方向 [3]
        intrinsics: 相机内参
        image_name: 图像名称
        uid: 相机ID
        device: 设备

    返回:
        camera: 虚拟相机
    """
    if intrinsics is None:
        intrinsics = CameraIntrinsics(width=512, height=512)

    # 计算相机坐标系
    forward = look_at - position
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # 构建相机到世界变换
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = position

    # 创建相机
    return create_camera_from_pose(
        pose=c2w,
        intrinsics=intrinsics,
        image_name=image_name,
        uid=uid,
        device=device
    )


def create_spherical_camera_trajectory(center: np.ndarray,
                                       radius: float = 5.0,
                                       num_cameras: int = 60,
                                       intrinsics: Optional[CameraIntrinsics] = None,
                                       device: str = 'cuda') -> List[OptimizedCamera]:
    """
    创建球形相机轨迹

    参数:
        center: 场景中心 [3]
        radius: 球半径
        num_cameras: 相机数量
        intrinsics: 相机内参
        device: 设备

    返回:
        cameras: 相机列表
    """
    if intrinsics is None:
        intrinsics = CameraIntrinsics(width=512, height=512)

    cameras = []

    for i in range(num_cameras):
        # 球形坐标
        theta = 2 * np.pi * i / num_cameras
        phi = np.pi / 4 + np.pi / 8 * np.sin(2 * np.pi * i / num_cameras)

        # 计算相机位置
        x = center[0] + radius * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * np.cos(phi)
        z = center[2] + radius * np.sin(phi) * np.sin(theta)

        camera = create_virtual_camera(
            position=np.array([x, y, z]),
            look_at=center,
            up=np.array([0, -1, 0]),
            intrinsics=intrinsics,
            image_name=f"virtual_{i:04d}",
            uid=i,
            device=device
        )

        cameras.append(camera)

    return cameras


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("🧪 测试相机类...")

    # 测试内参类
    print("\n1. 测试相机内参:")
    intrinsics = CameraIntrinsics(
        fx=500.0, fy=500.0,
        cx=256.0, cy=256.0,
        width=512, height=512
    )

    K = intrinsics.to_tensor('cpu')
    print(f"   内参矩阵:\n{K}")

    fov_x, fov_y = intrinsics.get_fov()
    print(f"   视野角度: {math.degrees(fov_x):.1f}°, {math.degrees(fov_y):.1f}°")

    # 测试相机类
    print("\n2. 测试相机类:")

    # 创建测试相机
    test_world2cam = torch.eye(4)
    test_world2cam[2, 3] = -5.0  # 相机在z=-5位置

    test_K = intrinsics.to_tensor('cpu')
    test_image = torch.rand(3, 512, 512)

    camera = OptimizedCamera(
        world2cam=test_world2cam,
        K=test_K,
        image=test_image,
        H=512,
        W=512,
        image_name="test_camera",
        uid=0,
        device='cpu'
    )

    print(camera)

    # 测试投影
    print("\n3. 测试投影:")
    test_points = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    points_2d, depths = camera.project_points(test_points)
    print(f"   3D点: {test_points.tolist()}")
    print(f"   2D投影: {points_2d.tolist()}")
    print(f"   深度: {depths.tolist()}")

    # 测试可见性
    print("\n4. 测试可见性:")
    visible = camera.is_point_visible(torch.tensor([0, 0, 0]))
    print(f"   原点是否可见: {visible}")

    # 测试虚拟相机
    print("\n5. 测试虚拟相机:")
    virtual_camera = create_virtual_camera(
        position=np.array([3, 2, 1]),
        look_at=np.array([0, 0, 0]),
        intrinsics=intrinsics,
        device='cpu'
    )

    print(virtual_camera)

    print("\n✅ 相机类测试完成!")


