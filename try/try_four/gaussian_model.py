import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class GaussianModel(nn.Module):
    """
    3D高斯泼溅的核心模型。
    管理一组可优化的3D高斯参数：位置、不透明度、缩放、旋转和球谐系数。
    """

    def __init__(self, sh_degree: int = 3):
        super().__init__()
        self.sh_degree = sh_degree
        self.max_sh_degree = sh_degree

        # 高斯参数（将在初始化后分配）
        self._xyz = None  # 位置 [N, 3]
        self._opacity = None  # 不透明度 [N, 1]
        self._scaling = None  # 缩放（对数空间）[N, 3]
        self._rotation = None  # 旋转（四元数）[N, 4]
        self._features_dc = None  # 球谐系数（0阶，即基础颜色）[N, 3]
        self._features_rest = None  # 球谐系数（高阶）[N, (sh_degree+1)^2 - 1, 3]

        # 优化器将直接设置这些参数
        self.optimizer = None

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)  # 从对数空间转换回来

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def get_features(self):
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    def create_from_pcd(self, pcd: torch.Tensor, spatial_lr_scale: float = 1.0):
        """
        从点云初始化高斯模型。
        pcd: [N, 3] 点云坐标
        spatial_lr_scale: 用于调整位置学习率的空间尺度
        """
        N = pcd.shape[0]

        # 1. 位置：直接使用点云坐标
        self._xyz = nn.Parameter(pcd.clone().requires_grad_(True))

        # 2. 不透明度：初始化为中等不透明值（经过sigmoid后约为0.1）
        opacities = torch.logit(0.1 * torch.ones(N, 1, dtype=torch.float, device=pcd.device))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        # 3. 缩放：初始化为较小的对数缩放（对应约exp(-3.0)=0.05的物理尺寸）
        scales = torch.log(torch.ones(N, 3, dtype=torch.float, device=pcd.device) * 0.05)
        self._scaling = nn.Parameter(scales.requires_grad_(True))

        # 4. 旋转：初始化为单位四元数 [1, 0, 0, 0]
        rots = torch.zeros(N, 4, dtype=torch.float, device=pcd.device)
        rots[:, 0] = 1.0  # 实部为1
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        # 5. 球谐系数：0阶（基础颜色）初始化为随机颜色，高阶初始化为0
        shs = torch.zeros((N, (self.sh_degree + 1) ** 2 - 1, 3), dtype=torch.float, device=pcd.device)

        # 从随机角度观看点云时猜测的基础颜色（此处简化：随机颜色）
        rgb = torch.rand(N, 3, dtype=torch.float, device=pcd.device) * 0.5 + 0.3  # 在[0.3, 0.8]之间
        self._features_dc = nn.Parameter(rgb.unsqueeze(1).requires_grad_(True))  # [N, 1, 3]
        self._features_rest = nn.Parameter(shs.requires_grad_(True))

        # 为位置优化设置空间学习率缩放
        self.spatial_lr_scale = spatial_lr_scale

        print(f"Initialized Gaussian Model with {N} Gaussians")

    def setup_optimizer(self, cfg):
        """为高斯参数配置优化器，采用不同的学习率。"""
        lr_params = [
            {'params': [self._xyz], 'lr': cfg.position_lr_init * self.spatial_lr_scale, 'name': 'xyz'},
            {'params': [self._opacity], 'lr': 0.05, 'name': 'opacity'},
            {'params': [self._scaling], 'lr': 0.005, 'name': 'scaling'},
            {'params': [self._rotation], 'lr': 0.001, 'name': 'rotation'},
            {'params': [self._features_dc], 'lr': 0.0025, 'name': 'f_dc'},
            {'params': [self._features_rest], 'lr': 0.0025 / 20.0, 'name': 'f_rest'},
        ]

        self.optimizer = torch.optim.Adam(lr_params, lr=0.0, eps=1e-15)

        # 位置学习率调度器（余弦衰减）
        self.xyz_scheduler_args = {
            'lr_init': cfg.position_lr_init * self.spatial_lr_scale,
            'lr_final': cfg.position_lr_final * self.spatial_lr_scale,
            'lr_delay_mult': cfg.position_lr_delay_mult,
            'max_steps': cfg.position_lr_max_steps
        }

    def update_learning_rate(self, iteration):
        """根据迭代次数衰减位置学习率。"""
        if self.optimizer is None:
            return

        # 仅更新位置参数的学习率（余弦衰减）
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr_init = self.xyz_scheduler_args['lr_init']
                lr_final = self.xyz_scheduler_args['lr_final']
                lr_delay_mult = self.xyz_scheduler_args['lr_delay_mult']
                max_steps = self.xyz_scheduler_args['max_steps']

                # 延迟期的线性预热
                delay_steps = int(max_steps * lr_delay_mult)
                if iteration < delay_steps:
                    t = iteration / delay_steps
                    lr = lr_init * t
                else:
                    # 余弦衰减
                    t = (iteration - delay_steps) / (max_steps - delay_steps)
                    t = min(t, 1.0)
                    lr = lr_final + 0.5 * (lr_init - lr_final) * (1 + np.cos(t * np.pi))

                param_group['lr'] = lr
                break

    def densify_and_prune(self, grad_threshold: float, scene_extent: float, iteration: int,
                          densify_until: int, percent_dense: float):
        """
        自适应密度控制：克隆梯度大的高斯，分裂大高斯，修剪不透明的高斯。
        """
        if iteration > densify_until:
            return

        grads = self._xyz.grad.norm(dim=-1)  # 每个高斯的梯度幅度
        # 1. 选择梯度大的高斯进行克隆（在场景边界内）
        grad_condition = grads >= grad_threshold
        spatial_condition = (self._xyz.abs().max(dim=-1).values <= scene_extent)
        selected = torch.logical_and(grad_condition, spatial_condition)

        if selected.sum() > 0:
            # 克隆选中的高斯：复制所有属性
            self.clone_gaussians(selected)

        # 2. 选择大高斯进行分裂（缩放过大）
        scale = self.get_scaling
        scale_condition = torch.max(scale, dim=-1).values > 0.01 * scene_extent  # 阈值
        selected = torch.logical_and(scale_condition, spatial_condition)

        if selected.sum() > 0:
            self.split_gaussians(selected, scene_extent)

        # 3. 修剪不透明度过低的高斯
        opacity = torch.sigmoid(self._opacity)
        prune_mask = opacity.squeeze() < 0.005  # 不透明度阈值
        if prune_mask.sum() > 0:
            self.prune_gaussians(prune_mask)

        # 4. 随机修剪以控制总数
        total = self._xyz.shape[0]
        target_num = int(total * (1 - percent_dense))
        if total > target_num:
            # 按不透明度排序，修剪最透明的
            _, indices = torch.sort(opacity.squeeze())
            prune_num = total - target_num
            prune_mask = torch.zeros(total, dtype=torch.bool, device=self._xyz.device)
            prune_mask[indices[:prune_num]] = True
            self.prune_gaussians(prune_mask)

    def clone_gaussians(self, selected):
        """克隆选中的高斯。"""
        new_xyz = self._xyz[selected].detach()
        new_opacity = self._opacity[selected].detach()
        new_scaling = self._scaling[selected].detach()
        new_rotation = self._rotation[selected].detach()
        new_features_dc = self._features_dc[selected].detach()
        new_features_rest = self._features_rest[selected].detach()

        # 连接到现有参数
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, new_opacity], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rotation], dim=0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, new_features_dc], dim=0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, new_features_rest], dim=0))

        # 重新创建优化器（简化处理，实际实现需要更精细）
        print(f"Cloned {selected.sum().item()} Gaussians. Total now: {self._xyz.shape[0]}")

    def split_gaussians(self, selected, scene_extent):
        """分裂选中的大高斯为两个较小的高斯。"""
        N = selected.sum().item()
        if N == 0:
            return

        # 获取选中的高斯
        stds = self.get_scaling[selected].detach()
        means = torch.zeros((N, 3), device=self._xyz.device)

        # 在缩放的标准差范围内采样偏移
        samples = torch.normal(mean=means, std=stds)
        # 创建两个新位置：原位置 +/- 偏移
        new_xyz1 = self._xyz[selected].detach() + samples * 0.2  # 缩小偏移
        new_xyz2 = self._xyz[selected].detach() - samples * 0.2

        # 新缩放：将原缩放减半（在对数空间）
        new_scaling = self._scaling[selected].detach() - np.log(2.0)

        # 复制其他属性
        new_opacity = self._opacity[selected].detach()
        new_rotation = self._rotation[selected].detach()
        new_features_dc = self._features_dc[selected].detach()
        new_features_rest = self._features_rest[selected].detach()

        # 连接到现有参数
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz1, new_xyz2], dim=0))
        self._opacity = nn.Parameter(torch.cat([self._opacity, new_opacity, new_opacity], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling, new_scaling], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rotation, new_rotation], dim=0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, new_features_dc, new_features_dc], dim=0))
        self._features_rest = nn.Parameter(
            torch.cat([self._features_rest, new_features_rest, new_features_rest], dim=0))

        print(f"Split {N} Gaussians. Total now: {self._xyz.shape[0]}")

    def prune_gaussians(self, prune_mask):
        """修剪掩码选中的高斯。"""
        keep_mask = ~prune_mask

        self._xyz = nn.Parameter(self._xyz[keep_mask])
        self._opacity = nn.Parameter(self._opacity[keep_mask])
        self._scaling = nn.Parameter(self._scaling[keep_mask])
        self._rotation = nn.Parameter(self._rotation[keep_mask])
        self._features_dc = nn.Parameter(self._features_dc[keep_mask])
        self._features_rest = nn.Parameter(self._features_rest[keep_mask])

        print(f"Pruned {prune_mask.sum().item()} Gaussians. Total now: {self._xyz.shape[0]}")

    def save_ply(self, path):
        """将高斯模型保存为PLY文件（用于可视化）。"""
        from plyfile import PlyData, PlyElement
        import numpy as np

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)  # 法线占位符
        f_dc = self._features_dc.detach().cpu().numpy().squeeze()  # [N, 3]

        # 计算不透明度（sigmoid后）
        opacity = torch.sigmoid(self._opacity).detach().cpu().numpy().squeeze()

        # 计算缩放（指数后）
        scale = self.get_scaling.detach().cpu().numpy()

        # 计算旋转（四元数）
        rotation = self.get_rotation.detach().cpu().numpy()

        # 构建顶点数据
        vertex_data = np.empty(xyz.shape[0], dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ])

        vertex_data['x'] = xyz[:, 0].astype('f4')
        vertex_data['y'] = xyz[:, 1].astype('f4')
        vertex_data['z'] = xyz[:, 2].astype('f4')
        vertex_data['nx'] = normals[:, 0].astype('f4')
        vertex_data['ny'] = normals[:, 1].astype('f4')
        vertex_data['nz'] = normals[:, 2].astype('f4')
        vertex_data['f_dc_0'] = f_dc[:, 0].astype('f4')
        vertex_data['f_dc_1'] = f_dc[:, 1].astype('f4')
        vertex_data['f_dc_2'] = f_dc[:, 2].astype('f4')
        vertex_data['opacity'] = opacity.astype('f4')
        vertex_data['scale_0'] = scale[:, 0].astype('f4')
        vertex_data['scale_1'] = scale[:, 1].astype('f4')
        vertex_data['scale_2'] = scale[:, 2].astype('f4')
        vertex_data['rot_0'] = rotation[:, 0].astype('f4')
        vertex_data['rot_1'] = rotation[:, 1].astype('f4')
        vertex_data['rot_2'] = rotation[:, 2].astype('f4')
        vertex_data['rot_3'] = rotation[:, 3].astype('f4')

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(path)
        print(f"Saved Gaussian model to {path}")


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaussianModel(sh_degree=3)
    # 创建模拟点云
    pcd = torch.randn(1000, 3, device=device)
    model.create_from_pcd(pcd)
    print(f"Number of Gaussians: {model._xyz.shape[0]}")
    print(f"Features DC shape: {model._features_dc.shape}")
