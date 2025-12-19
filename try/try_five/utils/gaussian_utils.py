import torch
import torch.nn as nn
import numpy as np
import math


def build_rotation(r):
    """
    从四元数构建旋转矩阵
    """
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def strip_lowerdiag(L):
    """
    提取下三角矩阵的非零元素
    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]

    return uncertainty


def strip_symmetric(sym):
    """
    提取对称矩阵的上三角部分
    """
    return strip_lowerdiag(sym)


def build_scaling_rotation(s, r):
    """
    构建缩放和旋转矩阵
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    """
    从缩放和旋转构建协方差矩阵
    """
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)

    return symm


def inverse_sigmoid(x):
    """
    sigmoid函数的反函数
    """
    return torch.log(x / (1 - x))


def build_covariance_3d_from_7d(params):
    """
    从7维参数构建3D协方差矩阵
    """
    scaling = params[:, :3]
    rotation = params[:, 3:7]

    # 构建缩放矩阵
    S = torch.zeros((scaling.shape[0], 3, 3), device=scaling.device)
    S[:, 0, 0] = scaling[:, 0]
    S[:, 1, 1] = scaling[:, 1]
    S[:, 2, 2] = scaling[:, 2]

    # 构建旋转矩阵
    R = build_rotation(rotation)

    # 构建协方差矩阵
    M = S @ R.transpose(1, 2)
    covariance = M @ M.transpose(1, 2)

    return covariance


def get_exponental_coefficient(x):
    """
    获取指数系数
    """
    return torch.exp(-0.5 * x)


def compute_sh_color(shs, viewdir, sh_degree=3):
    """
    计算球谐函数颜色
    """
    # 简化版的球谐函数计算
    # 实际实现需要完整的球谐函数基函数
    if sh_degree == 0:
        return shs[:, 0:3]

    # 这里实现简化的球谐函数计算
    # 完整的实现需要SH基函数的计算
    result = shs[:, 0:3]  # DC项

    if sh_degree > 0:
        x, y, z = viewdir[:, 0], viewdir[:, 1], viewdir[:, 2]

        # 一阶球谐函数
        result += shs[:, 3:4] * y
        result += shs[:, 4:5] * z
        result += shs[:, 5:6] * x

        if sh_degree > 1:
            # 二阶球谐函数（简化版）
            xy = x * y
            yz = y * z
            xz = x * z
            xx = x * x
            yy = y * y
            zz = z * z

            result += shs[:, 6:7] * xy
            result += shs[:, 7:8] * yz
            result += shs[:, 8:9] * (2 * zz - xx - yy)
            result += shs[:, 9:10] * xz
            result += shs[:, 10:11] * (xx - yy)

            if sh_degree > 2:
                # 三阶球谐函数（简化版）
                result += shs[:, 11:12] * (3 * xx - yy) * y
                result += shs[:, 12:13] * xy * z
                result += shs[:, 13:14] * (4 * zz - xx - yy) * y
                result += shs[:, 14:15] * (4 * zz - 3 * xx - 3 * yy) * z
                result += shs[:, 15:16] * (4 * zz - xx - yy) * x
                result += shs[:, 16:17] * (xx - yy) * z
                result += shs[:, 17:18] * (xx - 3 * yy) * x

    return torch.sigmoid(result)


class GaussianModel:
    """
    3D高斯模型
    """

    def __init__(self, sh_degree=3):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0

        # 高斯参数
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        # 优化器设置
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        # 优化配置
        self.optimizer = None
        self.spatial_lr_scale = 1

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        """
        获取协方差矩阵
        """
        return build_covariance_from_scaling_rotation(
            self.get_scaling, scaling_modifier, self.get_rotation)

    def activation(self, x):
        """
        激活函数
        """
        return torch.exp(x)

    def scaling_activation(self, x):
        """
        缩放激活函数
        """
        return torch.exp(x)

    def rotation_activation(self, x):
        """
        旋转激活函数
        """
        return x / torch.norm(x, dim=1, keepdim=True)

    def opacity_activation(self, x):
        """
        不透明度激活函数
        """
        return torch.sigmoid(x)

    def create_from_pcd(self, pcd, spatial_lr_scale=1):
        """
        从点云创建高斯模型
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 初始化参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(fused_color.requires_grad_(True))
        self._features_rest = nn.Parameter(
            torch.zeros((fused_color.shape[0], 3 * (self.max_sh_degree + 1) ** 2 - 3)).cuda().requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.log(torch.ones((fused_point_cloud.shape[0], 3), device="cuda") * 0.01).requires_grad_(True))
        self._rotation = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 4), device="cuda").requires_grad_(True))
        self._rotation.data[:, 0] = 1.0
        self._opacity = nn.Parameter(inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")).requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        """
        设置训练参数
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = {
            'init_lr': training_args.position_lr_init * self.spatial_lr_scale,
            'final_lr': training_args.position_lr_final * self.spatial_lr_scale,
            'delay_mult': training_args.position_lr_delay_mult,
            'max_steps': training_args.position_lr_max_steps
        }

    def update_learning_rate(self, iteration):
        """
        更新学习率
        """
        # 更新位置学习率
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args['init_lr']
                lr = max(
                    lr * self.xyz_scheduler_args['delay_mult'] ** (iteration / self.xyz_scheduler_args['max_steps']),
                    self.xyz_scheduler_args['final_lr'])
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        """
        构建属性列表
        """
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        # 所有颜色
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))

        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))

        l.append('opacity')

        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))

        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        return l

    def save_ply(self, path):
        """
        保存为PLY文件
        """
        import plyfile
        import sklearn

        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = plyfile.PlyElement.describe(elements, 'vertex')

        plyfile.PlyData([el]).write(path)

    def load_ply(self, path):
        """
        从PLY文件加载
        """
        import plyfile

        plydata = plyfile.PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree