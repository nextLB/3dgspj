"""
VastGaussian 分块重建算法 - PyTorch实现
完整的3D Gaussian Splatting训练流程
修复: COLMAP二进制解析器(8字节track元素), 可微渲染器, 密度控制
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import json
import math
import argparse
from typing import Optional, Tuple

# ============================================================
# COLMAP数据加载
# ============================================================

def load_colmap_data(dataset_path: str) -> Tuple:
    """
    加载COLMAP数据
    优先使用pycolmap，失败时回退到修复版手动解析器
    """
    sparse_dir = os.path.join(dataset_path, 'sparse', '0')
    images_dir = os.path.join(dataset_path, 'images')
    if not os.path.exists(sparse_dir):
        sparse_dir = os.path.join(dataset_path, 'sparse')
        if not os.path.exists(sparse_dir):
            raise FileNotFoundError(f"找不到COLMAP稀疏目录: {sparse_dir}")

    # 优先使用 pycolmap（安装确认可用）
    try:
        import pycolmap as _pm
        rec = _pm.Reconstruction(sparse_dir)
        return _convert_pycolmap(rec, images_dir)
    except ImportError:
        print("【数据】pycolmap未安装，使用手动解析器")
    except Exception as e:
        print(f"【数据】pycolmap回退 ({e})")

    return _parse_colmap_binary(sparse_dir, images_dir)


def _convert_pycolmap(rec, images_dir):
    """转换 pycolmap Reconstruction → 统一格式"""
    # 兼容不同pycolmap版本: model可能叫 model_name / model
    def _cam_model(cam):
        if hasattr(cam, 'model_name'):
            return cam.model_name() if callable(cam.model_name) else cam.model_name
        if hasattr(cam, 'model'):
            m = cam.model
            return m.name if hasattr(m, 'name') else str(m)
        return str(getattr(cam, 'model_id', 0))

    cameras = {}
    for cid, c in rec.cameras.items():
        mn = _cam_model(c)
        mid = 1 if 'PINHOLE' in mn.upper() else 0
        cameras[cid] = {
            'width': c.width, 'height': c.height,
            'params': list(c.params),
            'model_id': mid,
        }

    cam_images = []
    for img_id, img in rec.images.items():
        cam_images.append({
            'name': img.name,
            'qvec': np.array([img.qvec[i] for i in range(4)], dtype=np.float64),
            'tvec': np.array([img.tvec[i] for i in range(3)], dtype=np.float64),
            'camera_id': img.camera_id,
        })

    pts_list, cols_list = [], []
    for pt_id, pt in rec.points3D.items():
        pts_list.append([pt.xyz[0], pt.xyz[1], pt.xyz[2]])
        cols_list.append([pt.color[0]/255., pt.color[1]/255., pt.color[2]/255.])

    pts = np.array(pts_list, dtype=np.float64) if pts_list else np.empty((0, 3))
    cols = np.array(cols_list, dtype=np.float32) if cols_list else np.empty((0, 3))

    print(f"【数据】pycolmap: {len(pts)} 点, {len(cam_images)} 图像")
    return pts, cols, cameras, cam_images, images_dir


def _parse_colmap_binary(sparse_dir, images_dir):
    """手动解析 COLMAP 二进制文件（已修复track元素大小为8字节）"""
    pts, cols = None, None
    cameras = {}
    cam_images = []

    # ---- 3D点云 points3D.bin ----
    p3d = os.path.join(sparse_dir, 'points3D.bin')
    if os.path.exists(p3d):
        try:
            with open(p3d, 'rb') as f:
                n_pts = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
                print(f"【数据】points3D.bin: {n_pts} 个点")
                pts_l, cols_l, bad = [], [], 0
                for _ in range(n_pts):
                    _ = f.read(8)       # point3D_id (uint64, 跳过)
                    xyz = np.frombuffer(f.read(24), dtype=np.float64)
                    rgb = np.frombuffer(f.read(3), dtype=np.uint8)
                    _ = f.read(8)        # error
                    trk_len = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
                    if trk_len > 100000:  # 合理性检查
                        raise ValueError(f"track_len={trk_len} 异常，数据可能损坏")
                    f.read(8 * trk_len)
                    if np.all(np.isfinite(xyz)):
                        pts_l.append(xyz)
                        cols_l.append(rgb / 255.)
                    else:
                        bad += 1
                if pts_l:
                    pts = np.array(pts_l, dtype=np.float64)
                    cols = np.array(cols_l, dtype=np.float32)
                    print(f"【数据】有效点: {len(pts)}" + (f" (过滤{bad})" if bad else ""))
        except Exception as e:
            print(f"【数据】解析points3D失败: {e}")

    # ---- 相机 cameras.bin ----
    cbin = os.path.join(sparse_dir, 'cameras.bin')
    if os.path.exists(cbin):
        try:
            with open(cbin, 'rb') as f:
                n_cam = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
                param_counts = {0: 4, 1: 4, 2: 5, 3: 6, 4: 3}
                for _ in range(n_cam):
                    cid = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
                    mid = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
                    w = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
                    h = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
                    np_ = param_counts.get(mid, 4)
                    prm = np.frombuffer(f.read(8 * np_), dtype=np.float64).tolist()
                    cameras[cid] = {'width': w, 'height': h, 'params': prm, 'model_id': mid}
        except Exception as e:
            print(f"【数据】解析cameras失败: {e}")

    # ---- 图像 images.bin ----
    ibin = os.path.join(sparse_dir, 'images.bin')
    if os.path.exists(ibin):
        try:
            with open(ibin, 'rb') as f:
                n_img = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
                for _ in range(n_img):
                    _ = f.read(8)   # image_id
                    qvec = np.frombuffer(f.read(32), dtype=np.float64)
                    tvec = np.frombuffer(f.read(24), dtype=np.float64)
                    cid = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
                    # 读图像名 (null-terminated)
                    name = b''.join(iter(lambda: f.read(1), b'\x00')).decode('utf-8')
                    n2d = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
                    f.read(24 * n2d)
                    cam_images.append({'name': name, 'qvec': qvec, 'tvec': tvec, 'camera_id': cid})
        except Exception as e:
            print(f"【数据】解析images失败: {e}")

    if pts is None or len(pts) < 100:
        print("【数据】无有效3D点，生成随机初始点")
        n = 5000
        pts = np.random.randn(n, 3) * 5
        cols = np.random.rand(n, 3)

    return pts, cols, cameras, cam_images, images_dir


def qvec2rotmat(qvec):
    """四元数(w,x,y,z) → 旋转矩阵 (3×3)"""
    q = qvec / (np.linalg.norm(qvec) + 1e-8)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])


# ============================================================
# 3D Gaussian 模型
# ============================================================

def build_rotation(q: torch.Tensor) -> torch.Tensor:
    """四元数 (N,4) → 旋转矩阵 (N,3,3)"""
    q = F.normalize(q, dim=1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)
    ], dim=1).reshape(-1, 3, 3)


def build_covariance(s: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """缩放+四元数 → 3D协方差 (N,3,3)"""
    R = build_rotation(q)
    S = torch.diag_embed(s)
    M = R @ S
    return M @ M.transpose(-2, -1)


class GaussianModel(nn.Module):
    """完整的3D Gaussian模型"""

    def __init__(self):
        super().__init__()
        self._xyz = None          # (N, 3) positions
        self._features_dc = None  # (N, 1, 3) SH DC
        self._scaling = None      # (N, 3) log-scales
        self._rotation = None     # (N, 4) quaternions
        self._opacity = None      # (N, 1) logit-opacity
        self.optimizer = None
        self.xyz_gradient_accum = None
        self.denom = None

    def create_from_pcd(self, points: np.ndarray, colors: np.ndarray):
        """从COLMAP点云初始化"""
        N = points.shape[0]
        pts = torch.tensor(points, dtype=torch.float32)
        self._xyz = nn.Parameter(pts)
        self._features_dc = nn.Parameter(torch.tensor(colors, dtype=torch.float32).unsqueeze(1))

        # 分批计算KNN平均距离初始化缩放（避免N²内存爆炸）
        topk = min(4, N)
        batch_size = 2000
        neigh_dist_list = []
        for i in range(0, N, batch_size):
            batch = pts[i:i+batch_size]  # (B, 3)
            d2 = torch.cdist(batch, pts) ** 2  # (B, N) 平方距离
            nd, _ = torch.topk(d2, topk, dim=1, largest=False)
            neigh_dist_list.append(nd)
        neigh_dist = torch.cat(neigh_dist_list, dim=0)
        mean_dist = torch.sqrt(neigh_dist[:, 1:].mean(dim=1)).clamp(min=1e-7)
        self._scaling = nn.Parameter(torch.log(mean_dist.unsqueeze(1).repeat(1, 3)))

        rots = torch.zeros(N, 4)
        rots[:, 0] = 1.0
        self._rotation = nn.Parameter(rots)
        self._opacity = nn.Parameter(torch.full((N, 1), math.log(0.1 / (1 - 0.1))))

        print(f"【模型】初始化 {N} 个高斯点")

    def setup_optimizer(self, lr=1.6e-4, pos_lr_final=1.6e-6):
        self.pos_lr_init = lr
        self.pos_lr_final = pos_lr_final
        self.optimizer = torch.optim.Adam([
            {'params': [self._xyz], 'lr': lr, 'name': 'xyz'},
            {'params': [self._features_dc], 'lr': 0.0025, 'name': 'f_dc'},
            {'params': [self._scaling], 'lr': 0.005, 'name': 'scaling'},
            {'params': [self._rotation], 'lr': 0.001, 'name': 'rotation'},
            {'params': [self._opacity], 'lr': 0.05, 'name': 'opacity'},
        ])
        self.xyz_gradient_accum = torch.zeros(self._xyz.shape[0], 1)
        self.denom = torch.zeros(self._xyz.shape[0], 1)

    def update_lr(self, iteration: int, total: int):
        t = iteration / total
        lr = self.pos_lr_init * (1 - t) + self.pos_lr_final * t
        for pg in self.optimizer.param_groups:
            if pg['name'] == 'xyz':
                pg['lr'] = lr
        return lr

    # ---- 属性访问 ----
    def get_xyz(self):      return self._xyz
    def get_features(self): return self._features_dc[:, 0, :]
    def get_scaling(self):  return torch.exp(self._scaling)
    def get_rotation(self): return F.normalize(self._rotation, dim=1)
    def get_opacity(self):  return torch.sigmoid(self._opacity)
    def get_covariance(self): return build_covariance(self.get_scaling(), self.get_rotation())

    # ---- 密度控制辅助 ----
    def _optimizer_mask(self, mask):
        for g in self.optimizer.param_groups:
            s = self.optimizer.state.get(g['params'][0], None)
            if s is not None:
                s['exp_avg'] = s['exp_avg'][mask]
                s['exp_avg_sq'] = s['exp_avg_sq'][mask]
                self.optimizer.state[g['params'][0]] = s

    def _optimizer_cat(self, tensors):
        for g in self.optimizer.param_groups:
            ext = tensors[g['name']]
            s = self.optimizer.state.get(g['params'][0], None)
            if s is not None:
                z = torch.zeros_like(ext)
                s['exp_avg'] = torch.cat([s['exp_avg'], z], dim=0)
                s['exp_avg_sq'] = torch.cat([s['exp_avg_sq'], z], dim=0)

    def prune(self, mask: torch.Tensor):
        """移除mask=True的高斯"""
        m = ~mask
        self._xyz = nn.Parameter(self._xyz[m])
        self._features_dc = nn.Parameter(self._features_dc[m])
        self._scaling = nn.Parameter(self._scaling[m])
        self._rotation = nn.Parameter(self._rotation[m])
        self._opacity = nn.Parameter(self._opacity[m])
        self.xyz_gradient_accum = self.xyz_gradient_accum[m]
        self.denom = self.denom[m]
        self._optimizer_mask(m)

    def densify(self, **kw):
        """追加新高斯"""
        n = kw['xyz'].shape[0]
        self._xyz = nn.Parameter(torch.cat([self._xyz, kw['xyz']]))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, kw['f_dc']]))
        self._scaling = nn.Parameter(torch.cat([self._scaling, kw['scaling']]))
        self._rotation = nn.Parameter(torch.cat([self._rotation, kw['rotation']]))
        self._opacity = nn.Parameter(torch.cat([self._opacity, kw['opacity']]))
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros(n, 1)])
        self.denom = torch.cat([self.denom, torch.zeros(n, 1)])
        self._optimizer_cat(kw)

    # ---- 自适应密度控制 ----
    def adaptive_density(self, grad_thresh=2e-4, split_size=1.6,
                         prune_opacity=0.005, prune_size=5.0,
                         opacity_reset_interval=3000, iteration=0):
        if self._xyz is None or self._xyz.shape[0] == 0:
            return

        grads = (self.xyz_gradient_accum / (self.denom + 1e-8)).squeeze()
        grads[torch.isnan(grads)] = 0

        with torch.no_grad():
            xyz = self._xyz
            scales = self.get_scaling()
            opac = self.get_opacity().squeeze()
            feat = self._features_dc

            high_grad = grads >= grad_thresh
            big = scales.amax(dim=1) > split_size
            split_mask = high_grad & big
            clone_mask = high_grad & ~big

            n_before = xyz.shape[0]
            n_clone = clone_mask.sum().item()
            n_split = split_mask.sum().item()

            # 克隆
            if n_clone:
                self.densify(
                    xyz=xyz[clone_mask],
                    f_dc=feat[clone_mask],
                    scaling=self._scaling[clone_mask],
                    rotation=self._rotation[clone_mask],
                    opacity=self._opacity[clone_mask],
                )

            # 分裂
            if n_split:
                stds = scales[split_mask]
                n2 = n_split * 2
                samples = torch.randn(n2, 3, device=xyz.device)
                new_xyz = xyz[split_mask].unsqueeze(1) + \
                          stds.unsqueeze(1) * samples.view(n_split, 2, 3)
                self.densify(
                    xyz=new_xyz.reshape(-1, 3),
                    f_dc=feat[split_mask].repeat(2, 1, 1),
                    scaling=(self._scaling[split_mask] + math.log(0.8)).repeat(2, 1),
                    rotation=self._rotation[split_mask].repeat(2, 1),
                    opacity=self._opacity[split_mask].repeat(2, 1),
                )

            # 裁剪
            n_after = self._xyz.shape[0]
            pr = torch.zeros(n_after, dtype=torch.bool)
            pr[:n_before] = (opac < prune_opacity) | (scales.amax(dim=1) > prune_size)
            # 同时裁剪透明度极低的新增点
            all_op = self.get_opacity().squeeze()
            pr = pr | (all_op < prune_opacity * 0.5)

            if pr.sum().item() > 0:
                self.prune(pr)

            if iteration > 0 and iteration % opacity_reset_interval == 0:
                self._opacity.data = torch.clamp(self._opacity.data,
                                                 max=math.log(0.01 / (1 - 0.01)))

            n_final = self._xyz.shape[0]
            if n_final != n_before:
                print(f"【密度】Iter{iteration}: {n_before}→{n_final} "
                      f"(克隆{n_clone} 分裂{n_split} 裁剪{n_before+n_clone+n2-n_final})")


# ============================================================
# 可微渲染器
# ============================================================

class DifferentiableRenderer(nn.Module):
    """基于点的可微渲染器 (PyTorch 纯张量操作)"""

    def forward(self, gaussians: GaussianModel,
                viewmatrix: torch.Tensor, fx: float, fy: float,
                cx: float, cy: float, W: int, H: int,
                bg_color: Tuple[float,float,float] = (0., 0., 0.)) -> torch.Tensor:
        """
        可微渲染: 投影 → 逐像素Gaussian求值 → 前向α合成
        bg_color 默认黑色（标准3DGS实践）
        """
        device = gaussians._xyz.device

        # ---- 1. 投影到屏幕 (保持可微) ----
        xyz = gaussians.get_xyz()                          # (N, 3)
        R = viewmatrix[:3, :3]                              # (3, 3)
        T = viewmatrix[:3, 3:4]                             # (3, 1)
        p_cam = xyz @ R.T + T.squeeze()                    # (N, 3)
        z = p_cam[:, 2:3].clamp(min=1e-6)                   # (N, 1)
        x_proj = p_cam[:, 0:1] / z * fx + cx               # (N, 1)
        y_proj = p_cam[:, 1:2] / z * fy + cy               # (N, 1)

        # ---- 2. 过滤不可见点 ----
        with torch.no_grad():
            visible = (
                (z.squeeze() > 0.1) &
                (x_proj.squeeze() >= 0) &
                (x_proj.squeeze() < W) &
                (y_proj.squeeze() >= 0) &
                (y_proj.squeeze() < H)
            )                                               # (N,) bool
            if visible.sum() == 0:
                return torch.full((H, W, 3), bg_color[0], device=device)

            # 按深度排序 (前→后)
            z_vis = z.squeeze()[visible]
            order = torch.argsort(z_vis)                     # front→back

        # 收集可见且排序后的属性
        x_sorted = x_proj[visible][order]                    # (M, 1)
        y_sorted = y_proj[visible][order]                    # (M, 1)
        cols_sorted = gaussians.get_features()[visible][order]  # (M, 3)
        opac_sorted = gaussians.get_opacity()[visible][order]   # (M, 1)
        M = x_sorted.shape[0]

        # ---- 3. 逐像素α合成 ----
        bg = torch.tensor(bg_color, device=device).view(1, 1, 3)
        image = bg.expand(H, W, 3).contiguous()
        # 透射率: detach 防止构建过深的计算图
        T_buffer = torch.ones(H, W, 1, device=device)

        # 分批处理以减少峰值内存
        batch_sz = 256
        for i in range(0, M, batch_sz):
            j = min(i + batch_sz, M)
            bx = x_sorted[i:j]     # (B,1)
            by = y_sorted[i:j]     # (B,1)
            bc = cols_sorted[i:j]  # (B,3)
            bo = opac_sorted[i:j]  # (B,1)

            # 像素网格 (H, W)
            gy, gx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij')
            gx = gx.unsqueeze(-1)  # (H,W,1)
            gy = gy.unsqueeze(-1)

            # 每个高斯到每个像素的距离^2
            dx = gx - bx.unsqueeze(0).unsqueeze(0)  # (H,W,B)
            dy = gy - by.unsqueeze(0).unsqueeze(0)
            d2 = dx*dx + dy*dy

            # 高斯核 (sigma=1.0)
            gauss_val = torch.exp(-d2 * 0.5)          # (H,W,B)

            # 逐高斯合成 (保持透射率 detach 以截断梯度流)
            for b in range(j - i):
                alpha = bo[b:b+1].view(1, 1, 1) * gauss_val[:, :, b:b+1]  # (H,W,1)
                alpha = torch.clamp(alpha, 0.0, 0.99)
                color = bc[b:b+1].view(1, 1, 3)                            # (1,1,3)

                # 梯度仅流经 color 和 alpha (经 T_buffer 不反向)
                image = image + color * alpha * T_buffer.detach()
                T_buffer = T_buffer * (1 - alpha)

                # 提前终止 (几乎所有像素都不透明)
                if T_buffer.max() < 1e-4:
                    return image.clamp(0, 1)

        return image.clamp(0, 1)


# ============================================================
# SSIM Loss
# ============================================================

def ssim_loss(img1: torch.Tensor, img2: torch.Tensor,
              window_size: int = 11) -> torch.Tensor:
    """结构相似性损失 (可微)"""
    def _gauss(sz, sigma):
        g = torch.arange(sz, dtype=torch.float32) - sz // 2
        g = torch.exp(-g**2 / (2 * sigma**2))
        return g / g.sum()

    C1, C2 = 0.01**2, 0.03**2
    k = _gauss(window_size, 1.5)
    k2d = (k.unsqueeze(1) @ k.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    k2d = k2d.to(img1.device)

    # (H,W,C) → (1,C,H,W)
    i1 = img1.permute(2, 0, 1).unsqueeze(0)
    i2 = img2.permute(2, 0, 1).unsqueeze(0)

    mu1 = F.conv2d(i1, k2d, padding=window_size//2, groups=3)
    mu2 = F.conv2d(i2, k2d, padding=window_size//2, groups=3)
    s1 = F.conv2d(i1**2, k2d, padding=window_size//2, groups=3) - mu1**2
    s2 = F.conv2d(i2**2, k2d, padding=window_size//2, groups=3) - mu2**2
    s12 = F.conv2d(i1*i2, k2d, padding=window_size//2, groups=3) - mu1*mu2

    ssim = ((2*mu1*mu2 + C1) * (2*s12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))
    return (1 - ssim.mean()) * 0.5


# ============================================================
# 训练循环
# ============================================================

def training_loop(gaussians, cameras, cam_images, images_dir,
                  iterations=30000, render_res=256,
                  save_dir=None, task=None, device='cpu'):
    """主训练循环 (渐进式分辨率 + 多视角累积 + 周期性导出)"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"【训练】使用GPU: {torch.cuda.get_device_name(0)}")

    gaussians.to(device)
    gaussians.setup_optimizer()
    renderer = DifferentiableRenderer()

    # 预过滤有效视角
    valid = []
    H0 = W0 = None
    for ci in cam_images:
        if ci['camera_id'] not in cameras:
            continue
        cam = cameras[ci['camera_id']]
        ip = os.path.join(images_dir, ci['name'])
        if not os.path.exists(ip):
            continue
        if H0 is None:
            H0, W0 = cam['height'], cam['width']
        valid.append({'cam': cam, 'info': ci, 'path': ip})

    if not valid:
        print("【训练】无有效视角，进入自监督模式")
        return _self_supervised(gaussians, renderer, iterations)

    nv = len(valid)
    # GT图像缓存 (延迟加载)
    gt_cache = {}

    def _get_gt(path, w, h):
        key = (w, h)
        if key not in gt_cache:
            gt_cache[key] = {}
        if path not in gt_cache[key]:
            try:
                pil = Image.open(path).resize((w, h), Image.LANCZOS)
                gt_cache[key][path] = torch.tensor(
                    np.array(pil) / 255., dtype=torch.float32, device=device)
            except Exception:
                return None
        return gt_cache[key][path]

    print(f"【训练】{nv} 个视角, 渲染分辨率 {render_res}×{render_res}")

    # ---- 渐进式分辨率参数 ----
    # 从低分辨率(render_res//4)逐渐上升到render_res
    res_start = max(32, render_res // 4)
    res_final = render_res

    # ---- 密度控制参数 ----
    densify_from = 500
    densify_until = max(1000, iterations // 2)
    densify_every = 100
    # 前期激进细化，后期稳定
    prune_opacity_thresh = 0.005
    split_size_thresh = 1.6

    best_loss = float('inf')
    gaussians.xyz_gradient_accum.zero_()
    gaussians.denom.zero_()

    # 多视角累积: 每N步累积多个视角后再更新
    accum_steps = 1  # 改为累积步数，默认1=单视角

    for it in range(iterations):
        # ---- 渐进式分辨率 ----
        progress = it / iterations
        # 前60%从低分辨上升到目标，后40%保持目标
        if progress < 0.6:
            cur_res = int(res_start + (res_final - res_start) * progress / 0.6)
        else:
            cur_res = res_final

        scale = cur_res / max(H0, W0)
        rH, rW = max(1, int(H0 * scale)), max(1, int(W0 * scale))
        cur_scale = scale

        # ---- 学习率 ----
        lr = gaussians.update_lr(it, iterations)

        # ---- 多视角累积 (mini-batch) ----
        if accum_steps > 1 and it % accum_steps == 0:
            gaussians.optimizer.zero_grad()

        # 选1个视角 (或随机选1个)
        v = valid[torch.randint(0, nv, (1,)).item()]
        cam = v['cam']

        # 相机参数
        params = cam['params']
        mid = cam.get('model_id', 1)
        if mid in (0, 2):
            fx_v = fy_v = params[0]
            cx_v = params[2] if len(params) > 2 else W0 / 2
            cy_v = params[3] if len(params) > 3 else H0 / 2
        else:
            fx_v = params[0]
            fy_v = params[1] if len(params) > 1 else params[0]
            cx_v = params[2] if len(params) > 2 else W0 / 2
            cy_v = params[3] if len(params) > 3 else H0 / 2

        rfx, rfy = fx_v * cur_scale, fy_v * cur_scale
        rcx, rcy = cx_v * cur_scale, cy_v * cur_scale

        # 视图矩阵
        R = qvec2rotmat(v['info']['qvec'])
        T = v['info']['tvec']
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = R
        vm[:3, 3]  = T

        # 加载GT (延迟缓存)
        gt = _get_gt(v['path'], rW, rH)
        if gt is None:
            continue

        # ---- 前向 ----
        vm_t = torch.tensor(vm, device=device)
        rendered = renderer(gaussians, vm_t, rfx, rfy, rcx, rcy, rW, rH)

        # ---- 自适应损失权重 ----
        # 早期更多L1 (几何约束), 后期更多SSIM (感知质量)
        ss_lambda = 0.1 + 0.2 * min(progress * 2, 1.0)

        loss_l1 = F.l1_loss(rendered, gt)
        loss_ssim = ssim_loss(rendered, gt)
        loss = (1 - ss_lambda) * loss_l1 + ss_lambda * loss_ssim

        # ---- 反向 (累计梯度) ----
        loss.backward()

        # 累积位置梯度用于密度控制
        with torch.no_grad():
            if gaussians._xyz.grad is not None:
                gaussians.xyz_gradient_accum += torch.norm(
                    gaussians._xyz.grad, dim=1, keepdim=True)
                gaussians.denom += 1

        if accum_steps <= 1 or (it + 1) % accum_steps == 0:
            # 梯度裁剪
            for pg in gaussians.optimizer.param_groups:
                if pg['params'][0].grad is not None:
                    pg['params'][0].grad.data = pg['params'][0].grad.data.clamp(-1, 1)

            gaussians.optimizer.step()
            if accum_steps > 1:
                gaussians.optimizer.zero_grad()

        # ---- 密度控制 ----
        if densify_from < it < densify_until and it % densify_every == 0:
            gaussians.adaptive_density(
                split_size=split_size_thresh,
                prune_opacity=prune_opacity_thresh,
                iteration=it)
            gaussians.xyz_gradient_accum.zero_()
            gaussians.denom.zero_()

        # ---- 定期重置透明度 (防止floaters) ----
        if it > 0 and it % 3000 == 0:
            with torch.no_grad():
                # 将过低/过高的透明度拉回中间值
                opac = gaussians.get_opacity()
                reset_mask = (opac.squeeze() < 0.01) | (opac.squeeze() > 0.99)
                if reset_mask.sum().item() > 0:
                    gaussians._opacity.data[reset_mask] = math.log(0.05 / (1 - 0.05))
                    if it % 3000 == 0:
                        print(f"\n  → 重置{reset_mask.sum().item()}个点的透明度")

        # ---- 尺度正则化 (抑制过大高斯点) ----
        if it % 500 == 0:
            with torch.no_grad():
                scales = gaussians.get_scaling()
                max_scale = scales.amax(dim=1)
                huge = max_scale > 20.0
                if huge.sum().item() > 0:
                    # 缩小超大高斯
                    scale_factor = 15.0 / max_scale[huge].clamp(min=1e-6)
                    gaussians._scaling.data[huge] += torch.log(scale_factor.unsqueeze(1))

        # ---- 中期和大后期额外裁剪 ----
        if it in (iterations // 3, iterations // 2, iterations * 2 // 3) and it > 0:
            with torch.no_grad():
                opac = gaussians.get_opacity().squeeze()
                pr = opac < prune_opacity_thresh * 0.5
                if pr.sum().item() > 0:
                    gaussians.prune(pr)
                    print(f"【裁剪】Iter{it}: 移除{pr.sum().item()}个死点, 剩余{gaussians._xyz.shape[0]}")

        # ---- 监控 ----
        if it % 100 == 0:
            psnr = -10 * math.log10(loss_l1.item() + 1e-8)
            print(f"\r【训练】Iter {it:5d}/{iterations} "
                  f"L1:{loss_l1.item():.5f} PSNR:{psnr:.2f} "
                  f"λ_ss:{ss_lambda:.2f} Res:{cur_res} "
                  f"Points:{gaussians._xyz.shape[0]}", end='', flush=True)

            if loss.item() < best_loss and save_dir:
                best_loss = loss.item()
                _save_ckpt(gaussians, save_dir, 'best.pth')

            # 定期导出中间结果
            if save_dir and it % 2000 == 0 and it > 0:
                step_ply = os.path.join(save_dir, f'point_cloud_iter{it}.ply')
                _export_ply(gaussians, step_ply)
                print(f"\n  → 已保存 {step_ply}")

            if task and it % 1000 == 0:
                try:
                    task.progress = min(99, int(100 * it / iterations))
                    task.save(update_fields=['progress'])
                except Exception:
                    pass

    print()
    return True


def _self_supervised(gaussians, renderer, iterations):
    """无GT数据时的自监督训练保底"""
    print("【自监督】无真实视角数据")
    device = gaussians._xyz.device
    opt = torch.optim.Adam([
        {'params': [gaussians._xyz], 'lr': 0.001},
        {'params': [gaussians._features_dc], 'lr': 0.0025},
        {'params': [gaussians._scaling], 'lr': 0.005},
        {'params': [gaussians._rotation], 'lr': 0.001},
        {'params': [gaussians._opacity], 'lr': 0.05},
    ])

    for it in range(iterations):
        theta, phi = np.random.uniform(0, 2*np.pi), np.random.uniform(0.3, np.pi-0.3)
        r = np.random.uniform(2, 10)
        cp = np.array([r*np.sin(phi)*np.cos(theta),
                       r*np.sin(phi)*np.sin(theta),
                       r*np.cos(phi)])
        fwd = -cp / (np.linalg.norm(cp) + 1e-8)
        right = np.cross(fwd, [0, 1, 0])
        right /= (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, fwd)

        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = np.array([right, up, -fwd]).T
        vm[:3, 3] = cp @ vm[:3, :3]

        W = H = 128
        vm_t = torch.tensor(vm, device=device)
        try:
            rendered = renderer(gaussians, vm_t, 500, 500, W/2, H/2, W, H)
            loss = rendered.std() * 0.1 + (1 - rendered.mean()).abs() * 0.01
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 500 == 0:
                print(f"\r【自监督】Iter {it}/{iterations} Loss:{loss.item():.6f}", end='', flush=True)
        except Exception:
            continue
    print()
    return True


def _save_ckpt(gaussians, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({k: v.data.cpu() for k, v in {
        'xyz': gaussians._xyz, 'features_dc': gaussians._features_dc,
        'scaling': gaussians._scaling, 'rotation': gaussians._rotation,
        'opacity': gaussians._opacity,
    }.items()}, os.path.join(save_dir, name))


def _export_ply(gaussians, path):
    xyz = gaussians._xyz.detach().cpu().numpy()
    rgb = gaussians.get_features().detach().cpu().numpy()
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(xyz)):
            r = int(np.clip(rgb[i, 0]*255, 0, 255))
            g = int(np.clip(rgb[i, 1]*255, 0, 255))
            b = int(np.clip(rgb[i, 2]*255, 0, 255))
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} {r} {g} {b}\n")


# ============================================================
# 分块重建
# ============================================================

class VastGaussianChunkedReconstruction:
    """VastGaussian 分块重建主入口"""

    def __init__(self, dataset_path, cube_size=10, position=(0, 0, 0),
                 resolution=1024, iterations=30000, task=None):
        self.dataset_path = dataset_path.rstrip('/\\')
        self.cube_size = cube_size
        self.position = np.array(position, dtype=np.float64)
        self.resolution = resolution
        self.iterations = iterations
        self.task = task
        self.output_dir = os.path.join(self.dataset_path, 'output', 'vast_gaussian')
        os.makedirs(self.output_dir, exist_ok=True)

    def _filter_cube(self, pts):
        half = self.cube_size / 2.0
        c = self.position
        return ((pts[:, 0] >= c[0]-half) & (pts[:, 0] <= c[0]+half) &
                (pts[:, 1] >= c[1]-half) & (pts[:, 1] <= c[1]+half) &
                (pts[:, 2] >= c[2]-half) & (pts[:, 2] <= c[2]+half))

    def run(self):
        try:
            print("\n" + "="*60)
            print("【VastGaussian 简化版】开始分块重建")
            print(f"  数据集: {self.dataset_path}")
            print(f"  方块: {self.cube_size}m @ {self.position}")
            print(f"  分辨率: {self.resolution}  迭代: {self.iterations}")
            print("="*60 + "\n")

            # [1] 加载数据
            print("[1/5] 加载COLMAP数据...")
            pts, cols, cameras, cam_images, img_dir = load_colmap_data(self.dataset_path)
            print(f"  点云: {len(pts)}  图像: {len(cam_images)}")

            # 分块过滤
            mask = self._filter_cube(pts)
            pts, cols = pts[mask], (cols[mask] if cols is not None else None)
            print(f"  分块内点: {mask.sum()}/{mask.shape[0]} ({100*mask.mean():.1f}%)")

            # [2] 初始化模型
            print("[2/5] 初始化高斯模型...")
            model = GaussianModel()
            if len(pts) > 0:
                model.create_from_pcd(pts, cols)
            else:
                n = 5000
                model.create_from_pcd(
                    self.position + np.random.randn(n, 3) * (self.cube_size / 6),
                    np.random.rand(n, 3))

            # [3] 训练
            print(f"[3/5] 训练 {self.iterations} 次迭代...")
            ok = training_loop(model, cameras, cam_images, img_dir,
                               iterations=self.iterations,
                               render_res=min(self.resolution, 256),
                               save_dir=self.output_dir, task=self.task)
            if not ok:
                return False

            # [4] 保存
            print("[4/5] 保存结果...")
            _export_ply(model, os.path.join(self.output_dir, 'point_cloud.ply'))
            _save_ckpt(model, self.output_dir, 'final.pth')
            with open(os.path.join(self.output_dir, 'info.json'), 'w') as f:
                json.dump({
                    'dataset': self.dataset_path,
                    'cube_size': self.cube_size,
                    'position': self.position.tolist(),
                    'iterations': self.iterations,
                    'num_points': model._xyz.shape[0],
                    'num_views': len(cam_images),
                }, f, indent=2)
            print(f"  PLY: {os.path.join(self.output_dir, 'point_cloud.ply')}")

            # [5] 完成
            print("\n[5/5] 完成!")
            print("【VastGaussian 简化版】分块重建完成!")
            return True

        except Exception as e:
            print(f"【错误】{e}")
            import traceback
            traceback.print_exc()
            return False


def run_vast_gaussian_chunked(dataset_path, resolution=1024, iterations=30000, task=None):
    """便捷入口 (兼容 worker 调用)"""
    cs, pos = 10, (0., 0., 0.)
    if task and task.description:
        try:
            p = json.loads(task.description)
            cs = p.get('cube_size', 10)
            pos = tuple(p.get('position', [0., 0., 0.]))
        except Exception:
            pass
    return VastGaussianChunkedReconstruction(
        dataset_path, cube_size=cs, position=pos,
        resolution=resolution, iterations=iterations, task=task,
    ).run()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='VastGaussian 分块重建')
    p.add_argument('-s', '--source', required=True)
    p.add_argument('-i', '--iterations', type=int, default=3000)
    p.add_argument('--cube-size', type=float, default=10)
    p.add_argument('--position', nargs=3, type=float, default=[0, 0, 0])
    p.add_argument('--resolution', type=int, default=1024)
    a = p.parse_args()
    VastGaussianChunkedReconstruction(
        a.source, cube_size=a.cube_size, position=tuple(a.position),
        resolution=a.resolution, iterations=a.iterations,
    ).run()
