import torch
import numpy as np
from scipy.spatial import KDTree


def distCUDA2(points):
    """
    计算点云中每个点到其最近邻点的距离的平方
    参数:
        points: torch.Tensor, 形状为 (N, 3) 的点云
    返回:
        torch.Tensor, 形状为 (N,) 的距离平方
    """
    # 确保输入是浮点数类型
    points_np = points.cpu().numpy() if points.is_cuda else points.numpy()

    # 构建KD树来快速找到最近邻
    tree = KDTree(points_np)

    # 对于每个点，找到第二个最近邻（第一个是自己）
    distances, indices = tree.query(points_np, k=2)

    # 取第二个最近邻的距离（第一个是点到自身的距离，为0）
    # 计算距离的平方
    dist2 = distances[:, 1] ** 2

    # 返回CUDA tensor
    return torch.from_numpy(dist2).float().to(points.device)





