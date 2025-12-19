import torch
import numpy as np

def quaternion_to_mat(q):
    w, x, y, z = q.unbind(-1)
    mat = torch.stack([
        1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
        2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w,
        2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2
    ], dim=-1).view(q.shape[0], 3, 3)
    return mat

def get_cov3d(scales, rotations):
    S = torch.diag_embed(torch.exp(scales))
    R = quaternion_to_mat(rotations)
    cov = R @ S @ S @ R.transpose(-1, -2)
    return cov

def project_cov3d_to2d(means3d, cov3d, w2c, K):
    view = w2c[:3, :3]
    t = w2c[:3, 3]
    mean_view = view @ means3d.t() + t
    depths = mean_view[2].clamp(min=1e-6)
    mean_view = mean_view / depths[:, None]
    x, y = mean_view[0], mean_view[1]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    mean2d = torch.stack([fx * x + cx, fy * y + cy], dim=-1)

    J = torch.zeros((means3d.shape[0], 2, 3), device=means3d.device)
    J[:, 0, 0] = fx / depths
    J[:, 0, 2] = -fx * x / depths**2
    J[:, 1, 1] = fy / depths
    J[:, 1, 2] = -fy * y / depths**2

    cov_view = view @ cov3d @ view.transpose(-1, -2)
    cov2d = J @ cov_view @ J.transpose(-1, -2)
    cov2d = cov2d[:, :2, :2]
    cov2d += torch.eye(2, device=cov2d.device) * 1e-6

    return mean2d, cov2d, depths

def sh_basis(dirs):
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = 1.0925484305920792
    C3 = 0.31539156525252005
    C4 = 0.5462742152960396
    C5 = 0.5900435899266435
    C6 = 2.890611442640554
    C7 = 0.45704579946446567
    C8 = 0.3731763325901154
    C9 = 1.445305721320277

    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    x2, y2, z2 = x**2, y**2, z**2
    xy, xz, yz = x*y, x*z, y*z

    result = torch.empty((dirs.shape[0], 16), dtype=dirs.dtype, device=dirs.device)
    result[:, 0] = C0

    result[:, 1] = -C1 * y
    result[:, 2] = C1 * z
    result[:, 3] = -C1 * x

    result[:, 4] = C2 * xy
    result[:, 5] = -C2 * yz
    result[:, 6] = C3 * (3 * z2 - 1)
    result[:, 7] = -C2 * xz
    result[:, 8] = C4 * (x2 - y2)

    result[:, 9] = -C5 * y * (3 * x2 - y2)
    result[:, 10] = C6 * xy * z
    result[:, 11] = -C7 * y * (5 * z2 - 1)
    result[:, 12] = C8 * z * (5 * z2 - 3)
    result[:, 13] = -C7 * x * (5 * z2 - 1)
    result[:, 14] = C9 * z * (x2 - y2)
    result[:, 15] = -C5 * x * (x2 - 3 * y2)

    return result

def render(model, camera, bg_color=1.0, pixel_batch_size=10000):
    device = model.means.device
    h, w = camera['h'], camera['w']
    p = h * w
    x, y = torch.meshgrid(torch.arange(w, device=device).float(), torch.arange(h, device=device).float(), indexing='xy')
    pixels = torch.stack((x, y), -1).reshape(-1, 2)

    cov3d = get_cov3d(model.scales, model.rotations)
    mean2d, cov2d, depths = project_cov3d_to2d(model.means, cov3d, camera['w2c'], camera['K'])

    mask = (depths > 0) & ((mean2d[:, 0] > -0.5 * w) & (mean2d[:, 0] < 1.5 * w)) & ((mean2d[:, 1] > -0.5 * h) & (mean2d[:, 1] < 1.5 * h))
    mean2d = mean2d[mask]
    cov2d = cov2d[mask]
    depths = depths[mask]
    sh = model.sh[mask]
    opacities = torch.sigmoid(model.opacities[mask])

    sort_idx = torch.argsort(depths)
    mean2d = mean2d[sort_idx]
    cov2d = cov2d[sort_idx]
    depths = depths[sort_idx]
    sh = sh[sort_idx]
    opacities = opacities[sort_idx]

    n = len(depths)
    if n == 0:
        return torch.ones((h, w, 3), device=device) * bg_color

    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1]**2
    det = det.clamp(min=1e-6)
    inv_cov = torch.zeros_like(cov2d)
    inv_cov[:, 0, 0] = cov2d[:, 1, 1] / det
    inv_cov[:, 0, 1] = -cov2d[:, 0, 1] / det
    inv_cov[:, 1, 0] = -cov2d[:, 0, 1] / det
    inv_cov[:, 1, 1] = cov2d[:, 0, 0] / det

    cam_pos = camera['c2w'][:3, 3]
    view_dirs = cam_pos - model.means[mask][sort_idx]
    view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True)
    basis = sh_basis(view_dirs)
    colors = (basis[:, :, None] * sh[:, None, :]).sum(dim=1) + 0.5
    colors = colors.clamp(0, 1)

    final_color = torch.zeros((p, 3), device=device)
    for start in range(0, p, pixel_batch_size):
        end = min(start + pixel_batch_size, p)
        pixels_b = pixels[start:end]
        b = end - start

        d = pixels_b[:, None, :] - mean2d[None, :, :]  # b n 2
        tmp = torch.einsum('bni,nij->bnj', d, inv_cov)  # b n 2
        mahalanobis = (d * tmp).sum(-1)  # b n
        power = -0.5 * mahalanobis
        alpha = opacities[None, :] * torch.exp(power)  # b n
        alpha = alpha.clamp(0, 0.999)

        one_minus_alpha = 1 - alpha
        T = torch.cat([torch.ones((b, 1), device=device), torch.cumprod(one_minus_alpha[:, :-1], dim=1)], dim=1)
        weights = alpha * T  # b n

        rendered_color = (weights[:, :, None] * colors[None, :, :]).sum(dim=1)  # b 3
        bg_contrib = torch.prod(one_minus_alpha, dim=1)[:, None] * bg_color  # b 1 * 3 if bg vector
        rendered_color += bg_contrib

        final_color[start:end] = rendered_color

    final_image = final_color.reshape(h, w, 3)
    return final_image