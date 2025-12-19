import torch

class GaussianModel:
    def __init__(self, points3D, colors3D):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n = len(points3D)
        self.means = torch.from_numpy(points3D).float().to(device).requires_grad_(True)
        self.scales = torch.log(torch.ones((n, 3), device=device) * 0.01).requires_grad_(True)
        self.rotations = torch.zeros((n, 4), device=device)
        self.rotations[:, 0] = 1.0
        self.rotations.requires_grad_(True)
        self.opacities = torch.logit(torch.ones((n,), device=device) * 0.1).requires_grad_(True)
        self.sh = torch.zeros((n, 16, 3), device=device).requires_grad_(True)
        C0 = 0.28209479177387814
        self.sh[:, 0, :] = (torch.from_numpy(colors3D).float().to(device) - 0.5) / C0