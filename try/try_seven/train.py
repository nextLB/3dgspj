import torch
from torch.optim import Adam
from data_loader import load_data
from gaussian_model import GaussianModel
from renderer import render
import argparse
import imageio
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--scene_path', type=str, required=True, help='Path to the scene folder, e.g., Mip_NeRF360/360_extra_scenes/flowers')
parser.add_argument('--iters', type=int, default=100, help='Number of training iterations (small for testing)')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
args = parser.parse_args()

points3D, colors3D, images, cameras = load_data(args.scene_path)

model = GaussianModel(points3D, colors3D)

params = [
    {'params': [model.means], 'lr': args.lr * 0.5},
    {'params': [model.scales], 'lr': args.lr},
    {'params': [model.rotations], 'lr': args.lr * 0.1},
    {'params': [model.opacities], 'lr': args.lr * 5},
    {'params': [model.sh], 'lr': args.lr * 0.02}
]
optimizer = Adam(params, lr=args.lr)

os.makedirs('outputs', exist_ok=True)

for iter in tqdm(range(args.iters), desc="Training"):
    losses = 0
    num_views = 0
    for img_id in images:
        gt = images[img_id].to(model.means.device)
        pred = render(model, cameras[img_id])
        loss = torch.mean(torch.abs(pred - gt))
        losses += loss
        num_views += 1
        if iter % 10 == 0:
            imageio.imwrite(f'outputs/iter{iter}_view{img_id}.png', (pred.cpu().numpy() * 255).astype(np.uint8))
    loss = losses / num_views
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Iter {iter}: Loss {loss.item()}")

print("Training complete. Check outputs/ for rendered images.")