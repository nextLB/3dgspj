import os
import numpy as np
import torch
from pycolmap import Reconstruction
import cv2
from tqdm import tqdm

def load_data(scene_path, downsample_factor=8):
    sparse_path = os.path.join(scene_path, 'sparse/0')
    recon = Reconstruction(sparse_path)

    # Load points3D
    points3D = np.array([p.xyz for p in recon.points3D.values()])
    colors3D = np.array([p.color / 255.0 for p in recon.points3D.values()])

    # Subsample points if too many
    if len(points3D) > 100000:
        idx = np.random.permutation(len(points3D))[:100000]
        points3D = points3D[idx]
        colors3D = colors3D[idx]

    # Load images and cameras
    if downsample_factor == 1:
        image_dir = os.path.join(scene_path, 'images')
    else:
        image_dir = os.path.join(scene_path, f'images_{downsample_factor}')

    images = {}
    cameras = {}
    for img_id, img in tqdm(recon.images.items(), desc="Loading images"):
        cam = recon.cameras[img.camera_id]
        R = img.rotation_matrix()
        t = img.translation_vector()
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c)

        params = cam.params
        K = np.array([[params[0], 0, params[2]],
                      [0, params[1], params[3]],
                      [0, 0, 1]])
        K[:2, :] /= downsample_factor  # Scale intrinsics for downsample

        image_name = img.name
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        h, w = image.shape[:2]

        images[img_id] = torch.from_numpy(image).float()
        cameras[img_id] = {'K': torch.from_numpy(K).float(), 'w2c': torch.from_numpy(w2c).float(),
                           'c2w': torch.from_numpy(c2w).float(), 'h': h, 'w': w}

    return points3D, colors3D, images, cameras