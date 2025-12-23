"""
    训练优化程序文件
"""

import sys
import random
import numpy as np
import torch
import socket
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
from plyfile import PlyData, PlyElement

from system_utils import searchForMaxIteration
from dataset_readers import sceneLoadTypeCallbacks
from camera_utils import cameraList_from_camInfos, camera_to_JSON
from general_utils import get_expon_lr_func


IP = '127.0.0.1'
PORT = 6009
DEBUG_FROM = 1
DETECT_ANOMALY = False
TEST_ITERATIONS = [7_000, 30_000]
SAVE_ITERATIONS = [7_000, 30_000]
QUIET = False
DISABLE_VIEWER = False
CHECKPOINT_ITERATIONS = []
START_CHECKPOINT = None

TENSORBOARD_FOUND = True
SUMMARY_WRITER_OUTPUT_DIR = './summary_writer_out_put'
WARNED = False

sh_degree = 3
optimizer_type = "default"
percent_dense = 0.01
white_background = False
depth_l1_weight_init = 1.0
depth_l1_weight_final = 0.01
lr_delay_steps=0
lr_delay_mult=1.0
iterations = 30_000
source_path = "/home/next_lb/桌面/无人机影像三维重建任务/Mip_NeRF360/360_v2/bicycle/"
images = "images"
depths = ""
evalF = False
train_test_exp = False
shuffle = True
resolution_scales=[1.0]
resolution = -1
data_device = "cuda"

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)






def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def init():
    global listener
    listener.bind((IP, PORT))
    listener.listen()
    listener.settimeout(0)



class GaussianModel:
    def __init__(self, sh_degree, optimizer_type):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree

    def training_setup(self):
        self.percent_dense = percent_dense


class Scene:
    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, images, depths, evalF, train_test_exp)
        else:
            scene_info = None

        if shuffle and scene_info:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, scene_info.is_nerf_synthetic, False, resolution, train_test_exp, data_device)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,  scene_info.is_nerf_synthetic, True, resolution, train_test_exp, data_device)


    def getTrainCameras(self, scale):
        return self.train_cameras[scale]



def train():
    if TENSORBOARD_FOUND:
        os.makedirs(SUMMARY_WRITER_OUTPUT_DIR, exist_ok = True)
        tb_writer = SummaryWriter(SUMMARY_WRITER_OUTPUT_DIR)
    else:
        tb_writer = None

    gaussians = GaussianModel(sh_degree, optimizer_type)

    scene = Scene(gaussians)

    gaussians.training_setup()

    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    depth_l1_weight = get_expon_lr_func(depth_l1_weight_init, depth_l1_weight_final, lr_delay_steps, lr_delay_mult, iterations)

    # viewpoint_stack = scene.getTrainCameras(1.0).copy()




def main():
    print('test version')

    # Initialize system state (RNG)
    safe_state(QUIET)

    # Start GUI server, configure and run training
    if not DISABLE_VIEWER:
        init()
    torch.autograd.set_detect_anomaly(DETECT_ANOMALY)

    train()



if __name__ == '__main__':
    main()






