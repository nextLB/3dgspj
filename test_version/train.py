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


sh_degree = 3
optimizer_type = "default"



listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


class GaussianModel:
    def __init__(self, sh_degree, optimizer_type):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree


class Scene:
    def __init__(self, gaussians):
        self.gaussians = gaussians



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



def train():
    if TENSORBOARD_FOUND:
        os.makedirs(SUMMARY_WRITER_OUTPUT_DIR, exist_ok = True)
        tb_writer = SummaryWriter(SUMMARY_WRITER_OUTPUT_DIR)
    else:
        tb_writer = None

    gaussians = GaussianModel(sh_degree, optimizer_type)

    scene = Scene(gaussians)

    print(tb_writer)
    print(gaussians)
    print(scene)






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






