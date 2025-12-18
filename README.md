# 3dgspj

# 环境配置如下:

    conda create -n 3dgspj python=3.11
    
    conda activate 3dgspj

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple flask

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pymysql

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple django

    pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-image

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple plyfile

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple imageio

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple imageio-ffmpeg
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple kornia
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple lpips
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple trimesh

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple open3d

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple wandb

    pip install pycolmap -i https://pypi.tuna.tsinghua.edu.cn/simple

训练程序的运行:

    python train_3dgs_mip.py \
        --data_path /home/next_lb/桌面/无人机影像三维重建任务/archive/360_v2/ \
        --scene bonsai \
        --output_path ./output \
        --iterations 10000 \
        --resolution 1 \
        --mip_filter \
        --batch_size 1 \
        --position_lr_init 0.00016 \
        --position_lr_final 0.0000016 \
        --feature_lr 0.0025 \
        --opacity_lr 0.05 \
        --scaling_lr 0.005 \
        --rotation_lr 0.001


显存使用的可视化:

    watch -n 2 nvidia-smi



