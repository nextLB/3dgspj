# 3dgsjs


## 基于conda的虚拟环境的配置

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
    
    conda install cudatoolkit=11.8 -c pytorch -c nvidia

    conda update -n base -c defaults conda

接下来需要到cuda官网下载指定的nvcc包，进行安装

    https://developer.nvidia.com/cuda-11-8-0-download-archive
    选择对应的包后，运行官网页面提供的下列形式的命令
        wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.runsudo 
        sh cuda_11.8.0_520.61.05_linux.run


之后来到gaussian-splatting文件夹下，运行




## 关于本项目的运行与指南



## 关于显存使用的查看

    watch -n 2 nvidia-smi

