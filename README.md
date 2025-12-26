# 3dgsjs


## 基于conda的虚拟环境的配置(linux)   version: 1.1

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
    
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pycolmap 

    conda install cudatoolkit=11.8 -c pytorch -c nvidia
    
    conda update -n base -c defaults conda

    conda install cuda-toolkit=12.1 -c nvidia

    pip install gsplat

    pip install tyro

    pip install viser

    pip install git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e


关于GPU显存是否可使用的验证

    python verify_pytorch_gpu.py
    然后仔细查看核对输出信息即可





## 基于conda的虚拟环境的配置(linux)   version: 1.2

    conda create -n sharp python=3.13

    conda activate sharp

然后来到ml-sharp文件夹下，执行如下命令









## 关于本项目的运行与指南

关于数据集的下载可以参考kaggle网址，下载Mip_NeRF360数据集

参考官网版本的网址

https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/



### 测试版本

来到test_version文件夹下，启动配置好的虚拟环境

运行训练和优化程序
    
    python train.py

运行渲染与可视化程序

    python ./render.py --model_path ./out_put



### 自主实现版本

来到independently_achieved文件夹下，启动配置好的虚拟环境

运行训练和优化程序
    
    python train.py
    

运行系统UI

    python ./manage.py runserver







## 关于显存使用的查看

    watch -n 2 nvidia-smi

