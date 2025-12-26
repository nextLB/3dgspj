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

    pip install -r requirements.txt

验证环境是否配置成功可以执行下面的命令

    sharp --help

输出类似下面的内容即可

    (sharp) next_lb@NEXT:~/桌面/无人机影像三维重建任务/ml-sharp$ sharp --help
    Usage: sharp [OPTIONS] COMMAND [ARGS]...
    
      Run inference for SHARP model.
    
    Options:
      --help  Show this message and exit.
    
    Commands:
      predict  Predict Gaussians from input images.
      render   Predict Gaussians from input images.
    




## 关于本项目的运行与指南  version: 1.1

关于数据集的下载可以参考kaggle网址，下载Mip_NeRF360数据集

参考官网版本的网址

https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/


### 测试版本

来到test_version文件夹下，启动配置好的虚拟环境

运行训练和优化程序
    
    python train.py

运行渲染与可视化程序

    python ./render.py --model_path ./out_put




## 关于本项目的运行与指南  version: 1.2


### 最简单的使用示例说明

启动配置完善的虚拟环境

    conda activate sharp

执行重构自己的图像数据集

    sharp predict -i ./_DSC9040.JPG -o output

第一次运行时，它会自动下载一个模型文件，请耐心等待

等待完成：如果看到进度条走完，或者提示 Success，恭喜你！转换成功了。

The model checkpoint will be downloaded automatically on first run and cached locally at ~/.cache/torch/hub/checkpoints/.

Alternatively, you can download the model directly:
    
    wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt

接下来就可以打开这个output文件夹，再打开浏览器，访问在线查看器，

例如: https://playcanvas.com/products/supersplat 或者 https://antimatter15.com/splat/

然后将生成的.ply文件直接拖进网页里即可




### 自主实现版本

来到independently_achieved文件夹下，启动配置好的虚拟环境

运行系统UI

    python ./manage.py runserver

目前系统关于图像数据上传的功能由两部分，单张图像的数据上传和多张图像的数据上传









## 关于显存使用的查看

    watch -n 2 nvidia-smi

