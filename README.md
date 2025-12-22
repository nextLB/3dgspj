# 3dgsjs


## 关于依赖于conda的python环境的配置

可以来到gaussian-splatting文件夹下，运行如下命令:

    SET DISTUTILS_USE_SDK=1 # Windows only
    conda env create --file environment.yml
    conda activate gaussian_splatting

配置过虚拟环境后，可以试一下如下命令来验证是否是完整配置成功了

    python train.py -s <path to COLMAP or NeRF Synthetic dataset>

若成功运行起来，即环境配置成功


## 关于官方程序的运行与指南

来到gaussian-splatting文件夹下

运行优化，使用如下命令

        python train.py -s <path to COLMAP or NeRF Synthetic dataset>

评估与渲染，使用如下命令

    python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # Train with train/test split
    python render.py -m <path to trained model> # Generate renderings
    python metrics.py -m <path to trained model> # Compute error metrics on renderings

    python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
    python metrics.py -m <path to pre-trained model>

    python full_eval.py -m360 <mipnerf360 folder> -tat <tanks and temples folder> -db <deep blending folder>

    python full_eval.py -o <directory with pretrained models> --skip_training -m360 <mipnerf360 folder> -tat <tanks and temples folder> -db <deep blending folder>

    python full_eval.py -m <directory with evaluation images>/garden ... --skip_training --skip_rendering



## 关于本项目的运行与指南



