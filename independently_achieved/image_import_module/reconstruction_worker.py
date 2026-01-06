
# reconstruction_worker.py
import os
import time
import tempfile
from django.utils import timezone
from django.core.files import File
from .models import ReconstructionTask
from django.conf import settings
import subprocess



def process_reconstruction_task(task_id):
    """处理三维重建任务的函数（在后台线程中运行）"""
    try:
        task = ReconstructionTask.objects.get(id=task_id)

        # 获取任务参数
        resolution = task.resolution
        iterations = task.iterations
        task_name = task.name

        print(f'正在处理任务 {task_id}')
        print(f'任务名称: {task_name}')
        print(f'重建参数 - 分辨率: {resolution}, 迭代次数: {iterations}')

        # 获取任务中的图像
        images = task.images.all()
        print(f'正在处理任务 {task_id}，图像数量: {images.count()}')

        # 检查图像数量
        if images.count() == 0:
            raise ValueError("任务中没有图像")

        # 单图重建逻辑
        if images.count() == 1:
            firstImage = images.first()
            imagePath = firstImage.image.path  # 获取绝对路径
            print(f"执行单图重建算法，图像: {imagePath}")
            # TODO: 调用单图重建算法
            # 定义基础的sharp三维重构的命令
            outputDir = './output/reconstruction_results'
            sharpBaseCommand = ['sharp', 'predict', '-i', f'{imagePath}', '-o',
                                outputDir]


            # 执行命令
            baseCommandResult = subprocess.run(
                sharpBaseCommand,
                capture_output=True,
                text=True,
                check=True  # 如果命令返回非零退出码，抛出异常
            )
            print(f'命令输出: {baseCommandResult.stdout}')
            if baseCommandResult.stderr:
                print(f'命令错误: {baseCommandResult.stderr}')

            # task.progress = 100
            # task.save()
            # task.status = 'completed'
            # task.completed_at = timezone.now()
            # 检查sharp是否成功执行
            if baseCommandResult.returncode == 0:
                print("Sharp重建成功完成！")

                # 查找生成的ply文件
                ply_files = []
                for root, dirs, files in os.walk(outputDir):
                    for file in files:
                        if file.endswith('.ply'):
                            ply_files.append(os.path.join(root, file))

                if ply_files:
                    # 使用第一个找到的ply文件
                    ply_file = ply_files[0]
                    print(f"找到PLY文件: {ply_file}")

                    # 保存结果文件到数据库
                    with open(ply_file, 'rb') as f:
                        task.result_ply.save(f'result_{task.id}.ply', File(f), save=False)

                    print(f"结果文件已保存到数据库")
                else:
                    print(f"警告: 在 {outputDir} 中未找到PLY文件")

                # 更新任务状态
                task.progress = 100
                task.status = 'completed'
                task.completed_at = timezone.now()
                task.save()

                print(f"任务 {task_id} 完成！状态已更新为 completed")
                print(f"可以在Supersplat中编辑: https://superspl.at/editor/")
            else:
                print(f"Sharp重建失败，返回码: {baseCommandResult.returncode}")
                task.status = 'failed'
                task.error_message = f"Sharp重建失败: {baseCommandResult.stderr[:500]}"
                task.save()


        else:
            print(f"执行多图重建算法，图像数量: {images.count()}")
            # TODO: 调用多图重建算法
            # 获取数据集路径
            dataset_path = task.dataset_path
            print(f'正在处理任务 {task_id}')
            print(f'数据集路径: {dataset_path}')

            # 获取路径最后的一个字段
            normPath = os.path.normpath(dataset_path)
            finalPathName = os.path.basename(normPath)

            # TODO:创建相应的config.txt文件再去运行
            configPath = f'../pytorch/configs/{finalPathName}.txt'
            configContent = f"""expname = {finalPathName}_test
basedir = ./output
datadir = {dataset_path}
dataset_type = llff

factor = 8
llffhold = 8

N_rand = 1024
N_samples = 64
N_importance = 64

use_viewdirs = True
raw_noise_std = 1e0
"""
            with open(configPath, 'w', encoding='utf-8') as f:
                f.write(configContent)


            nerfBaseCommand = ['python',  '../pytorch/run_nerf.py', '--config', f'../pytorch/configs/{finalPathName}.txt']
            print(nerfBaseCommand)
            # 执行命令
            baseCommandResult = subprocess.run(
                nerfBaseCommand,
                capture_output=True,
                text=True,
                check=True  # 如果命令返回非零退出码，抛出异常
            )
            print(f'命令输出: {baseCommandResult.stdout}')
            if baseCommandResult.stderr:
                print(f'命令错误: {baseCommandResult.stderr}')



    except Exception as e:
        task = ReconstructionTask.objects.get(id=task_id)
        task.status = 'failed'
        task.error_message = str(e)
        task.save()
        print(f"任务 {task_id} 失败: {e}")


