
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
            sharpBaseCommand = ['sharp', 'predict', '-i', f'{imagePath}', '-o',
                                './output/reconstruction_results']


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
            task.progress = 100
            task.save()
            task.status = 'completed'
            task.completed_at = timezone.now()

        else:
            print(f"执行多图重建算法，图像数量: {images.count()}")
            # TODO: 调用多图重建算法
            # 获取数据集路径
            dataset_path = task.dataset_path
            print(f'正在处理任务 {task_id}')
            print(f'数据集路径: {dataset_path}')

            nerfBaseCommand = ['python',  '../pytorch/run_nerf.py', '--config', '../pytorch/configs/bicycle.txt']
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