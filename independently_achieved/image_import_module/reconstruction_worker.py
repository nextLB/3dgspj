# reconstruction_worker.py
import os
import time
from django.utils import timezone
from .models import ReconstructionTask
from django.conf import settings


def process_reconstruction_task(task_id):
    """处理三维重建任务的函数（在后台线程中运行）"""
    try:
        task = ReconstructionTask.objects.get(id=task_id)

        # 模拟重建过程
        images = task.images.all()

        # 这里应该调用你的3D高斯溅射算法
        # 例如：
        # 1. 准备图像数据
        # 2. 调用外部命令或库进行重建
        # 3. 生成PLY文件和预览图

        # 模拟进度更新
        for i in range(1, 101):
            time.sleep(0.5)  # 模拟处理时间
            task.progress = i
            task.save()

            if i % 10 == 0:
                print(f"任务 {task_id} 进度: {i}%")

        # 模拟完成
        task.status = 'completed'
        task.completed_at = timezone.now()

        # 模拟生成结果文件（实际应该保存算法生成的文件）
        # task.result_ply = 'reconstruction_results/ply/sample.ply'
        # task.preview_image = 'reconstruction_previews/sample.png'

        task.save()
        print(f"任务 {task_id} 完成")

    except Exception as e:
        task = ReconstructionTask.objects.get(id=task_id)
        task.status = 'failed'
        task.error_message = str(e)
        task.save()
        print(f"任务 {task_id} 失败: {e}")