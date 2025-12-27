
# reconstruction_worker.py
import os
import time
import tempfile
from django.utils import timezone
from django.core.files import File
from .models import ReconstructionTask
from django.conf import settings


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
            print(f"执行单图重建算法，图像: {images.first().filename()}")
            # TODO: 调用单图重建算法
            # 例如：单图深度估计 + 三维重建
        else:
            print(f"执行多图重建算法，图像数量: {images.count()}")
            # TODO: 调用多图重建算法（原有的）
            # 例如：SFM + 3D高斯溅射

        # 创建输出目录
        output_dir = os.path.join(settings.MEDIA_ROOT, 'reconstruction_results', str(task_id))
        os.makedirs(output_dir, exist_ok=True)

        # 模拟进度更新
        for i in range(1, 101):
            time.sleep(0.5)  # 模拟处理时间
            task.progress = i
            task.save()

            if i % 10 == 0:
                print(f"任务 {task_id} 进度: {i}%")

        # 模拟完成后生成结果文件
        task.status = 'completed'
        task.completed_at = timezone.now()

        # 创建一个简单的PLY文件作为示例
        ply_content = """ply
format ascii 1.0
element vertex 8
property float x
property float y
property float z
element face 6
property list uchar int vertex_index
end_header
0 0 0
0 0 1
0 1 1
0 1 0
1 0 0
1 0 1
1 1 1
1 1 0
4 0 1 2 3
4 7 6 5 4
4 0 4 5 1
4 1 5 6 2
4 2 6 7 3
4 3 7 4 0
"""

        # 保存PLY文件
        ply_filename = f"reconstruction_{task_id}.ply"
        ply_path = os.path.join(output_dir, ply_filename)

        with open(ply_path, 'w') as f:
            f.write(ply_content)

        # 将文件保存到数据库
        with open(ply_path, 'rb') as f:
            task.result_ply.save(ply_filename, File(f))

        print(f"任务 {task_id} 完成，结果文件已保存: {ply_filename}")
        task.save()

    except Exception as e:
        task = ReconstructionTask.objects.get(id=task_id)
        task.status = 'failed'
        task.error_message = str(e)
        task.save()
        print(f"任务 {task_id} 失败: {e}")