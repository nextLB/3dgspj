
# reconstruction_worker.py
import os
import time
import tempfile
from django.utils import timezone
from django.core.files import File
from .models import ReconstructionTask
from django.conf import settings
import subprocess
import threading
import re





# 解析模型训练输出，更新任务进度
def parse_output(line, task):
    trainPattern = r'\[TRAIN\] Iter:\s*(\d+)\s+Loss:\s*([\d.]+)\s+PSNR:\s*([\d.]+)'

    progressPattern = r'\|\s*(\d+)/(\d+)\s+\[\d+:\d+<'

    simplePattern = r'Iter(?:ation)?\s*[:=]?\s*(\d+).*?Loss\s*[:=]\s*([\d.]+)'

    # 先尝试匹配训练信息
    trainMatch = re.search(trainPattern, line)
    if trainMatch:
        iteration = int(trainMatch.group(1))
        loss = float(trainMatch.group(2))
        psnr = float(trainMatch.group(3))

        # 计算进度百分比
        totalIterations = task.iterations
        progress = min(100.0, (iteration / totalIterations) * 100)

        # 更新任务进度
        task.progress = progress
        task.save()

        print(f"训练进度: {iteration}/{totalIterations} ({progress:.1f}%), Loss: {loss:.6f}, PSNR: {psnr:.2f}")
        return iteration, loss, psnr

    # 尝试匹配进度条信息
    progressMatch = re.search(progressPattern, line)
    if progressMatch:
        iteration = int(trainMatch.group(1))
        totalIterations = int(progressMatch.group(2))

        # 如果任务中没有设置总迭代次数，则使用进度条中的
        if task.iterations == 0 or task.iterations != totalIterations:
            task.iterations = totalIterations
            task.save()
            print(f"从进度条获取总迭代次数: {totalIterations}")

        # 计算进度百分比
        progress = min(100.0, (iteration / totalIterations) * 100)

        # 更新任务进度
        task.progress = progress
        task.save()

        print(f"进度条更新: {iteration}/{totalIterations} ({progress:.1f}%)")
        return iteration, None, None

    return None, None, None




# 运行多图构建并实时更新进度
def run_mul_pic_train_with_progress(task, configPath):
    try:
        nerfBaseCommand = ["python", "../pytorch/run_nerf.py", "--config", configPath]
        print(f"执行命令: {' '.join(nerfBaseCommand)}")

        # 启动进程
        process = subprocess.Popen(
            nerfBaseCommand,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )


        # 实时读取输出
        outputLines = []

        # 读取输出流的线程函数
        def read_output(pipe, name):
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        line = line.strip()
                        outputLines.append(f"[{name}] {line}")

                        # 只打印重要的训练信息，避免输出太多
                        if name == 'STDOUT' and ('[TRAIN]' in line or '/1000000' in line):
                            print(f"[{name}] {line}")

                        # 尝试解析进度
                        parse_output(line, task)

            except Exception as e:
                print(f"读取{name}时出错: {e}")
            finally:
                pipe.close()

        # 创建线程读取stdout和stderr
        stdoutThread = threading.Thread(target=read_output, args=(process.stdout, 'STDOUT'))
        stderrThread = threading.Thread(target=read_output, args=(process.stderr, 'STDERR'))

        stdoutThread.daemon = True
        stderrThread.daemon = True

        stdoutThread.start()
        stderrThread.start()

        # 等待进程结束
        while True:
            returnCode = process.poll()
            if returnCode is not None:
                break
            time.sleep(1)

            # 等待输出线程结束
            stdoutThread.join(timeout=5)
            stderrThread.join(timeout=5)

            return returnCode, outputLines





    except Exception as e:
        print(f"运行多图重构算法时出错: {e}")
        return -1, [f"错误: {str(e)}"]




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


            # nerfBaseCommand = ['python',  '../pytorch/run_nerf.py', '--config', f'../pytorch/configs/{finalPathName}.txt']
            # print(nerfBaseCommand)
            # # 执行命令
            # baseCommandResult = subprocess.run(
            #     nerfBaseCommand,
            #     capture_output=True,
            #     text=True,
            #     check=True  # 如果命令返回非零退出码，抛出异常
            # )
            # print(f'命令输出: {baseCommandResult.stdout}')
            # if baseCommandResult.stderr:
            #     print(f'命令错误: {baseCommandResult.stderr}')



            # 基于终端输出进行百分比显示的新方案
            returnCode, outputLines  = run_mul_pic_train_with_progress(task, configPath)



    except Exception as e:
        task = ReconstructionTask.objects.get(id=task_id)
        task.status = 'failed'
        task.error_message = str(e)
        task.save()
        print(f"任务 {task_id} 失败: {e}")


