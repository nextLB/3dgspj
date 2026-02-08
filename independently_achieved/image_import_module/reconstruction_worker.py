


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
import signal

# 在 views.py 中定义的 running_tasks 需要在模块间共享
# 我们将使用一个全局字典来跟踪进程
from . import views


def check_task_cancelled(task_id):
    """检查任务是否已被取消"""
    try:
        task = ReconstructionTask.objects.get(id=task_id)
        return task.status == 'cancelled'
    except:
        return True  # 如果任务不存在，认为已取消


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
        iteration = int(progressMatch.group(1))
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

        # 将进程存储在全局字典中
        task_id_str = str(task.id)
        views.running_tasks[task_id_str] = process

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

                        # 检查任务是否被取消
                        if check_task_cancelled(task.id):
                            print(f"检测到任务 {task.id} 被取消，正在终止进程...")
                            process.terminate()
                            break

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

            # 检查任务是否被取消
            if check_task_cancelled(task.id):
                print(f"任务 {task.id} 已被取消，正在终止进程...")
                process.terminate()
                break

            time.sleep(1)

        # 等待输出线程结束
        stdoutThread.join(timeout=5)
        stderrThread.join(timeout=5)

        # 从运行任务字典中移除
        if task_id_str in views.running_tasks:
            views.running_tasks.pop(task_id_str)

        return returnCode, outputLines

    except Exception as e:
        print(f"运行多图重构算法时出错: {e}")
        return -1, [f"错误: {str(e)}"]


def process_reconstruction_task(task_id):
    """处理三维重建任务的函数（在后台线程中运行）"""
    process = None
    try:
        task = ReconstructionTask.objects.get(id=task_id)

        # 在开始前再次检查是否已取消
        if check_task_cancelled(task_id):
            print(f"任务 {task_id} 在开始前已被取消")
            task.status = 'cancelled'
            task.save()
            return

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
        if images.count() == 0 and not task.dataset_path:
            task.status = 'failed'
            task.error_message = "任务中没有图像且未提供数据集路径"
            task.save()
            return

        # 单图重建逻辑
        if images.count() == 1 and not task.dataset_path:
            firstImage = images.first()
            imagePath = firstImage.image.path  # 获取绝对路径
            print(f"执行单图重建算法，图像: {imagePath}")

            # 定义基础的sharp三维重构的命令
            outputDir = './output/reconstruction_results'
            sharpBaseCommand = ['sharp', 'predict', '-i', f'{imagePath}', '-o', outputDir]

            # 创建子进程
            process = subprocess.Popen(
                sharpBaseCommand,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )

            # 将进程存储在全局字典中
            task_id_str = str(task.id)
            views.running_tasks[task_id_str] = process

            # 定义读取输出的函数
            def read_process_output():
                output_lines = []
                try:
                    for line in process.stdout:
                        line = line.strip()
                        if line:
                            output_lines.append(line)
                            print(f"[Sharp] {line}")

                            # 检查任务是否被取消
                            if check_task_cancelled(task_id):
                                print(f"检测到任务 {task_id} 被取消，正在终止Sharp进程...")
                                process.terminate()
                                break
                except:
                    pass

                return output_lines

            # 读取输出并等待进程完成
            output_lines = read_process_output()
            process.wait()

            # 检查进程返回码
            if process.returncode == 0:
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
                # 检查是否是被取消的
                if check_task_cancelled(task_id):
                    task.status = 'cancelled'
                    task.error_message = "任务被用户取消"
                else:
                    print(f"Sharp重建失败，返回码: {process.returncode}")
                    task.status = 'failed'
                    error_output = '\n'.join(output_lines[-5:]) if output_lines else "无输出"
                    task.error_message = f"Sharp重建失败: {error_output[:500]}"
                task.save()

        else:
            # 多图重建逻辑
            print(f"执行多图重建算法，图像数量: {images.count()}")

            # 检查是否有数据集路径或上传的图像
            if task.dataset_path:
                dataset_path = task.dataset_path
                print(f'数据集路径: {dataset_path}')
            elif images.count() > 0:
                # 如果有上传的图像但没有数据集路径，可以创建一个临时目录
                dataset_path = tempfile.mkdtemp(prefix='uploaded_images_')
                for img in images:
                    img_path = img.image.path
                    dest_path = os.path.join(dataset_path, os.path.basename(img_path))
                    os.system(f"cp '{img_path}' '{dest_path}'")
                print(f'创建临时数据集目录: {dataset_path}')
            else:
                task.status = 'failed'
                task.error_message = "没有可用的图像数据"
                task.save()
                return

            # 获取路径最后的一个字段
            normPath = os.path.normpath(dataset_path)
            finalPathName = os.path.basename(normPath)

            # 创建相应的config.txt文件再去运行
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

            # 基于终端输出进行百分比显示的新方案
            returnCode, outputLines = run_mul_pic_train_with_progress(task, configPath)

            # 检查返回码
            if returnCode == 0:
                # 任务成功完成
                if not check_task_cancelled(task_id):  # 如果不是被取消的
                    task.progress = 100
                    task.status = 'completed'
                    task.completed_at = timezone.now()
                    task.save()
                    print(f"多图重建任务 {task_id} 完成！")
            elif returnCode is None:
                # 进程被终止
                if check_task_cancelled(task_id):
                    task.status = 'cancelled'
                    task.error_message = "任务被用户取消"
                    task.save()
                else:
                    task.status = 'failed'
                    task.error_message = "进程被意外终止"
                    task.save()
            else:
                # 任务失败
                if not check_task_cancelled(task_id):  # 如果不是被取消的
                    task.status = 'failed'
                    error_msg = outputLines[-1] if outputLines else f"进程返回码: {returnCode}"
                    task.error_message = error_msg[:500]
                    task.save()

    except ReconstructionTask.DoesNotExist:
        print(f"任务 {task_id} 不存在")
    except Exception as e:
        # 更新任务状态为失败
        try:
            task = ReconstructionTask.objects.get(id=task_id)
            if check_task_cancelled(task_id):
                task.status = 'cancelled'
                task.error_message = "任务被用户取消"
            else:
                task.status = 'failed'
                task.error_message = str(e)
            task.save()
        except:
            pass
        print(f"任务 {task_id} 处理过程中出错: {e}")
    finally:
        # 从运行任务字典中移除
        task_id_str = str(task_id)
        if task_id_str in views.running_tasks:
            views.running_tasks.pop(task_id_str, None)