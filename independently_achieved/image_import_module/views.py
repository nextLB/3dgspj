


# views.py 替换内容 - 修改 upload_image 函数中的多图上传部分
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
import uuid
import os
from .forms import SingleImageUploadForm, MultipleImageUploadForm, ReconstructionSettingsForm, CubeReconstructionForm
from .models import UploadedImage, ReconstructionTask


# 存储正在运行的任务进程
running_tasks = {}

def upload_image(request):
    """处理图像上传和显示"""
    max_size = 20 * 1024 * 1024  # 20MB

    # 初始化表单
    single_form = SingleImageUploadForm()
    multiple_form = MultipleImageUploadForm()
    cube_form = CubeReconstructionForm()  # 新增：分块重建表单
    recent_images = UploadedImage.objects.all().order_by('-uploaded_at')[:10]
    recent_tasks = ReconstructionTask.objects.all().order_by('-created_at')[:5]

    context = {
        'single_form': single_form,
        'multiple_form': multiple_form,
        'cube_form': cube_form,  # 新增：分块重建表单
        'recent_images': recent_images,
        'recent_tasks': recent_tasks,
        'max_size': max_size,
        'active_tab': 'single'  # 默认激活单图上传标签页
    }

    if request.method == 'POST':
        upload_type = request.POST.get('upload_type', 'single')

        if upload_type == 'single':
            form = SingleImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                # 保存图像但先不提交到数据库
                image_instance = form.save(commit=False)

                # 创建重建任务
                task_name = form.cleaned_data.get('task_name', f'单图重建-{uuid.uuid4().hex[:8]}')
                task = ReconstructionTask.objects.create(
                    name=task_name,
                    status='pending',
                    created_at=timezone.now()
                )

                # 将任务与图像关联并保存
                image_instance.task = task
                image_instance.save()

                # 准备显示的数据
                saved_images = [image_instance]

                context.update({
                    'uploaded_images': saved_images,
                    'task': task,
                    'success_message': '单张图像上传成功！已创建重建任务。',
                    'active_tab': 'single'
                })

                # 重新初始化表单
                context['single_form'] = SingleImageUploadForm()
            else:
                context['single_form'] = form
                context['active_tab'] = 'single'

        elif upload_type == 'multiple':
            # 手动处理多文件上传
            form = MultipleImageUploadForm(request.POST)
            if form.is_valid():
                task_name = form.cleaned_data.get('task_name', f'批量上传任务-{uuid.uuid4().hex[:8]}')
                dataset_path = form.cleaned_data.get('dataset_path', '')  # 获取数据集路径

                # 获取上传的文件
                files = request.FILES.getlist('images')

                if not files and not dataset_path:
                    context['multiple_form'] = form
                    context['active_tab'] = 'multiple'
                    context['error_message'] = '请至少选择一个文件或提供数据集路径'
                    return render(request, 'image_import_module/upload.html', context)

                # 创建重建任务（包含数据集路径）
                task = ReconstructionTask.objects.create(
                    name=task_name,
                    dataset_path=dataset_path,  # 保存数据集路径
                    status='pending',
                    created_at=timezone.now()
                )

                saved_images = []
                errors = []

                # 如果有上传的文件，保存它们
                if files:
                    if len(files) > 1000:  # 从 50 改为 1000
                        context['multiple_form'] = form
                        context['active_tab'] = 'multiple'
                        context['error_message'] = '一次最多上传1000张图像'  # 更新错误消息
                        task.delete()  # 删除刚创建的任务
                        return render(request, 'image_import_module/upload.html', context)

                    # 验证并保存每个文件
                    allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif']

                    for file in files:
                        # 检查文件大小
                        if file.size > max_size:
                            errors.append(f'文件 {file.name} 大小超过20MB限制')
                            continue

                        # 检查文件类型
                        ext = os.path.splitext(file.name)[1].lower()
                        if ext not in allowed_extensions:
                            errors.append(f'文件 {file.name} 格式不支持')
                            continue

                        # 保存图像
                        uploaded_image = UploadedImage(
                            image=file,
                            task=task
                        )
                        uploaded_image.save()
                        saved_images.append(uploaded_image)

                if errors:
                    # 如果验证失败，删除任务
                    task.delete()
                    context['multiple_form'] = form
                    context['active_tab'] = 'multiple'
                    context['error_message'] = '; '.join(errors[:3])  # 只显示前3个错误
                    return render(request, 'image_import_module/upload.html', context)

                if saved_images or dataset_path:
                    context.update({
                        'uploaded_images': saved_images,
                        'task': task,
                        'success_message': f'成功创建任务！{len(saved_images)} 张图像已上传，数据集路径已保存。',
                        'active_tab': 'multiple'
                    })

                    # 重新初始化表单
                    context['multiple_form'] = MultipleImageUploadForm()
            else:
                context['multiple_form'] = form
                context['active_tab'] = 'multiple'

        elif upload_type == 'cube':
            form = CubeReconstructionForm(request.POST)
            if form.is_valid():
                # 获取表单数据
                task_name = form.cleaned_data.get('task_name', f'方块重建任务-{uuid.uuid4().hex[:8]}')
                cube_size = form.cleaned_data.get('cube_size', 10)
                position_x = form.cleaned_data.get('position_x', 0.0)
                position_y = form.cleaned_data.get('position_y', 0.0)
                position_z = form.cleaned_data.get('position_z', 0.0)

                # ==============================================
                # 注意：这里只是创建任务，具体方块重建逻辑需要您自己实现
                # 您可以根据需要：
                # 1. 调用外部的方块生成算法
                # 2. 生成3D模型文件
                # 3. 保存参数到数据库的新字段中
                # ==============================================

                # 创建重建任务（使用现有模型，可以添加cube_type字段区分）
                task = ReconstructionTask.objects.create(
                    name=task_name,
                    status='pending',
                    created_at=timezone.now()
                )

                # 在任务描述中保存方块参数（临时方案，建议在模型中添加专用字段）
                cube_params = {
                    'cube_size': cube_size,
                    'position': [position_x, position_y, position_z],
                    'type': 'cube_reconstruction'
                }
                task.description = json.dumps(cube_params, ensure_ascii=False, indent=2)
                task.save()

                context.update({
                    'task': task,
                    'success_message': f'方块重建任务 "{task_name}" 已创建！立方体尺寸：{cube_size}米，位置：[{position_x}, {position_y}, {position_z}]',
                    'active_tab': 'cube',
                    'uploaded_images': []  # 方块重建没有图像
                })

                # 重新初始化表单
                context['cube_form'] = CubeReconstructionForm()
            else:
                context['cube_form'] = form
                context['active_tab'] = 'cube'

    return render(request, 'image_import_module/upload.html', context)

    return render(request, 'image_import_module/upload.html', context)



# views.py - 修改 start_reconstruction 函数
@csrf_exempt
@require_POST
def start_reconstruction(request):
    """开始三维重建任务"""
    try:
        data = json.loads(request.body)
        task_id = data.get('task_id')

        if not task_id:
            return JsonResponse({
                'success': False,
                'error': '未提供任务ID'
            })

        task = get_object_or_404(ReconstructionTask, id=task_id)

        # 检查是否可以开始重建
        if not task.images.exists() and not task.dataset_path:
            return JsonResponse({
                'success': False,
                'error': '任务中没有图像且未提供数据集路径'
            })

        if task.status != 'pending':
            return JsonResponse({
                'success': False,
                'error': f'任务状态为{task.get_status_display()}，无法开始重建'
            })

        # 更新任务状态
        task.status = 'processing'
        task.progress = 0
        task.started_at = timezone.now()
        task.save()

        # 启动后台线程处理重建任务
        import threading
        from .reconstruction_worker import process_reconstruction_task

        # 传递任务参数到工作线程
        thread = threading.Thread(
            target=process_reconstruction_task,
            args=(task.id,)
        )
        thread.daemon = True
        thread.start()

        return JsonResponse({
            'success': True,
            'message': '三维重建任务已开始',
            'task_id': str(task.id),
            'estimated_time': task.estimated_time(),
            'resolution': task.resolution,  # 返回分辨率参数
            'iterations': task.iterations   # 返回迭代次数参数
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


def get_task_status(request, task_id):
    """获取重建任务状态"""
    try:
        task = get_object_or_404(ReconstructionTask, id=task_id)

        return JsonResponse({
            'id': str(task.id),
            'name': task.name,
            'dataset_path': task.dataset_path or '',
            'status': task.status,
            'progress': task.progress,
            'image_count': task.images.count(),
            'created_at': task.created_at.isoformat(),
            'error_message': task.error_message,
            'estimated_time': task.estimated_time()
        })
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=400)


def task_detail(request, task_id):
    """任务详情页面"""
    task = get_object_or_404(ReconstructionTask, id=task_id)
    images = task.images.all()
    settings_form = ReconstructionSettingsForm(instance=task)

    if request.method == 'POST':
        form = ReconstructionSettingsForm(request.POST, instance=task)
        if form.is_valid():
            form.save()
            return redirect('image_import_module:task_detail', task_id=task_id)

    return render(request, 'image_import_module/task_detail.html', {
        'task': task,
        'images': images,
        'settings_form': settings_form,
        'can_start_reconstruction': task.status == 'pending' and (task.images.exists() or task.dataset_path)
    })


def home(request):
    """首页，重定向到上传页面"""
    return redirect('image_import_module:upload_image')




@csrf_exempt
@require_POST
def cancel_reconstruction(request):
    """取消正在进行的重建任务"""
    try:
        data = json.loads(request.body)
        task_id = data.get('task_id')

        if not task_id:
            return JsonResponse({
                'success': False,
                'error': '未提供任务ID'
            })

        task = get_object_or_404(ReconstructionTask, id=task_id)

        # 检查任务是否可以取消
        if not task.can_be_cancelled():
            return JsonResponse({
                'success': False,
                'error': f'任务状态为{task.get_status_display()}，无法取消'
            })

        # 更新任务状态
        task.status = 'cancelled'
        task.error_message = '任务已被用户取消'
        task.save()

        # 如果任务正在运行，尝试终止进程
        if str(task_id) in running_tasks:
            process = running_tasks[str(task_id)]
            try:
                process.terminate()  # 尝试终止进程
                print(f"任务 {task_id} 的进程已被终止")
            except:
                pass
            finally:
                # 从运行任务字典中移除
                running_tasks.pop(str(task_id), None)

        return JsonResponse({
            'success': True,
            'message': '任务已成功取消',
            'status': 'cancelled'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@csrf_exempt
@require_POST
def delete_reconstruction_task(request):
    """删除重建任务及其相关文件"""
    try:
        data = json.loads(request.body)
        task_id = data.get('task_id')

        if not task_id:
            return JsonResponse({
                'success': False,
                'error': '未提供任务ID'
            })

        task = get_object_or_404(ReconstructionTask, id=task_id)

        # 检查任务是否可以删除
        if task.status == 'processing':
            return JsonResponse({
                'success': False,
                'error': '处理中的任务不能删除，请先取消任务'
            })

        # 存储任务名称用于返回消息
        task_name = task.name
        task_id_str = str(task.id)

        # 删除关联的文件
        file_fields = ['result_ply', 'result_mesh', 'preview_image', 'log_file']
        for field_name in file_fields:
            field = getattr(task, field_name)
            if field:
                try:
                    field.delete(save=False)  # 删除文件
                except:
                    pass  # 如果文件不存在，继续

        # 删除任务记录
        task.delete()

        # 如果任务在运行列表中，移除
        running_tasks.pop(task_id_str, None)

        return JsonResponse({
            'success': True,
            'message': f'任务 "{task_name}" 已成功删除',
            'redirect_url': '/upload/'  # 删除后重定向到上传页面
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


def get_task_list(request):
    """获取任务列表（用于删除页面）"""
    tasks = ReconstructionTask.objects.all().order_by('-created_at')

    task_list = []
    for task in tasks:
        task_list.append({
            'id': str(task.id),
            'name': task.name,
            'status': task.status,
            'status_display': task.get_status_display(),
            'created_at': task.created_at.strftime('%Y-%m-%d %H:%M'),
            'image_count': task.images.count(),
            'can_delete': task.can_be_deleted(),
            'progress': task.progress,
        })

    return JsonResponse({
        'success': True,
        'tasks': task_list
    })