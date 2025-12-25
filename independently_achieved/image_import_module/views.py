# views.py 替换内容
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
import uuid
from .forms import SingleImageUploadForm, MultipleImageUploadForm, ReconstructionSettingsForm
from .models import UploadedImage, ReconstructionTask
import os


def upload_image(request):
    """处理图像上传和显示"""
    max_size = 20 * 1024 * 1024  # 20MB

    # 初始化表单
    single_form = SingleImageUploadForm()
    multiple_form = MultipleImageUploadForm()
    recent_images = UploadedImage.objects.all().order_by('-uploaded_at')[:10]
    recent_tasks = ReconstructionTask.objects.all().order_by('-created_at')[:5]

    context = {
        'single_form': single_form,
        'multiple_form': multiple_form,
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
                image = form.save()
                context.update({
                    'uploaded_image': image,
                    'image_url': image.image.url,
                    'success_message': '单张图像上传成功！',
                    'active_tab': 'single'
                })
            else:
                context['single_form'] = form
                context['active_tab'] = 'single'

        elif upload_type == 'multiple':
            # 手动处理多文件上传
            form = MultipleImageUploadForm(request.POST)
            if form.is_valid():
                task_name = form.cleaned_data.get('task_name', f'批量上传任务-{uuid.uuid4().hex[:8]}')

                # 获取上传的文件
                files = request.FILES.getlist('images')

                if not files:
                    context['multiple_form'] = form
                    context['active_tab'] = 'multiple'
                    context['error_message'] = '请至少选择一个文件'
                    return render(request, 'image_import_module/upload.html', context)

                if len(files) > 50:
                    context['multiple_form'] = form
                    context['active_tab'] = 'multiple'
                    context['error_message'] = '一次最多上传50张图像'
                    return render(request, 'image_import_module/upload.html', context)

                if len(files) < 2:
                    context['multiple_form'] = form
                    context['active_tab'] = 'multiple'
                    context['error_message'] = '多图上传至少需要2张图像'
                    return render(request, 'image_import_module/upload.html', context)

                # 验证每个文件
                allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif']
                saved_images = []
                errors = []

                # 创建重建任务
                task = ReconstructionTask.objects.create(
                    name=task_name,
                    status='pending'
                )

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

                if saved_images:
                    context.update({
                        'uploaded_images': saved_images,
                        'task': task,
                        'success_message': f'成功上传 {len(saved_images)} 张图像！已创建重建任务。',
                        'active_tab': 'multiple'
                    })
            else:
                context['multiple_form'] = form
                context['active_tab'] = 'multiple'

    return render(request, 'image_import_module/upload.html', context)


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
        if not task.is_ready_for_reconstruction():
            return JsonResponse({
                'success': False,
                'error': '任务无法开始重建：需要至少2张图像且状态为等待中'
            })

        # 更新任务状态
        task.status = 'processing'
        task.progress = 0
        task.started_at = timezone.now()
        task.save()

        # TODO: 这里需要调用实际的三维重建算法
        # 例如：调用你的3D高斯溅射算法
        # 1. 获取任务的所有图像
        # 2. 调用重建算法（可能是外部命令或Python库）
        # 3. 保存结果文件到任务对象中

        # 示例：模拟重建过程
        import threading
        from .reconstruction_worker import process_reconstruction_task

        # 启动后台线程处理重建任务
        thread = threading.Thread(target=process_reconstruction_task, args=(task.id,))
        thread.daemon = True
        thread.start()

        return JsonResponse({
            'success': True,
            'message': '三维重建任务已开始',
            'task_id': str(task.id),
            'estimated_time': task.estimated_time()
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
            'status': task.status,
            'progress': task.progress,
            'image_count': task.image_count(),
            'created_at': task.created_at.isoformat(),
            'error_message': task.error_message
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
        'can_start_reconstruction': task.is_ready_for_reconstruction()
    })


def home(request):
    """首页，重定向到上传页面"""
    return redirect('image_import_module:upload_image')