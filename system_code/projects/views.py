import os
import sys
import json
import subprocess
import threading
import time
import signal
import platform
import tempfile
from pathlib import Path

from django.utils import timezone

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST
from django.db.models import Q

from .models import Project, Dataset, Job, EvaluationResult, DatasetDirectory
from .forms import ProjectCreateForm, DatasetForm, TrainingConfigForm, RenderConfigForm, PreprocessForm


# ─── Cross-platform process group helpers ────────────────────────────────────

def _popen_kwargs():
    """Return kwargs for subprocess.Popen to create a new process group.

    Unix: use preexec_fn=os.setsid so os.killpg can kill the whole tree.
    Windows: use CREATE_NEW_PROCESS_GROUP flag.
    """
    if platform.system() == 'Windows':
        return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    setsid = getattr(os, 'setsid', None)
    if setsid is not None:
        return {'preexec_fn': setsid}
    return {}


def _kill_process_group(pid):
    """Kill an entire process group by its leader PID (cross-platform)."""
    if platform.system() == 'Windows':
        try:
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                           capture_output=True, timeout=5)
        except Exception:
            pass  # Process already gone or taskkill unavailable
    else:
        try:
            killpg = getattr(os, 'killpg', None)
            getpgid = getattr(os, 'getpgid', None)
            if killpg is not None and getpgid is not None:
                killpg(getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass  # Process already gone


def _build_train_cmd(project):
    """Build the full train_vast.py command for a project."""
    vast_base = settings.VASTGAUSSIAN_BASE
    train_script = os.path.join(vast_base, 'train_vast.py')
    cmd = [sys.executable, train_script, '-s', project.source_path]

    args_map = {
        '--exp_name': project.exp_name or project.name,
        '--resolution': project.resolution if project.resolution > 0 else None,
        '--llffhold': project.llffhold,
        '--m_region': project.m_region,
        '--n_region': project.n_region,
        '--extend_rate': project.extend_rate,
        '--visible_rate': project.visible_rate,
        '--iterations': project.iterations,
    }
    for flag, val in args_map.items():
        if val is not None and val != '' and val != -1:
            cmd.extend([flag, str(val)])

    if project.eval_mode:
        cmd.append('--eval')
    if project.white_background:
        cmd.append('--white_background')
    if project.quiet:
        cmd.append('--quiet')
    if project.manhattan:
        cmd.append('--manhattan')
        cmd.extend(['--platform', project.platform])
        if project.pos:
            cmd.extend(['--pos', project.pos])
        if project.rot:
            cmd.extend(['--rot', project.rot])

    return cmd


def _build_render_cmd(project, load_iteration=60_000):
    """Build the render.py command."""
    vast_base = settings.VASTGAUSSIAN_BASE
    render_script = os.path.join(vast_base, 'render.py')
    cmd = [sys.executable, render_script, '-s', project.source_path,
           '--exp_name', project.exp_name or project.name,
           '--load_iteration', str(load_iteration)]

    if project.resolution > 0:
        cmd.extend(['--resolution', str(project.resolution)])
    if project.eval_mode:
        cmd.append('--eval')
    if project.manhattan:
        cmd.append('--manhattan')
        if project.pos:
            cmd.extend(['--pos', project.pos])
        if project.rot:
            cmd.extend(['--rot', project.rot])

    return cmd


def _build_eval_cmd(project):
    """Build the metrics.py command."""
    vast_base = settings.VASTGAUSSIAN_BASE
    eval_script = os.path.join(vast_base, 'metrics.py')
    model_path = project.get_output_path()
    cmd = [sys.executable, eval_script, '-m', model_path]
    return cmd


def _build_convert_cmd(source_path, camera='OPENCV', resize=False):
    """Build the convert.py command for COLMAP preprocessing."""
    vast_base = settings.VASTGAUSSIAN_BASE
    convert_script = os.path.join(vast_base, 'convert.py')
    cmd = [sys.executable, convert_script, '-s', source_path, '--camera', camera]
    if resize:
        cmd.append('--resize')
    return cmd


def _run_convert_job_thread(job_id):
    """Background thread for COLMAP preprocessing (convert) jobs."""
    try:
        job = Job.objects.get(id=job_id)
    except Job.DoesNotExist:
        return

    job.status = 'running'
    job.started_at = timezone.now()
    job.save()

    dataset = job.dataset
    if dataset:
        dataset.status = 'processing'
        dataset.save()

    try:
        log_dir = os.path.dirname(dataset.source_path.rstrip('/\\')) if dataset else tempfile.gettempdir()
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'preprocess_{job.id}.log')
        job.log_file = log_file

        with open(log_file, 'w') as fp:
            fp.write(f"[{timezone.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting COLMAP preprocessing\n")
            fp.write(f"[{timezone.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset: {dataset.name if dataset else 'N/A'}\n")
            fp.write(f"[{timezone.now().strftime('%Y-%m-%d %H:%M:%S')}] Source: {dataset.source_path if dataset else 'N/A'}\n")
            fp.write(f"[{timezone.now().strftime('%Y-%m-%d %H:%M:%S')}] Command: {job.command}\n")
            fp.write("=" * 60 + "\n")
            fp.flush()

        process = subprocess.Popen(
            job.command,
            shell=True if isinstance(job.command, str) else False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            cwd=settings.VASTGAUSSIAN_BASE,
            **_popen_kwargs(),
        )
        job.pid = process.pid
        job.save()

        output_lines = []
        for line in iter(process.stdout.readline, ''):
            output_lines.append(line)
            with open(log_file, 'a') as fp:
                fp.write(line)
            if len(output_lines) % 100 == 0:
                job.output = ''.join(output_lines[-500:])
                job.save(update_fields=['output'])

        process.wait()
        job.output = ''.join(output_lines[-500:])

        if process.returncode == 0:
            job.status = 'completed'
            job.finished_at = timezone.now()
            job.duration_seconds = (job.finished_at - job.started_at).total_seconds()
            with open(log_file, 'a') as fp:
                fp.write(f"\n[{timezone.now().strftime('%Y-%m-%d %H:%M:%S')}] Preprocessing completed successfully.\n")
            if dataset:
                # 统计图片数量
                images_dir = os.path.join(dataset.source_path, 'images')
                if os.path.exists(images_dir):
                    dataset.image_count = len([f for f in os.listdir(images_dir)
                                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                dataset.status = 'ready'
                dataset.save()
        else:
            job.status = 'failed'
            job.finished_at = timezone.now()
            job.error_message = f'Process exited with code {process.returncode}'
            with open(log_file, 'a') as fp:
                fp.write(f"\n[{timezone.now().strftime('%Y-%m-%d %H:%M:%S')}] Preprocessing FAILED with code {process.returncode}\n")
            if dataset:
                dataset.status = 'failed'
                dataset.save()

    except Exception as e:
        job.status = 'failed'
        job.finished_at = timezone.now()
        job.error_message = str(e)
        if dataset:
            dataset.status = 'failed'
            dataset.save()

    job.save()


def _run_job_thread(job_id):
    """Background thread to execute a training/render/eval job."""
    try:
        job = Job.objects.get(id=job_id)
    except Job.DoesNotExist:
        return

    job.status = 'running'
    job.started_at = timezone.now()
    job.save()

    # Update project status
    project = job.project
    if project:
        project.status = 'training' if job.job_type == 'train' else (
            'rendering' if job.job_type == 'render' else 'evaluating'
        )
        project.save()

    try:
        log_dir = project.get_output_path() if project else settings.VASTGAUSSIAN_OUTPUT
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'{job.job_type}_{job.id}.log')
        job.log_file = log_file

        proj_name = project.name if project else '(独立任务)'
        with open(log_file, 'w') as fp:
            fp.write(f"[{timezone.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting {job.job_type} job for project: {proj_name}\n")
            fp.write(f"[{timezone.now().strftime('%Y-%m-%d %H:%M:%S')}] Command: {job.command}\n")
            fp.write("=" * 60 + "\n")
            fp.flush()

        process = subprocess.Popen(
            job.command,
            shell=True if isinstance(job.command, str) else False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            cwd=settings.VASTGAUSSIAN_BASE,
            **_popen_kwargs(),
        )
        job.pid = process.pid
        job.save()

        output_lines = []
        for line in iter(process.stdout.readline, ''):
            output_lines.append(line)
            with open(log_file, 'a') as fp:
                fp.write(line)
            # Periodically save output
            if len(output_lines) % 100 == 0:
                job.output = ''.join(output_lines[-500:])
                job.save(update_fields=['output'])

        process.wait()
        job.output = ''.join(output_lines[-500:])

        if process.returncode == 0:
            job.status = 'completed'
            job.finished_at = timezone.now()
            job.duration_seconds = (job.finished_at - job.started_at).total_seconds()
            with open(log_file, 'a') as fp:
                fp.write(f"\n[{timezone.now().strftime('%Y-%m-%d %H:%M:%S')}] Job completed successfully.\n")

            if project:
                if job.job_type == 'train':
                    project.status = 'completed'
                elif job.job_type == 'render':
                    project.status = 'completed'
                elif job.job_type == 'eval':
                    project.status = 'evaluated'
                    _parse_and_save_eval_results(project, job)
                project.save()
        else:
            job.status = 'failed'
            job.finished_at = timezone.now()
            job.duration_seconds = (job.finished_at - job.started_at).total_seconds()
            if project:
                project.status = 'failed'
                project.error_message = f'Process exited with code {process.returncode}'
            with open(log_file, 'a') as fp:
                fp.write(f"\n[{timezone.now().strftime('%Y-%m-%d %H:%M:%S')}] Job FAILED with code {process.returncode}\n")

    except Exception as e:
        job.status = 'failed'
        job.finished_at = timezone.now()
        job.error_message = str(e)
        if project:
            project.status = 'failed'
            project.error_message = str(e)
            project.save()

    project.save()
    job.save()


def _parse_and_save_eval_results(project, job):
    """Parse evaluation results from results.json and save to EvaluationResult model."""
    results_path = project.get_eval_results_path()
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
            for scene, methods in data.items():
                for method, metrics in methods.items():
                    EvaluationResult.objects.create(
                        project=project,
                        iteration=60000,
                        psnr=metrics.get('PSNR'),
                        ssim=metrics.get('SSIM'),
                        lpips=metrics.get('LPIPS'),
                        raw_data=json.dumps(metrics),
                        eval_job=job,
                    )
        except Exception as e:
            print(f"Failed to parse eval results: {e}")


# ─── Dashboard & Project Views ────────────────────────────────────────────────

@login_required
def dashboard(request):
    base_projects = Project.objects.filter(user=request.user)
    projects = base_projects.order_by('-created_at')[:10]
    recent_jobs = Job.objects.filter(project__user=request.user).order_by('-created_at')[:5]
    total_projects = base_projects.count()
    completed = base_projects.filter(status='completed').count()
    running = base_projects.filter(status__in=['training', 'rendering', 'evaluating', 'partitioning']).count()

    context = {
        'projects': projects,
        'recent_jobs': recent_jobs,
        'total_projects': total_projects,
        'completed_count': completed,
        'running_count': running,
        'vast_base': settings.VASTGAUSSIAN_BASE,
        'output_base': settings.VASTGAUSSIAN_OUTPUT,
    }
    return render(request, 'projects/dashboard.html', context)


@login_required
def project_list(request):
    projects = Project.objects.filter(user=request.user).order_by('-created_at')
    query = request.GET.get('q', '')
    if query:
        projects = projects.filter(Q(name__icontains=query) | Q(exp_name__icontains=query))
    return render(request, 'projects/project_list.html', {'projects': projects, 'query': query})


@login_required
def project_create(request):
    if request.method == 'POST':
        form = ProjectCreateForm(request.POST, user=request.user)
        if form.is_valid():
            project = form.save(commit=False)
            project.user = request.user
            project.save()

            # Create a dataset record if manual path was entered
            if form.cleaned_data.get('source_path_manual') and not form.cleaned_data.get('dataset'):
                Dataset.objects.create(
                    name=project.name + '_dataset',
                    source_path=form.cleaned_data['source_path_manual'],
                    created_by=request.user,
                )
            messages.success(request, f'项目 "{project.name}" 创建成功！')
            return redirect('projects:project_detail', pk=project.id)
    else:
        form = ProjectCreateForm(user=request.user)

    # Pre-configured datasets
    datasets = Dataset.objects.filter(created_by=request.user)
    presets = DatasetDirectory.objects.filter(is_active=True)

    return render(request, 'projects/project_create.html', {
        'form': form,
        'datasets': datasets,
        'presets': presets,
    })


@login_required
def project_detail(request, pk):
    project = get_object_or_404(Project, pk=pk, user=request.user)
    jobs = project.jobs.all().order_by('-created_at')[:20]
    results = project.results.all().order_by('-evaluated_at')

    # Check log file
    log_content = ''
    log_path = project.get_log_path()
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                log_content = f.read()[-50000:]
        except:
            pass

    # Check if output directory exists
    output_dir = project.get_output_path()
    output_exists = os.path.exists(output_dir)
    output_contents = []
    if output_exists:
        try:
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                output_contents.append({
                    'name': item,
                    'is_dir': os.path.isdir(item_path),
                    'size': os.path.getsize(item_path) if os.path.isfile(item_path) else 0,
                })
        except:
            pass

    context = {
        'project': project,
        'jobs': jobs,
        'results': results,
        'log_content': log_content[:30000] if log_content else '',
        'output_dir': output_dir,
        'output_exists': output_exists,
        'output_contents': sorted(output_contents, key=lambda x: (not x['is_dir'], x['name'])),
    }
    return render(request, 'projects/project_detail.html', context)


@login_required
def project_delete(request, pk):
    project = get_object_or_404(Project, pk=pk, user=request.user)
    if request.method == 'POST':
        name = project.name
        project.delete()
        messages.success(request, f'项目 "{name}" 已删除。')
        return redirect('projects:project_list')
    return render(request, 'projects/project_confirm_delete.html', {'project': project})


# ─── Training Operations ──────────────────────────────────────────────────────

@login_required
@require_POST
def project_train(request, pk):
    project = get_object_or_404(Project, pk=pk, user=request.user)

    if not os.path.exists(project.source_path):
        messages.error(request, f'数据路径不存在: {project.source_path}')
        return redirect('projects:project_detail', pk=pk)

    if project.jobs.filter(job_type='train', status__in=['pending', 'running']).exists():
        messages.warning(request, '该项目已有正在运行的训练任务。')
        return redirect('projects:project_detail', pk=pk)

    cmd = _build_train_cmd(project)
    cmd_str = subprocess.list2cmdline(cmd)

    job = Job.objects.create(
        project=project,
        job_type='train',
        status='pending',
        command=cmd_str,
    )

    # Start background thread
    t = threading.Thread(target=_run_job_thread, args=(job.id,), daemon=True)
    t.start()

    # Update project
    project.started_at = timezone.now()
    project.save()

    messages.success(request, f'训练任务已启动！Job ID: {job.id}')
    return redirect('projects:project_detail', pk=pk)


@login_required
def project_render(request, pk):
    project = get_object_or_404(Project, pk=pk, user=request.user)

    if request.method == 'POST':
        form = RenderConfigForm(request.POST)
        if form.is_valid():
            load_iteration = form.cleaned_data['load_iteration']
            cmd = _build_render_cmd(project, load_iteration)
            cmd_str = subprocess.list2cmdline(cmd)

            job = Job.objects.create(
                project=project,
                job_type='render',
                status='pending',
                command=cmd_str,
            )

            t = threading.Thread(target=_run_job_thread, args=(job.id,), daemon=True)
            t.start()

            messages.success(request, f'渲染任务已启动！Job ID: {job.id}')
            return redirect('projects:project_detail', pk=pk)
    else:
        form = RenderConfigForm()

    return render(request, 'projects/render_config.html', {'project': project, 'form': form})


@login_required
@require_POST
def project_eval(request, pk):
    project = get_object_or_404(Project, pk=pk, user=request.user)

    if project.jobs.filter(job_type='eval', status__in=['pending', 'running']).exists():
        messages.warning(request, '该项目已有正在运行的评估任务。')
        return redirect('projects:project_detail', pk=pk)

    cmd = _build_eval_cmd(project)
    cmd_str = subprocess.list2cmdline(cmd)

    job = Job.objects.create(
        project=project,
        job_type='eval',
        status='pending',
        command=cmd_str,
    )

    t = threading.Thread(target=_run_job_thread, args=(job.id,), daemon=True)
    t.start()

    messages.success(request, f'评估任务已启动！Job ID: {job.id}')
    return redirect('projects:project_detail', pk=pk)


# ─── Preprocessing (COLMAP Convert) ───────────────────────────────────────────

@login_required
def preprocess(request):
    """Preprocess raw images into VastGaussian-ready dataset via COLMAP."""
    if request.method == 'POST':
        form = PreprocessForm(request.POST)
        if form.is_valid():
            source_path = os.path.abspath(os.path.expanduser(form.cleaned_data['source_path']))
            dataset_name = form.cleaned_data['dataset_name']
            merge_subdirs = form.cleaned_data['merge_subdirs']
            camera = form.cleaned_data['camera_model']
            resize = form.cleaned_data['resize']

            if not os.path.exists(source_path):
                messages.error(request, f'路径不存在: {source_path}')
                return render(request, 'projects/preprocess.html', {'form': form})

            # Step 1: Optionally merge images from subdirectories into input/
            input_dir = os.path.join(source_path, 'input')
            if merge_subdirs:
                os.makedirs(input_dir, exist_ok=True)
                image_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
                copied = 0
                for item in os.listdir(source_path):
                    item_path = os.path.join(source_path, item)
                    if os.path.isdir(item_path) and item not in ('input', 'images', 'sparse', 'distorted'):
                        for fname in os.listdir(item_path):
                            if fname.lower().endswith(image_exts):
                                src = os.path.join(item_path, fname)
                                dst = os.path.join(input_dir, fname)
                                if not os.path.exists(dst):
                                    import shutil
                                    shutil.copy2(src, dst)
                                    copied += 1
                if copied == 0:
                    # Try copying from root level
                    for fname in os.listdir(source_path):
                        if fname.lower().endswith(image_exts):
                            src = os.path.join(source_path, fname)
                            dst = os.path.join(input_dir, fname)
                            if not os.path.exists(dst):
                                import shutil
                                shutil.copy2(src, dst)
                                copied += 1
            else:
                # Check if input/ already exists, if not, try to create from root images
                if not os.path.exists(input_dir):
                    os.makedirs(input_dir, exist_ok=True)
                    image_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
                    copied = 0
                    for fname in os.listdir(source_path):
                        if fname.lower().endswith(image_exts):
                            import shutil
                            shutil.copy2(os.path.join(source_path, fname), os.path.join(input_dir, fname))
                            copied += 1
                    if copied > 0:
                        messages.info(request, f'已将 {copied} 张图片复制到 input/ 目录。')

            # Step 2: Create dataset record
            dataset = Dataset.objects.create(
                name=dataset_name,
                source_path=source_path,
                format_type='colmap',
                status='processing',
                created_by=request.user,
            )

            # Step 3: Build and launch convert command
            cmd = _build_convert_cmd(source_path, camera, resize)
            cmd_str = subprocess.list2cmdline(cmd)

            job = Job.objects.create(
                project=None,
                dataset=dataset,
                job_type='convert',
                status='pending',
                command=cmd_str,
            )

            t = threading.Thread(target=_run_convert_job_thread, args=(job.id,), daemon=True)
            t.start()

            messages.success(request, f'预处理任务已启动！数据集 "{dataset_name}" 正在处理中。')
            return redirect('projects:dataset_list')
    else:
        form = PreprocessForm()

    return render(request, 'projects/preprocess.html', {'form': form})


@login_required
def dataset_list(request):
    datasets = Dataset.objects.filter(created_by=request.user)
    presets = DatasetDirectory.objects.filter(is_active=True)
    return render(request, 'projects/dataset_list.html', {
        'datasets': datasets,
        'presets': presets,
    })


@login_required
def dataset_create(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.created_by = request.user

            # Normalize path
            path = os.path.abspath(os.path.expanduser(dataset.source_path))
            if not os.path.exists(path):
                messages.warning(request, f'路径 "{path}" 不存在，但数据集仍会创建。')
            dataset.source_path = path
            dataset.save()
            messages.success(request, f'数据集 "{dataset.name}" 创建成功！')
            return redirect('projects:dataset_list')
    else:
        form = DatasetForm()
    return render(request, 'projects/dataset_form.html', {'form': form, 'title': '导入数据集'})


@login_required
def dataset_delete(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk, created_by=request.user)
    if request.method == 'POST':
        dataset.delete()
        messages.success(request, f'数据集 "{dataset.name}" 已删除。')
    return redirect('projects:dataset_list')


# ─── Job & Status Views ──────────────────────────────────────────────────────

def _user_owns_job(job, user):
    """Check if user owns the project or dataset associated with a job."""
    if job.project and job.project.user == user:
        return True
    if job.dataset and job.dataset.created_by == user:
        return True
    return False


@login_required
def job_detail(request, pk):
    job = get_object_or_404(Job, pk=pk)
    if not _user_owns_job(job, request.user):
        from django.http import Http404
        raise Http404()

    log_content = ''
    if job.log_file and os.path.exists(job.log_file):
        try:
            with open(job.log_file, 'r') as f:
                log_content = f.read()
        except:
            log_content = job.output

    return render(request, 'projects/job_detail.html', {
        'job': job,
        'log_content': log_content,
    })


@login_required
def job_list(request):
    project_jobs = Job.objects.filter(project__user=request.user)
    dataset_jobs = Job.objects.filter(dataset__created_by=request.user, project__isnull=True)
    from itertools import chain
    combined = sorted(
        chain(project_jobs, dataset_jobs),
        key=lambda j: j.created_at,
        reverse=True
    )[:50]
    return render(request, 'projects/job_list.html', {'jobs': combined})


@login_required
def job_stop(request, pk):
    job = get_object_or_404(Job, pk=pk)
    if not _user_owns_job(job, request.user):
        from django.http import Http404
        raise Http404()
    if job.status == 'running' and job.pid:
        _kill_process_group(job.pid)
        job.status = 'failed'
        job.error_message = '手动停止'
        job.save()
        messages.success(request, '任务已停止。')
    return redirect('projects:job_detail', pk=pk)


# ─── API-like JSON endpoints for real-time updates ────────────────────────────

@login_required
def api_project_status(request, pk):
    project = get_object_or_404(Project, pk=pk, user=request.user)
    running_job = project.jobs.filter(status__in=['pending', 'running']).first()
    return JsonResponse({
        'status': project.status,
        'status_display': project.get_status_display(),
        'job_running': running_job is not None,
        'job_id': running_job.id if running_job else None,
        'error_message': project.error_message,
    })


@login_required
def api_job_status(request, pk):
    job = get_object_or_404(Job, pk=pk)
    if not _user_owns_job(job, request.user):
        return JsonResponse({'error': 'permission denied'}, status=403)
    return JsonResponse({
        'id': job.id,
        'status': job.status,
        'status_display': job.get_status_display(),
        'job_type': job.job_type,
        'job_type_display': job.get_job_type_display(),
        'pid': job.pid,
        'duration_seconds': job.duration_seconds,
        'error_message': job.error_message,
    })


@login_required
def api_job_log(request, pk):
    job = get_object_or_404(Job, pk=pk)
    if not _user_owns_job(job, request.user):
        return JsonResponse({'error': 'permission denied'}, status=403)
    offset = int(request.GET.get('offset', 0))
    max_lines = 200

    if job.log_file and os.path.exists(job.log_file):
        try:
            with open(job.log_file, 'r') as f:
                content = f.read()
        except:
            content = job.output
    else:
        content = job.output

    lines = content.split('\n')
    total = len(lines)
    start = max(0, total - max_lines)
    sliced = lines[start:]
    new_content = '\n'.join(sliced)

    return JsonResponse({
        'total_lines': total,
        'lines': sliced,
        'content': new_content,
        'status': job.status,
    })


@login_required
def api_check_output(request, pk):
    """Check if output directory has point cloud data."""
    project = get_object_or_404(Project, pk=pk, user=request.user)
    output_dir = project.get_output_path()
    status_info = 'not_found'
    iterations = []

    if os.path.exists(output_dir):
        pc_dir = os.path.join(output_dir, 'point_cloud')
        if os.path.exists(pc_dir):
            for d in os.listdir(pc_dir):
                if d.startswith('iteration_'):
                    it = d.replace('iteration_', '')
                    iterations.append(it)
            status_info = 'has_data' if iterations else 'no_iterations'
        else:
            status_info = 'no_point_cloud'
    return JsonResponse({
        'status': status_info,
        'output_dir': output_dir,
        'exists': os.path.exists(output_dir),
        'iterations': sorted(iterations),
    })
