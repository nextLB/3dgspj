import os
import json
from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
from django.utils import timezone


class Dataset(models.Model):
    FORMAT_CHOICES = [
        ('colmap', 'COLMAP'),
        ('blender', 'Blender/NeRF'),
    ]
    STATUS_CHOICES = [
        ('ready', '就绪'),
        ('processing', '预处理中'),
        ('failed', '预处理失败'),
    ]
    name = models.CharField('数据集名称', max_length=100)
    source_path = models.CharField('数据路径', max_length=500, help_text='包含COLMAP或Blender数据的目录路径')
    format_type = models.CharField('数据格式', max_length=20, choices=FORMAT_CHOICES, default='colmap')
    status = models.CharField('状态', max_length=20, choices=STATUS_CHOICES, default='ready')
    image_count = models.IntegerField('图片数量', null=True, blank=True)
    description = models.TextField('描述', blank=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='创建者')
    created_at = models.DateTimeField('创建时间', auto_now_add=True)

    class Meta:
        verbose_name = '数据集'
        verbose_name_plural = '数据集'

    def __str__(self):
        return f'{self.name} [{self.get_status_display()}]'

    def exists(self):
        return os.path.exists(self.source_path)


class Project(models.Model):
    STATUS_CHOICES = [
        ('created', '已创建'),
        ('partitioning', '数据分区中'),
        ('training', '训练中'),
        ('completed', '训练完成'),
        ('rendering', '渲染中'),
        ('evaluating', '评估中'),
        ('evaluated', '已评估'),
        ('failed', '失败'),
    ]
    name = models.CharField('项目名称', max_length=200)
    description = models.TextField('项目描述', blank=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='创建者', related_name='projects')
    dataset = models.ForeignKey(Dataset, on_delete=models.SET_NULL, null=True, blank=True, verbose_name='数据集')
    status = models.CharField('状态', max_length=20, choices=STATUS_CHOICES, default='created')

    source_path = models.CharField('数据源路径', max_length=500, blank=True)
    exp_name = models.CharField('实验名称', max_length=200, blank=True)
    resolution = models.IntegerField('分辨率缩放', default=-1, help_text='-1=自动, 1=原始, 2=1/2, 4=1/4, 8=1/8')
    eval_mode = models.BooleanField('评估模式', default=False)
    llffhold = models.IntegerField('LLFF测试间隔', default=83)
    white_background = models.BooleanField('白色背景', default=False)
    sh_degree = models.IntegerField('SH度数', default=3)
    data_device = models.CharField('数据设备', max_length=10, default='cuda')

    # Manhattan alignment
    manhattan = models.BooleanField('曼哈顿对齐', default=False)
    platform = models.CharField('对齐平台', max_length=10, default='cc', choices=[('cc', 'CloudCompare'), ('tj', 'Three.js')])
    pos = models.CharField('平移向量', max_length=200, blank=True, default='', help_text='例如: "25.6 0.0 -12.0"')
    rot = models.CharField('旋转矩阵/向量', max_length=500, blank=True, default='', help_text='cc:9个元素 tj:3个元素')

    # Data partition
    m_region = models.IntegerField('X方向分区数', default=3)
    n_region = models.IntegerField('Z方向分区数', default=3)
    extend_rate = models.FloatField('边界扩展率', default=0.2)
    visible_rate = models.FloatField('可见性比率', default=0.25)

    # Training params
    iterations = models.IntegerField('训练迭代数', default=30_000)
    test_iterations = models.CharField('测试迭代点', max_length=100, default='7_000 30_000 60_000')
    save_iterations = models.CharField('保存迭代点', max_length=100, default='7_000 30_000 60_000')
    checkpoint = models.CharField('检查点', max_length=500, blank=True, null=True)
    quiet = models.BooleanField('安静模式', default=False)

    # Paths
    model_path = models.CharField('模型输出路径', max_length=500, blank=True)

    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)
    started_at = models.DateTimeField('开始时间', null=True, blank=True)
    finished_at = models.DateTimeField('完成时间', null=True, blank=True)
    error_message = models.TextField('错误信息', blank=True, default='')

    class Meta:
        verbose_name = '项目'
        verbose_name_plural = '项目'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.name} ({self.get_status_display()})'

    def get_params_dict(self):
        """Build command-line argument dict for train_vast.py / render.py"""
        params = {
            '-s': self.source_path,
            '--exp_name': self.exp_name or self.name,
        }
        if self.resolution > 0:
            params['--resolution'] = str(self.resolution)
        if self.eval_mode:
            params['--eval'] = ''
        params['--llffhold'] = str(self.llffhold)
        if self.white_background:
            params['--white_background'] = ''
        if self.manhattan:
            params['--manhattan'] = ''
            params['--platform'] = self.platform
            if self.pos:
                params['--pos'] = f'"{self.pos}"'
            if self.rot:
                params['--rot'] = f'"{self.rot}"'
        params['--m_region'] = str(self.m_region)
        params['--n_region'] = str(self.n_region)
        params['--extend_rate'] = str(self.extend_rate)
        params['--visible_rate'] = str(self.visible_rate)
        params['--iterations'] = str(self.iterations)
        if self.quiet:
            params['--quiet'] = ''
        return params

    def get_output_path(self):
        base = settings.VASTGAUSSIAN_OUTPUT
        exp = self.exp_name or self.name
        return os.path.join(base, exp)

    def get_log_path(self):
        out = self.get_output_path()
        return os.path.join(out, 'training.log')

    def get_eval_results_path(self):
        out = self.get_output_path()
        return os.path.join(out, 'results.json')


class Job(models.Model):
    JOB_TYPES = [
        ('train', '训练'),
        ('render', '渲染'),
        ('eval', '评估'),
        ('convert', '格式转换'),
        ('merge', '无缝合并'),
    ]
    STATUS_CHOICES = [
        ('pending', '等待中'),
        ('running', '运行中'),
        ('completed', '已完成'),
        ('failed', '失败'),
    ]
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='jobs', verbose_name='项目',
                                 null=True, blank=True)
    job_type = models.CharField('任务类型', max_length=20, choices=JOB_TYPES)
    status = models.CharField('状态', max_length=20, choices=STATUS_CHOICES, default='pending')
    command = models.TextField('命令', blank=True)
    pid = models.IntegerField('进程ID', null=True, blank=True)
    log_file = models.CharField('日志文件', max_length=500, blank=True)
    output = models.TextField('输出', blank=True, default='')
    error_message = models.TextField('错误信息', blank=True, default='')
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.FloatField('耗时(秒)', null=True, blank=True)
    # 预处理专用 - 关联创建的数据集
    dataset = models.ForeignKey('Dataset', on_delete=models.SET_NULL, null=True, blank=True, verbose_name='生成的数据集')

    class Meta:
        verbose_name = '任务'
        verbose_name_plural = '任务'
        ordering = ['-created_at']

    def __str__(self):
        proj_name = self.project.name if self.project else '(预处理)'
        return f'{self.get_job_type_display()} - {proj_name} [{self.get_status_display()}]'


class EvaluationResult(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='results', verbose_name='项目')
    iteration = models.IntegerField('迭代次数', default=60_000)
    psnr = models.FloatField('PSNR', null=True, blank=True)
    ssim = models.FloatField('SSIM', null=True, blank=True)
    lpips = models.FloatField('LPIPS', null=True, blank=True)
    raw_data = models.TextField('原始数据', blank=True, default='')
    evaluated_at = models.DateTimeField('评估时间', auto_now_add=True)
    eval_job = models.ForeignKey(Job, on_delete=models.SET_NULL, null=True, blank=True, verbose_name='评估任务')

    class Meta:
        verbose_name = '评估结果'
        verbose_name_plural = '评估结果'

    def __str__(self):
        return f'{self.project.name} @{self.iteration} - PSNR:{self.psnr:.2f}'


class DatasetDirectory(models.Model):
    """Pre-configured dataset directories for quick selection"""
    name = models.CharField('名称', max_length=100)
    path = models.CharField('路径', max_length=500)
    description = models.TextField('描述', blank=True)
    is_active = models.BooleanField('启用', default=True)

    class Meta:
        verbose_name = '预置数据集目录'
        verbose_name_plural = '预置数据集目录'

    def __str__(self):
        return self.name
