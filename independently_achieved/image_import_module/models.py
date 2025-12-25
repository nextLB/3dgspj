# models.py 替换内容
from django.db import models
import os
import uuid
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator


class ReconstructionTask(models.Model):
    """三维重建任务模型"""
    STATUS_CHOICES = [
        ('pending', '等待中'),
        ('processing', '处理中'),
        ('completed', '已完成'),
        ('failed', '失败'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, verbose_name='任务名称', default='3D重建任务')
    description = models.TextField(blank=True, verbose_name='任务描述')

    # 重建参数
    resolution = models.IntegerField(
        default=1024,
        validators=[MinValueValidator(256), MaxValueValidator(4096)],
        verbose_name='重建分辨率'
    )
    iterations = models.IntegerField(
        default=30000,
        validators=[MinValueValidator(1000), MaxValueValidator(100000)],
        verbose_name='迭代次数'
    )

    # 任务状态
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        verbose_name='任务状态'
    )
    progress = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        verbose_name='进度百分比'
    )

    # 时间和结果
    created_at = models.DateTimeField(default=timezone.now, verbose_name='创建时间')
    started_at = models.DateTimeField(null=True, blank=True, verbose_name='开始时间')
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name='完成时间')

    # 结果存储
    result_ply = models.FileField(
        upload_to='reconstruction_results/ply/',
        null=True,
        blank=True,
        verbose_name='PLY格式结果'
    )
    result_mesh = models.FileField(
        upload_to='reconstruction_results/mesh/',
        null=True,
        blank=True,
        verbose_name='网格模型'
    )
    preview_image = models.ImageField(
        upload_to='reconstruction_previews/',
        null=True,
        blank=True,
        verbose_name='预览图'
    )

    # 日志和错误信息
    log_file = models.FileField(
        upload_to='reconstruction_logs/',
        null=True,
        blank=True,
        verbose_name='日志文件'
    )
    error_message = models.TextField(blank=True, verbose_name='错误信息')

    class Meta:
        verbose_name = '三维重建任务'
        verbose_name_plural = '三维重建任务'
        ordering = ['-created_at']

    def __str__(self):
        return f"任务 {self.id} - {self.name} ({self.get_status_display()})"

    def image_count(self):
        return self.images.count()

    def is_ready_for_reconstruction(self):
        """检查是否可以进行三维重建（至少需要2张图像）"""
        return self.images.count() >= 2 and self.status == 'pending'

    def estimated_time(self):
        """估算处理时间（基于图像数量和参数）"""
        base_time = 5  # 基础时间（分钟）
        per_image_time = 0.5  # 每张图像额外时间
        total_images = self.images.count()

        # 考虑迭代次数的影响
        time_multiplier = self.iterations / 30000

        estimated_minutes = (base_time + per_image_time * total_images) * time_multiplier
        return round(estimated_minutes, 1)


class UploadedImage(models.Model):
    """存储上传的图像模型"""
    image = models.ImageField(
        upload_to='uploaded_images/%Y/%m/%d/',  # 按日期组织目录
        verbose_name='上传的图像'
    )
    uploaded_at = models.DateTimeField(
        default=timezone.now,
        verbose_name='上传时间'
    )
    task = models.ForeignKey(
        ReconstructionTask,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='images',
        verbose_name='关联的重建任务'
    )

    class Meta:
        verbose_name = '上传的图像'
        verbose_name_plural = '上传的图像'
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"图像 {self.id} - {os.path.basename(self.image.name)}"

    def filename(self):
        return os.path.basename(self.image.name)