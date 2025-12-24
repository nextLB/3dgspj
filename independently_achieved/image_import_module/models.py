from django.db import models
import os
from django.utils import timezone


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

    class Meta:
        verbose_name = '上传的图像'
        verbose_name_plural = '上传的图像'
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"图像 {self.id} - {os.path.basename(self.image.name)}"

    def filename(self):
        return os.path.basename(self.image.name)