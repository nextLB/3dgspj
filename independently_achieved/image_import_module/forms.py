# forms.py 替换内容
from django import forms
from .models import UploadedImage, ReconstructionTask
import os


class SingleImageUploadForm(forms.ModelForm):
    """单图像上传表单"""

    class Meta:
        model = UploadedImage
        fields = ['image']
        widgets = {
            'image': forms.ClearableFileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'id': 'singleImageInput'
            })
        }
        labels = {
            'image': '选择单个图像文件'
        }

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            max_size = 20 * 1024 * 1024  # 20MB
            if image.size > max_size:
                raise forms.ValidationError(f'图像文件大小不能超过20MB（当前：{image.size / (1024 * 1024):.1f}MB）')

            allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif']
            ext = os.path.splitext(image.name)[1].lower()
            if ext not in allowed_extensions:
                raise forms.ValidationError('只支持以下图像格式：JPG, PNG, GIF, BMP, TIFF')

        return image


# 简单化：多文件上传表单只处理任务名称，文件在视图中处理
class MultipleImageUploadForm(forms.Form):
    """多图像上传表单"""
    task_name = forms.CharField(
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '可选：为这组图像命名任务'
        }),
        label='任务名称'
    )


class ReconstructionSettingsForm(forms.ModelForm):
    """三维重建参数设置表单"""

    class Meta:
        model = ReconstructionTask
        fields = ['name', 'description', 'resolution', 'iterations']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '输入任务名称'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': '可选：输入任务描述'
            }),
            'resolution': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 256,
                'max': 4096,
                'step': 256
            }),
            'iterations': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 1000,
                'max': 100000,
                'step': 1000
            })
        }
        labels = {
            'name': '任务名称',
            'description': '任务描述',
            'resolution': '重建分辨率',
            'iterations': '迭代次数'
        }