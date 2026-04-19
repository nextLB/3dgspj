


# forms.py 替换内容 - 在 MultipleImageUploadForm 中添加 dataset_path 字段
from django import forms
from .models import UploadedImage, ReconstructionTask
import os


class SingleImageUploadForm(forms.ModelForm):
    """单图像上传表单"""
    task_name = forms.CharField(
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '可选：为这张图像命名任务'
        }),
        label='任务名称',
        help_text='如果不填写，将自动生成任务名称'
    )

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


# 修改：多文件上传表单添加数据集路径字段
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

    # 新增：本地数据集路径字段
    dataset_path = forms.CharField(
        max_length=500,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '可选：输入本地数据集文件夹路径，如：/home/user/dataset/images/'
        }),
        label='本地数据集路径',
        help_text='如果提供此路径，将优先使用本地文件进行重建'
    )





# ============ 新增：分块重建表单 ============
class CubeReconstructionForm(forms.Form):
    """分块重建表单"""
    task_name = forms.CharField(
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '可选：为方块重建任务命名'
        }),
        label='任务名称',
        help_text='如果不填写，将自动生成任务名称'
    )

    # 新增：本地数据集路径字段
    dataset_path = forms.CharField(
        max_length=500,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '可选：输入本地数据集文件夹路径，如：/home/user/dataset/images/'
        }),
        label='本地数据集路径',
        help_text='如果提供此路径，将使用本地文件进行分块重建'
    )

    # 这里可以添加方块重建特有的参数，例如：
    cube_size = forms.IntegerField(
        required=False,
        initial=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '方块尺寸（米）'
        }),
        label='方块尺寸',
        help_text='方块的边长（单位：米）'
    )

    position_x = forms.FloatField(
        required=False,
        initial=0.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'X坐标'
        }),
        label='X坐标'
    )

    position_y = forms.FloatField(
        required=False,
        initial=0.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Y坐标'
        }),
        label='Y坐标'
    )

    position_z = forms.FloatField(
        required=False,
        initial=0.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Z坐标'
        }),
        label='Z坐标'
    )

# 您可以根据需要添加更多字段
    # 例如：颜色选择器、纹理上传、旋转角度等




# ============ 新增：新分块重建表单 ============
class NewCubeReconstructionForm(forms.Form):
    """新分块重建表单"""
    task_name = forms.CharField(
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '可选：为新分块重建任务命名'
        }),
        label='任务名称',
        help_text='如果不填写，将自动生成任务名称'
    )

    dataset_path = forms.CharField(
        max_length=500,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '可选：输入本地数据集文件夹路径'
        }),
        label='本地数据集路径',
        help_text='如果提供此路径，将使用本地文件进行新分块重建'
    )

    cube_size = forms.IntegerField(
        required=False,
        initial=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '方块尺寸（米）'
        }),
        label='方块尺寸',
        help_text='方块的边长（单位：米）'
    )

    grid_resolution = forms.IntegerField(
        required=False,
        initial=3,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': 1,
            'max': 10,
            'placeholder': '网格分辨率'
        }),
        label='网格分辨率',
        help_text='分块网格的分辨率（如3x3x3）'
    )

    position_x = forms.FloatField(
        required=False,
        initial=0.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'X坐标'
        }),
        label='X坐标'
    )

    position_y = forms.FloatField(
        required=False,
        initial=0.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Y坐标'
        }),
        label='Y坐标'
    )

    position_z = forms.FloatField(
        required=False,
        initial=0.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Z坐标'
        }),
        label='Z坐标'
    )




class ReconstructionSettingsForm(forms.ModelForm):
    """三维重建参数设置表单"""

    class Meta:
        model = ReconstructionTask
        fields = ['name', 'description', 'dataset_path', 'resolution', 'iterations']
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
            'dataset_path': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '可选：输入本地数据集路径'
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
            'dataset_path': '本地数据集路径',
            'resolution': '重建分辨率',
            'iterations': '迭代次数'
        }