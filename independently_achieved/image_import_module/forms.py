from django import forms
from .models import UploadedImage


class ImageUploadForm(forms.ModelForm):
    """图像上传表单"""

    class Meta:
        model = UploadedImage
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            })
        }
        labels = {
            'image': '选择图像文件'
        }

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            # 检查文件大小（限制为5MB）
            max_size = 5 * 1024 * 1024  # 5MB
            if image.size > max_size:
                raise forms.ValidationError('图像文件大小不能超过5MB')

            # 检查文件类型
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
            import os
            ext = os.path.splitext(image.name)[1].lower()
            if ext not in allowed_extensions:
                raise forms.ValidationError('只支持以下图像格式：JPG, PNG, GIF, BMP')

        return image