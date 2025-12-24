from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from .models import UploadedImage



def upload_image(request):
    """处理图像上传和显示"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()

            # 构建图像URL用于显示
            image_url = uploaded_image.image.url

            # 获取最近上传的所有图像
            recent_images = UploadedImage.objects.all().order_by('-uploaded_at')[:10]

            return render(request, 'image_import_module/upload.html', {
                'form': ImageUploadForm(),
                'uploaded_image': uploaded_image,
                'image_url': image_url,
                'recent_images': recent_images,
                'success_message': '图像上传成功！'
            })
    else:
        form = ImageUploadForm()

    # 获取最近上传的图像
    recent_images = UploadedImage.objects.all().order_by('-uploaded_at')[:10]

    max_size = 20 * 1024 * 1024  # 20MB

    return render(request, 'image_import_module/upload.html', {
        'form': form,
        'recent_images': recent_images,
        'max_size': max_size
    })


def home(request):
    """首页，重定向到上传页面"""
    return redirect('image_import_module:upload_image')