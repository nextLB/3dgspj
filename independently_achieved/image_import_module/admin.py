from django.contrib import admin
from .models import UploadedImage


# Register your models here.
@admin.register(UploadedImage)
class UploadedImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'image', 'uploaded_at', 'file_size']
    list_filter = ['uploaded_at']
    search_fields = ['image']

    def file_size(self, obj):
        if obj.image:
            try:
                size = obj.image.size
                if size < 1024:
                    return f"{size} B"
                elif size < 1024 * 1024:
                    return f"{size / 1024:.1f} KB"
                else:
                    return f"{size / (1024 * 1024):.1f} MB"
            except:
                return "N/A"
        return "N/A"

    file_size.short_description = '文件大小'