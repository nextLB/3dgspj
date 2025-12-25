# admin.py 替换内容
from django.contrib import admin
from .models import UploadedImage, ReconstructionTask



@admin.register(UploadedImage)
class UploadedImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'image', 'task_link', 'uploaded_at', 'file_size']
    list_filter = ['uploaded_at', 'task']
    search_fields = ['image', 'task__name']

    def task_link(self, obj):
        if obj.task:
            return obj.task.name
        return '-'

    task_link.short_description = '关联任务'

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


@admin.register(ReconstructionTask)
class ReconstructionTaskAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'status', 'progress', 'image_count', 'created_at', 'estimated_time_display']
    list_filter = ['status', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['progress', 'started_at', 'completed_at']
    fieldsets = (
        ('基本信息', {
            'fields': ('name', 'description', 'status')
        }),
        ('重建参数', {
            'fields': ('resolution', 'iterations')
        }),
        ('处理进度', {
            'fields': ('progress', 'started_at', 'completed_at')
        }),
        ('结果文件', {
            'fields': ('result_ply', 'result_mesh', 'preview_image')
        }),
        ('日志信息', {
            'fields': ('log_file', 'error_message')
        }),
    )

    def image_count(self, obj):
        return obj.images.count()

    image_count.short_description = '图像数量'

    def estimated_time_display(self, obj):
        return f"{obj.estimated_time()} 分钟"

    estimated_time_display.short_description = '预计时间'