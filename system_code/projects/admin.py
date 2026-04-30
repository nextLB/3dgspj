from django.contrib import admin
from .models import Project, Dataset, Job, EvaluationResult, DatasetDirectory


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'status', 'exp_name', 'iterations', 'created_at']
    list_filter = ['status', 'manhattan', 'eval_mode']
    search_fields = ['name', 'exp_name', 'user__username']
    readonly_fields = ['created_at', 'updated_at', 'started_at', 'finished_at']


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'format_type', 'status', 'image_count', 'source_path', 'created_by', 'created_at']
    list_filter = ['status', 'format_type']
    search_fields = ['name', 'source_path']


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ['id', 'project', 'dataset', 'job_type', 'status', 'created_at', 'started_at', 'finished_at']
    list_filter = ['job_type', 'status']
    readonly_fields = ['created_at']


@admin.register(EvaluationResult)
class EvaluationResultAdmin(admin.ModelAdmin):
    list_display = ['project', 'iteration', 'psnr', 'ssim', 'lpips', 'evaluated_at']


@admin.register(DatasetDirectory)
class DatasetDirectoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'path', 'is_active']
    list_filter = ['is_active']
