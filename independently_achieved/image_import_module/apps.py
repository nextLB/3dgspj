from django.apps import AppConfig

class ImageImportModuleConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'  # 修正路径
    name = 'image_import_module'
    verbose_name = '图像导入模块'