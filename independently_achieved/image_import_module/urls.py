# urls.py 替换内容
from django.urls import path
from . import views

app_name = 'image_import_module'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_image, name='upload_image'),
    path('start-reconstruction/', views.start_reconstruction, name='start_reconstruction'),
    path('task/<uuid:task_id>/', views.task_detail, name='task_detail'),
    path('task/<uuid:task_id>/status/', views.get_task_status, name='get_task_status'),
]