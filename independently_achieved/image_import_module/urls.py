
# urls.py - 添加终止和删除任务的路由
from django.urls import path
from . import views

app_name = 'image_import_module'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_image, name='upload_image'),
    path('start-reconstruction/', views.start_reconstruction, name='start_reconstruction'),
    path('cancel-reconstruction/', views.cancel_reconstruction, name='cancel_reconstruction'),
    path('delete-task/', views.delete_reconstruction_task, name='delete_reconstruction_task'),
    path('task-list/', views.get_task_list, name='get_task_list'),
    path('task/<uuid:task_id>/', views.task_detail, name='task_detail'),
    path('task/<uuid:task_id>/status/', views.get_task_status, name='get_task_status'),
]