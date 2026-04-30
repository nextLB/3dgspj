from django.urls import path
from . import views

app_name = 'projects'

urlpatterns = [
    # Dashboard
    path('dashboard/', views.dashboard, name='dashboard'),

    # Project management
    path('', views.project_list, name='project_list'),
    path('create/', views.project_create, name='project_create'),
    path('<int:pk>/', views.project_detail, name='project_detail'),
    path('<int:pk>/delete/', views.project_delete, name='project_delete'),

    # Training
    path('<int:pk>/train/', views.project_train, name='project_train'),
    path('<int:pk>/render/', views.project_render, name='project_render'),
    path('<int:pk>/eval/', views.project_eval, name='project_eval'),

    # Datasets
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('datasets/create/', views.dataset_create, name='dataset_create'),
    path('datasets/<int:pk>/delete/', views.dataset_delete, name='dataset_delete'),

    # Jobs
    path('jobs/', views.job_list, name='job_list'),
    path('jobs/<int:pk>/', views.job_detail, name='job_detail'),
    path('jobs/<int:pk>/stop/', views.job_stop, name='job_stop'),

    # API endpoints
    path('api/project/<int:pk>/status/', views.api_project_status, name='api_project_status'),
    path('api/job/<int:pk>/status/', views.api_job_status, name='api_job_status'),
    path('api/job/<int:pk>/log/', views.api_job_log, name='api_job_log'),
    path('api/project/<int:pk>/check_output/', views.api_check_output, name='api_check_output'),
]
