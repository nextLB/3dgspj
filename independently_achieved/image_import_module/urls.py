from django.urls import path
from . import views

app_name = 'image_import_module'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_image, name='upload_image'),
]