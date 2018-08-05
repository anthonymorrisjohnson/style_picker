from django.urls import path

from . import views

urlpatterns = [
    path('upload', views.upload_file, name='upload_file'),
    path('', views.index, name='index'),
    path('activate/<int:id>', views.activate, name='activate'),
    path('current_model', views.current_model)
]