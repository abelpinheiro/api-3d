from django.urls import path
from . import views

urlpatterns = [
    path('', views.reconstruct, name='reconstruct'),
]