from django.conf.urls import include, url
from django.urls import path
from . import views

urlpatterns = [
    path('', views.hello, name = 'home'),
    path('add', views.add, name = 'add'),
]