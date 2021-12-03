from django.conf.urls import include, url
from django.urls import path
from . import views

urlpatterns = [
    path('', views.digit_rec_model, name = 'digit_rec_model'),
    path('index', views.index, name = 'index'),
]