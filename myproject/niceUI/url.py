from django.conf.urls import include, url
from django.urls import path
from . import views

urlpatterns = [
    path('', views.load_model, name = 'load_model'),
    path('canvas', views.canvas, name = 'canvas'),
    path('crop', views.crop, name = 'crop'),
    path('digit_rec_model', views.digit_rec_model, name = 'digit_rec_model'),
    path('predict_digit', views.predict_digit, name = 'predict_digit'),
    
]