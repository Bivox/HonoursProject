from django.db import models
from django import forms

# Create your models here.
class Image(models.Model):
    #canvas = models.
    image = models.ImageField(upload_to='images')

    def __str__(self):
        return self.title

