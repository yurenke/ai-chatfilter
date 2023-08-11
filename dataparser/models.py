from django.db import models

# Create your models here.

class CustomDictionaryWord(models.Model):
    text = models.CharField(max_length=32)
    pinyin = models.CharField(max_length=16, default="")
    freq = models.IntegerField(default=1)
    date = models.DateTimeField(auto_now_add=True, blank=True)
