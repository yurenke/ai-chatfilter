from django.urls import path
from django.templatetags.static import static

from . import views

urlpatterns = [
    # path('', views.index, name='index'),
    # path('', views.room, name='room'),
    path('zh', views.controller_zh, name='controller_zh'),
    path('en', views.controller_en, name='controller_en'),
    path('axios.min.js', views.axiosjs, name='axios')
]