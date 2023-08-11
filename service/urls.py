"""service URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.http import Http404, HttpResponse, JsonResponse
from django.views.static import serve
from service import instance
from .views import ServiceJSONDataAPIView, ServiceUploadAPIView, ServiceRemoveAPIView, ServicePinyinBlockListAPIView
from .views import ServiceNicknamePinyinBlockListAPIView, ServiceAlertWordsListAPIView, TwiceServiceAPIView, NicknameTwiceServiceAPIView, ServiceCommandAPIView, TrainServiceAPIView, train_val_complete_handler
import os

def read_model_path(request, name):
    main_service = instance.get_main_service(is_admin=True)
    nickname_filter = instance.get_nickname_filter(is_admin=True)

    if name == 'chat':
        _path = main_service.get_chat_model_path()
    elif name == 'nickname':
        _path = nickname_filter.get_nickname_model_path()
    else:
        raise Http404('Model Not Found.')

    if _path:
        return serve(request, os.path.basename(_path), os.path.dirname(_path))
        # return HttpResponse('admin {} model open minded'.format(name))
    else:
        return Http404('Model Path Not Found.')


def read_data_path(request, name):
    main_service = instance.get_main_service(is_admin=True)
    nickname_filter = instance.get_nickname_filter(is_admin=True)
    result_data = None
    if name == 'vocabulary':
        result_data = main_service.get_vocabulary_data()
    elif name == 'dpinyinblist':
        result_data = main_service.get_dynamic_pinyin_block_list()
    elif name == 'dalertwordslist':
        result_data = main_service.get_dynamic_alert_words_list()
    elif name == 'dnpinyinblist':
        result_data = nickname_filter.get_dynamic_nickname_pinyin_block_list()
    # elif name == 'textbook':
    #     page_number = request.GET.get("page", 1)
    #     per_page = request.GET.get("per_page", 500)
    #     result_data = main_service.get_textbook_sentense_list(page=page_number, per_page=per_page)
    elif name == 'textbookall':
        result_data = main_service.get_textbook_sentense_all()

    elif name == 'nickname_textbookall':
        result_data = nickname_filter.get_nickname_textbook_all()

    # print('result_data: ', result_data)

    if result_data is None:
        raise Http404('Data Not Found.')
    else:
        return JsonResponse(result_data, safe=False)

def read_textbook(request, page):
    main_service = instance.get_main_service(is_admin=True)
    result_data = None

    # page_number = request.GET.get("page", 1)
    # per_page = request.GET.get("per_page", 500)
    page_number = int(page)
    per_page = 100
    result_data = main_service.get_textbook_sentense_list(page=page_number, per_page=per_page)

    if result_data is None:
        raise Http404('Data Not Found.')
    else:
        return JsonResponse(result_data, safe=False)

def read_nickname_textbook(request, page):
    nickname_filter = instance.get_nickname_filter(is_admin=True)
    result_data = None

    # page_number = request.GET.get("page", 1)
    # per_page = request.GET.get("per_page", 500)
    page_number = int(page)
    per_page = 100
    result_data = nickname_filter.get_nickname_textbook_list(page=page_number, per_page=per_page)

    if result_data is None:
        raise Http404('Data Not Found.')
    else:
        return JsonResponse(result_data, safe=False)

urlpatterns = [
    path('chat/', include('chat.urls')),
    path('admin/', admin.site.urls),
    path('auth/', include('djoser.urls')),
    path('auth/', include('djoser.urls.authtoken')),
    path('api/model/<slug:name>', read_model_path),
    path('api/data/<slug:name>', read_data_path),
    path('api/textbook/<slug:page>', read_textbook),
    path('api/nickname_textbook/<slug:page>', read_nickname_textbook),
    path('api/jsondata/<slug:name>', ServiceJSONDataAPIView.as_view()),
    path('api/upload/<slug:name>', ServiceUploadAPIView.as_view()),
    path('api/remove/<slug:name>/<slug:id>', ServiceRemoveAPIView.as_view()),
    path('api/pinyinblock/<slug:id>', ServicePinyinBlockListAPIView.as_view()),
    path('api/nicknamepinyinblock/<slug:id>', ServiceNicknamePinyinBlockListAPIView.as_view()),
    path('api/alertwords/<slug:id>', ServiceAlertWordsListAPIView.as_view()),
    path('api/cmd/<slug:name>', ServiceCommandAPIView.as_view()),
    path('api/twice/<slug:fn>', TwiceServiceAPIView.as_view()),
    path('api/nickname_twice/<slug:fn>', NicknameTwiceServiceAPIView.as_view()),
    path('api/train/<slug:fn>', TrainServiceAPIView.as_view()),
    path('api/complete/<slug:name>', train_val_complete_handler),
    path('ai/', include('ai.urls')),
]
