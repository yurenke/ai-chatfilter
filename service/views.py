from numpy import isin
from rest_framework.parsers import FileUploadParser, FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from django.http import Http404, JsonResponse, HttpResponseForbidden, HttpResponse
from django.apps import apps
from django.utils.timezone import datetime
from django.core.paginator import Paginator

import csv, codecs, json, re

from dataparser.apps import ExcelParser
from dataparser.jsonparser import JsonParser
from .instance import get_main_service, get_nickname_filter, get_remote_twice_service
from datetime import date

import redis
from configparser import RawConfigParser
import os
import threading
from http.client import HTTPConnection
import logging

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = RawConfigParser()
config.read(BASE_DIR+'/setting.ini')

class ServiceJSONDataAPIView(APIView):
    """

    """

    # permission_classes = [IsAuthenticated]

    def post(self, request, name):
        data = request.data
        if name:
            try:
                app_name = data.get('app_name', 'service')
                _filter = data.get('filter', {})
                _columns = data.get('columns', {})
                _page = data.get('page', 1)
                _pagination = data.get('pagination', 100)
                
                Model = apps.get_model(app_label=app_name, model_name=name)
                q = Model.objects.filter(**_filter).values(*_columns).order_by('-id')

                if _pagination:
                    paginator = Paginator(q, _pagination)
                    _res = paginator.get_page(_page)
                else:
                    paginator = False
                    _res = q
                
                if _columns:
                    _result = []
                    for _ in _res:
                        _r = []
                        for _col in _columns:
                            _r.append(_[_col])
                        _result.append(_r)
                else:
                    _result = list(_res)

                return JsonResponse({
                    'datetime': datetime.today(),
                    'columns': _columns,
                    'result': _result,
                    'total_page': paginator.num_pages if paginator else 1,
                    'total_count': paginator.count if paginator else q.count(),
                })
                
            except Exception as err:
                return HttpResponseForbidden(str(err))
        else:
            return JsonResponse({'name': 'none'}, safe=False)
        
        

class ServiceUploadAPIView(APIView):
    """
    """

    parser_classes = [MultiPartParser,FormParser,FileUploadParser]
    # permission_classes = [IsAuthenticated]


    def post(self, request, name):
        data = request.data
        try:
            if name == 'textbook':
               
                _data = []
                _file = request.FILES['file']
                _ep = ExcelParser(file_content=_file.read())
                _data = _ep.get_row_list(column=['发言内容', '状态', '權重'])
                _service = get_main_service(is_admin=True)
                _data = [_ for _ in _data if  not bool(re.search(r'[^\u0020-\uffff]', _[0]))]
                
                _done = _service.add_textbook_sentense(origin=_file.name, sentenses=_data)

                # if isinstance(_done, bool):
                #     # _data = _data[:500]
                #     if _done:
                #         _twice_service = get_remote_twice_service()
                #         _done = _twice_service.add_textbook_sentense(origin=_file.name, sentenses=_data)
                    
                return JsonResponse({'done': _done, 'data': _data}, safe=False)

            elif name == 'textbookjson':
                _data = []
                _file = request.FILES['file']
                _jp = JsonParser(file_content=_file.read())
                data_list = _jp.get_data() # [id, origin, text, state, weight]
                _data = [_[2:] for _ in data_list]
                _done = False
                if len(_data) > 0 and len(_data[0]) == 3:

                    _service = get_main_service(is_admin=True)
                    _done = _service.add_textbook_sentense(origin=_file.name, sentenses=_data)

                    # if isinstance(_done, bool):
                    #     if _done:
                    #         _twice_service = get_remote_twice_service()
                    #         _done = _twice_service.add_textbook_sentense(origin=_file.name, sentenses=_data)
                    
                return JsonResponse({'done': _done, 'data': _data}, safe=False)

            elif name == 'nickname_textbook':
               
                _data = []
                _file = request.FILES['file']
                _ep = ExcelParser(file_content=_file.read())
                _data = _ep.get_row_list(column=['昵称', '状态'])
                _service = get_nickname_filter(is_admin=True)
                _data = [_ for _ in _data if  not bool(re.search(r'[^\u0020-\uffff]', _[0]))]
                
                _done = _service.add_to_nickname_textbook(origin=_file.name, nicknames=_data)

                # if isinstance(_done, bool):
                #     # _data = _data[:500]
                #     if _done:
                #         _twice_service = get_remote_twice_service()
                #         _done = _twice_service.add_to_nickname_textbook(origin=_file.name, nicknames=_data)
                    
                return JsonResponse({'done': _done, 'data': _data}, safe=False)

            elif name == 'nickname_textbookjson':
                _data = []
                _file = request.FILES['file']
                _jp = JsonParser(file_content=_file.read())
                data_list = _jp.get_data() # [id, origin, text, state, weight]
                _data = [_[2:] for _ in data_list]
                _done = False
                if len(_data) > 0 and len(_data[0]) == 3:

                    _service = get_nickname_filter(is_admin=True)
                    _done = _service.add_to_nickname_textbook(origin=_file.name, nicknames=_data)

                    # if isinstance(_done, bool):
                    #     if _done:
                    #         _twice_service = get_remote_twice_service()
                    #         _done = _twice_service.add_to_nickname_textbook(origin=_file.name, nicknames=_data)
                    
                return JsonResponse({'done': _done, 'data': _data}, safe=False)

            else:
                return JsonResponse({'name': 'none'}, safe=False)
        
        except Exception as err:
            return HttpResponseForbidden(str(err))


class ServiceRemoveAPIView(APIView):
    """
    """
    # permission_classes = [IsAuthenticated]

    def delete(self, request, name, id):
        data = request.data
        if name:
            try:
                if name == 'textbook':

                    _done = get_main_service(is_admin=True).remove_textbook_sentense(int(id))

                    if not _done:
                        return HttpResponseForbidden('Delete Failed.')

                return JsonResponse({
                    'id': id,
                    'datetime': datetime.today(),
                })
                
            except Exception as err:
                return HttpResponseForbidden(str(err))
        else:

            return JsonResponse({'name': 'none'}, safe=False)


class ServicePinyinBlockListAPIView(APIView):
    """
    """
    model = apps.get_model(app_label='service', model_name='DynamicPinyinBlock')

    def get(self, request, id):
        if id == 'list':
            today = date.today()
            _date = today.strftime("%Y%m%d")
            filename = '{}_block_list.csv'.format(_date)
            reslist = get_main_service(is_admin=True).get_dynamic_pinyin_block_list()
            
            response = HttpResponse(content_type='text/csv; charset=utf-8')
            response['Content-Disposition'] = "attachment; filename=" + filename
            response.write(codecs.BOM_UTF8)
            writer = csv.writer(response)
            # writer.writeheader()
            for r in reslist:
                writer.writerow(r)
            return response
        
        return HttpResponseForbidden('Add Failed.')

    def post(self, request, id):
        result = []
        if id == 'add':
            texts = request.data.get('text', [])
            print('===============ServicePinyinBlockListAPIView texts : ', texts)
            if texts:
                result = get_main_service(is_admin=True).add_pinyin_block(texts)

            return JsonResponse({'result': result}, safe=False)
        
        elif id == 'file':
            pass
        
        return HttpResponseForbidden('Add Failed.')
        

    def delete(self, request, id):
        try:
            id = int(id)
            if id > 0:

                _done = get_main_service(is_admin=True).remove_pinyin_block(id)

                if not _done:
                    return HttpResponseForbidden('Delete Failed.')

            return JsonResponse({
                'id': id,
                'datetime': datetime.today(),
            })
            
        except Exception as err:
            return HttpResponseForbidden(str(err))

class ServiceNicknamePinyinBlockListAPIView(APIView):
    """
    """
    model = apps.get_model(app_label='service', model_name='DynamicNicknamePinyinBlock')

    def get(self, request, id):
        if id == 'list':
            today = date.today()
            _date = today.strftime("%Y%m%d")
            filename = '{}_nickname_block_list.csv'.format(_date)
            reslist = get_nickname_filter(is_admin=True).get_dynamic_nickname_pinyin_block_list()
            
            response = HttpResponse(content_type='text/csv; charset=utf-8')
            response['Content-Disposition'] = "attachment; filename=" + filename
            response.write(codecs.BOM_UTF8)
            writer = csv.writer(response)
            # writer.writeheader()
            for r in reslist:
                writer.writerow(r)
            return response
        
        return HttpResponseForbidden('Add Failed.')

    def post(self, request, id):
        result = []
        if id == 'add':
            texts = request.data.get('text', [])
            print('===============ServiceNicknamePinyinBlockListAPIView texts : ', texts)
            if texts:
                result = get_nickname_filter(is_admin=True).add_nickname_pinyin_block(texts)

            return JsonResponse({'result': result}, safe=False)
        
        elif id == 'file':
            pass
        
        return HttpResponseForbidden('Add Failed.')
        

    def delete(self, request, id):
        try:
            id = int(id)
            if id > 0:

                _done = get_nickname_filter(is_admin=True).remove_nickname_pinyin_block(id)

                if not _done:
                    return HttpResponseForbidden('Delete Failed.')

            return JsonResponse({
                'id': id,
                'datetime': datetime.today(),
            })
            
        except Exception as err:
            return HttpResponseForbidden(str(err))

class ServiceAlertWordsListAPIView(APIView):
    """
    """
    model = apps.get_model(app_label='service', model_name='DynamicAlertWords')

    def get(self, request, id):
        if id == 'list':
            today = date.today()
            _date = today.strftime("%Y%m%d")
            filename = '{}_alert_words_list.csv'.format(_date)
            reslist = get_main_service(is_admin=True).get_dynamic_alert_words_list()
            
            response = HttpResponse(content_type='text/csv; charset=utf-8')
            response['Content-Disposition'] = "attachment; filename=" + filename
            response.write(codecs.BOM_UTF8)
            writer = csv.writer(response)
            # writer.writeheader()
            for r in reslist:
                writer.writerow(r)
            return response
        
        return HttpResponseForbidden('Add Failed.')

    def post(self, request, id):
        result = []
        if id == 'add':
            texts = request.data.get('text', [])
            print('===============ServiceAlertWordsListAPIView texts : ', texts)
            if texts:
                result = get_main_service(is_admin=True).add_alert_words(texts)

            return JsonResponse({'result': result}, safe=False)
        
        elif id == 'file':
            pass
        
        return HttpResponseForbidden('Add Failed.')
        

    def delete(self, request, id):
        try:
            id = int(id)
            if id > 0:

                _done = get_main_service(is_admin=True).remove_alert_words(id)

                if not _done:
                    return HttpResponseForbidden('Delete Failed.')

            return JsonResponse({
                'id': id,
                'datetime': datetime.today(),
            })
            
        except Exception as err:
            return HttpResponseForbidden(str(err))


class TwiceServiceAPIView(APIView):
    """
    """
    

    def get(self, request, fn):
        
        return HttpResponseForbidden('Get Failed.')

    def post(self, request, fn):
        result = None
        data = request.data
        try:
            next_data = {}
            if data:
                data = dict(data)
                # print('[TwiceServiceAPIView] first data: ', data)
                
                for idx, key in enumerate(data):
                    if isinstance(data[key], list):
                        _loc = data[key][0]
                    elif isinstance(data[key], str):
                        _loc = data[key]

                    _loc = _loc.replace("'", '"')
                    _loc = _loc.replace('\\xa0', '')
                    _loc = re.sub(r'[\r\n]+', '' ,_loc)
                    # print('[TwiceServiceAPIView] _loc: ', _loc)
                    next_data[key] = json.loads(_loc) if (_loc[0] == '[' or _loc[0] == '{') else _loc
                
                # print('[TwiceServiceAPIView] next_data: ', next_data)
            _service = get_main_service(is_admin=True)
            _fn = getattr(_service, fn)
            
            result = _fn(**next_data)
        
        except Exception as err:

            return HttpResponseForbidden(str(err))

        return JsonResponse({
            'mode': 'TwiceServiceAPI',
            'result': result,
            'datetime': datetime.today(),
        })

class NicknameTwiceServiceAPIView(APIView):
    """
    """
    def get(self, request, fn):
        
        return HttpResponseForbidden('Get Failed.')

    def post(self, request, fn):
        result = None
        data = request.data
        try:
            next_data = {}
            if data:
                data = dict(data)
                
                for idx, key in enumerate(data):
                    if isinstance(data[key], list):
                        _loc = data[key][0]
                    elif isinstance(data[key], str):
                        _loc = data[key]

                    _loc = _loc.replace("'", '"')
                    _loc = _loc.replace('\\xa0', '')
                    _loc = re.sub(r'[\r\n]+', '' ,_loc)
                    next_data[key] = json.loads(_loc) if (_loc[0] == '[' or _loc[0] == '{') else _loc
                
            _service = get_nickname_filter(is_admin=True)
            _fn = getattr(_service, fn)
            
            result = _fn(**next_data)
        
        except Exception as err:

            return HttpResponseForbidden(str(err))

        return JsonResponse({
            'mode': 'NicknameTwiceServiceAPI',
            'result': result,
            'datetime': datetime.today(),
        })
        



class ServiceCommandAPIView(APIView):
    """
    """
    def get(self, request, name):
        _service = get_main_service(is_admin=True)
        _nickname_service = get_nickname_filter(is_admin=True)
        _twice_service = get_remote_twice_service()
        if name == 'aitrainer':
            chat_rslt = json.loads(_twice_service.get_ai_train_data())['result']
            nickname_rslt = json.loads(_twice_service.get_nickname_ai_train_data())['result']
            # print('result: ', result)
            return JsonResponse({
                'mode': 'ServiceCommandAPI',
                'result': {'chat': chat_rslt, 'nickname': nickname_rslt},
                'datetime': datetime.today(),
            })
        elif name == 'ai_test_result':
            result = _service.get_ai_test_result()
            # print('result: ', result)
            return JsonResponse({
                'mode': 'ServiceCommandAPI',
                'result': result,
                'datetime': datetime.today(),
            })
        elif name == 'nickname_ai_test_result':
            result = _nickname_service.get_nickname_ai_test_result()
            # print('result: ', result)
            return JsonResponse({
                'mode': 'ServiceCommandAPI',
                'result': result,
                'datetime': datetime.today(),
            })
        
        return HttpResponseForbidden('Get Failed.')


    def post(self, request, name):
        result = None
        data = request.data

        try:
            # _service = get_remote_twice_service()
            _service = get_main_service(is_admin=True)
            _nickname_service = get_nickname_filter(is_admin=True)
            _twice_service = get_remote_twice_service()
            
            if name == 'trainstart':
                result = _twice_service.fit_chat_model()
            
            elif name == 'nickname_trainstart':
                result = _twice_service.fit_nickname_model()

            elif name == 'testaccuracy' or name == 'nickname_testaccuracy':
                _busy = _service.is_testing or _nickname_service.is_testing

                if _busy:
                    return HttpResponseForbidden('currently busy.')
                
                _origin = data.get('origin', False)
                if not _origin:
                    result = 'not recived origin !!'

                else:
                    if name == 'testaccuracy':
                        _service.is_testing = True
                        fn = 'test_chinese_chat'
                    else:
                        _nickname_service.is_testing = True
                        fn = 'test_chinese_nickname'

                    r = redis.Redis(host='localhost', port=config.get('CHANNEL', 'CHANNEL_PORT'), db=0)
                    command_dict = {}
                    command_dict['command'] = fn
                    command_dict['origin'] = _origin
                    command_dict_str = json.dumps(command_dict)
                    r.publish('training_request', command_dict_str)

                    # return JsonResponse({
                    #     'mode': 'AsyncTrainAPI',
                    #     'result': 'start {}'.format(fn),
                    #     'datetime': datetime.today(),
                    # })
                    result = '{} ok!! origin: {}'.format(fn, _origin)

            # elif name == 'nickname_testaccuracy':

            #     _origin = data.get('origin', False)
            #     if _origin:
            #         result = _service.get_nickname_test_accuracy_by_origin(origin=_origin)
            #     else:
            #         result = 'no recive origin: {}'.format(_origin)
        
        except Exception as err:

            return HttpResponseForbidden(str(err))

        return JsonResponse({
            'mode': 'ServiceCommandAPI',
            'result': result,
            'datetime': datetime.today(),
        })

class TrainServiceAPIView(APIView):
    """
    """  

    def get(self, request, fn):
        
        return HttpResponseForbidden('Get Failed.')

    def post(self, request, fn):
        result = None
        data = request.data
        command_dict = {}
        _service = get_main_service(is_admin=True)
        _nickname_filter = get_nickname_filter(is_admin=True)
        _busy = _service.is_training or _nickname_filter.is_training
        try:
            if not _busy:
                if fn == 'train_chinese_chat' or fn == 'train_chinese_nickname':
                    if fn == 'train_chinese_chat':
                        _service.is_training = True
                    else:
                        _nickname_filter.is_training = True

                    r = redis.Redis(host='localhost', port=config.get('CHANNEL', 'CHANNEL_PORT'), db=0)

                    command_dict['command'] = fn
                    command_dict_str = json.dumps(command_dict)
                    r.publish('training_request', command_dict_str)

                    return JsonResponse({
                        'mode': 'AsyncTrainAPI',
                        'result': 'start {}'.format(fn),
                        'datetime': datetime.today(),
                    })

                elif fn == 'test_chinese_chat' or fn == 'test_chinese_nickname':
                    if not data:
                        return HttpResponseForbidden('no origin specified.')
                    next_data = {}
                    data = dict(data)
                    # print('[TwiceServiceAPIView] first data: ', data)
                        
                    for idx, key in enumerate(data):
                        if isinstance(data[key], list):
                            _loc = data[key][0]
                        elif isinstance(data[key], str):
                            _loc = data[key]

                        _loc = _loc.replace("'", '"')
                        _loc = _loc.replace('\\xa0', '')
                        _loc = re.sub(r'[\r\n]+', '' ,_loc)
                        # print('[TwiceServiceAPIView] _loc: ', _loc)
                        next_data[key] = json.loads(_loc) if (_loc[0] == '[' or _loc[0] == '{') else _loc

                    if fn == 'test_chinese_chat':
                        _service.is_testing = True
                    else:
                        _nickname_filter.is_testing = True

                    r = redis.Redis(host='localhost', port=config.get('CHANNEL', 'CHANNEL_PORT'), db=0)

                    command_dict['command'] = fn
                    command_dict['origin'] = next_data['origin']
                    command_dict_str = json.dumps(command_dict)
                    r.publish('training_request', command_dict_str)

                    return JsonResponse({
                        'mode': 'AsyncTrainAPI',
                        'result': 'start {}'.format(fn),
                        'datetime': datetime.today(),
                    })
            else:
                return HttpResponseForbidden('currently busy.')
        
        except Exception as err:

            return HttpResponseForbidden(str(err))

def train_val_complete_handler(request, name):
    if name == 'train_chinese_chat':
        _service = get_main_service(is_admin=True)
        _service.is_training = False
        _twice_service = get_remote_twice_service()

        _twice_service.notify_chat_train_complete()

        return JsonResponse({
                'mode': 'AsyncTrainAPI',
                'result': 'chinese chat model training finished',
                'datetime': datetime.today(),
            })

    elif name == 'test_chinese_chat':
        _service = get_main_service(is_admin=True)
        _service.is_testing = False        

        return JsonResponse({
                'mode': 'AsyncTrainAPI',
                'result': 'chinese chat model testing finished',
                'datetime': datetime.today(),
            })
    elif name == 'train_chinese_nickname':
        _service = get_nickname_filter(is_admin=True)
        _service.is_training = False
        _twice_service = get_remote_twice_service()

        _twice_service.notify_nickname_train_complete()

        return JsonResponse({
                'mode': 'AsyncTrainAPI',
                'result': 'chinese nickname model training finished',
                'datetime': datetime.today(),
            })
    elif name == 'test_chinese_nickname':
        _service = get_nickname_filter(is_admin=True)
        _service.is_testing = False

        return JsonResponse({
                'mode': 'AsyncTrainAPI',
                'result': 'chinese nickname model testing finished',
                'datetime': datetime.today(),
            })
    
def get_model_file(target):
    _twice_service = get_remote_twice_service()
    remote_ip = _twice_service.service_addr
    port = _twice_service.service_web_port
    _http_cnn = HTTPConnection(remote_ip, port)

    def _save_file_by_http_response(response, path):
        with open(path, 'wb+') as f:
            while True:
                _buf = response.read()
                if _buf:
                    f.write(_buf)
                else:
                    break

    _http_cnn.request('GET', '/api/model/' + target)
    _http_res = _http_cnn.getresponse()
    if _http_res.status == 200:
        if target == 'chat':
            _model_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/ai/_models/chinese_chat_model'
        elif target == 'nickname':
            _model_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/ai/_models/chinese_nickname_model'
        _save_file_by_http_response(response=_http_res, path=_model_path+'/model.h5')
        logging.info('[sync ai model] Download Remote Chinese {} Model Done.'.format(target))
    else:
        logging.error('[sync ai model] Download Remote Chinese {} Model Failed. will retry'.format(target))
        timer = threading.Timer(5, get_model_file, (target,))
        timer.start()
    
def train_complete_notification_handler(request, name):
    if name == 'chat_complete':
        target = 'chat'

    elif name == 'nickname_complete':
        target = 'nickname'

    timer = threading.Timer(5, get_model_file, (target,))
    timer.start()

    return JsonResponse({
                'mode': 'NotificationAPI',
                'result': 'get ready to sync model',
                'datetime': datetime.today(),
            })