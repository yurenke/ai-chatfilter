from dis import findlinestarts
from django.utils import timezone
from django.conf import settings
from django.forms.models import model_to_dict
from django.core.paginator import Paginator
# from django.core.files.base import ContentFile
from http.client import HTTPConnection
from ai.apps import MainAiApp
from ai.service_impact import get_all_vocabulary_from_models
from ai.helper import get_pinyin_path, get_multi_lingual_bert_chat_model_path, get_chinese_chat_model_path, get_vocabulary_dictionary_path
from ai.classes.translator_pinyin import translate_by_string
from ai.models import TextbookSentense
from ai.train import train_pinyin_by_list, get_row_list_by_json_path
from dataparser.apps import MessageParser, EnglishParser
from dataparser.classes.store import ListPickle
from tensorflow.keras.callbacks import Callback

import numpy as np
import time, re, logging, json, threading
from os import path, listdir

from .classes.prefilter import PreFilter
# from .classes.fuzzycenter import FuzzyCenter
from .classes.chatstore import ChatStore
from .models import GoodSentence, BlockedSentence, AnalyzingData, UnknownWord, ChangeNicknameRequest, Blockword, DynamicPinyinBlock, DynamicAlertWords





class MainService():
    """

    """
    pre_filter = None
    ai_app = None
    message_parser = None
    english_parser = None
    fuzzy_center = None
    chat_store = None
    main_admin_server_addr = None
    is_open_mind = False
    is_admin_server = False

    timestamp_ymdh = [0, 0, 0, 0]
    service_avoid_filter_lv = 8
    lang_mode = 1
    vocabulary_data = {}
    is_training = False
    is_testing = False
    train_thread = None

    
    STATUS_PREDICTION_NO_MSG = 0
    STATUS_PREDICTION_ADVERTISING = 1
    STATUS_PREDICTION_HUMAN_DELETE = 3
    STATUS_PREDICTION_DIRTY_WORD = 4
    STATUS_PREDICTION_OTHER_AI = 5
    STATUS_PREDICTION_SEPCIFY_BLOCK = 8
    STATUS_PREDICTION_UNKNOWN_MEANING = 9
    STATUS_PREDICTION_SPECIAL_CHAR = 11
    STATUS_PREDICTION_NONSENSE = 12
    STATUS_PREDICTION_WEHCAT_SUSPICION = 13
    STATUS_PREDICTION_BLOCK_WORD = 14
    STATUS_PREDICTION_SUSPECT_WATER_ARMY = 15
    STATUS_PREDICTION_NOT_ALLOW = 16
    STATUS_PREDICTION_SAME_LOGINNAME_IN_SHORTTIME = 17
    STATUS_PREDICTION_GRAMMARLY = 21

    STATUS_MODE_CHINESE = 1
    STATUS_MODE_ENGLISH = 2
    STATUS_MODE_BERT = 3

    REMOTE_ROUTE_CHAT_MODEL = '/api/model/chat'
    # REMOTE_ROUTE_TRANSFORMER_MODEL = '/api/model/transformer'
    # REMOTE_ROUTE_ENGLISH_MODEL = '/api/model/english'
    REMOTE_ROUTE_VOCABULARY_DATA = '/api/data/vocabulary'
    REMOTE_ROUTE_DYNAMIC_PINYIN_BLOCK = '/api/data/dpinyinblist'
    REMOTE_ROUTE_DYNAMIC_ALERT_WORDS = '/api/data/dalertwordslist'

    regex_all_english_word = re.compile("^[a-zA-Z\s\r\n]+$")
    regex_has_gap = re.compile("[a-zA-Z]+\s+[a-zA-Z]+")


    def __init__(self, is_admin_server = False):

        self.init_language()
        self.message_parser = MessageParser()
        self.english_parser = EnglishParser()
        self.chat_store = ChatStore()
        self.pre_filter = PreFilter()
        self.do_not_filter_rooms = settings.DO_NOT_FILTER_ROOMS

        if is_admin_server:
            self.is_admin_server = True
            self.check_analyzing()
            if self.lang_mode == self.STATUS_MODE_CHINESE:
                self.pre_filter.set_pinyin_block_list(self.get_dynamic_pinyin_block_list())
                self.pre_filter.set_alert_words_list(self.get_dynamic_alert_words_list())

        logging.info('=============  Main Service Activated. Time Zone: [ {} ] ============='.format(settings.TIME_ZONE))


    def init_language(self):
        _setting = settings.LANGUAGE_MODE
        logging.info('Service Main Language [ {} ]'.format(_setting))
        if _setting == 'EN':
            self.lang_mode = self.STATUS_MODE_ENGLISH
        elif _setting == 'CH' or _setting == 'ZH':
            self.lang_mode = self.STATUS_MODE_CHINESE
        elif _setting == 'BERT':
            self.lang_mode = self.STATUS_MODE_BERT
        else:
            raise Exception('No Specify Right Language')


    def open_mind(self):
        if self.is_open_mind:
            return True
    
        if self.lang_mode == self.STATUS_MODE_CHINESE:
            
            _voca_data = self.get_vocabulary_data()

            _vocabulary_english = _voca_data.get('english', [])

            self.english_parser.set_vocabulary(_vocabulary_english)

            self.ai_app = MainAiApp(english_data=_vocabulary_english)
            
            self.ai_app.load_chinese_chat_model()
            self.ai_app.model.save(is_check=True, is_continue=False)

        elif self.lang_mode == self.STATUS_MODE_ENGLISH:
            
            self.ai_app = MainAiApp()
            self.ai_app.load_multi_lingual_chat_model()
            self.ai_app.model.save(is_check=True, is_continue=False)

        else:

            logging.error('Language Mode Not Found :: {}'.format(self.lang_mode))
        

        self.is_open_mind = True
        # self.fuzzy_center = FuzzyCenter()

        return True
    

    def parse_message(self, string):
        _, lv, ac = self.message_parser.parse(string)
        # _ = self.english_parser.replace_to_origin_english(_)
        return _, lv, ac


    def trim_text(self, text):
        return self.message_parser.trim_only_general_and_chinese(text).strip()


    def think(self, message, user = '', room = '', silence=False, detail=False):
        st_time = time.time()
        if not self.is_open_mind:
            logging.warning('AI Is Not Ready..')
            return self.return_reslut(0, message=message, st_time=st_time)
        
        text = ''
        lv = 0
        anchor = 0
        reason_char = ''
        prediction = 0
        is_suspicious = 0
        # print('receive message :', message)

        if len(self.do_not_filter_rooms) > 1 or self.do_not_filter_rooms[0] != 'None':
            if room in self.do_not_filter_rooms:
                return self.return_reslut(prediction, message=message, user=user, room=room, text=text, reason=reason_char, is_suspicious=is_suspicious, silence=silence, st_time=st_time)

        if message:
            text, lv, anchor = self.parse_message(message)
            # print('parse_message text: ', text)
            if self.lang_mode == self.STATUS_MODE_CHINESE:
                if text:
                    suspiciousWords = self.pre_filter.find_alert_words(text)
                    if suspiciousWords:
                        is_suspicious = 1

                    if len(text) > 25:
                        reason_char = 'too many words'
                        prediction = self.STATUS_PREDICTION_NOT_ALLOW
                        return self.return_reslut(prediction, message=message, user=user, room=room, text=text, reason=reason_char, is_suspicious=is_suspicious, silence=silence, detail=detail, st_time=st_time)

                    reason_char = self.find_prefilter_reject_reason_with_nonparsed_msg(text)
                    if reason_char:
                        prediction = self.STATUS_PREDICTION_NOT_ALLOW
                        return self.return_reslut(prediction, message=message, user=user, room=room, text=text, reason=reason_char, is_suspicious=is_suspicious, silence=silence, detail=detail, st_time=st_time)
        
        if user[:3] == 'TST':
            if anchor > 0 or room == 'BG01':
                return self.return_reslut(prediction, message=message, user=user, room=room, text=text, reason=reason_char, is_suspicious=is_suspicious, silence=silence, detail=detail, st_time=st_time)

        if text:
            if lv >= self.service_avoid_filter_lv:
                return self.return_reslut(prediction, message=message, user=user, room=room, text=text, reason=reason_char, is_suspicious=is_suspicious, silence=silence, detail=detail, st_time=st_time)                

            reason_char = self.chat_store.check_same_room_conversation(text, room)
            if reason_char:
                prediction = self.STATUS_PREDICTION_SUSPECT_WATER_ARMY
                return self.return_reslut(prediction, message=message, user=user, room=room, text=text, reason=reason_char, is_suspicious=is_suspicious, silence=silence, detail=detail, st_time=st_time)

            if self.pre_filter.check_loginname_shorttime_saying(user):
                reason_char = 'Speak Too Quickly'
                prediction = self.STATUS_PREDICTION_SAME_LOGINNAME_IN_SHORTTIME
                return self.return_reslut(prediction, message=message, user=user, room=room, text=text, reason=reason_char, is_suspicious=is_suspicious, silence=silence, detail=detail, st_time=st_time)

            if self.lang_mode == self.STATUS_MODE_CHINESE:
                text = self.trim_text(text)
                if len(text) == 0 :
                    return self.return_reslut(prediction, message=message, user=user, room=room, text=text, reason=reason_char, is_suspicious=is_suspicious, silence=silence, detail=detail, st_time=st_time)
                
                # block sentences with >= 3 spaces
                if text.count(' ') >= 5:
                    prediction = self.STATUS_PREDICTION_NOT_ALLOW
                    reason_char = 'too many spaces'
                    return self.return_reslut(prediction, message=message, user=user, room=room, text=text, reason=reason_char, is_suspicious=is_suspicious, silence=silence, detail=detail, st_time=st_time)
            #main ai
            prediction, reason_char = self.ai_app.predict(text, lv=lv, with_reason=self.is_admin_server)
            
            # save message to room store
            if prediction == 0:
                self.store_temporary_text(
                    text=text,
                    user=user,
                    room=room,
                )

            return self.return_reslut(prediction, message=message, user=user, room=room, text=text, reason=reason_char, is_suspicious=is_suspicious, silence=silence, detail=detail, st_time=st_time)

        else:

            prediction = self.STATUS_PREDICTION_NO_MSG

        return self.return_reslut(prediction, message=message, user=user, room=room, reason=reason_char, is_suspicious=is_suspicious, silence=silence, st_time=st_time)

    def return_reslut(self, prediction, message, user='', room='', text='', reason='', is_suspicious = 0, silence=True, detail=False, st_time=0):
        result = {}
        detail_data = {}
        
        if detail:

            detail_data = self.ai_app.get_details(text)

        ed_time = time.time()
        
        result['user'] = user
        result['room'] = room
        result['message'] = message
        result['text'] = text
        result['prediction'] = int(prediction)
        result['reason_char'] = reason
        result['is_suspicious'] = is_suspicious
        result['detail'] = detail_data
        result['spend_time'] = ed_time - st_time
        if result['spend_time'] > 0.2:
            logging.error('Spend Time Of Think Result = {}'.format(result['spend_time']))
        logging.debug('Think Result [ msg: {} prediction: {} is_suspicious: {} time: {} ] '.format(result['message'], result['prediction'], result['is_suspicious'], result['spend_time']))
        return result


    def find_prefilter_reject_reason_with_nonparsed_msg(self, msg):
        # print('find_prefilter_reject_reason_with_nonparsed_msg: ', msg)
        methods = [
            self.pre_filter.find_suspect_digits_symbol,
            self.pre_filter.find_not_allowed_chat,
            # self.pre_filter.find_korea_mixed,
            self.pre_filter.find_emoji_word_mixed,
            self.pre_filter.find_unallow_eng,
            self.pre_filter.find_pinyin_blocked,
        ]
        for m in methods:
            r = m(msg)
            if r:
                return r

        return False

    def store_temporary_text(self, text, user, room):
        self.chat_store.upsert_text(text, user, room)


    def saveRecord(self, prediction, message, text='', reason=''):
        try:

            if prediction == 0:
                # save to good sentence
                record = GoodSentence(
                    message=message[:95],
                    text=text[:63],
                )
            else:
                # save to blocked
                if text:
                    _text = text
                else:
                    _text, lv, anchor = self.parse_message(message)
                
                record = BlockedSentence(
                    message=message[:95],
                    text=_text[:63],
                    reason=reason[:63] if reason else '',
                    status=int(prediction),
                )

            record.save()
        
            self.check_analyzing()

        except Exception as ex:

            logging.error('Save Record Failed,  message :: {}'.format(message))
            print(ex, flush=True)

    
    # def saveNicknameRequestRecord(self, nickname, status):
    #     if self.is_admin_server:

    #         try:
    #             record = ChangeNicknameRequest(
    #                 nickname=nickname,
    #                 status=status,
    #             )
    #             record.save()
    #         except Exception as ex:
    #             logging.error('Save NicknameRequestRecord Failed, nickname: [{}].'.format(nickname))
    #             print(ex)



    def check_analyzing(self):
        _now = timezone.now()
        today_datetime = timezone.localtime(_now)
        _ymdh = [_now.year, _now.month, _now.day, _now.hour]
        _not_day_matched = _ymdh[2] != self.timestamp_ymdh[2]
        _not_hour_matched = _ymdh[3] != self.timestamp_ymdh[3]


        if _not_day_matched:
            today_date = today_datetime.replace(hour=0,minute=0,second=0)
            yesterday_date = today_date + timezone.timedelta(days=-1)
            yesterday_goods = GoodSentence.objects.filter(date__gte=yesterday_date, date__lte=today_date).count()
            yesterday_blockeds = BlockedSentence.objects.filter(date__gte=yesterday_date, date__lte=today_date).count()

            yesterday_analyzing = AnalyzingData.objects.filter(
                year=yesterday_date.year,
                month=yesterday_date.month,
                day=yesterday_date.day,
            )

            if yesterday_analyzing:

                yesterday_analyzing.update(
                    good_sentence=yesterday_goods,
                    blocked_sentence=yesterday_blockeds,
                )

            else:

                yesterday_analyzing = AnalyzingData(
                    year=yesterday_date.year,
                    month=yesterday_date.month,
                    day=yesterday_date.day,
                    good_sentence=yesterday_goods,
                    blocked_sentence=yesterday_blockeds,
                )

                yesterday_analyzing.save()

            self.timestamp_ymdh[2] = _now.day

        if _not_hour_matched:
            today_date = today_datetime.replace(hour=0,minute=0,second=0)
            today_goods = GoodSentence.objects.filter(date__gte=today_date).count()
            today_blockeds = BlockedSentence.objects.filter(date__gte=today_date).count()
            today_analyzing = AnalyzingData.objects.filter(
                year=today_date.year,
                month=today_date.month,
                day=today_date.day,
            )

            if today_analyzing:

                today_analyzing.update(
                    good_sentence=today_goods,
                    blocked_sentence=today_blockeds,
                )

            else:
                today_analyzing = AnalyzingData(
                    year=today_date.year,
                    month=today_date.month,
                    day=today_date.day,
                    good_sentence=today_goods,
                    blocked_sentence=today_blockeds,
                )

                today_analyzing.save()
                
            self.timestamp_ymdh = _ymdh

    def get_vocabulary_data(self):
        if self.vocabulary_data:
            return self.vocabulary_data
        elif self.is_admin_server:
            
            _unknowns = [[_['unknown'], _['text']] for _ in UnknownWord.objects.values('unknown', 'text')]
            _voca = get_all_vocabulary_from_models()
            _voca['unknowns'] = _unknowns
            # self.vocabulary_data = _voca
            return _voca
        else:
            _data_pk = ListPickle(get_vocabulary_dictionary_path() + '/data.pickle')
            return _data_pk.get_list()[0] or {}


    def get_dynamic_pinyin_block_list(self):
        return list(DynamicPinyinBlock.objects.values_list('id', 'text', 'pinyin').order_by('-id').all())
            
    def get_dynamic_alert_words_list(self):
        return list(DynamicAlertWords.objects.values_list('id', 'text').order_by('-id').all())

    def get_vocabulary_data_remotely(self, http_connection):
        
        http_connection.request('GET', self.REMOTE_ROUTE_VOCABULARY_DATA, headers={'Content-type': 'application/json'})
        _http_res = http_connection.getresponse()
        if _http_res.status == 200:

            _json_data = json.loads(_http_res.read().decode(encoding='utf-8'))
            logging.info('[get_vocabulary_data_remotely] Download Data Done.')

        else:

            _json_data = None
            logging.error('[get_vocabulary_data_remotely] Download Failed.')

        _data_pk = ListPickle(get_vocabulary_dictionary_path() + '/data.pickle')
        if _json_data:
            _data_pk.save([_json_data])
        else:
            # _json_data = _data_pk.get_list()[0]
            _json_data = {}
        
        return _json_data

    
    def get_dynamic_pinyin_block_list_remotely(self, http_connection):
        http_connection.request('GET', self.REMOTE_ROUTE_DYNAMIC_PINYIN_BLOCK, headers={'Content-type': 'application/json'})
        _http_res = http_connection.getresponse()
        json_data = None
        if _http_res.status == 200:

            json_data = json.loads(_http_res.read().decode(encoding='utf-8'))
            logging.info('[get_dynamic_pinyin_block_list_remotely] Download Data Done.')
        else:
            logging.error('[get_dynamic_pinyin_block_list_remotely] Download Failed.')
        
        return json_data

    def get_dynamic_alert_words_list_remotely(self, http_connection):
        http_connection.request('GET', self.REMOTE_ROUTE_DYNAMIC_ALERT_WORDS, headers={'Content-type': 'application/json'})
        _http_res = http_connection.getresponse()
        json_data = None
        if _http_res.status == 200:

            json_data = json.loads(_http_res.read().decode(encoding='utf-8'))
            logging.info('[get_dynamic_alert_words_list_remotely] Download Data Done.')
        else:
            logging.error('[get_dynamic_alert_words_list_remotely] Download Failed.')
        
        return json_data

    
    def get_textbook_sentense_list(self, page=1, per_page=100):
        logging.info('[get_textbook_sentense_list] is called. page: {}'.format(page))

        result = {
            'total': 0,
            'page': {
                'current': 1,
                'has_next': False,
                'has_previous': False
            },
            'data': []
        }

        _first = TextbookSentense.objects.order_by('-id').first()
        if _first:
            _latest_origin = model_to_dict(_first, ['origin'])['origin']
            print('_latest_origin: ', _latest_origin)
            _model = TextbookSentense.objects.order_by('-id')
            _sentences = _model.filter(origin=_latest_origin)
            paginator = Paginator(_sentences, per_page)
            page_obj = paginator.get_page(page)
            # data = list(page_obj.values_list('id', 'origin', 'text', 'status', 'weight'))
            data = [(ts.id, ts.origin, ts.text, ts.status, ts.weight) for ts in page_obj.object_list]

            result = {
                'total': paginator.count,
                "page": {
                    "current": page_obj.number,
                    "has_next": page_obj.has_next(),
                    "has_previous": page_obj.has_previous(),
                },
                "data": data
            }

        return result

    def get_textbook_sentense_all(self):
        _model = TextbookSentense.objects.values_list('id', 'origin', 'text', 'status', 'weight').order_by('-id')
        return list(_model.all())
    
    def get_chat_model_path(self):
        self.open_mind()
        return self.ai_app.model.get_model_path() if self.ai_app.model else None
    
    def get_ai_train_data(self):
        _app = self.ai_app
        if _app:
            return _app.get_train_data()
        return {}

    def get_ai_test_result(self):
        _app = self.ai_app
        if _app:
            return _app.get_test_result()
        return {}


    def fetch_ai_model_data(self, remote_ip, port = 80):
        if self.is_admin_server:
            logging.error('Admin Server Can Not Fetch Data From Anywhere.')
            return exit(2)

        _http_cnn = HTTPConnection(remote_ip, port)
        self.main_admin_server_addr = (remote_ip, port)

        def _save_file_by_http_response(response, path):
            with open(path, 'wb+') as f:
                while True:
                    _buf = response.read()
                    if _buf:
                        f.write(_buf)
                    else:
                        break

        

        if self.lang_mode == self.STATUS_MODE_CHINESE:

            self.vocabulary_data = self.get_vocabulary_data_remotely(_http_cnn)
            self.reload_pinyin_block(_http_cnn)
            self.reload_alert_words(_http_cnn)
            _http_cnn.request('GET', self.REMOTE_ROUTE_CHAT_MODEL)
            _http_res = _http_cnn.getresponse()
            if _http_res.status == 200:
                _chinese_model_path = get_chinese_chat_model_path()
                _save_file_by_http_response(response=_http_res, path=_chinese_model_path+'/model.remote.h5')
                logging.info('[fetch_ai_model_data] Download Remote Chinese Chat Model Done.')
            else:

                logging.error('[fetch_ai_model_data] Download Remote Chinese Chat Model Failed.')

        elif self.lang_mode == self.STATUS_MODE_ENGLISH:
            #
            _http_cnn.request('GET', self.REMOTE_ROUTE_CHAT_MODEL)
            _http_res = _http_cnn.getresponse()
            if _http_res.status == 200:
                _save_file_by_http_response(response=_http_res, path=get_multi_lingual_bert_chat_model_path()+'/model.remote.h5')
                logging.info('[fetch_ai_model_data] Download Remote Multi Lingual Chat Model Done.')

            else:

                logging.error('[fetch_ai_model_data] Download Remote Multi Lingual Chat Model Failed.')
        

    def add_textbook_sentense(self, origin='', sentenses=[]):
        limit_tbs_size = 100
        try:
            _exist_texts = list(TextbookSentense.objects.filter(origin=origin).values_list('text', flat=True))
            tbs = []
            for _sen in sentenses:
                _text = _sen[0]
                _status = int(_sen[1])
                _weight = int(_sen[2]) if _sen[2] else 1
                _text, _lv, _a = self.parse_message(_text)
                if self.lang_mode == self.STATUS_MODE_CHINESE:
                    _text = self.trim_text(_text)
                if _text and _status >= 0 and _text not in _exist_texts:
                    _textbook = TextbookSentense(
                        origin=origin,
                        text=_text,
                        status=_status,
                        weight=_weight,
                    )
                    tbs.append(_textbook)
                # _textbook.save()
                if len(tbs) >= limit_tbs_size:
                    TextbookSentense.objects.bulk_create(tbs)
                    tbs = []

            if len(tbs) > 0:
                TextbookSentense.objects.bulk_create(tbs)

            return True
        except Exception as err:
            logging.error(str(err))
            raise Exception(err)


    def add_pinyin_block(self, text):
        _num_max = 255
        try:
            if isinstance(text, list):
                _dps = []
                for _ in text:
                    _py = translate_by_string(_)
                    _dps.append(DynamicPinyinBlock(text=_, pinyin=_py))
                DynamicPinyinBlock.objects.bulk_create(_dps, batch_size=100)
                _ids = list(DynamicPinyinBlock.objects.values_list('pk', flat=True)[_num_max:])
                if len(_ids) > 0:
                    DynamicPinyinBlock.objects.filter(id__in=_ids).delete()
                return [model_to_dict(_) for _ in _dps]
            else:
                _py = translate_by_string(text)
                qs = DynamicPinyinBlock.objects.filter(pinyin=_py)
                if len(qs) == 0:
                    qs = DynamicPinyinBlock.objects.create(
                        text=text,
                        pinyin=_py,
                    )
                    _count = DynamicPinyinBlock.objects.count()
                    if _count > _num_max:
                        DynamicPinyinBlock.objects.all().first().delete()
                    return model_to_dict(qs)
                else:
                    return None
        except Exception as err:
            print('add_pinyin_block error: ', err)
            return None

    def add_alert_words(self, text):
        _num_max = 255
        try:
            if isinstance(text, list):
                _dps = []
                for _ in text:
                    _dps.append(DynamicAlertWords(text=_))
                DynamicAlertWords.objects.bulk_create(_dps, batch_size=100)
                _ids = list(DynamicAlertWords.objects.values_list('pk', flat=True)[_num_max:])
                if len(_ids) > 0:
                    DynamicAlertWords.objects.filter(id__in=_ids).delete()
                return [model_to_dict(_) for _ in _dps]
            else:
                qs = DynamicAlertWords.objects.filter(text=text)
                if len(qs) == 0:
                    qs = DynamicAlertWords.objects.create(
                        text=text
                    )
                    _count = DynamicAlertWords.objects.count()
                    if _count > _num_max:
                        DynamicAlertWords.objects.all().first().delete()
                    return model_to_dict(qs)
                else:
                    return None
        except Exception as err:
            print('add_alert_words error: ', err)
            return None


    def remove_textbook_sentense(self, id):
        try:
            if isinstance(id, int):
                TextbookSentense.objects.get(pk=id).delete()
            elif id == 'all':
                TextbookSentense.objects.all().delete()
            return True
        except Exception as err:
            return False


    def remove_pinyin_block(self, id):
        try:
            DynamicPinyinBlock.objects.get(pk=id).delete()
            return True
        except Exception as err:
            return False

    def remove_alert_words(self, id):
        try:
            DynamicAlertWords.objects.get(pk=id).delete()
            return True
        except Exception as err:
            return False


    def reload_pinyin_block(self, conn = None):
        print('[Reload_pinyin_block] Triggered.')
        if conn:
            _dpb_list = self.get_dynamic_pinyin_block_list_remotely(conn)
        elif self.main_admin_server_addr:
            _ip, _port = self.main_admin_server_addr
            _dpb_list = self.get_dynamic_pinyin_block_list_remotely(HTTPConnection(_ip, _port))
        else:
            _dpb_list = self.get_dynamic_pinyin_block_list()
        
        self.pre_filter.set_pinyin_block_list(_dpb_list)

    def reload_alert_words(self, conn = None):
        print('[Reload_alert_words] Triggered.')
        if conn:
            _dpb_list = self.get_dynamic_alert_words_list_remotely(conn)
        elif self.main_admin_server_addr:
            _ip, _port = self.main_admin_server_addr
            _dpb_list = self.get_dynamic_alert_words_list_remotely(HTTPConnection(_ip, _port))
        else:
            _dpb_list = self.get_dynamic_alert_words_list()
        
        self.pre_filter.set_alert_words_list(_dpb_list)


    
            
