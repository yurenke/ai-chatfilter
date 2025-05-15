from django.utils import timezone
from django.conf import settings
from django.forms.models import model_to_dict
from django.core.paginator import Paginator
import numpy as np
import time, re, logging, json
from http.client import HTTPConnection
from service.widgets import printt
from service.classes.unicode import contains_not_allowed_chars, is_not_allowed_chinese_radicals
from ai.apps import NicknameAiApp
from dataparser.apps import MessageParser
from .models import DynamicNicknamePinyinBlock, ChangeNicknameRequest
from ai.models import NicknameTextbook
from .classes.prefilter import PreFilter
from ai.helper import get_chinese_nickname_model_path
from ai.classes.translator_pinyin import translate_by_string
import langid
from .preprocesstext import cc

LANG_OTHERS = 0
LANG_CH = 1

class NicknameFilter():
    """

    """
    # 0 ~ 6 are handled and returned by model
    CODE_OK = 0
    CODE_DIRTY_WORDS = 1
    CODE_ADS = 2
    CODE_POLITICAL_WORDS = 3
    CODE_CUSTOMER_SERVICE_RELATED = 4
    CODE_SUSPICIOUS_NUMBERS_BY_MODEL = 5
    CODE_UNCOMMON_WORDS_BY_MODEL = 6
    # 7 ~ 10 are handled and returned by rule
    CODE_INVALID_FORMAT = 7
    CODE_BLOCKED_WORDS = 8
    CODE_SUSPICIOUS_NUMBERS_BY_RULE = 9
    CODE_UNCOMMON_WORDS_BY_RULE = 10

    STATUS_MODE_CHINESE = 1
    STATUS_MODE_ENGLISH = 2

    REMOTE_ROUTE_NICKNAME_MODEL = '/api/model/nickname'
    REMOTE_ROUTE_DYNAMIC_NICKNAME_PINYIN_BLOCK = '/api/data/dnpinyinblist'

    main_admin_server_addr = None

    regex_is_eng = re.compile('[a-zA-Z]')
    lang_mode = 1
    english_parser = None

    def __init__(self, is_admin=True, lang_mode=0):
        if lang_mode == 0:
            self.init_language()

        self.is_open_mind = False
        self.is_training = False
        self.is_testing = False
        self.is_admin_server = False

        self.message_parser = MessageParser()
        self.pre_filter = PreFilter()

        if is_admin:
            self.is_admin_server = True
            # self.check_analyzing()
            if self.lang_mode == self.STATUS_MODE_CHINESE:
                self.pre_filter.set_nickname_pinyin_block_list(self.get_dynamic_nickname_pinyin_block_list())

        logging.info('=============  Nickname Filter Activated. Time Zone: [ {} ] ============='.format(settings.TIME_ZONE))


    def init_language(self):
        _setting = settings.LANGUAGE_MODE
        if _setting == 'EN':
            self.lang_mode = self.STATUS_MODE_ENGLISH
        else:
            self.lang_mode = self.STATUS_MODE_CHINESE

    def open_mind(self):
        if self.is_open_mind:
            return True

        self.ai_app = NicknameAiApp()

        if self.lang_mode == self.STATUS_MODE_CHINESE:
            self.ai_app.load_chinese_nickname_model()
            self.ai_app.model.save(is_check=True, is_training=False)

        self.is_open_mind = True

        return True

    # def transform_text(self, text):
    #     return self.message_parser.transform_full_char_to_half(text).strip()

    def think(self, nickname, detail=False):
        

        if self.lang_mode == self.STATUS_MODE_CHINESE:

            # pred, reason = self.think_chinese(nickname=nickname)
            return self.think_chinese(nickname=nickname, detail=detail)

        elif self.lang_mode == self.STATUS_MODE_ENGLISH:

            # pred, reason = self.think_english(nickname=nickname)
            return self.think_english(nickname=nickname, detail=detail)

        # return self.return_result(prediction=pred, text=nickname, reason=reason, detail=detail, st_time=st_time)


    def think_chinese(self, nickname, detail=False):
        st_time = time.time()
        pred = self.CODE_OK
        reason = ''
        digits = 0
        num_of_ch_chars = 0
        # lang = LANG_OTHERS
        suspicious_pinyin_number_count = 0

        if len(nickname) == 0:
            pred = self.CODE_INVALID_FORMAT
            reason = 'empty'
            return self.return_result(prediction=pred, text='', reason=reason, detail=detail, st_time=st_time)
        
        if contains_not_allowed_chars(nickname):
            pred = self.CODE_INVALID_FORMAT
            reason = 'not allowed chars'
            return self.return_result(prediction=pred, text=nickname, reason=reason, detail=detail, st_time=st_time)
        
        text = nickname.encode('utf-8', errors="ignore").decode('utf-8')
        # text = text.lower()
        text = cc.convert(text)
        # lang = LANG_CH if langid.classify(text)[0] == 'zh' else LANG_OTHERS
        
        for uchar in text:
            if uchar.isdigit():
                digits += 1
                if digits >= 3:
                    pred = self.CODE_INVALID_FORMAT
                    reason = 'number of digits >= 3'
                    return self.return_result(prediction=pred, text=text, reason=reason, detail=detail, st_time=st_time)
            
            # if lang == LANG_CH:
            if u'\u4e00' <= uchar and uchar <= u'\u9fa5':
                num_of_ch_chars += 1
                if is_not_allowed_chinese_radicals(uchar) or uchar not in self.ai_app.model.vocab:
                    pred = self.CODE_UNCOMMON_WORDS_BY_RULE
                    reason = uchar
                    return self.return_result(prediction=pred, text=text, reason=reason, detail=detail, st_time=st_time)
                
                elif self.pre_filter.is_number_pinyin(uchar):
                    suspicious_pinyin_number_count += 1
                    if suspicious_pinyin_number_count >= 3: # 3 consecutive chinese words which are suspicious pinyin numbers
                        pred = self.CODE_SUSPICIOUS_NUMBERS_BY_RULE
                        reason = 'suspicious pinyin numbers'
                        return self.return_result(prediction=pred, text=text, reason=reason, detail=detail, st_time=st_time)
                
            else:
                suspicious_pinyin_number_count = 0

        if num_of_ch_chars > 0 and (len(text) - num_of_ch_chars) >= 3:
            pred = self.CODE_INVALID_FORMAT
            reason = 'mixed with chinese, number of non-chinese chars >= 3'
            return self.return_result(prediction=pred, text=text, reason=reason, detail=detail, st_time=st_time)

        reason = self.pre_filter.find_unallow_nickname_eng(text)
        if reason:
            pred = self.CODE_BLOCKED_WORDS
            return self.return_result(prediction=pred, text=text, reason=reason, detail=detail, st_time=st_time)

        # if digits + eng == len(text):
        #     return self.CODE_OK, '' # if all chars are digits or eng, pass over to human judgement now

        # if eng >= 3:
        #     return self.CODE_INVALID_FORMAT, 'with chinese, eng >= 3'

        # check pinyin blocked list
        if num_of_ch_chars > 0:
            reason = self.pre_filter.find_nickname_pinyin_blocked(text)
            if reason:
                pred = self.CODE_BLOCKED_WORDS
                return self.return_result(prediction=pred, text=text, reason=reason, detail=detail, st_time=st_time)

        # model prediction
        pred = self.ai_app.predict(text)

        return self.return_result(prediction=pred, text=text, reason='', detail=detail, st_time=st_time)

    def think_english(self, nickname, detail=False):
        st_time = time.time()
        return self.return_result(prediction=self.CODE_OK, text=nickname, reason='', detail=detail, st_time=st_time)


    # def set_english_parser(self, parser_instance):
    #     self.english_parser = parser_instance
    #     print('[NicknameFilter] Set English Parser Vocabulary Length: ', len(self.english_parser.get_vocabulary()))

    def return_result(self, prediction, text='', reason='', detail=False, st_time=0):
        result = {}
        detail_data = {}
        
        if detail:

            detail_data = self.ai_app.get_details(text)

        ed_time = time.time()
        result['text'] = text
        result['code'] = int(prediction)
        result['reason_char'] = reason
        result['detail'] = detail_data
        result['spend_time'] = ed_time - st_time
        if result['spend_time'] > 0.2:
            logging.error('Nickname spend Time Of Think Result = {}'.format(result['spend_time']))
        logging.debug('Nickname Think Result [ text: {} prediction: {} time: {} ] '.format(result['text'], result['code'], result['spend_time']))
        return result

    def get_dynamic_nickname_pinyin_block_list(self):
        return list(DynamicNicknamePinyinBlock.objects.values_list('id', 'text', 'pinyin').order_by('-id').all())

    def get_dynamic_nickname_pinyin_block_list_remotely(self, http_connection):
        http_connection.request('GET', self.REMOTE_ROUTE_DYNAMIC_NICKNAME_PINYIN_BLOCK, headers={'Content-type': 'application/json'})
        _http_res = http_connection.getresponse()
        json_data = None
        if _http_res.status == 200:

            json_data = json.loads(_http_res.read().decode(encoding='utf-8'))
            logging.info('[get_dynamic_nickname_pinyin_block_list_remotely] Download Data Done.')
        else:
            logging.error('[get_dynamic_nickname_pinyin_block_list_remotely] Download Failed.')
        
        return json_data

    def remove_from_nickname_textbook(self, id):
        try:
            if isinstance(id, int):
                NicknameTextbook.objects.get(pk=id).delete()
            elif id == 'all':
                NicknameTextbook.objects.all().delete()
            return True
        except Exception as err:
            return False

    def remove_nickname_pinyin_block(self, id):
        try:
            DynamicNicknamePinyinBlock.objects.get(pk=id).delete()
            return True
        except Exception as err:
            return False

    def reload_nickname_pinyin_block(self, conn = None):
        print('[Reload_nickname_pinyin_block] Triggered.')
        if conn:
            _dpb_list = self.get_dynamic_nickname_pinyin_block_list_remotely(conn)
        elif self.main_admin_server_addr:
            _ip, _port = self.main_admin_server_addr
            _dpb_list = self.get_dynamic_nickname_pinyin_block_list_remotely(HTTPConnection(_ip, _port))
        else:
            _dpb_list = self.get_dynamic_nickname_pinyin_block_list()
        
        self.pre_filter.set_nickname_pinyin_block_list(_dpb_list)

    def get_nickname_model_path(self):
        self.open_mind()
        return self.ai_app.model.get_model_path() if self.ai_app.model else None

    def fetch_nickname_ai_model_data(self, remote_ip, port = 80):
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

        # self.vocabulary_data = self.get_vocabulary_data_remotely(_http_cnn)

        if self.lang_mode == self.STATUS_MODE_CHINESE:
            #
            self.reload_nickname_pinyin_block(_http_cnn)
            _http_cnn.request('GET', self.REMOTE_ROUTE_NICKNAME_MODEL)
            _http_res = _http_cnn.getresponse()
            if _http_res.status == 200:
                _nickname_model_path = get_chinese_nickname_model_path()
                _save_file_by_http_response(response=_http_res, path=_nickname_model_path+'/model.remote.h5')
                logging.info('[fetch_nickname_ai_model_data] Download Remote Nickname Model Done.')

            else:

                logging.error('[fetch_nickname_ai_model_data] Download Remote Nickname Model Failed.')

        elif self.lang_mode == self.STATUS_MODE_ENGLISH:
            logging.info('[fetch_nickname_ai_model_data] Currently remote nickname model for english is not available.')

        else:

            logging.error('[fetch_nickname_ai_model_data] Download Remote English Model Failed.')

    def get_nickname_textbook_list(self, page=1, per_page=100):
        logging.info('[get_nickname_textbook_sentense_list] is called. page: {}'.format(page))

        result = {
            'total': 0,
            'page': {
                'current': 1,
                'has_next': False,
                'has_previous': False
            },
            'data': []
        }

        _first = NicknameTextbook.objects.order_by('-id').first()
        if _first:
            _latest_origin = model_to_dict(_first, ['origin'])['origin']
            print('_latest_origin: ', _latest_origin)
            _model = NicknameTextbook.objects.order_by('-id')
            _sentences = _model.filter(origin=_latest_origin)
            paginator = Paginator(_sentences, per_page)
            page_obj = paginator.get_page(page)
            data = [(ts.id, ts.origin, ts.text, ts.status) for ts in page_obj.object_list]

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

    def get_nickname_textbook_all(self):
        _model = NicknameTextbook.objects.values_list('id', 'origin', 'text', 'status').order_by('-id')
        return list(_model.all())

    def get_nickname_ai_train_data(self):
        _app = self.ai_app
        if _app:
            return _app.get_train_data()
        return {}

    def get_nickname_ai_test_result(self):
        _app = self.ai_app
        if _app:
            return _app.get_test_result()
        return {}

    def add_to_nickname_textbook(self, origin='', nicknames=[]):
        limit_tbs_size = 100
        try:
            _exist_texts = list(NicknameTextbook.objects.filter(origin=origin).values_list('text', flat=True))
            tbs = []
            for _nic in nicknames:
                _text = _nic[0]
                _status = int(_nic[1])
                # if self.lang_mode == self.STATUS_MODE_CHINESE:
                #     _text = self.transform_text(_text)
                if _text and _status >= 0 and _text not in _exist_texts:
                    _textbook = NicknameTextbook(
                        origin=origin,
                        text=_text,
                        status=_status,
                    )
                    tbs.append(_textbook)
                # _textbook.save()
                if len(tbs) >= limit_tbs_size:
                    NicknameTextbook.objects.bulk_create(tbs)
                    tbs = []

            if len(tbs) > 0:
                NicknameTextbook.objects.bulk_create(tbs)

            return True
        except Exception as err:
            logging.error(str(err))
            raise Exception(err)


    def add_nickname_pinyin_block(self, text):
        _num_max = 255
        try:
            if isinstance(text, list):
                _dps = []
                for _ in text:
                    _ = cc.convert(_)
                    _py = translate_by_string(_)
                    _dps.append(DynamicNicknamePinyinBlock(text=_, pinyin=_py))
                DynamicNicknamePinyinBlock.objects.bulk_create(_dps, batch_size=100)
                _ids = list(DynamicNicknamePinyinBlock.objects.values_list('pk', flat=True)[_num_max:])
                if len(_ids) > 0:
                    DynamicNicknamePinyinBlock.objects.filter(id__in=_ids).delete()
                return [model_to_dict(_) for _ in _dps]
            else:
                text = cc.convert(text)
                _py = translate_by_string(text)
                qs = DynamicNicknamePinyinBlock.objects.filter(pinyin=_py)
                if len(qs) == 0:
                    qs = DynamicNicknamePinyinBlock.objects.create(
                        text=text,
                        pinyin=_py,
                    )
                    _count = DynamicNicknamePinyinBlock.objects.count()
                    if _count > _num_max:
                        DynamicNicknamePinyinBlock.objects.all().first().delete()
                    return model_to_dict(qs)
                else:
                    return None
        except Exception as err:
            print('add_nickname_pinyin_block error: ', err)
            return None

    def saveNicknameRequestRecord(self, nickname, status):
        if self.is_admin_server:

            try:
                record = ChangeNicknameRequest(
                    nickname=nickname,
                    status=status,
                )
                record.save()
            except Exception as ex:
                logging.error('Save NicknameRequestRecord Failed, nickname: [{}].'.format(nickname))
                print(ex)