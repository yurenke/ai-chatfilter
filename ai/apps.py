from django.apps import AppConfig
from tensorflow import keras
from .helper import get_chinese_chat_model_path, get_multi_lingual_bert_chat_model_path

from .classes.english_filter_basic import BasicEnglishFilter
from .classes.chinese_chat_filter import ChineseChatFilter
from .classes.chinese_nickname_model import ChineseNicknameModel
from .classes.multi_lingual_bert_chat_filter import MultiLingualBertChatFilter

import tensorflow as tf
import os
from datetime import datetime

chinese_chat_model_path = get_chinese_chat_model_path()
multi_lingual_bert_chat_model_path = get_multi_lingual_bert_chat_model_path()


class MainAiApp():
    # transformer_model = None
    # multi_lingual_chat_model = None
    model = None

    pinyin_data = []
    english_data = []
    chinese_data = []

    # loaded_models = []
    # loaded_model_names = []

    def __init__(self, pinyin_data=[], english_data=[], chinese_data=[]):
        print('=============  A.I Init  =============')

        if pinyin_data:
            self.pinyin_data = pinyin_data

        if english_data:
            self.english_data = english_data

        if chinese_data:
            self.chinese_data = chinese_data
        
        print('using tensorflow version: ', tf.__version__)

    def load_chinese_chat_model(self, folder=None):
        _model_path = folder if folder else chinese_chat_model_path
        self.model = ChineseChatFilter(load_folder=_model_path)

        # self.loaded_models.append(self.transformer_model)
        # self.loaded_model_names.append('transformer')

    def load_multi_lingual_chat_model(self, folder=None):
        # print('[load_english]: english_data: ', self.english_data)
        _model_path = folder if folder else multi_lingual_bert_chat_model_path
        # _english_vocabulary = [_[0] for _ in self.english_data]
        self.model = MultiLingualBertChatFilter(load_folder=_model_path)
        # self.loaded_models.append(self.multi_lingual_chat_model)
        # self.loaded_model_names.append('multi_lingual_chat')

    def predict(self, txt, lv=0, with_reason=False):
        prediction = 0
        reason = ''
        # _certainty_reason = 'SURE'
        prediction, reason = self.model.predictText(txt, lv, with_reason=with_reason)

        # for model in self.loaded_models:
        #     _predict, reason = model.predictText(txt, lv, with_reason=with_reason)
        #     if _predict == 0:
        #         if reason == _certainty_reason:
        #             break
        #     else:
        #         prediction = _predict
        #         break

        return prediction, reason

    
    def get_details(self, txt):
        details_result = {'text': txt}
        if txt:
            # _i = 0
            # for model in self.loaded_models:
            #     _detail = model.get_details(txt)
            #     details_result[self.loaded_model_names[_i]] = _detail
            #     _i += 1
            _detail = self.model.get_details(txt)
        details_result['details'] = _detail

        return details_result


    def get_ai_dir(self):
        return os.path.dirname(os.path.realpath(__file__))

    def get_train_data(self):
        if self.model:
            return self.model.get_last_history()
        else:
            return {
                    'accuracy': 0,
                    'loss': 0,
                    'validation': 0,
                    'ontraining': False,
                    'ETA': 0,
                    'timestamp': datetime.now().isoformat(),
                    'origin': 'NA',
                }

    def get_test_result(self):
        if self.model:    
            return self.model.get_test_result()
        else:
            return {'status': 'done', 'acc': 0, 'length': 0, 'right': 0, 'acc_rv': 0}


class NicknameAiApp():
    model = None

    def __init__(self):
        print('============= Nickname A.I Init  =============')
        print('using tensorflow version: ', tf.__version__)

    def load_chinese_nickname_model(self):
        self.model = ChineseNicknameModel()

    def predict(self, txt):
        prediction = self.model.predict(txt)

        return prediction
    
    def get_details(self, txt):
        details_result = {'text': txt}
        if txt:
            _detail = self.model.get_details(txt)
            details_result['details'] = _detail

        return details_result

    def get_train_data(self):
        if self.model:
            return self.model.get_last_history()
        else:
            return {
                    'accuracy': 0,
                    'loss': 0,
                    'validation': 0,
                    'ontraining': False,
                    'ETA': 0,
                    'timestamp': datetime.now().isoformat(),
                    'origin': 'NA',
                }

    def get_test_result(self):
        if self.model:
            return self.model.get_test_result()
        else:
            return {'status': 'done', 'acc': 0, 'length': 0, 'right': 0, 'acc_rv': 0}