from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime
import os
from configparser import RawConfigParser

AI_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(AI_DIR)

# config_setting = RawConfigParser()
# config_setting.read(BASE_DIR+'/setting.ini')
config_version = RawConfigParser()
config_version.read(BASE_DIR+'/version.cfg')
version = '{}.{}'.format(config_version.get('MAIN', 'V'), config_version.get('MAIN', 'SUR'))




def print_spend_time(_st_time):
    _ed_time = datetime.now() #
    _spend_seconds = (_ed_time - _st_time).total_seconds() #
    _left_seconds = int(_spend_seconds % 60)
    _spend_minutes = int(_spend_seconds // 60)
    _left_minutes = int(_spend_minutes % 60)
    _left_hours = int(_spend_minutes // 60)
    print('==== spend time: {:d} h: {:d} m: {:d} s'.format(_left_hours, _left_minutes, _left_seconds))
    return _spend_seconds

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_multi_lingual_bert_nickname_model_path():
    _path = os.path.dirname(os.path.abspath(__file__)) + '/_models/multi_lingual_bert_nickname_model'
    return check_path(_path)

def get_multi_lingual_bert_chat_model_path():
    _path = os.path.dirname(os.path.abspath(__file__)) + '/_models/multi_lingual_bert_chat_model'
    return check_path(_path)

def get_chinese_nickname_model_path():
    _path = os.path.dirname(os.path.abspath(__file__)) + '/_models/chinese_nickname_model'
    return check_path(_path)

def get_chinese_chat_model_path():
    _path = os.path.dirname(os.path.abspath(__file__)) + '/_models/chinese_chat_model'
    return check_path(_path)

def get_ss_path():
    _path = os.path.dirname(os.path.abspath(__file__)) + '/_models/ss_model'
    return check_path(_path)

def get_pinyin_path(is_version=False):
    _path = os.path.dirname(os.path.abspath(__file__)) + '/_models/pinyin_model'
    if is_version:
        _path += '/vers'
    return check_path(_path)

def get_pinyin_re_path(is_version=False):
    _path = os.path.dirname(os.path.abspath(__file__)) + '/_models/pinyin_re_model'
    if is_version:
        _path += '/vers'
    return check_path(_path)

def get_pinyin_multiple_version_path():
    _path = os.path.dirname(os.path.abspath(__file__)) + '/_model_vers/pinyin_model/' + version + datetime.now().strftime("_%Y_%m_%d")
    return check_path(_path)

def get_grammar_path():
    _path = os.path.dirname(os.path.abspath(__file__)) + '/_models/grammar_model'
    return check_path(_path)

def get_english_model_path():
    _path = os.path.dirname(os.path.abspath(__file__)) + '/_models/english_model'
    return check_path(_path)

def get_chinese_path(is_version=False):
    _path = os.path.dirname(os.path.abspath(__file__)) + '/_models/chinese_model'
    if is_version:
        _path += '/vers'
    return check_path(_path)

def get_vocabulary_dictionary_path():
    _path = os.path.dirname(os.path.abspath(__file__)) + '/_pickles/vocabulary'
    return check_path(_path)

