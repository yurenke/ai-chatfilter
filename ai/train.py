from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys, getopt
import tensorflow as tf
import tensorflow_datasets as tfds

from datetime import datetime

from dataparser.apps import ExcelParser, MessageParser
from dataparser.jsonparser import JsonParser
from dataparser.classes.store import ListPickle
from .classes.chinese_filter_pinyin import PinYinFilter, PinYinReverseStateFilter
from .classes.chinese_filter_grammar import GrammarFilter
from .classes.english_filter_basic import BasicEnglishFilter
from .classes.chinese_filter_basic import BasicChineseFilter
from .helper import print_spend_time, get_pinyin_path, get_grammar_path, get_english_model_path, get_pinyin_multiple_version_path, get_pinyin_re_path, get_chinese_path



def get_row_list_by_excel_path(excel_file_path):
    ep = ExcelParser(file=excel_file_path)
    basic_model_columns = [['VID', '房號'], ['LOGINNAME', '會員號'], ['MESSAGE', '聊天信息', '禁言内容', '发言内容'], ['STATUS', '審核結果', '状态']]
    result_list = ep.get_row_list(column=basic_model_columns)

    message_parser = MessageParser()
    for res in result_list:
        msg = res[2]
        text, lv, anchor = message_parser.parse(msg)

        res.append(text)
        res.append(lv)
        res.append(anchor)

    return result_list



def get_row_list_by_json_path(json_file_path, allowed_weight=0):
    _jp = JsonParser(file=json_file_path)
    data_list = _jp.load()
    if allowed_weight and allowed_weight > 0:
        data_list = [_ for _ in data_list if _[1] >= allowed_weight]
    return data_list



def train_pinyin_by_excel_path(excel_file_path = None, final_accuracy = None, max_spend_time=0):
    
    if excel_file_path is not None:

        result_list = get_row_list_by_excel_path(excel_file_path)
    else:

        result_list = []

    if len(result_list) == 0:
        print('Wrong with no file path input.')
        return

    print('The result list length: ', len(result_list))

    train_pinyin_by_list(result_list, final_accuracy, max_spend_time)



def train_pinyin_by_json_path(json_file_path, final_accuracy = None, max_spend_time=0 ,allowed_weight=0 , is_re_mode=False):
    result_list = get_row_list_by_json_path(json_file_path, allowed_weight=allowed_weight)
    train_pinyin_by_list(result_list, final_accuracy, max_spend_time, is_re_mode)



def train_pinyin_by_list(train_data_list, final_accuracy = None, max_spend_time=0, is_re_mode=False):

    _st_time = datetime.now() #

    if is_re_mode:

        # train_data_list = [[_[0], _[1], _[2], _[3] if _[3]>0 else STATE_RE_VERSION_PASS] for _ in train_data_list]
        piny = PinYinReverseStateFilter(load_folder=get_pinyin_re_path())

    else:

        piny = PinYinFilter(load_folder=get_pinyin_path())

    history = piny.fit_model(train_data=train_data_list, stop_accuracy=final_accuracy, stop_hours=max_spend_time)

    print('=== history ===')
    print(history)
    print_spend_time(_st_time)

    return piny



def train_grammar_by_excel_path(excel_file_path = None, final_accuracy = None, max_spend_time=0):
    
    result_list = get_row_list_by_excel_path(excel_file_path)

    print('The result list length: ', len(result_list))

    train_grammar_by_list(result_list, final_accuracy, max_spend_time)



def train_grammar_by_json_path(json_file_path, final_accuracy = None, max_spend_time=0 ,allowed_weight=0):
    result_list = get_row_list_by_json_path(json_file_path, allowed_weight=allowed_weight)
    train_grammar_by_list(result_list, final_accuracy, max_spend_time)



def train_grammar_by_list(train_data_list = None, final_accuracy = None, max_spend_time=0):

    _saved_folder = get_grammar_path()

    _st_time = datetime.now() #

    model = GrammarFilter(load_folder=_saved_folder)

    history = model.fit_model(train_data=train_data_list, stop_accuracy=final_accuracy, stop_hours=max_spend_time)

    print('=== history ===')
    print(history)
    print_spend_time(_st_time)



def train_english_by_json_path(json_file_path, final_accuracy = None, max_spend_time=0 ,allowed_weight=0):
    result_list = get_row_list_by_json_path(json_file_path, allowed_weight=allowed_weight)
    train_english_by_list(result_list, final_accuracy, max_spend_time)


def train_english_by_list(train_data_list = None, final_accuracy = None, max_spend_time=0):

    _saved_folder = get_english_model_path()

    _st_time = datetime.now() #

    model = BasicEnglishFilter(load_folder=_saved_folder)

    history = model.fit_model(train_data=train_data_list, stop_accuracy=final_accuracy, stop_hours=max_spend_time)

    print('=== history ===')
    print(history)
    print_spend_time(_st_time)


def train_pinyin_to_next_version(train_data_list, jieba_vocabulary, jieba_freqs, stop_hours = 1):

    pinyin_saved_folder = get_pinyin_path()

    piny = PinYinFilter(load_folder=pinyin_saved_folder, jieba_vocabulary=jieba_vocabulary, jieba_freqs=jieba_freqs)

    next_version_saved_folder = get_pinyin_multiple_version_path()

    history = piny.fit_model(train_data=train_data_list, stop_hours=stop_hours, save_folder=next_version_saved_folder)

    return history


def train_chinese_by_json_path(json_file_path, final_accuracy = None, max_spend_time=0 ,allowed_weight=0):
    result_list = get_row_list_by_json_path(json_file_path, allowed_weight=allowed_weight)
    train_chinese_by_list(result_list, final_accuracy, max_spend_time)


def train_chinese_by_list(train_data_list = None, final_accuracy = None, max_spend_time=0):

    _st_time = datetime.now() #

    model = BasicChineseFilter(load_folder=get_chinese_path())

    history = model.fit_model(train_data=train_data_list, stop_accuracy=final_accuracy, stop_hours=max_spend_time)

    print('=== history ===')
    print(history)
    print_spend_time(_st_time)