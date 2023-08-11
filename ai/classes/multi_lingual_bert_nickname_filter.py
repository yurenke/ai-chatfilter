from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import json, logging
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import TimeDistributed, Embedding, Multiply, Add
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, Lambda
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from opencc import OpenCC

import string
import jieba
import pandas as pd

from ai import GRAPH_DIR
from ai.classes.basic_filter import BasicFilter
from datetime import datetime, timedelta

from ai.helper import get_multi_lingual_nickname_model_path

class MultiLingualBertNicknameFilter():
    def __init__(self):
        self.model_path = get_multi_lingual_nickname_model_path()
        # self.parser = MessageParser()
        # self.load()
        # super().__init__(load_folder=load_folder)

    # def load_vocab(self):
    #     with open(os.path.join(get_nickname_model_path(), "nickname_filter_vocab.txt"), encoding='UTF-8') as f:
    #         idx = 0
    #         for line in f:
    #             word = line.strip()
    #             self.vocab[word] = idx
    #             idx += 1
    #     print('nickname filter vocab loaded. {} words in total'.format(len(self.vocab)))

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

        return self.model

    def load(self):
        # model_path = self.get_model_path()
        
        # logging.info('Starting load multi-lingual nickname model: {}'.format(model_path))
        
        # self.load_model(model_path)
        
        # logging.info('Successful load multi-lingual nickname model. ')

        logging.info('currently no multi-lingual nickname model ')

    def get_model_path(self):
        _path = self.model_path + '/model.h5'
        if not os.path.exists(_path):
            _path = self.model_path + '/model.remote.h5'
        return _path


    def save(self, is_check = False, history = None, is_training= False, eta=0, origin='none'):
        
        folder = get_multi_lingual_nickname_model_path()
        
        if not is_check:
            # self.model.save(folder + '/model.h5')
        
            print('Successful saved. ')

        if history:
            print('Saved History: ', history)
            
            with open(folder + '/last.history', 'w+') as f:
                _acc = max(history.get('accuracy', [0]))
                _los = min(history.get('loss', [0]))
                _val_acc = max(history.get('val_accuracy', [0]))
                _acc = int(_acc * 10000) / 10000
                _los = int(_los * 10000) / 10000
                _val_acc = int(_val_acc * 10000) / 10000

                f.write(json.dumps({
                    'accuracy': _acc,
                    'loss': _los,
                    'validation': _val_acc,
                    'ontraining': is_training,
                    'ETA': eta if is_training else 0,
                    'timestamp': datetime.now().isoformat(),
                    'origin': origin,
                }, indent = 2))
        
        else:

            _json = {}
            try:
                with open(folder + '/last.history', 'r') as f:
                    _string = f.read()
                    if _string:
                        _json = json.loads(_string)
            except Exception as err:
                print(err)
            
            print('Save No History, last.history: ', _json)

            with open(folder + '/last.history', 'w+') as f:
                f.write(json.dumps({
                    'accuracy': _json.get('accuracy', 0),
                    'loss': _json.get('loss', 0),
                    'validation': _json.get('validation', 0),
                    'ontraining': is_training,
                    'ETA': eta if is_training else 0,
                    'timestamp': datetime.now().isoformat(),
                    'origin': origin,
                }, indent = 2))

        return self

    def get_details(self, text):
        # transformed_words = self.transform(text)
        # encoded_words = self.get_encode_word(transformed_words)

        # x = tf.expand_dims(tf.convert_to_tensor(encoded_words), 0)
        # predicted = self.model(x)[0]
        
        # return {
        #     'transformed_words': transformed_words,
        #     'encoded_words': encoded_words.tolist(),
        #     'predicted_ratios': ['{:2.2%}'.format(_) for _ in list(predicted)]
        # }
        return {
            'transformed_words': [],
            'encoded_words': [],
            'predicted_ratios': []
        }

    def predict(self, text):
        # possible = 0

        # _words = self.transform(text)
        # if len(_words) == 0:
        #     return possible
        
        # _result_text = self.get_encode_word(_words)

        # x = tf.expand_dims(tf.convert_to_tensor(_result_text), 0)
        # predicted = self.model(x)[0]

        # possible = np.argmax(predicted)
                
        # return possible
        return 0

    # def transform(self, data):

    #     text = self.cc.convert(data)
    #     return NicknameModel.tokenize_sentence(text)

    # def get_encode_word(self, _words):

    #     return NicknameModel.transform_to_id(_words, self.vocab, self.parameters['sentence_maxlen'])

    def get_last_history(self):
        data = {}
        _path = get_multi_lingual_nickname_model_path() + '/last.history'
        try:
            with open(_path, 'r') as f:
                data = json.load(f)
        except Exception as err:
            print(err)
        
        return data

    def get_test_result(self):
        data = {}
        _path = get_multi_lingual_nickname_model_path() + '/test.rslt'
        try:
            with open(_path, 'r') as f:
                data = json.load(f)
        except Exception as err:
            print(err)
        
        return data

    # @staticmethod
    # def tokenize_sentence(s):
    #     def _preprocessing(s):
    #         # remove punctuation
    #         table = str.maketrans('', '', string.punctuation)
    #         s = s.translate(table)

    #         # to_lower
    #         s = s.lower()

    #         # split by digits
    #         s = ' '.join(re.split('(\d+)', s))

    #         # seperate each chinese characters
    #         s = re.sub(r'[\u4e00-\u9fa5\uf970-\ufa6d]', '\g<0> ', s)

    #         return s

    #     tokens = jieba.cut(_preprocessing(s))

    #     # transform all digits to special token
    #     tokens = ['[NUM]' if w.isdigit() else w for w in tokens]

    #     # remove space
    #     tokens = [w for w in tokens if w != ' ']

    #     return tokens

    # @staticmethod
    # def transform_to_id(tokens_list, vocab, sentence_maxlen):
        
    #     token_ids = np.zeros((sentence_maxlen,), dtype=np.int32)
    #     # token_ids = [0] * sentence_maxlen
    #     token_ids[0] = vocab['[CLS]']
    #     for i, token in enumerate(tokens_list[: sentence_maxlen - 1]): # -1 for [CLS]
    #         if token in vocab:
    #             token_ids[i + 1] = vocab[token]
    #         else:
    #             token_ids[i + 1] = vocab['[UNK]']

    #     return token_ids