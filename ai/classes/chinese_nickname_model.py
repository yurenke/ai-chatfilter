from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import json, logging
import numpy as np

import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

import transformers
from transformers import DistilBertTokenizer
# from opencc import OpenCC

from ai.classes.basic_filter import BasicFilter
from datetime import datetime, timedelta

import string
import pandas as pd

from ai.helper import get_chinese_nickname_model_path, get_chinese_chat_model_path
# from dataparser.apps import MessageParser

class ChineseNicknameModel():
    parameters = {
        'lr': 1e-5,
        # 'sentence_maxlen': 10
        'sentence_maxlen': 40,
        'num_classes': 7
    }

    def __init__(self):
        self.vocab = {}
        self.load_vocab()
        self.load_tokenizer()
        # self.vf_embeddings = {}
        # self.cc = OpenCC('t2s')
        self.model_path = get_chinese_nickname_model_path()
        # self.parser = MessageParser()
        self.load()
        # super().__init__(load_folder=load_folder)

    def load_vocab(self):
        with open(os.path.join(get_chinese_nickname_model_path(), "nickname_filter_vocab.txt"), encoding='UTF-8') as f:
            idx = 0
            for line in f:
                word = line.strip()
                self.vocab[word] = idx
                idx += 1
        print('nickname filter vocab loaded. {} words in total'.format(len(self.vocab)))

    def load_tokenizer(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(get_chinese_chat_model_path(), 'tokenizer'), local_files_only=True)
        print('chinese nickname model tokenzier loaded')

    def build_model(self):
        max_len = self.parameters['sentence_maxlen']
        transformer_layer = (
            transformers.TFDistilBertModel
            .from_pretrained(os.path.join(get_chinese_chat_model_path(), 'pretrained'), local_files_only=True)
        )
    
        input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        sequence_output = transformer_layer(input_word_ids)[0]
        cls_token = sequence_output[:, 0, :]
        out = Dense(self.parameters['num_classes'], activation='softmax')(cls_token)
        
        self.model = Model(inputs=input_word_ids, outputs=out)
        self.model.compile(Adam(learning_rate=self.parameters['lr']), loss='sparse_categorical_crossentropy', weighted_metrics=['accuracy'])
        
        self.model.summary() 

    def load_model(self, path):
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path, custom_objects={'TFDistilBertModel': transformers.TFDistilBertModel})
        else:
            self.build_model()

        return self.model

    def load(self):
        model_path = self.get_model_path()
        
        logging.info('Starting load nickname model: {}'.format(model_path))
        
        self.load_model(model_path)
        
        logging.info('Successful load nickname model. ')

    def get_model_path(self):
        _path = self.model_path + '/model.h5'
        if not os.path.exists(_path):
            _path = self.model_path + '/model.remote.h5'
        return _path


    def save(self, is_check = False, history = None, is_training= False, eta=0, origin='none'):
        
        folder = get_chinese_nickname_model_path()
        
        if not is_check:
            self.model.save(folder + '/model.h5')
        
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
        encoded_words = self.get_encode_word(text)

        x = tf.expand_dims(tf.convert_to_tensor(encoded_words), axis=0)
        predicted = self.model(x, training=False)[0]
        
        return {
            'predicted_ratios': ['{:2.2%}'.format(_) for _ in list(predicted)]
        }

    def predict(self, text):
        possible = 0
        
        _result_text = self.get_encode_word(text)

        x = tf.expand_dims(tf.convert_to_tensor(_result_text), axis=0)
        predicted = self.model(x, training=False)[0]

        possible = np.argmax(predicted)
                
        return possible

    # def transform(self, data):

    #     text = self.cc.convert(data)
    #     return ChineseNicknameModel.tokenize_sentence(text)

    def get_encode_word(self, text):

        enc_di = self.tokenizer(text, return_attention_mask=False, 
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            max_length=self.parameters['sentence_maxlen'])

        return np.array(enc_di['input_ids'])

    def get_last_history(self):
        data = {}
        _path = get_chinese_nickname_model_path() + '/last.history'
        try:
            with open(_path, 'r') as f:
                data = json.load(f)
        except Exception as err:
            print(err)
        
        return data

    def get_test_result(self):
        data = {}
        _path = get_chinese_nickname_model_path() + '/test.rslt'
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