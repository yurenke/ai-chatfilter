import os
import re
import json
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

from ai.helper import get_chinese_chat_model_path

# from dataparser.apps import MessageParser

# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import TimeDistributed, Embedding, Multiply, Add
# from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, Lambda
# from tensorflow.keras.constraints import MinMaxNorm
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import string
import jieba
import pandas as pd

# from ai import GRAPH_DIR
from ai.classes.basic_filter import BasicFilter
from datetime import datetime, timedelta

from ai.helper import get_chinese_chat_model_path

# from .transformer.encoder import TransformerBlock
# from .transformer.embedding import TokenAndPositionEmbedding
# from .transformer.optimization import WarmUp

class ChineseChatFilter(BasicFilter):
    parameters = {
        'lr': 1e-5,
        # 'sentence_maxlen': 20,
        'sentence_maxlen': 40,
        'num_classes': 3
    }
    enforced_stop = False

    def __init__(self, load_folder=None):
        # self.cc = OpenCC('t2s')
        self.load_tokenizer()
        # self.parser = MessageParser()
        super().__init__(load_folder=load_folder)

    def load_tokenizer(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(get_chinese_chat_model_path(), 'tokenizer'), local_files_only=True)
        print('chinese chat model tokenzier loaded')

    # def load_vocab(self):
    #     with open(os.path.join(get_chinese_chat_model_path(), "transformer_vocab.txt"), encoding='UTF-8') as f:
    #         idx = 0
    #         for line in f:
    #             word = line.strip()
    #             self.vocab[word] = idx
    #             idx += 1
    #     print('vocab loaded. {} words in total'.format(len(self.vocab)))  

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

    def save(self, folder = None, is_check = False, history = None, is_continue= False, eta=0, origin='none'):
        if folder is not None:
            self.saved_folder = folder
        elif self.saved_folder:
            folder = self.saved_folder
        else:
            print('Error Save, because folder is not specify')
            return None
        
        if not os.path.isdir(folder) or not os.path.exists(folder):
            os.makedirs(folder)
        
        if not is_check:
            self.save_model(folder + '/model.h5')
        
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
                    'ontraining': is_continue,
                    'ETA': eta if is_continue else 0,
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
                    'ontraining': is_continue,
                    'ETA': eta if is_continue else 0,
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

    def predictText(self, text, lv = 0, with_reason=False):
        possible = 0
        reason = ''
        
        _result_text = self.get_encode_word(text)

        x = tf.expand_dims(tf.convert_to_tensor(_result_text), axis=0)
        predicted = self.model(x, training=False)[0]

        possible = np.argmax(predicted)
        max_prob = predicted[possible]
        # possible = 1 if predicted >= 0.5 else 0
        if max_prob < 0.8:
            possible = 0

        possible = possible if possible < 2 else 4
                
        return possible, reason

    def get_encode_word(self, text):
        
        # text = self.cc.convert(text)
        enc_di = self.tokenizer(text, return_attention_mask=False, 
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            max_length=self.parameters['sentence_maxlen'])

        return np.array(enc_di['input_ids'])