import os
import re
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dense, SpatialDropout1D, Reshape, Flatten, Concatenate, Dropout, Lambda
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from opencc import OpenCC

import string
import jieba
import pandas as pd

from ai.classes.basic_filter import BasicFilter
# from ai.service_impact import get_all_vocabulary_from_models
from datetime import datetime, timedelta

from .elmo.custom_layers import SampledSoftmax
from .custom_layers import ScalarMix
from ai.helper import get_ss_path

from dataparser.apps import MessageParser

class SSFilter(BasicFilter):
    parameters = {
        'sentence_maxlen': 20,
        'patience': 3,
        'batch_size': 64,
        'proj_clip': 5,
        'lr': 0.001,
        'n_lstm_layers': 2,
        'cnn_filter_sizes': [3, 4, 5],
        'num_cnn_filters': 128,
        'num_classes': 3, # 0, 1, 4
        'lstm_units_size': 128,
        'embedding_dim': 128,
        'dropout_rate': 0.1
    }
    enforced_stop = False

    def __init__(self, load_folder=None):
        self.vocab = {}
        self.load_vocab()
        self.cc = OpenCC('t2s')
        self.parser = MessageParser()
        super().__init__(load_folder=load_folder)

    def load_vocab(self):
        with open(os.path.join(get_ss_path(), "ss_vocab.txt"), encoding='UTF-8') as f:
            idx = 0
            for line in f:
                word = line.strip()
                self.vocab[word] = idx
                idx += 1
        print('vocab loaded. {} words in total'.format(len(self.vocab)))

    def set_data(self, data):
        def _transform_status(x):
            return x if x < 2 else 2
        
        if self.check_data_shape(data):

            self.data = pd.DataFrame(data, columns =self.columns)
            self.data = self.data[self.data.STATUS != 3]
            self.data['TEXT'] = self.data['TEXT'].apply(lambda x: self.cc.convert(x))
            self.data['TEXT'] = self.data['TEXT'].apply(lambda x: self.parser.trim_only_general_and_chinese(x).strip())
            self.data = self.data.drop_duplicates(subset=['TEXT'], keep='last')
            self.data_length = len(self.data.index)
            print('dataset size: '.format(self.data_length))
            
            self.data['TARGET'] = self.data['STATUS'].apply(lambda x: _transform_status(x))
            self.data['TOKENIZED'] = self.data['TEXT'].apply(lambda x: SSFilter.tokenize_sentence(x))
            self.data['TOKEN_IDS'] = self.data['TOKENIZED'].apply(lambda x: SSFilter.transform_to_id(x, self.vocab, self.parameters['sentence_maxlen']))
            self.data['WEIGHT'] = self.data['WEIGHT'].astype(float)

        else:
            
            raise Exception('Set data failed.')

    def get_train_batchs(self):
        BATCH_SIZE = self.parameters['batch_size']
        X, y= self.data.pop('TOKEN_IDS'), self.data.pop('TARGET')

        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2)
        # X_test, X_val, y_test, y_val, w_test, w_val = train_test_split(X_test, y_test, w_test, stratify=y_test, test_size=0.5)

        print('number of train data: {}'.format(len(X_train)))
        print('number of val data: {}'.format(len(X_val)))

        class_weights = [(i, _) for (i, _) in enumerate(class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train))]
        class_weight_dict = dict(class_weights)
        print('class weights: {}'.format(class_weight_dict))

        train_data = tf.data.Dataset.from_tensor_slices((np.stack(X_train), y_train)).shuffle(buffer_size=1024, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(2)
        val_data = tf.data.Dataset.from_tensor_slices((np.stack(X_val), y_val)).batch(BATCH_SIZE).prefetch(2)

        return train_data, val_data, class_weight_dict
        

    def build_model(self):
        maxlen = self.parameters['sentence_maxlen']
        filter_sizes = self.parameters['cnn_filter_sizes']
        num_filters = self.parameters['num_cnn_filters']
        embed_size = self.parameters['embedding_dim']
        dropout_rate = self.parameters['dropout_rate']

        elmo_model = load_model(os.path.join(get_ss_path(), 'elmo/ELMo_LM.h5'), custom_objects={'SampledSoftmax': SampledSoftmax})

        inputs = elmo_model.get_layer('word_indices').input
        input_mask = elmo_model.get_layer('w2v_encoding').compute_mask(inputs)

        # combine 3 kinds of embedding
        elmo_output = list()
        w2v_vf_concate = Concatenate(name='w2v_vf_combined')([elmo_model.get_layer('combined_inputs').output, elmo_model.get_layer('combined_inputs').output])
        elmo_output.append(w2v_vf_concate)

        for i in range(self.parameters['n_lstm_layers']):
            elmo_output.append(elmo_model.get_layer('bi_lstm_block_{}'.format(i + 1)).output)

        # gamma * (sum of weight_i * elmo_embedding_i)
        ss_embedding = ScalarMix(name='ss_embedding')(elmo_output)
        ss_proj = TimeDistributed(Dense(embed_size, activation='linear',
                                         kernel_constraint=MinMaxNorm(-1 * self.parameters['proj_clip'],
                                                                      self.parameters['proj_clip'])
                                         ), name='ss_proj')(ss_embedding, mask=input_mask)

        # drop_inputs = SpatialDropout1D(dropout_rate, name='ss_spatial_dropout')(ss_proj)
        # cnn_inputs = Reshape((maxlen, embed_size, 1))(drop_inputs)
        cnn_inputs = Reshape((maxlen, embed_size, 1))(ss_proj)

        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='glorot_normal',
                                                                                    activation=tf.keras.layers.LeakyReLU(alpha=0.1), name='conv_0')(cnn_inputs)
                                                                                    # activation='elu', name='conv_0')(cnn_inputs)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='glorot_normal',
                                                                                    activation=tf.keras.layers.LeakyReLU(alpha=0.1), name='conv_1')(cnn_inputs)
                                                                                    # activation='elu', name='conv_1')(cnn_inputs)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='glorot_normal',
                                                                                    activation=tf.keras.layers.LeakyReLU(alpha=0.1), name='conv_2')(cnn_inputs)
                                                                                    # activation='elu', name='conv_2')(cnn_inputs)

        maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1), name='maxpool_0')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1), name='maxpool_1')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1), name='maxpool_2')(conv_2)

        z = Concatenate(axis=1, name='all_maxpooling')([maxpool_0, maxpool_1, maxpool_2])   
        z = Flatten()(z)
        z = Dropout(dropout_rate, name='cnn_dropout')(z)

        outp = Dense(self.parameters['num_classes'], activation="softmax", name='final_output')(z)

        self.model = Model(inputs=inputs, outputs=outp)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.parameters['lr'],
                decay_steps=6000,
                decay_rate=0.9)

        # optimizer = tf.keras.optimizers.Adam(learning_rate=self.parameters['lr'], amsgrad=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, amsgrad=True)

        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            # metrics=['accuracy'],
            weighted_metrics=['accuracy']
        )
        
        self.model.summary()

    # override
    def fit_model(self, epochs=10, verbose=1, save_folder=None, train_data=None, stop_hours=None, origin='', callback=None):
        self.enforced_stop = False

        if save_folder is not None:
            self.saved_folder = save_folder
        
        if train_data is not None:
            self.set_data(train_data)

        train_dataset, val_dataset, cls_weights = self.get_train_batchs()

        history = None
        best_acc = 0.0
        _round = 0
        _is_better = True

        try:
            weights_file = os.path.join(get_ss_path(), "ss_best_weights.h5")
            save_best_model = ModelCheckpoint(filepath=weights_file, monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='auto')
            early_stopping = EarlyStopping(patience=self.parameters['patience'], restore_best_weights=True)
            _callbacks = [save_best_model, early_stopping]
            
            if callback:
                _callbacks.append(callback)

            _eta = stop_hours
            _start = datetime.now()
            
            if stop_hours:
                _end = _start + timedelta(hours=stop_hours)

            while True:
                history = self.model.fit(
                        x=train_dataset,
                        class_weight=cls_weights,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=val_dataset,
                        callbacks=_callbacks,
                    )

                current_best_acc = max(history.history.get('val_accuracy'))
                _round += 1

                print('round_{}: current_best_acc: {:.4f}, best_acc: {:.4f}'.format(_round, current_best_acc, best_acc))

                # first round
                if _round == 1:
                    best_acc = current_best_acc

                else:
                    if current_best_acc < best_acc:
                        _is_better = False
                    else:
                        _is_better = True
                        best_acc = current_best_acc
                    

                if self.enforced_stop:
                    whether_continue = False
                elif not _is_better:
                    whether_continue = False
                else:
                    whether_continue = True

                if stop_hours:
                    _now = datetime.now()
                    _eta = int((_end - _now).total_seconds() / 60 / 6 * 10) / 100
                    if _eta <= 0:
                        whether_continue = False
                        _eta = 0
                
                print('whether continue: {}'.format(whether_continue))

                if _is_better:
                    self.load_model(weights_file)
                    self.save(history=history.history, is_continue=whether_continue, eta=_eta, origin=origin)
                else:
                    self.load() # restore
                    self.save(is_check=True, is_continue=whether_continue, eta=_eta, origin=origin)

                if not whether_continue:
                    break

        except KeyboardInterrupt:
            print('Keyboard pressed. Stop Tranning.')
        except Exception as err:
            print('Exception on Fit model.')
            print(err)
        
        return history

    def load_model(self, path):
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path, custom_objects={'ScalarMix': ScalarMix})
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
        transformed_words = self.transform(text)
        encoded_words = self.get_encode_word(transformed_words)

        x = tf.expand_dims(tf.convert_to_tensor(encoded_words), 0)
        predicted = self.model(x)[0]
        
        return {
            'transformed_words': transformed_words,
            'encoded_words': encoded_words.tolist(),
            'predicted_ratios': ['{:2.2%}'.format(_) for _ in list(predicted)]
        }

    def predictText(self, text, lv = 0, with_reason=False):
        possible = 0
        reason = ''

        _words = self.transform(text)
        if len(_words) == 0:
            return possible, reason
        
        _result_text = self.get_encode_word(_words)

        x = tf.expand_dims(tf.convert_to_tensor(_result_text), 0)
        predicted = self.model(x)[0]

        possible = np.argmax(predicted)

        possible = possible if possible < 2 else 4
                
        return possible, reason

    def set_stop(self):
        self.enforced_stop = True
        return self.enforced_stop

    def transform(self, data):

        text = self.cc.convert(data)
        return SSFilter.tokenize_sentence(text)

    def get_encode_word(self, _words):

        return SSFilter.transform_to_id(_words, self.vocab, self.parameters['sentence_maxlen'])

    @staticmethod
    def tokenize_sentence(s):
        def _preprocessing(s):
            # remove punctuation
            table = str.maketrans('', '', string.punctuation)
            s = s.translate(table)

            # to_lower
            s = s.lower()

            # split by digits
            s = ' '.join(re.split('(\d+)', s))

            # seperate each chinese characters
            s = re.sub(r'[\u4e00-\u9fa5\uf970-\ufa6d]', '\g<0> ', s)

            return s

        tokens = jieba.cut(_preprocessing(s))

        # transform all digits to special token
        tokens = ['[NUM]' if w.isdigit() else w for w in tokens]

        # remove space
        tokens = [w for w in tokens if w != ' ']

        return tokens

    @staticmethod
    def transform_to_id(tokens_list, vocab, sentence_maxlen):
        
        token_ids = np.zeros((sentence_maxlen,), dtype=np.int32)
        # token_ids = [0] * sentence_maxlen
        token_ids[0] = vocab['[BOS]']
        for i, token in enumerate(tokens_list[: sentence_maxlen - 2]): # -2 for [BOS], [EOS]
            if token in vocab:
                token_ids[i + 1] = vocab[token]
            else:
                token_ids[i + 1] = vocab['[UNK]']

        if token_ids[1]:
            token_ids[i + 2] = vocab['[EOS]']

        return token_ids