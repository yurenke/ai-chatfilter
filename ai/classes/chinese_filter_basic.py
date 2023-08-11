from __future__ import absolute_import, division, print_function, unicode_literals
import ast

# from copy import deepcopy

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os, re
from ai.classes.basic_filter import BasicFilter
from ai.service_impact import get_all_vocabulary_from_models
from datetime import datetime, timedelta



class BasicChineseFilter(BasicFilter):
    """
    """

    full_words_length = 16
    full_vocab_size = 65536
    # full_vocab_size = 131072
    basic_num_dataset = 5000
    unknown_position = 1
    alphabet_position = 2
    numeric_position = 3
    tokenizer_vocabularies = []
    encoder = None
    encoder_size = 0
    alpha_pattern = re.compile('[A-Za-z]+')
    confirmed_rate = 0.8
    enforced_stop = False

    # override
    def __init__(self, data = [], load_folder=None, vocabulary=[], freqs=[]):
        super().__init__(data=data, load_folder=load_folder)
        if len(vocabulary) == 0:
            vocabulary_data = get_all_vocabulary_from_models(english=False, pinyin=False, chinese=True)
            _data = vocabulary_data['chinese']
            for _d in _data:
                if len(_d[0]) == 1:
                    vocabulary.append(_d[0])
                    freqs.append(_d[1])

        self.load_tokenizer_vocabularies(vocabulary)
        

    
    def load_tokenizer_vocabularies(self, vocabulary_dataset = []):
        _vocabularies = vocabulary_dataset
        vocabulary_length = len(_vocabularies)
        print('[BasicChineseFilter][Load_Tokenizer_Vocabularies] Vocabulary length: ', vocabulary_length)
        assert vocabulary_length > 0 and vocabulary_length < self.full_vocab_size
        self.tokenizer_vocabularies = _vocabularies
        try:
            self.encoder = tfds.features.text.TokenTextEncoder(_vocabularies)
        except:
            self.encoder = tfds.deprecated.text.TokenTextEncoder(_vocabularies)
        self.encoder_size = vocabulary_length


    # override
    def transform_str(self, _string):
        words = list(_string.replace(' ', ''))
        return words


    # override
    def build_model(self):
        full_words_length = self.full_words_length
        all_scs = self.num_status_classs

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(self.full_vocab_size, full_words_length, mask_zero=True))
        # model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(full_words_length, return_sequences=True)))
        model.add(tf.keras.layers.LSTM(full_words_length, return_sequences=True))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(tf.keras.layers.Dense(full_words_length, activation=tf.nn.relu))
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(all_scs, return_sequences=True)))
        # model.add(tf.keras.layers.Dense(full_words_length, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(all_scs, activation=tf.nn.softmax))

        model.summary()
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        self.model = model

        return self

    
    # override
    def fit_model(self, epochs=5, verbose=1, save_folder=None, train_data=None, validation_data=None, stop_accuracy=None, stop_hours=None, origin='', callback=None):
        self.enforced_stop = False
        if save_folder is not None:
            self.saved_folder = save_folder
        
        if train_data is not None:
            self.set_data(train_data)

        # return exit(2)
        batch_data = self.get_train_batchs(check_duplicate = False)

        _length_of_data = self.length_x

        BUFFER_SIZE = _length_of_data
        BATCH_SIZE = self.full_words_length
        VALIDATION_SIZE = int(_length_of_data / 8) if _length_of_data > 5000 else int(_length_of_data / 2)
        TRAIN_SIZE = _length_of_data - VALIDATION_SIZE

        batch_data = batch_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

        if validation_data is None:

            batch_train_data = batch_data.take(TRAIN_SIZE).repeat(epochs)
            # batch_train_data = batch_data.take(TRAIN_SIZE)
            batch_test_data = batch_data.skip(TRAIN_SIZE).take(VALIDATION_SIZE)

        else:
            print('Can Not Give Validation Data.')
            exit(2)

        history = None
        batch_train_data = batch_train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[],[]))
        batch_test_data = batch_test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[],[]))

        # print(batch_train_data)
        # for __ in batch_train_data.take(1):
        #     print(__)

        print('==== batch_train_data ====')
        print('BUFFER_SIZE :: ', BUFFER_SIZE)
        print('BATCH_SIZE :: ', BATCH_SIZE)
        print('TRAIN_SIZE :: ', TRAIN_SIZE)
        print('VALIDATION_SIZE :: ', VALIDATION_SIZE)

        steps = int(TRAIN_SIZE / BATCH_SIZE)
        vaildation_steps = int(VALIDATION_SIZE / BATCH_SIZE)
        # vaildation_steps = int(VALIDATION_SIZE / BATCH_SIZE / epochs)
        print('steps [{}]  val steps [{}]'.format(steps, vaildation_steps))

        try:
            _eta = stop_hours
            _start = datetime.now()
            _callbacks = [callback] if callback else []
            if stop_hours:
                _end = _start + timedelta(hours=stop_hours)
            while True: 
                history = self.model.fit(
                    batch_train_data,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=batch_test_data,
                    steps_per_epoch=steps,
                    validation_steps=vaildation_steps,
                    callbacks=_callbacks,
                )
                
                whether_continue = False if self.enforced_stop else True

                if stop_accuracy:
                    acc = max(history.history.get('accuracy'))
                    print('Now Accuracy: {:.4f} / Target Accuracy: {:.4f}'.format(acc, stop_accuracy))
                    if acc >= stop_accuracy:
                        whether_continue = False

                if stop_hours:
                    _now = datetime.now()
                    _eta = int((_end - _now).total_seconds() / 60 / 6 * 10) / 100
                    if _eta <= 0:
                        whether_continue = False
                        _eta = 0

                self.save(history=history.history, is_continue=whether_continue, eta=_eta, origin=origin)
                if whether_continue is False:
                    break
                
        except KeyboardInterrupt:
            print('Keyboard pressed. Stop Tranning.')
        except Exception as err:
            print('Exception on Fit model.')
            print(err)
        
        return history


    # override
    def get_train_batchs(self, check_duplicate= True):
        
        x, y, w = self.get_xyw_data(to_numpy=True)

        print('[get_train_batchs] x length: ', len(x))

        tokenized_list = self.tokenize_data(x)

        if check_duplicate:

            _i = 0
            _check_map = {}
            _check_map_idx = {}
            _all_duplicate_zipstr = []

            for _ in tokenized_list:
                if len(_)==0:
                    _i += 1
                    continue
                _zip_str = '|'.join(str(__) for __ in _)
                _map_value = _check_map.get(_zip_str, None)
                _y_value = 0 if y[_i] == 0 else 1
                # print(_i, ': ', [self.transform_back_str(xx) for xx in x[_i]], _)

                if _map_value:
                    if _map_value != _y_value:
                        if _zip_str not in _all_duplicate_zipstr:
                            _all_duplicate_zipstr.append(_zip_str)

                        _origin = self.data[_i][2]
                        _against_idx = _check_map_idx[_zip_str]
                        _against_data = self.data[_against_idx][2]
                        print('_origin: {}   _against_data: {}  _zip_str: {}'.format(_origin, _against_data, _zip_str))
                    
                else:
                    _check_map[_zip_str] = _y_value
                    _check_map_idx[_zip_str] = _i
                
                _i += 1

            if len(_all_duplicate_zipstr) > 0:
                print('[Error] Failed To Start Train Because Data is Confusion.')
                my_input = input('Do you wanna continue ? (y/n)')
                if my_input == 'y':
                    pass
                else:
                    exit(2)

        # _basic = int(self.basic_num_dataset / len(tokenized_list))

        # if _basic >= 1:
        #     tokenized_list = tokenized_list * (_basic+1)
        #     y = y * (_basic+1)
        #     w = w * (_basic+1)

        self.length_x = len(tokenized_list)

        labeled_dataset = self.bathchs_labeler(tokenized_list, y, w)

        return labeled_dataset


    def tokenize_data(self, datalist):
        print('Start Tokenize Data.')
        # unknowns = self.unknown_words
        _i = 0
        _total = len(datalist)
        tokenized = []
        
        for words in datalist:
            _i += 1
            if _i % 1000 == 0:
                _percent = _i / _total * 100
                print(" {:.2f}%".format(_percent), end="\r")

            _list, _has_unknown = self.get_encode_word(words)
            # if len(_list) ==0:
            #     continue
            _ary = np.asarray(_list).astype(np.int32)

            tokenized.append(_ary)
        
        print('Tokenize Done.')
        return np.asarray(tokenized)


    def get_encode_word(self, _words, ignore_english = True):
        _result_text = []
        _found_other_unknown = False

        for _ in _words:
            if ignore_english and self.alpha_pattern.fullmatch(_):
                continue
            _code = self.sub_encode(_)
            _result_text.append(_code)
            if _code == self.unknown_position:
                _found_other_unknown = True

        # if _found_other_unknown:
        #     print('_found_other_unknown: ', _words)
        
        return _result_text, _found_other_unknown

    
    def get_details(self, text):

        encoded_words, _has_unknown = self.get_encode_word(text)

        if encoded_words:
            predicted = self.model.predict([encoded_words])[0]
        else:
            predicted = []

        # print('encoded_words: ', encoded_words)
        
        return {
            'encoded_words': encoded_words,
            'predicted_ratios': ['{:2.2%}'.format(_) for _ in list(predicted)]
        }


    def sub_encode(self, word):
        _encoder = self.encoder
        _max_size = self.encoder_size
        _loc = _encoder.encode(word)
            
        if len(_loc) > 0:
            __code = _loc[0]

            if __code > _max_size:
                # find the new word
                if len(word) <= 2 and self.alpha_pattern.fullmatch(word):
                    return self.alphabet_position
                elif word.isnumeric():
                    return self.numeric_position
                else:
                    return self.unknown_position
                
            elif __code >= 0:

                return __code
        
        return self.unknown_position


    def bathchs_labeler(self, x, y, w):
        assert len(x) == len(y)
        # encoder = self.encoder
        full_words_length = self.full_words_length

        x_list = []
        y_list = []
        w_list = []

        for idx, texts in enumerate(x):
            _len = len(texts)
            if _len == 0:
                continue

            st = y[idx] if y[idx] else 0
            weight = w[idx] if w[idx] else 1
            npts = np.pad(texts[:full_words_length], (0, max(0, full_words_length - _len)), 'constant')
            
            x_list.append(npts)
            y_list.append(np.int64(st))
            w_list.append(np.int64(weight))

        self.length_x = len(x_list)

        print('x_list[-10:]: ', x_list[-10:])
        print('y_list[-10:]: ', y_list[-10:])
        print('w_list[-10:]: ', w_list[-10:])

        dataset = tf.data.Dataset.from_tensor_slices((x_list, y_list, w_list))
        
        return dataset
    

    # override
    def predictText(self, text, lv = 0, with_reason=False):
        possible = 0
        reason = ''
        
        _result_text, _has_unknown = self.get_encode_word(text, ignore_english=True)

        if len(_result_text) > 0 and lv < 5:

            predicted = self.model(np.array([_result_text]))[0]
            # print('[BasicChineseFilter] predicted: ', predicted)

            possible = np.argmax(predicted)

            if possible > 0:
                ratio = predicted[possible]
                if ratio < (self.confirmed_rate + lv*0.04):
                    possible = 0
            
        return possible, reason


    def set_stop(self):
        self.enforced_stop = True
        return self.enforced_stop