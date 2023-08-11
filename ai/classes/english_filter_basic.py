from __future__ import absolute_import, division, print_function, unicode_literals

from .chinese_filter_basic import BasicChineseFilter
from ai.models import Vocabulary, Language
from ai.service_impact import get_all_vocabulary_from_models

import tensorflow as tf
import tensorflow_datasets as tfds
import os, re
import numpy as np
from datetime import datetime, timedelta

CHARACTER_UNKNOWN = '#UNK#'
CHARACTER_PAD = '#PAD#'
CHARACTER_NUM = '#NUM#'
CHARACTER_ALPHABET = '#ALP#'
CHARACTER_RESERVE = '#RES#'


class BasicEnglishFilter(BasicChineseFilter):
    """
    """

    # re_is_chinese = re.compile('[\u4e00-\u9fa5]')
    # re_is_english = re.compile('[a-zA-Z]')
    # re_is_number = re.compile('[0-9]')
    # re_is_other = re.compile('[\u0020-\u0085]')

    status_classsets = 8
    full_words_length = 255

    basic_num_dataset = 5000
    full_vocab_size = 65536

    origin_vocabulary = [CHARACTER_PAD, CHARACTER_UNKNOWN, CHARACTER_NUM, CHARACTER_ALPHABET, CHARACTER_RESERVE]
    alphabet_position = 0
    unknown_position = 0
    eng_vocabulary = []
    map_first_eng_voca = {}
    encoder = None
    encoder_size = 0

    def __init__(self, data = [], load_folder=None, english_vocabulary=[]):

        if english_vocabulary:
            self.eng_vocabulary = english_vocabulary
        else:
            _voca_data = get_all_vocabulary_from_models(pinyin=False, chinese=False)
            self.eng_vocabulary = [_[0] for _ in _voca_data['english']]

        # print('[BasicEnglishFilter] eng_vocabulary: ', self.eng_vocabulary)

        for _eng in self.eng_vocabulary:
            _first_char = _eng[0]
            if self.map_first_eng_voca.get(_first_char):
                self.map_first_eng_voca[_first_char].append(_eng)
            else:
                self.map_first_eng_voca[_first_char] = [_eng]

        _full_vocabulary = self.get_vocabulary()

        
        try:
            self.encoder = tfds.features.text.TokenTextEncoder(_full_vocabulary)
        except:
            self.encoder = tfds.deprecated.text.TokenTextEncoder(_full_vocabulary)
        
        self.encoder_size = len(_full_vocabulary)

        self.alphabet_position = self.origin_vocabulary.index(CHARACTER_ALPHABET)
        self.unknown_position = self.origin_vocabulary.index(CHARACTER_UNKNOWN)
        
        super().__init__(data=data, load_folder=load_folder)
    

    # override return list
    def transform_str(self, _string):
        return re.split(r'[\s]+', _string)

    
    # override
    def build_model(self):
        full_words_length = self.full_words_length
        all_scs = self.status_classsets

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(self.full_vocab_size, all_scs, mask_zero=True))
        # model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(all_scs)))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(tf.keras.layers.Dense(full_words_length, activation=tf.nn.relu))
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(all_scs, return_sequences=True)))
        # model.add(tf.keras.layers.Dense(full_words_length, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(all_scs, activation=tf.nn.softmax))

        model.summary()
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, amsgrad=True)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        self.model = model

        return self


    # override
    def fit_model(self, epochs=3, verbose=1, save_folder=None, train_data=None, validation_data=None, stop_accuracy=None, stop_hours=None):
        if save_folder is not None:
            self.saved_folder = save_folder
        
        if train_data is not None:
            self.set_data(train_data)


        batch_train_data = self.get_train_batchs()

        _length_of_data = self.length_x
        
        BATCH_SIZE = 32
        BUFFER_SIZE = _length_of_data + 1
        VALIDATION_SIZE = int(_length_of_data / 8) if _length_of_data > 5000 else int(_length_of_data / 2)

        # print("batch_train_data: ", batch_train_data)
        # for _ in batch_train_data.take(2):
        #     print(_)
        # exit(2)

        if validation_data is None:

            batch_train_data = batch_train_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).repeat(epochs)
            batch_test_data = batch_train_data.take(VALIDATION_SIZE)


        history = None
        batch_train_data = batch_train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))
        batch_test_data = batch_test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

        # for _ in batch_train_data:                  
        #     print("2 _: ", _)

        # exit(2)

        print('==== batch_train_data ====')
        print('Length of Data :: ', _length_of_data)
        print('BATCH_SIZE :: ', BATCH_SIZE)
        print('BUFFER_SIZE :: ', BUFFER_SIZE)
        print('VALIDATION_SIZE :: ', VALIDATION_SIZE)

        steps = int(_length_of_data / BATCH_SIZE)
        vaildation_steps = int(VALIDATION_SIZE / BATCH_SIZE)

        try:
            _start = datetime.now()
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
                )
                self.save()

                acc = history.history.get('accuracy')[-1]
                
                if stop_accuracy:
                    print('Now Accuracy: {:.4f} / Target Accuracy: {:.4f}'.format(acc, stop_accuracy))
                    if acc >= stop_accuracy:
                        break
                
                if stop_hours:
                    _now = datetime.now()
                    if _now > _end:
                        break
                
        except KeyboardInterrupt:
            print('Keyboard pressed. Stop Tranning.')
        except Exception as err:
            print('Exception on Fit model.')
            print('Exception Message: {}'.format(err))
        
        return history

    

    def get_train_batchs(self, check_duplicate= True):
        
        x, y = self.get_xy_data()

        _parsed_x_list = [ self.get_encode_word(_)[0] for _ in x ]

        # print('_parsed_x_list: ', _parsed_x_list)
        # exit(2)

        if check_duplicate:

            _i = 0
            _check_map = {}
            _check_map_idx = {}
            _all_duplicate_zipstr = []

            for _ in _parsed_x_list:
                _zip_str = '|'.join(str(__) for __ in _)
                _map_value = _check_map.get(_zip_str, None)
                _y_value = 0 if y[_i] == 0 else 1
                # print(_i, ': ', [self.transform_back_str(xx) for xx in x[_i]], _)

                if _map_value:
                    if _map_value != _y_value:
                        if _zip_str not in _all_duplicate_zipstr:
                            _all_duplicate_zipstr.append(_zip_str)

                        _origin = self.data[_i][2]
                        _transformed = x[_i]
                        _against_idx = _check_map_idx[_zip_str]
                        _against = self.data[_against_idx][2]
                        print('[English Filter][get_train_batchs] Duplicate Data: {} | Idx: {} | against: {} | {}'.format(_origin, _i, _against_idx, _against))
                        
                    
                else:
                    _check_map[_zip_str] = _y_value
                    _check_map_idx[_zip_str] = _i
                
                _i += 1

            if len(_all_duplicate_zipstr) > 0:
                print('[Error] Failed To Start Train Because Data is Confusion.')
                exit(2)
        
        _basic = int(self.basic_num_dataset / len(x))

        if _basic >= 1:
            _parsed_x_list = _parsed_x_list * (_basic+1)
            y = y * (_basic+1)

        self.length_x = len(_parsed_x_list)

        return self.bathchs_labeler(x=_parsed_x_list, y=y)


    def bathchs_labeler(self, x, y):
        assert len(x) == len(y)

        full_words_length = self.full_words_length

        def gen():
            for idx, texts in enumerate(x):
                _len = len(texts)
                if _len == 0:
                    continue

                st = y[idx] if y[idx] else 0
                npts = np.pad(texts, (0, full_words_length - _len), 'constant')

                yield npts, np.int64(st)
                # yield npts, [0,0,0,0,0,0,0,0]
        
        dataset = tf.data.Dataset.from_generator(
            gen,
            ( tf.int64, tf.int64 ),
            ( tf.TensorShape([full_words_length]), tf.TensorShape([]) ),
        )

        return dataset


    
    def get_encode_word(self, _words):
        _result_text = []
        _encoder = self.encoder
        _max_size = self.encoder_size
        _found_other_unknown = False

        for _ in _words:
            _loc = _encoder.encode(_)
            
            if len(_loc) > 0:
                __code = _loc[0]

                if __code > _max_size:
                    # find the new word
                    if len(_) <= 2:

                        _result_text.append(self.alphabet_position)
                    
                    else:

                        _similar_words = self.serach_similar_word(_)
                        # print('_similar_words: ', _similar_words)

                        if _similar_words:

                            _s_word = _similar_words[0]
                            _loc = _encoder.encode(_s_word)
                            __code = _loc[0]
                            _result_text.append(__code)

                        else:
                            
                            _found_other_unknown = True
                            _result_text.append(self.unknown_position)
                        
                    
                elif __code >= 0:
                    _result_text.append(__code)
        
        return _result_text, _found_other_unknown


    def serach_similar_word(self, text):
        possibility_words = []
        _suffix = ['ies', 'es', 's', 'ive', 'ing', 'ed', 'en']
        _test_vars = ['y', 'e']
        _map = self.map_first_eng_voca
        for _sf in _suffix:
            if text.endswith(_sf):
                _first_char = text[0]
                _parsed_text = text[:-(len(_sf))]
                # print('_parsed_text: ', _parsed_text)
                _list_of_same_first = _map.get(_first_char)
                if _list_of_same_first:

                    if _parsed_text in _list_of_same_first:
                        possibility_words.append(_parsed_text)
                        break
                    
                    _test_good = False
                    for _tv in _test_vars:
                        __test_word = '{}{}'.format(_parsed_text, _tv)
                        if __test_word in _list_of_same_first:
                            possibility_words.append(__test_word)
                            _test_good = True
                            break
                    
                    if _test_good:
                        break

                    else:
                        _length_parsed_text = len(_parsed_text)
                        _length_parsed_text_end = _length_parsed_text + 1
                        _found_matched = False
                        for _word in _list_of_same_first:
                            _length_w = len(_word)
                            if _length_w >= _length_parsed_text and _length_w <= _length_parsed_text_end:
                                _i = 0
                                _right = 0
                                _lose_list = []
                                for _single_w_char in _word:
                                    _text_position_char = _parsed_text[_i]
                                    if _single_w_char == _text_position_char:
                                        _right += 1
                                    else:
                                        if _lose_list and _lose_list[0] == _text_position_char:
                                            _right += 1
                                            _lose_list.pop(0)
                                        
                                        _lose_list.append(_single_w_char)
                                        if len(_lose_list) > 1:
                                            break
                                    _i += 1
                                    if _i >= _length_parsed_text:
                                        break
                                
                                if _right >= _length_parsed_text - 1:
                                    _found_matched = True
                                    _parsed_text = _word
                                    break
                        
                        if _found_matched:
                            possibility_words.append(_parsed_text)

        return possibility_words
    

    # override
    def get_details(self, text):

        _words = self.transform(text)

        if len(_words) == 0:
            return 0

        _texts = self.get_encode_word(_words)[0]
        
        predicted = self.model.predict(np.array([_texts]))[0]

        # print('grammar predicted: ', predicted)

        return {
            'words': _words,
            'encoded': _texts,
            'predicted_ratios': ['{:2.2%}'.format(_) for _ in list(predicted)],
        }
    

    # override
    def predictText(self, text, lv=0, with_reason=False):
        reason = ''
        passible = 0
        
        if lv < self.avoid_lv:

            _words = self.transform(text)

            if len(_words) == 0:
                return 0

            _words = self.get_encode_word(_words)[0]
            
            predicted = self.model.predict(np.array([_words]))[0]
            passible = np.argmax(predicted)

        if passible > 0 and with_reason:
            reason = 'Deleted by English filter.'

        return passible, reason


    def get_vocabulary(self, pure = False):
        if pure:
            return self.eng_vocabulary
        else:
            return self.origin_vocabulary + self.eng_vocabulary