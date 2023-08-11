from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os, json



class BasicFilter():
    """
    """

    columns = ['ROOM', 'WEIGHT', 'TEXT', 'STATUS']
    appended_columns = ['TRANSFORMED_WORD']
    data = []
    data_length = 0
    model = None

    saved_folder = None
    
    full_words_length = 64
    num_status_classs = 8
    length_x = 0

    
    def __init__(self, data = [], load_folder=None):
        
        if len(data) > 0:

            self.set_data(data)

        elif load_folder:
            
            self.load(load_folder)


    def check_data_shape(self, data=[]):

        print('check_data_shape data length: ', len(data))

        if len(data) == 0:
            data = self.data

        _len_should = len(self.columns)
        _isright = len(data[0]) == _len_should

        if _isright:
            _idx_status = self.columns.index('STATUS')
            _status_map = {}
            for _ in data:
                _s = _[_idx_status]
                if _status_map.get(_s):
                    _status_map[_s] += 1
                else:
                    _status_map[_s] = 1
            print('Status Map: ', _status_map)
        else:
            print('Dataset length wrong of checking function. should be {} , But {}'.format(_len_should, data[0]))

        return _isright
    

    def set_data(self, data):
        if self.check_data_shape(data):

            self.data = data
            self.data_length = len(data)
            self.transform_column('TEXT')

        else:
            
            raise Exception('Set data failed.')

        return self


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
                _acc = history.get('accuracy', [0])[-1]
                _los = history.get('loss', [0])[-1]
                _val_acc = history.get('val_accuracy', [0])[-1]
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



    def load(self, folder):
        if folder:
            self.saved_folder = folder

        model_path = self.get_model_path()
        
        print('Starting load model: ', model_path)
        
        self.load_model(model_path)
        
        print('Successful load model. ')
        

    def save_model(self, path):
        return self.model.save(path)


    def load_model(self, path):
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path)
        else:
            self.build_model()

        return self.model



    def transform(self, data):
        
        if type(data) is str:
            return self.transform_str(data)
        elif type(data) is list:
            return [self.transform(_) for _ in data]
        
        return None



    def transform_column(self, column = 'TEXT'):
        assert len(self.data) > 0

        _full_columns = self.columns + self.appended_columns

        if type(column) is int:
            column_idx = column
        elif type(column) is str:
            column_idx = _full_columns.index(column) if column in _full_columns else -1

        assert column_idx >= 0 and column_idx < len(_full_columns)

        _transformed_idx = _full_columns.index('TRANSFORMED_WORD')
        _length_of_columns = len(_full_columns)
        _length_of_data = len(self.data)
        _i = 0

        print('Start Transform Data..')

        for d in self.data:
            _text = d[column_idx]
            _transformed_words = self.transform(_text)

            if len(d) == _length_of_columns:
                d[_transformed_idx] = _transformed_words
            else:
                d.append(_transformed_words)

            _i += 1
            if _i % 500 == 0:
                print(' {:.2f}%'.format(_i / _length_of_data * 100), end='\r')

        print('Transform Data Done.')

        return self


    # should be override
    def transform_str(self, _string):
        return _string


    # could be override
    def build_model(self):

        full_words_length = self.full_words_length
        all_scs = self.num_status_classs

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(full_words_length, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(all_scs, activation=tf.nn.softmax))
        model.summary()
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, amsgrad=True)

        model.compile(
            optimizer=optimizer,
            # loss='categorical_crossentropy',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        self.model = model

        return self


    # could be overreide
    def fit_model(self, epochs=1, verbose=1, save_folder=None, train_data=None, validation_data=None, stop_accuracy=None, stop_hours=None):
        if save_folder is not None:
            self.saved_folder = save_folder
        
        if train_data is not None:
            self.set_data(train_data)


        batch_train_data = self.get_train_batchs()

        _length_of_data = self.length_x

        BUFFER_SIZE = _length_of_data + 1
        BATCH_SIZE = self.full_words_length
        VALIDATION_SIZE = int(_length_of_data / 8) if _length_of_data > 5000 else int(_length_of_data / 3)

        # exit(2)

        if validation_data is None:

            batch_train_data = batch_train_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
            batch_test_data = batch_train_data.take(VALIDATION_SIZE)

        history = None
        batch_train_data = batch_train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[])).repeat(epochs)
        batch_test_data = batch_test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

        print('==== batch_train_data ====')
        print('Length of Data :: ', _length_of_data)
        print('BUFFER_SIZE :: ', BUFFER_SIZE)
        print('BATCH_SIZE :: ', BATCH_SIZE)
        print('VALIDATION_SIZE :: ', VALIDATION_SIZE)

        steps = int(_length_of_data / BATCH_SIZE)
        vaildation_steps = int(VALIDATION_SIZE / BATCH_SIZE)

        try:
            while True:
                history = self.model.fit(
                    batch_train_data,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=batch_test_data,
                    steps_per_epoch=steps,
                    validation_steps=vaildation_steps,
                )
                self.save(history = history.history)

                acc = history.history.get('accuracy')[-1]
                
                if stop_accuracy:
                    print('Now Accuracy: {:.4f} / Target Accuracy: {:.4f}'.format(acc, stop_accuracy))
                    if acc >= stop_accuracy:
                        break
                
        except KeyboardInterrupt:
            print('Keyboard pressed. Stop Tranning.')
        
        return history



    def get_xy_data(self):
        print('Starting get XY data..', end='\r')
        _full_columns = self.columns + self.appended_columns
        # x_idx = self.columns.index('TEXT') if 'TEXT' in self.columns else -1
        x_idx = _full_columns.index('TRANSFORMED_WORD') if 'TRANSFORMED_WORD' in _full_columns else -1
        y_idx = _full_columns.index('STATUS') if 'STATUS' in _full_columns else -1
        new_x = []
        new_y = []
        __auto_human_delete_if_not = 3

        data_length = self.data_length
        _i = 0
        _has_not_word_value = False

        for _d in self.data:
            _t = _d[x_idx]
            if _t:
                _status = _d[y_idx]
                if _status:
                    _status = int(_status)
                elif not isinstance(_status, int):
                    continue
                
                new_x.append(_t)
                new_y.append(_status)
            else:
                _has_not_word_value = True
                print('[get_xy_data] Not Found: ', _d)

            if _i % 1000 == 0:
                _percent = _i / data_length
                print("Getting XY data processing [{:2.1%}]".format(_percent), end="\r")

            _i += 1
        
        print("Getting XY data sets is done. Total count: ", len(new_x))
        if _has_not_word_value:
            exit(2)
        return new_x, new_y


    def get_xyw_data(self, to_numpy=False):
        print('Starting get XYW data..')
        _full_columns = self.columns + self.appended_columns
        # x_idx = self.columns.index('TEXT') if 'TEXT' in self.columns else -1
        x_idx = _full_columns.index('TRANSFORMED_WORD') if 'TRANSFORMED_WORD' in _full_columns else -1
        y_idx = _full_columns.index('STATUS') if 'STATUS' in _full_columns else -1
        w_idx = _full_columns.index('WEIGHT') if 'WEIGHT' in _full_columns else -1
        assert w_idx >= 0 and y_idx >= 0 and x_idx >= 0
        new_x = []
        new_y = []
        new_w = []
        data_length = self.data_length
        _i = 0

        for _d in self.data:
            _t = _d[x_idx]
            if _t:
                _status = int(_d[y_idx])
                _weight = _d[w_idx]
                if _status >= 0:
                    new_x.append(_t)
                    new_y.append(_status)
                    new_w.append(_weight)
            else:
                print('[get_xyw_data] Not Found data: ', _d)
                continue

            if _i % 1000 == 0:
                _percent = _i / data_length
                print("Getting XYW data processing [{:2.1%}]".format(_percent), end="\r")

            _i += 1
        
        print("Getting XYW data sets is done. Total count: ", len(new_x))
        if to_numpy:
            return np.asarray(new_x), np.asarray(new_y).astype(np.int32), np.asarray(new_w).astype(np.float64)
        return new_x, new_y, new_w


    # could be override
    def get_train_batchs(self):
        
        x, y, w = self.get_xyw_data(to_numpy=True)
        length_x = len(x)
        assert length_x > 0
        self.length_x = length_x
        dataset = tf.data.Dataset.from_tensor_slices((x, y, w))

        return dataset


    def predictText(self, text, lv=0, with_reason=False):
        predicted = self.model.predict([text])[0]
        reason = ''
        passible = np.argmax(predicted)
        return passible, reason


    
    def get_details(self, text):
        transformed_words = self.transform(text)
        encoded_words, _has_unknown = self.get_encode_word(transformed_words, ignore_english=False)

        if encoded_words:
            predicted = self.model.predict([encoded_words])[0]
        else:
            predicted = []

        # print('encoded_words: ', encoded_words)
        
        return {
            'encoded_words': encoded_words,
            'predicted_ratios': ['{:2.2%}'.format(_) for _ in list(predicted)],
            'transformed_words': transformed_words
        }


    def get_saved_folder(self):
        return self.saved_folder


    def get_model_path(self):
        _path = self.get_saved_folder() + '/model.h5'
        if not os.path.exists(_path):
            _path = self.get_saved_folder() + '/model.remote.h5'
        return _path


    def get_last_history(self):
        data = {}
        _path = self.get_saved_folder() + '/last.history'
        try:
            with open(_path, 'r') as f:
                data = json.load(f)
        except Exception as err:
            print(err)
        
        return data

    def get_test_result(self):
        data = {}
        _path = self.get_saved_folder() + '/test.rslt'
        try:
            with open(_path, 'r') as f:
                data = json.load(f)
        except Exception as err:
            print(err)
        
        return data

