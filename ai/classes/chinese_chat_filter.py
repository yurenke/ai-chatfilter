import os
import re
import json
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

from ai.helper import get_chinese_chat_model_path

from dataparser.apps import MessageParser

from .transformer.encoder import TransformerBlock
from .transformer.embedding import TokenAndPositionEmbedding
from .transformer.optimization import WarmUp

class ChineseChatFilter(BasicFilter):
    parameters = {
        'sentence_maxlen': 20,
        'vf_emb_dim': 128,
        'token_emb_dim': 384,
        'patience': 5,
        'batch_size': 64,
        'proj_clip': 5,
        'lr': 0.0001,
        'lr_decay_steps': 2500,
        'lr_decay_rate': 0.96,
        'num_warmup_steps': 2500,
        'transformer_num_heads': 2,
        'n_transformer_layers': 6,
        'num_classes': 3, # 0, 1, 4
        'dropout_rate': 0.1
    }
    enforced_stop = False

    def __init__(self, load_folder=None):
        self.vocab = {}
        self.load_vocab()
        self.vf_embeddings = {}
        self.cc = OpenCC('t2s')
        self.parser = MessageParser()
        super().__init__(load_folder=load_folder)

    def load_vocab(self):
        with open(os.path.join(get_chinese_chat_model_path(), "transformer_vocab.txt"), encoding='UTF-8') as f:
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
            self.data = self.data[self.data['TEXT'].astype(bool)] # remove rows containing empty string
            self.data = self.data.drop_duplicates(subset=['TEXT'])
            self.data_length = len(self.data.index)
            print('dataset size: '.format(self.data_length))
            
            self.data['TARGET'] = self.data['STATUS'].apply(lambda x: _transform_status(x))
            self.data['TOKENIZED'] = self.data['TEXT'].apply(lambda x: ChineseChatFilter.tokenize_sentence(x))
            self.data['TOKEN_IDS'] = self.data['TOKENIZED'].apply(lambda x: ChineseChatFilter.transform_to_id(x, self.vocab, self.parameters['sentence_maxlen']))
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

    def load_pretrained_embedding_weights(self):
        self.vf_embeddings = {}
        # load variation family-enhanced graph embedding
        with open(os.path.join(GRAPH_DIR, "vf_emb_all.txt"), encoding='UTF-8') as f:
            # skip header
            next(f)
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                self.vf_embeddings[word] = coefs

    def build_vf_embedding_matrix(self):

        self.load_pretrained_embedding_weights()
        # build pretrained embedding matrix
        vf_emb_dim = self.parameters['vf_emb_dim']
        vf_embedding_matrix = np.zeros((len(self.vocab), vf_emb_dim))

        for word, idx in self.vocab.items():
            if word in self.vf_embeddings:
                vf_vector = self.vf_embeddings.get(word)
                vf_embedding_matrix[idx] = vf_vector

        return vf_embedding_matrix    

    def build_model(self):
        token_emb_dim = self.parameters['token_emb_dim']
        vf_emb_dim = self.parameters['vf_emb_dim']
        num_heads = self.parameters['transformer_num_heads']
        transformer_dim = token_emb_dim + vf_emb_dim
        sentence_maxlen = self.parameters['sentence_maxlen']
        vocab_size = len(self.vocab)
        # dropout_rate = self.parameters['dropout_rate']
        # proj_dim = self.parameters['proj_dim']

        word_inputs = Input(shape=(sentence_maxlen,), name='word_indices', dtype='int32')

        vf_embedding_matrix = self.build_vf_embedding_matrix()

        # token_embeddings = Embedding(input_dim=self.vocab_size, output_dim=embed_dim, 
        #                         embeddings_initializer=tf.keras.initializers.GlorotNormal(),
        #                             trainable=True, name='token_encoding', mask_zero=True)
        token_embedding_layer = TokenAndPositionEmbedding(sentence_maxlen, vocab_size, token_emb_dim, name='token_position_encoding')

        vf_embedding_layer = Embedding(input_dim=vocab_size, output_dim=vf_emb_dim,
                                embeddings_initializer=tf.keras.initializers.Constant(vf_embedding_matrix),
                                    trainable=False, name='vf_encoding')

        # pos_embeddings = Embedding(input_dim=self.sentence_maxlen, output_dim=embed_dim)

        token_emb = token_embedding_layer(word_inputs)
        vf_emb = vf_embedding_layer(word_inputs)

        # position encoding
        # positions = tf.range(start=0, limit=self.sentence_maxlen, delta=1)
        # positions = pos_embeddings(positions)

        # input_mask = vf_embedding_layer.compute_mask(word_inputs)

        # combine token_emb and vf_emb
        transformer_inputs = Concatenate(name='token_vf_concat')([token_emb, vf_emb])
        # preference_dense = Dense(embed_dim, activation='sigmoid',
        #                         kernel_constraint=MinMaxNorm(-1 * self.parameters['proj_clip'],
        #                             self.parameters['proj_clip']))

        # token_weights = TimeDistributed(preference_dense, name='preference_token')(concatenated_emb, mask=input_mask)
        # vf_weights = Lambda(lambda x: 1-x, name='preference_vf')(token_weights)

        # weighted_token_emb = Multiply(name='weighted_token_emb')([token_weights, token_emb])
        # weighted_vf_emb = Multiply(name='weighted_vf_emb')([vf_weights, vf_emb])

        # transformer_inputs = Add(name='combined_emb')([weighted_token_emb, weighted_vf_emb])

        for i in range(self.parameters['n_transformer_layers']):
            transformer_inputs = TransformerBlock(transformer_dim, num_heads, transformer_dim, name='transformer_block_{}'.format(i + 1))(transformer_inputs)

        cls_emb = Lambda(lambda x: x[:, 0, :], name='cls_emb')(transformer_inputs)

        # x = Dense(proj_dim, activation="relu", name='cls_emb_proj')(cls_emb)
        # x = Dropout(dropout_rate, name='cls_emb_dropout')(cls_emb)
        # x = Dense(proj_dim, activation="relu", name='cls_emb_proj')(x)
        # x = Dropout(dropout_rate, name='cls_emb_dropout')(x)

        outp = Dense(self.parameters['num_classes'], activation="softmax", name='final_output')(cls_emb)

        self.model = Model(inputs=word_inputs, outputs=outp)

        learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.parameters['lr'],
                decay_steps=self.parameters['lr_decay_steps'],
                decay_rate=self.parameters['lr_decay_rate'] )

        lr_schedule = WarmUp(initial_learning_rate=self.parameters['lr'],
                              decay_schedule_fn=learning_rate_fn,
                              warmup_steps=self.parameters['num_warmup_steps'])

        # optimizer = tf.keras.optimizers.Adam(learning_rate=self.parameters['lr'], amsgrad=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, amsgrad=True)

        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            # metrics=['accuracy'],
            weighted_metrics=['accuracy']
        )
        
        self.model.summary()


    def load_model(self, path):
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path, custom_objects={'TransformerBlock': TransformerBlock,
                                                        'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                                        'WarmUp': WarmUp})
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
        return ChineseChatFilter.tokenize_sentence(text)

    def get_encode_word(self, _words):

        return ChineseChatFilter.transform_to_id(_words, self.vocab, self.parameters['sentence_maxlen'])

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
        token_ids[0] = vocab['[CLS]']
        for i, token in enumerate(tokens_list[: sentence_maxlen - 1]): # -1 for [CLS]
            if token in vocab:
                token_ids[i + 1] = vocab[token]
            else:
                token_ids[i + 1] = vocab['[UNK]']

        return token_ids