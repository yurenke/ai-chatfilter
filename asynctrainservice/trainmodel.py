import os
import re
import json
from datetime import datetime
import time
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import TimeDistributed, Embedding, Multiply, Add
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, Lambda
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

import transformers
from transformers import DistilBertTokenizer, TFDistilBertModel
from opencc import OpenCC

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import string
import jieba
import logging

import psycopg2
import pandas as pd

from transformer.encoder import TransformerBlock
from transformer.embedding import TokenAndPositionEmbedding
# from transformer.optimization import WarmUp

from configparser import RawConfigParser

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHINESE_CHAT_MODEL_PATH = os.path.join(BASE_DIR, 'ai/_models/chinese_chat_model')
CHINESE_NICKNAME_MODEL_PATH = os.path.join(BASE_DIR, 'ai/_models/chinese_nickname_model')
CH_TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'ai/assets/chinese')
ENG_TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'ai/assets/english')
THAI_TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'ai/assets/thai')
VIET_TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'ai/assets/vietnamese')

config = RawConfigParser()
config.read(BASE_DIR+'/setting.ini')

class SaveHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath, origin, monitor='val_accuracy', **kwargs):
        super().__init__()
        self.filepath = filepath
        self.origin = origin
        self.monitor = monitor
        self.monitor_op = np.greater
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        self._save_history(epoch=epoch, logs=logs)

    def _save_history(self, epoch, logs):
        logs = logs or {}

        # logs = tf_utils.sync_to_numpy_or_python_type(logs)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                "Can save history only with %s available, "
                "skipping.",
                self.monitor,
            )
        else:
            if self.monitor_op(current, self.best):
                self.best = current

                with open(self.filepath, 'w+') as f:
                    _acc = self.best
                    _los = 0
                    _val_acc = self.best
                    _acc = int(_acc * 10000) / 10000
                    _los = int(_los * 10000) / 10000
                    _val_acc = int(_val_acc * 10000) / 10000

                    f.write(json.dumps({
                        'accuracy': _acc,
                        'loss': _los,
                        'validation': _val_acc,
                        'ontraining': True,
                        'ETA': 20,
                        'timestamp': datetime.now().isoformat(),
                        'origin': self.origin,
                    }, indent = 2))


class BaseModel():
    def __init__(self):
        self.parameters = {
            'epochs': 30,
            'sentence_maxlen': 20,
            'vf_emb_dim': 128,
            'token_emb_dim': 384,
            'patience': 3,
            'batch_size': 64,
            'proj_clip': 5,
            'lr': 1e-5,
            # 'lr_decay_steps': 2500,
            # 'lr_decay_rate': 0.96,
            # 'num_warmup_steps': 2500,
            'transformer_num_heads': 2,
            'n_transformer_layers': 6,
            'num_classes': 3, # 0, 1, 4
            'dropout_rate': 0.1
        }
        self.vocab = {}
        self.latest_origin=''
        self.data = None
        self.test_data = None
        self.model_path = CHINESE_CHAT_MODEL_PATH
        self.model = None
        self.load_vocab()
        self.vf_embeddings = {}
        self.cc = OpenCC('t2s')

    def load_vocab(self):
        with open(os.path.join(self.model_path, "transformer_vocab.txt"), encoding='UTF-8') as f:
            idx = 0
            for line in f:
                word = line.strip()
                self.vocab[word] = idx
                idx += 1
        logging.info('vocab loaded. {} words in total'.format(len(self.vocab)))

    def load_pretrained_embedding_weights(self):
        self.vf_embeddings = {}
        # load variation family-enhanced graph embedding
        with open(os.path.join(BASE_DIR, "ai/assets/chinese/graph_emb/vf_emb_all.txt"), encoding='UTF-8') as f:
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

        word_inputs = Input(shape=(sentence_maxlen,), name='word_indices', dtype='int32')

        vf_embedding_matrix = self.build_vf_embedding_matrix()

        token_embedding_layer = TokenAndPositionEmbedding(sentence_maxlen, vocab_size, token_emb_dim, name='token_position_encoding')

        vf_embedding_layer = Embedding(input_dim=vocab_size, output_dim=vf_emb_dim,
                                embeddings_initializer=tf.keras.initializers.Constant(vf_embedding_matrix),
                                    trainable=False, name='vf_encoding')

        token_emb = token_embedding_layer(word_inputs)
        vf_emb = vf_embedding_layer(word_inputs)

        transformer_inputs = Concatenate(name='token_vf_concat')([token_emb, vf_emb])

        for i in range(self.parameters['n_transformer_layers']):
            transformer_inputs = TransformerBlock(transformer_dim, num_heads, transformer_dim, name='transformer_block_{}'.format(i + 1))(transformer_inputs)

        cls_emb = Lambda(lambda x: x[:, 0, :], name='cls_emb')(transformer_inputs)

        outp = Dense(self.parameters['num_classes'], activation="softmax", name='final_output')(cls_emb)

        self.model = Model(inputs=word_inputs, outputs=outp)

        # learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        #         initial_learning_rate=self.parameters['lr'],
        #         decay_steps=self.parameters['lr_decay_steps'],
        #         decay_rate=self.parameters['lr_decay_rate'] )

        # lr_schedule = WarmUp(initial_learning_rate=self.parameters['lr'],
        #                       decay_schedule_fn=learning_rate_fn,
        #                       warmup_steps=self.parameters['num_warmup_steps'])

        # # optimizer = tf.keras.optimizers.Adam(learning_rate=self.parameters['lr'], amsgrad=True)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, amsgrad=True)

        # self.model.compile(
        #     optimizer=optimizer,
        #     loss='sparse_categorical_crossentropy',
        #     # metrics=['accuracy'],
        #     weighted_metrics=['accuracy']
        # )
        self.model.compile(Adam(learning_rate=self.parameters['lr']), loss='sparse_categorical_crossentropy', weighted_metrics=['accuracy'])
        
        self.model.summary()  

    def load_model(self, path):
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path, custom_objects={'TransformerBlock': TransformerBlock,
                                                        'TokenAndPositionEmbedding': TokenAndPositionEmbedding})
        else:
            self.build_model()

        return self.model

    def save(self, is_check = False, history = None, is_training= False, eta=0, origin='none'):
        
        if not is_check and self.model:
            self.model.save(os.path.join(self.model_path, 'model.h5'))
            print('Successful saved. ')

        if history:
            print('Saved History: ', history)
            
            with open(os.path.join(self.model_path, 'last.history'), 'w+') as f:
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
                with open(os.path.join(self.model_path, 'last.history'), 'r') as f:
                    _string = f.read()
                    if _string:
                        _json = json.loads(_string)
            except Exception as err:
                print(err)
            
            print('Save No History, last.history: ', _json)

            with open(os.path.join(self.model_path, 'last.history'), 'w+') as f:
                f.write(json.dumps({
                    'accuracy': _json.get('accuracy', 0),
                    'loss': _json.get('loss', 0),
                    'validation': _json.get('validation', 0),
                    'ontraining': is_training,
                    'ETA': eta if is_training else 0,
                    'timestamp': datetime.now().isoformat(),
                    'origin': origin,
                }, indent = 2))

    def get_train_data(self, table_name):
        if table_name:
            logging.info(f"getting training data from DB")
            conn = psycopg2.connect(database=config.get('DATABASE', 'DATABASE_NAME'), 
                    user=config.get('DATABASE', 'DATABASE_USER'), 
                    password=config.get('DATABASE', 'DATABASE_PASSWORD'), 
                    host=config.get('DATABASE', 'DATABASE_HOST'), 
                    port=config.get('DATABASE', 'DATABASE_PORT'))

            with conn.cursor() as cur: 
                ## sql query
                sql = f"""
                SELECT id,
                    origin,
                    text,
                    status
                FROM {table_name}
                ORDER BY id DESC
                LIMIT 800000
                """
                
                ## 執行sql語法
                cur.execute(sql)
                
                ## 取得欄位名稱
                name = [desc[0] for desc in cur.description]
                
                ## 取得資料
                df = pd.DataFrame(cur.fetchall(),columns=name)

            self.latest_origin = df['origin'].iloc[0]

            logging.info(f"total records: {len(df.index)}")
            logging.info(f"latest_origin: {self.latest_origin}")

            return df

        return None

    def get_train_batchs(self):
        BATCH_SIZE = self.parameters['batch_size']
        X, y= self.data.pop('token_ids'), self.data.pop('target')

        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2)
        # X_test, X_val, y_test, y_val, w_test, w_val = train_test_split(X_test, y_test, w_test, stratify=y_test, test_size=0.5)

        logging.info('number of train data: {}'.format(len(X_train)))
        logging.info('number of val data: {}'.format(len(X_val)))

        class_weights = [(i, _) for (i, _) in enumerate(class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train))]
        class_weight_dict = dict(class_weights)
        logging.info('class weights: {}'.format(class_weight_dict))

        AUTO = tf.data.experimental.AUTOTUNE

        train_data = tf.data.Dataset.from_tensor_slices((np.stack(X_train), y_train)).shuffle(buffer_size=1024, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(AUTO)
        val_data = tf.data.Dataset.from_tensor_slices((np.stack(X_val), y_val)).batch(BATCH_SIZE).prefetch(AUTO)

        return train_data, val_data, class_weight_dict

    def write_test_rslt(self, rslt=None):
        print('Saved test rslt: ', rslt)

        if not rslt:
            rslt = {'status': 'ongoing', 'acc': 0, 'length': 0, 'right': 0, 'acc_rv': 0}

        with open(os.path.join(self.model_path, 'test.rslt'), 'w+') as f:
            f.write(json.dumps(rslt, indent = 2))

    def get_test_data(self, table_name, origin):
        if table_name:
            logging.info(f"getting testing data from DB")
            conn = psycopg2.connect(database=config.get('DATABASE', 'DATABASE_NAME'), 
                    user=config.get('DATABASE', 'DATABASE_USER'), 
                    password=config.get('DATABASE', 'DATABASE_PASSWORD'), 
                    host=config.get('DATABASE', 'DATABASE_HOST'), 
                    port=config.get('DATABASE', 'DATABASE_PORT'))

            with conn.cursor() as cur: 
                ## sql query
                sql = f"""
                    SELECT
                        text,
                        status
                    FROM {table_name}
                    WHERE origin = '{origin}'
                    """
                
                ## 執行sql語法
                cur.execute(sql)
                
                ## 取得欄位名稱
                name = [desc[0] for desc in cur.description]
                
                ## 取得資料
                df = pd.DataFrame(cur.fetchall(),columns=name)

            logging.info(f"total test records: {len(df.index)}")
            return df

        return None

    def get_test_batchs(self):
        BATCH_SIZE = self.parameters['batch_size']
        X = self.test_data.pop('token_ids')
        test_dataset = tf.data.Dataset.from_tensor_slices(np.stack(X)).batch(BATCH_SIZE)

        return test_dataset

    @staticmethod
    def trim_only_general_and_chinese(string):
        _result = ''

        for uc in string:
            _code = ord(uc)

            if _code >= 0x4e00 and _code <= 0x9faf:
                # chinese
                _result += uc
                continue
            elif _code == 0x3000 or _code == 0x00a0:
                # full space to half space
                _code = 0x0020
            elif _code > 0xfee0 and _code < 0xffff:
                # full char to half
                _code -= 0xfee0

            if _code == 0x0020 or (_code >= 0x0030 and _code <= 0x0039) or (_code >= 0x0041 and _code <= 0x005a) or (_code >= 0x0061 and _code <= 0x007a):
                _result += chr(_code)
            # if _code >= 0x0020 and _code <= 0x007f:
            #     _result += chr(_code)
            
        return _result

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
        token_ids[0] = vocab['[CLS]']
        for i, token in enumerate(tokens_list[: sentence_maxlen - 1]): # -1 for [CLS]
            if token in vocab:
                token_ids[i + 1] = vocab[token]
            else:
                token_ids[i + 1] = vocab['[UNK]']

        return token_ids
    
def remove_characters_emoji(text):
    ch_emojis = [':)', '=)', ':(', ':v', '-_-', ':3', '<3', '@@', ':D', ':>', ':">', '=]', ':<', '^_^', '^^', ':-)', '><', '>.<', '~~', ':p', ':-p']
    for e in ch_emojis:
        text = text.replace(e, " ")
    text = re.sub('  +', ' ', text).strip()
    return text

def replace_url(text):
    URL_PATTERN = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
    text = re.sub(URL_PATTERN, ' URL ', text)
    text = re.sub('  +', ' ', text).strip()
    return text

def replace_repeated_character(text):
    def _replace_rep(m):
        c,cc = m.groups()
        return f'{c}'
    re_rep = re.compile(r'(\S)(\1{2,})')
    return re_rep.sub(_replace_rep, text)

def remove_special_character(text):
    text = re.sub(r'\d+', lambda m: " ", text)
    # text = re.sub(r'\b(\w+)\s+\1\b',' ', text) #remove duplicate number word
    text = re.sub("[~!@#$%^&*()_+{}“”|:\"<>?`´\-=[\]\;\\\/.,]", " ", text)
    text = re.sub('  +', ' ', text).strip()
    return text

def transform_full_char_to_half(text):
    _result = ''

    for uc in text:
        _code = ord(uc)
        if _code == 0x3000 or _code == 0x00a0:
            # full space to half space
            _code = 0x0020
        elif _code > 0xfee0 and _code < 0xffff:
            # full char to half
            _code -= 0xfee0

        _result += chr(_code)

    return _result
    
def preprocess_text(text):
    text = text.encode('utf-8', errors="ignore").decode('utf-8')
    text = text.strip().replace('\n', ' ').replace('\r', ' ')
    text = transform_full_char_to_half(text)
    text = re.sub(r'<d>[a-zA-Z]</d>', '', text)
    text = remove_characters_emoji(text)
    text = replace_url(text)
    text = replace_repeated_character(text)
    text = remove_special_character(text)
    return text
    

class ChineseChatModel(BaseModel):
    def __init__(self):
        self.parameters = {
            'epochs': 20,
            'sentence_maxlen': 40,
            'patience': 4,
            'batch_size': 64,
            'lr': 1e-5,
            'num_classes': 3, # 0, 1, 4
        }
        # self.vocab = {}
        self.latest_origin=''
        self.data = None
        self.test_data = None
        self.model = None
        # self.vf_embeddings = {}
        self.cc = OpenCC('t2s')
        self.model_path = CHINESE_CHAT_MODEL_PATH

        self.tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(self.model_path, 'tokenizer'), local_files_only=True)

        # self.load_vocab()
        # self.load_model(os.path.join(self.model_path, 'model.h5'))

    def load_model(self, path):
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path, custom_objects={'TFDistilBertModel': transformers.TFDistilBertModel})
        else:
            self.build_model()

        return self.model
    
    def build_model(self):
        max_len = self.parameters['sentence_maxlen']
        transformer_layer = (
            transformers.TFDistilBertModel
            .from_pretrained(os.path.join(self.model_path, 'pretrained'), local_files_only=True)
        )
    
        input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        sequence_output = transformer_layer(input_word_ids)[0]
        cls_token = sequence_output[:, 0, :]
        out = Dense(self.parameters['num_classes'], activation='softmax')(cls_token)
        
        self.model = Model(inputs=input_word_ids, outputs=out)
        self.model.compile(Adam(learning_rate=self.parameters['lr']), loss='sparse_categorical_crossentropy', weighted_metrics=['accuracy'])
        
        self.model.summary() 

    def fit_model(self, verbose=0, callback=None):
        
        self.build_model()
        
        # self.enforced_stop = False
        # chinese data
        logging.info('fetching chinese training data...')
        self.data = self.get_train_data(config.get('TRAINING_DATA_TABLE', 'CHAT'))
        self.data = self.data.dropna()
        
        self.data = self.transform_data(df=self.data)
        logging.info('total length of chinese data: {}'.format(len(self.data.index)))
        x_ch = self.regular_encode(self.data.text.values.tolist(), self.parameters['sentence_maxlen'])
        y_ch = self.data.target.values
        x_train, x_val, y_train, y_val = train_test_split(x_ch, y_ch, stratify=y_ch, test_size=0.2)
        del self.data

        # english data
        logging.info('reading eng training data...')
        eng_train_df = pd.read_csv(os.path.join(ENG_TRAIN_DATA_PATH, 'eng_train.csv'))
        eng_train_df = eng_train_df.dropna()
        eng_val_df = pd.read_csv(os.path.join(ENG_TRAIN_DATA_PATH, 'eng_val.csv'))
        eng_val_df = eng_val_df.dropna()
        logging.info('total length of eng training data: {}'.format(len(eng_train_df.index)))
        logging.info('total length of eng val data: {}'.format(len(eng_val_df.index)))
        eng_train_df['target'] = eng_train_df['status'].apply(lambda x: 0 if x == 0 else 2)
        eng_val_df['target'] = eng_val_df['status'].apply(lambda x: 0 if x == 0 else 2)

        x_eng_train = self.regular_encode(eng_train_df.text.values.tolist(), self.parameters['sentence_maxlen'])
        x_eng_val = self.regular_encode(eng_val_df.text.values.tolist(), self.parameters['sentence_maxlen'])
        y_eng_train = eng_train_df.target.values
        y_eng_val = eng_val_df.target.values

        del eng_train_df, eng_val_df

        # thai data
        logging.info('reading thai training data...')
        thai_train_df = pd.read_csv(os.path.join(THAI_TRAIN_DATA_PATH, 'thai_train.csv'))
        thai_train_df = thai_train_df.dropna()
        thai_val_df = pd.read_csv(os.path.join(THAI_TRAIN_DATA_PATH, 'thai_val.csv'))
        thai_val_df = thai_val_df.dropna()
        thai_train_df = thai_train_df[thai_train_df['text'].astype(bool)] # remove rows containing empty string
        thai_train_df = thai_train_df.drop_duplicates(subset=['text'])
        logging.info('total length of thai training data: {}'.format(len(thai_train_df.index)))

        thai_val_df = thai_val_df[thai_val_df['text'].astype(bool)] # remove rows containing empty string
        thai_val_df = thai_val_df.drop_duplicates(subset=['text'])
        logging.info('total length of thai val data: {}'.format(len(thai_val_df.index)))
        thai_train_df['target'] = thai_train_df['status'].apply(lambda x: 0 if x == 0 else 2)
        thai_val_df['target'] = thai_val_df['status'].apply(lambda x: 0 if x == 0 else 2)

        x_thai_train = self.regular_encode(thai_train_df.text.values.tolist(), self.parameters['sentence_maxlen'])
        x_thai_val = self.regular_encode(thai_val_df.text.values.tolist(), self.parameters['sentence_maxlen'])
        y_thai_train = thai_train_df.target.values
        y_thai_val = thai_val_df.target.values

        del thai_train_df, thai_val_df

        # vietnamese data
        logging.info('reading vietnamese training data...')
        vi_train_df = pd.read_csv(os.path.join(VIET_TRAIN_DATA_PATH, 'vi_train.csv'))
        vi_val_df = pd.read_csv(os.path.join(VIET_TRAIN_DATA_PATH, 'vi_val.csv'))
        vi_train_df = vi_train_df.dropna()
        vi_val_df = vi_val_df.dropna()
        vi_train_df = vi_train_df[vi_train_df['text'].astype(bool)] # remove rows containing empty string
        vi_train_df = vi_train_df.drop_duplicates(subset=['text'])
        logging.info('total length of viet training data: {}'.format(len(vi_train_df.index)))
        vi_val_df = vi_val_df[vi_val_df['text'].astype(bool)] # remove rows containing empty string
        vi_val_df = vi_val_df.drop_duplicates(subset=['text'])
        logging.info('total length of viet val data: {}'.format(len(vi_val_df.index)))

        vi_train_df['target'] = vi_train_df['status'].apply(lambda x: 0 if x == 0 else 2)
        vi_val_df['target'] = vi_val_df['status'].apply(lambda x: 0 if x == 0 else 2)

        x_vi_train = self.regular_encode(vi_train_df.text.values.tolist(), self.parameters['sentence_maxlen'])
        x_vi_val = self.regular_encode(vi_val_df.text.values.tolist(), self.parameters['sentence_maxlen'])
        y_vi_train = vi_train_df.target.values
        y_vi_val = vi_val_df.target.values

        del vi_train_df, vi_val_df

        x_train = np.concatenate((x_train, x_eng_train, x_thai_train, x_vi_train), axis=0)
        x_val = np.concatenate((x_val, x_eng_val, x_thai_val, x_vi_val), axis=0)
        y_train = np.concatenate((y_train, y_eng_train, y_thai_train, y_vi_train), axis=0)
        y_val = np.concatenate((y_val, y_eng_val, y_thai_val, y_vi_val), axis=0)

        BATCH_SIZE = self.parameters['batch_size']

        logging.info('number of train data: {}'.format(len(x_train)))
        logging.info('number of val data: {}'.format(len(x_val)))

        class_weights = [(i, _) for (i, _) in enumerate(class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train))]
        class_weight_dict = dict(class_weights)
        logging.info('class weights: {}'.format(class_weight_dict))

        AUTO = tf.data.experimental.AUTOTUNE

        train_dataset = (
            tf.data.Dataset
            .from_tensor_slices((x_train, y_train))
            .shuffle(buffer_size=2048, reshuffle_each_iteration=True)
            .batch(BATCH_SIZE)
            .prefetch(AUTO) 
        )
        

        val_dataset = (
            tf.data.Dataset
            .from_tensor_slices((x_val, y_val))
            .batch(BATCH_SIZE)
            .cache()
            .prefetch(AUTO)
        )

        history = None

        try:
            weights_file = os.path.join(self.model_path, "best_weights.h5")
            save_best_model = ModelCheckpoint(filepath=weights_file, monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='auto')
            early_stopping = EarlyStopping(patience=self.parameters['patience'], restore_best_weights=True)
            save_history_cb = SaveHistoryCallback(filepath=os.path.join(self.model_path, 'last.history'), origin=self.latest_origin)
            _callbacks = [save_best_model, early_stopping, save_history_cb]
            
            if callback:
                _callbacks.append(callback)

            self.save(is_check=True, history={'validation': 0.0}, is_training=True, eta=20, origin=self.latest_origin)

            # n_steps = x_train.shape[0] // BATCH_SIZE
            history = self.model.fit(
                        x=train_dataset,
                        class_weight=class_weight_dict,
                        epochs=self.parameters['epochs'],
                        verbose=verbose,
                        validation_data=val_dataset,
                        callbacks=_callbacks,
                    )

            self.load_model(weights_file)
            self.save(history=history.history, is_training=False, eta=0, origin=self.latest_origin)

        except Exception as err:
            logging.error('Exception on Fit chat model.')
            logging.error(err)
        
        return history

    def regular_encode(self, texts, maxlen=512):
        enc_di = self.tokenizer.batch_encode_plus(
            texts, 
            return_attention_mask=False, 
            return_token_type_ids=False,
            # pad_to_max_length=True,
            truncation=True,
            padding='max_length',
            max_length=maxlen
        )
    
        return np.array(enc_di['input_ids'])

    def transform_data(self, df):
        def _transform_status(x):
            return x if x < 2 else 2

        df = df[df.status != 3]
        df['text'] = df['text'].apply(lambda x: self.cc.convert(x))
        df['text'] = df['text'].apply(lambda x: preprocess_text(x))
        df = df[df['text'].astype(bool)] # remove rows containing empty string
        df = df.drop_duplicates(subset=['text'])
            
        df['target'] = df['status'].apply(lambda x: _transform_status(x))
        return df
        
        # return x, y
    
    def get_train_batchs(self, x, y):
        BATCH_SIZE = self.parameters['batch_size']
        # X, y= self.data.pop('token_ids'), self.data.pop('target')

        x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=0.2)
        # X_test, X_val, y_test, y_val, w_test, w_val = train_test_split(X_test, y_test, w_test, stratify=y_test, test_size=0.5)

        logging.info('number of train data: {}'.format(len(x_train)))
        logging.info('number of val data: {}'.format(len(x_val)))

        class_weights = [(i, _) for (i, _) in enumerate(class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train))]
        class_weight_dict = dict(class_weights)
        logging.info('class weights: {}'.format(class_weight_dict))

        AUTO = tf.data.experimental.AUTOTUNE

        train_dataset = (
            tf.data.Dataset
            .from_tensor_slices((x_train, y_train))
            .shuffle(buffer_size=2048, reshuffle_each_iteration=True)
            .batch(BATCH_SIZE)
            .prefetch(AUTO) 
        )
        

        val_dataset = (
            tf.data.Dataset
            .from_tensor_slices((x_val, y_val))
            .batch(BATCH_SIZE)
            .cache()
            .prefetch(AUTO)
        )

        return train_dataset, val_dataset, class_weight_dict
    
    def get_test_batchs(self, x):
        BATCH_SIZE = self.parameters['batch_size']
        test_dataset = tf.data.Dataset.from_tensor_slices(x).batch(BATCH_SIZE)

        return test_dataset

    def test_by_origin(self, origin=''):
        if origin:
            t_start = time.time()
            self.write_test_rslt()
            self.load_model(os.path.join(self.model_path, 'model.h5'))
            self.test_data = self.get_test_data(table_name=config.get('TRAINING_DATA_TABLE', 'CHAT'), origin=origin)
            self.test_data = self.test_data.dropna()
            self.test_data = self.transform_data(df=self.test_data)
            self.test_data_length = len(self.test_data.index)
            x = self.regular_encode(self.test_data.text.values.tolist(), self.parameters['sentence_maxlen'])
            # y = df.target.values
            test_dataset = self.get_test_batchs(x)

            _pred = self.model.predict(x=test_dataset, verbose=0)

            max_probs = np.max(_pred, axis=1)  # 每行的最大值
            predicted_classes = np.argmax(_pred, axis=1)  # 每行的最大值索引
            results = np.where(max_probs >= 0.8, predicted_classes, 0)

            self.test_data['prediction'] = results
            self.test_data['b_status'] = self.test_data.target > 0
            self.test_data['b_prediction'] = self.test_data.prediction > 0
            right = self.test_data.loc[self.test_data.b_status == self.test_data.b_prediction]['prediction'].count()
            del_right = self.test_data.loc[(self.test_data.b_prediction) & (self.test_data.b_status)]['prediction'].count()
            total_del = self.test_data.loc[self.test_data.b_prediction]['prediction'].count()

            result = {
                'status': 'done',
                'acc': int(right) / int(self.test_data_length),
                'length': int(self.test_data_length),
                'right': int(right),
                'acc_rv': int(del_right) / int(total_del)
            }
            print('Testing took {0} sec'.format(str(time.time() - t_start)))

            self.write_test_rslt(rslt=result)


class ChineseNicknameModel(BaseModel):
    def __init__(self):
        self.parameters = {
            'epochs': 20,
            'sentence_maxlen': 40,
            'patience': 4,
            'batch_size': 64,
            'lr': 1e-5,
            'num_classes': 7 # 0, 1, 4
        }
        # self.vocab = {}
        self.latest_origin=''
        self.data = None
        self.test_data = None
        self.model = None
        self.cc = OpenCC('t2s')
        self.model_path = CHINESE_NICKNAME_MODEL_PATH
        self.tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(CHINESE_CHAT_MODEL_PATH, 'tokenizer'), local_files_only=True)
        # self.load_model(os.path.join(self.model_path, 'model.h5'))

    # def load_vocab(self):
    #     with open(os.path.join(self.model_path, "nickname_filter_vocab.txt"), encoding='UTF-8') as f:
    #         idx = 0
    #         for line in f:
    #             word = line.strip()
    #             self.vocab[word] = idx
    #             idx += 1
    #     logging.info('vocab loaded. {} words in total'.format(len(self.vocab)))

    # def load_pretrained_embedding_weights(self):
    #     self.vf_embeddings = {}
    #     # load variation family-enhanced graph embedding
    #     with open(os.path.join(BASE_DIR, "ai/assets/chinese/graph_emb/vf_emb_common.txt"), encoding='UTF-8') as f:
    #         # skip header
    #         next(f)
    #         for line in f:
    #             word, coefs = line.split(maxsplit=1)
    #             coefs = np.fromstring(coefs, "f", sep=" ")
    #             self.vf_embeddings[word] = coefs
    def load_model(self, path):
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path, custom_objects={'TFDistilBertModel': transformers.TFDistilBertModel})
        else:
            self.build_model()

        return self.model
    
    def build_model(self):
        max_len = self.parameters['sentence_maxlen']
        transformer_layer = (
            transformers.TFDistilBertModel
            .from_pretrained(os.path.join(CHINESE_CHAT_MODEL_PATH, 'pretrained'), local_files_only=True)
        )
    
        input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        sequence_output = transformer_layer(input_word_ids)[0]
        cls_token = sequence_output[:, 0, :]
        out = Dense(self.parameters['num_classes'], activation='softmax')(cls_token)
        
        self.model = Model(inputs=input_word_ids, outputs=out)
        self.model.compile(Adam(learning_rate=self.parameters['lr']), loss='sparse_categorical_crossentropy', weighted_metrics=['accuracy'])
        
        self.model.summary() 

    def fit_model(self, verbose=0, callback=None):
        self.build_model()
        # chinese data
        logging.info('fetching chinese training data...')
        self.data = self.get_train_data(config.get('TRAINING_DATA_TABLE', 'NICKNAME'))
        self.data = self.data[['text', 'status']]
        self.data = self.data.dropna()

        self.data = self.transform_data(df=self.data)
        logging.info('total length of chinese data: {}'.format(len(self.data.index)))
        x_ch = self.regular_encode(self.data.text.values.tolist(), self.parameters['sentence_maxlen'])
        y_ch = self.data.status.values

        x_train, x_val, y_train, y_val = train_test_split(x_ch, y_ch, stratify=y_ch, test_size=0.2)

        del self.data

        # english data
        logging.info('reading eng training data...')
        eng_train_df = pd.read_csv(os.path.join(ENG_TRAIN_DATA_PATH, 'eng_train.csv'))
        eng_train_df = eng_train_df.dropna()
        eng_val_df = pd.read_csv(os.path.join(ENG_TRAIN_DATA_PATH, 'eng_val.csv'))
        eng_val_df = eng_val_df.dropna()
        logging.info('total length of eng training data: {}'.format(len(eng_train_df.index)))
        logging.info('total length of eng val data: {}'.format(len(eng_val_df.index)))
        x_eng_train = self.regular_encode(eng_train_df.text.values.tolist(), self.parameters['sentence_maxlen'])
        x_eng_val = self.regular_encode(eng_val_df.text.values.tolist(), self.parameters['sentence_maxlen'])
        y_eng_train = eng_train_df.status.values
        y_eng_val = eng_val_df.status.values

        del eng_train_df, eng_val_df

        # thai data
        logging.info('reading thai training data...')
        thai_train_df = pd.read_csv(os.path.join(THAI_TRAIN_DATA_PATH, 'thai_train.csv'))
        thai_val_df = pd.read_csv(os.path.join(THAI_TRAIN_DATA_PATH, 'thai_val.csv'))
        thai_train_df = thai_train_df.dropna()
        thai_val_df = thai_val_df.dropna()
        thai_train_df = thai_train_df[thai_train_df['text'].astype(bool)] # remove rows containing empty string
        thai_train_df = thai_train_df.drop_duplicates(subset=['text'])
        logging.info('total length of thai training data: {}'.format(len(thai_train_df.index)))
        thai_val_df = thai_val_df[thai_val_df['text'].astype(bool)] # remove rows containing empty string
        thai_val_df = thai_val_df.drop_duplicates(subset=['text'])
        logging.info('total length of thai val data: {}'.format(len(thai_val_df.index)))
        x_thai_train = self.regular_encode(thai_train_df.text.values.tolist(), self.parameters['sentence_maxlen'])
        x_thai_val = self.regular_encode(thai_val_df.text.values.tolist(), self.parameters['sentence_maxlen'])
        y_thai_train = thai_train_df.status.values
        y_thai_val = thai_val_df.status.values

        del thai_train_df, thai_val_df

        # Vietnamese data
        logging.info('reading vietnamese training data...')
        vi_train_df = pd.read_csv(os.path.join(VIET_TRAIN_DATA_PATH, 'vi_train.csv'))
        vi_val_df = pd.read_csv(os.path.join(VIET_TRAIN_DATA_PATH, 'vi_val.csv'))
        vi_train_df = vi_train_df.dropna()
        vi_val_df = vi_val_df.dropna()
        vi_train_df = vi_train_df[vi_train_df['text'].astype(bool)] # remove rows containing empty string
        vi_train_df = vi_train_df.drop_duplicates(subset=['text'])
        logging.info('total length of viet training data: {}'.format(len(vi_train_df.index)))
        vi_val_df = vi_val_df[vi_val_df['text'].astype(bool)] # remove rows containing empty string
        vi_val_df = vi_val_df.drop_duplicates(subset=['text'])
        logging.info('total length of viet val data: {}'.format(len(vi_val_df.index)))
        x_vi_train = self.regular_encode(vi_train_df.text.values.tolist(), self.parameters['sentence_maxlen'])
        x_vi_val = self.regular_encode(vi_val_df.text.values.tolist(), self.parameters['sentence_maxlen'])
        y_vi_train = vi_train_df.status.values
        y_vi_val = vi_val_df.status.values

        del vi_train_df, vi_val_df

        x_train = np.concatenate((x_train, x_eng_train, x_thai_train, x_vi_train), axis=0)
        x_val = np.concatenate((x_val, x_eng_val, x_thai_val, x_vi_val), axis=0)
        y_train = np.concatenate((y_train, y_eng_train, y_thai_train, y_vi_train), axis=0)
        y_val = np.concatenate((y_val, y_eng_val, y_thai_val, y_vi_val), axis=0)

        BATCH_SIZE = self.parameters['batch_size']

        logging.info('number of train data: {}'.format(len(x_train)))
        logging.info('number of val data: {}'.format(len(x_val)))

        class_weights = [(i, _) for (i, _) in enumerate(class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train))]
        class_weight_dict = dict(class_weights)
        logging.info('class weights: {}'.format(class_weight_dict))

        AUTO = tf.data.experimental.AUTOTUNE

        train_dataset = (
            tf.data.Dataset
            .from_tensor_slices((x_train, y_train))
            .shuffle(buffer_size=2048, reshuffle_each_iteration=True)
            .batch(BATCH_SIZE)
            .prefetch(AUTO) 
        )
        

        val_dataset = (
            tf.data.Dataset
            .from_tensor_slices((x_val, y_val))
            .batch(BATCH_SIZE)
            .cache()
            .prefetch(AUTO)
        )

        history = None

        try:
            weights_file = os.path.join(self.model_path, "nickname_best_weights.h5")
            save_best_model = ModelCheckpoint(filepath=weights_file, monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='auto')
            early_stopping = EarlyStopping(patience=self.parameters['patience'], restore_best_weights=True)
            save_history_cb = SaveHistoryCallback(filepath=os.path.join(self.model_path, 'last.history'), origin=self.latest_origin)
            _callbacks = [save_best_model, early_stopping, save_history_cb]
            
            if callback:
                _callbacks.append(callback)

            self.save(is_check=True, history={'validation': 0.0}, is_training=True, eta=20, origin=self.latest_origin)

            history = self.model.fit(
                        x=train_dataset,
                        class_weight=class_weight_dict,
                        epochs=self.parameters['epochs'],
                        verbose=verbose,
                        validation_data=val_dataset,
                        callbacks=_callbacks,
                    )

            self.load_model(weights_file)
            self.save(history=history.history, is_training=False, eta=0, origin=self.latest_origin)

        except Exception as err:
            logging.error('Exception on Fit nickname model.')
            logging.error(err)
        
        return history
    
    def regular_encode(self, texts, maxlen=512):
        enc_di = self.tokenizer.batch_encode_plus(
            texts, 
            return_attention_mask=False, 
            return_token_type_ids=False,
            # pad_to_max_length=True,
            truncation=True,
            padding='max_length',
            max_length=maxlen
        )
    
        return np.array(enc_di['input_ids'])

    def transform_data(self, df):
        df['text'] = df['text'].apply(lambda x: self.cc.convert(x))
        df['text'] = df['text'].apply(lambda x: preprocess_text(x))
        # df['text'] = df['text'].apply(lambda x: ChineseChatModel.trim_only_general_and_chinese(x).strip())
        df = df[df['text'].astype(bool)] # remove rows containing empty string
        df = df.drop_duplicates(subset=['text'])
        return df
    
    def get_test_batchs(self, x):
        BATCH_SIZE = self.parameters['batch_size']
        test_dataset = tf.data.Dataset.from_tensor_slices(x).batch(BATCH_SIZE)
        return test_dataset


    def test_by_origin(self, origin=''):
        if origin:
            t_start = time.time()
            self.write_test_rslt()
            self.load_model(os.path.join(self.model_path, 'model.h5'))
            self.test_data = self.get_test_data(table_name=config.get('TRAINING_DATA_TABLE', 'NICKNAME'), origin=origin)
            self.test_data = self.transform_data(df=self.test_data)
            self.test_data_length = len(self.test_data.index)
            x = self.regular_encode(self.test_data.text.values.tolist(), self.parameters['sentence_maxlen'])
            # y = df.target.values
            test_dataset = self.get_test_batchs(x)

            _pred = self.model.predict(x=test_dataset, verbose=0)
            _pred = np.argmax(_pred, axis=1)

            self.test_data['prediction'] = _pred
            self.test_data['b_status'] = self.test_data.status > 0
            self.test_data['b_prediction'] = self.test_data.prediction > 0
            right = self.test_data.loc[self.test_data.b_status == self.test_data.b_prediction]['prediction'].count()
            del_right = self.test_data.loc[(self.test_data.b_prediction) & (self.test_data.b_status)]['prediction'].count()
            total_del = self.test_data.loc[self.test_data.b_prediction]['prediction'].count()

            result = {
                'status': 'done',
                'acc': int(right) / int(self.test_data_length),
                'length': int(self.test_data_length),
                'right': int(right),
                'acc_rv': int(del_right) / int(total_del)
            }
            print('Testing took {0} sec'.format(str(time.time() - t_start)))

            self.write_test_rslt(rslt=result)
