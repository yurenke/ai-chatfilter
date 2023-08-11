import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Lambda, Embedding
from tensorflow.keras.layers import Add, Concatenate, Multiply
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.utils import to_categorical

from gensim.models import Word2Vec

from ai import MODELS_DIR, GRAPH_DIR
from ai.helper import get_ss_path
from .custom_layers import SampledSoftmax


class ELMo(object):
    def __init__(self, parameters):
        self._model = None
        self._elmo_model = None
        self.parameters = parameters
        self.vf_embeddings = {}
        self.vocab = {}
        self.vocab_size = 0
        self.load_vocab()
        self.w2v_model = None
        self.sentence_maxlen = self.parameters['sentence_maxlen']
        self.load_pretrained_embedding_weights()
        # self.compile_elmo()

    def __del__(self):
        K.clear_session()
        del self._model

    def load_vocab(self):
        with open(os.path.join(get_ss_path(), "ss_vocab.txt"), encoding='UTF-8') as f:
            idx = 0
            for line in f:
                word = line.strip()
                self.vocab[word] = idx
                idx += 1
            self.vocab_size = len(self.vocab)
        print('vocab loaded. {} words in total'.format(self.vocab_size))

    def load_pretrained_embedding_weights(self):
        # load w2v model
        self.w2v_model = Word2Vec.load(os.path.join(MODELS_DIR, "ss_model/w2v/w2v.model"))
        
        # load variation family-enhanced graph embedding
        with open(os.path.join(GRAPH_DIR, "vf_emb_all.txt"), encoding='UTF-8') as f:
            # skip header
            next(f)
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                self.vf_embeddings[word] = coefs

    def build_embedding_matrices(self):
        embedding_dim = self.parameters['embedding_dim']

        # build pretrained embedding matrix
        w2v_embedding_matrix = np.zeros((self.vocab_size, embedding_dim))
        vf_embedding_matrix = np.zeros((self.vocab_size, embedding_dim))

        for word, idx in self.vocab.items():
            
            if word in self.w2v_model.wv.key_to_index:
                w2v_vector = self.w2v_model.wv.get_vector(word, norm=True)
                w2v_embedding_matrix[idx] = w2v_vector

            if word in self.vf_embeddings:
                vf_vector = self.vf_embeddings.get(word)
                vf_embedding_matrix[idx] = vf_vector

        return w2v_embedding_matrix, vf_embedding_matrix

    def compile_elmo(self, print_summary=False):
        """
        Compiles a Language Model RNN based on the given parameters
        """

        # load pretrained embedding weights
        word_inputs = Input(shape=(self.sentence_maxlen,), name='word_indices', dtype='int32')

        w2v_embedding_matrix, vf_embedding_matrix = self.build_embedding_matrices()

        w2v_embeddings = Embedding(self.vocab_size, self.parameters['embedding_dim'], 
                                embeddings_initializer=tf.keras.initializers.Constant(w2v_embedding_matrix),
                                    trainable=True, name='w2v_encoding', mask_zero=True)
        vf_embeddings = Embedding(self.vocab_size, self.parameters['embedding_dim'],
                                embeddings_initializer=tf.keras.initializers.Constant(vf_embedding_matrix),
                                    trainable=False, name='vf_encoding')

        w2v_inputs = w2v_embeddings(word_inputs)
        vf_inputs = vf_embeddings(word_inputs)

        input_mask = w2v_embeddings.compute_mask(word_inputs)

        # combine sg_inputs and vf_inputs
        concatenated_inputs = Concatenate()([w2v_inputs, vf_inputs])
        preference_dense = Dense(self.parameters['embedding_dim'], activation='sigmoid',
                                kernel_constraint=MinMaxNorm(-1 * self.parameters['proj_clip'],
                                    self.parameters['proj_clip']))

        w2v_weights = TimeDistributed(preference_dense, name='preference_w2v')(concatenated_inputs, mask=input_mask)
        vf_weights = Lambda(lambda x: 1-x, name='preference_vf')(w2v_weights)

        weighted_w2v_inputs = Multiply(name='weighted_w2v_inputs')([w2v_weights, w2v_inputs])
        weighted_vf_inputs = Multiply(name='weighted_vf_inputs')([vf_weights, vf_inputs])

        lstm_inputs = Add(name='combined_inputs')([weighted_w2v_inputs, weighted_vf_inputs])

        # Token embeddings for Input
        # drop_inputs = SpatialDropout1D(self.parameters['dropout_rate'])(inputs)
        # lstm_inputs = TimestepDropout(self.parameters['word_dropout_rate'])(drop_inputs)

        # Pass outputs as inputs to apply sampled softmax
        next_ids = Input(shape=(self.sentence_maxlen, 1), name='next_ids', dtype='float32')
        previous_ids = Input(shape=(self.sentence_maxlen, 1), name='previous_ids', dtype='float32')

        for i in range(self.parameters['n_lstm_layers']):
            forward_lstm = LSTM(units=self.parameters['lstm_units_size'], return_sequences=True,
                                    kernel_constraint=MinMaxNorm(-1 * self.parameters['cell_clip'],
                                                    self.parameters['cell_clip']),
                                    recurrent_constraint=MinMaxNorm(-1 * self.parameters['cell_clip'],
                                                    self.parameters['cell_clip']))
            backward_lstm = LSTM(units=self.parameters['lstm_units_size'], return_sequences=True, go_backwards=True,
                                    kernel_constraint=MinMaxNorm(-1 * self.parameters['cell_clip'],
                                                    self.parameters['cell_clip']),
                                    recurrent_constraint=MinMaxNorm(-1 * self.parameters['cell_clip'],
                                                    self.parameters['cell_clip']))
            lstm_inputs = Bidirectional(forward_lstm, backward_layer=backward_lstm, name='bi_lstm_block_{}'.format(i + 1))(lstm_inputs, mask=input_mask)

        # Project to Vocabulary with Sampled Softmax
        sampled_softmax = SampledSoftmax(num_classes=self.parameters['vocab_size'],
                                         num_sampled=int(self.parameters['num_sampled']))
        f_lstm_output = lstm_inputs[:, :, :self.parameters['lstm_units_size']]
        b_lstm_output = lstm_inputs[:, :, self.parameters['lstm_units_size']:]

        outputs = sampled_softmax([f_lstm_output, next_ids])
        re_outputs = sampled_softmax([b_lstm_output, previous_ids])

        self._model = Model(inputs=[word_inputs, next_ids, previous_ids],
                            outputs=[outputs, re_outputs])

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.parameters['lr'],
                decay_steps=10000,
                decay_rate=0.9)

        self._model.compile(optimizer=Adagrad(learning_rate=lr_schedule, clipvalue=self.parameters['clip_value']),
                            loss=None)
        if print_summary:
            self._model.summary()

    def train(self, train_data, valid_data):

        # Add callbacks (early stopping, model checkpoint)
        weights_file = os.path.join(MODELS_DIR, "ss_model/elmo/elmo_best_weights.hdf5")
        save_best_model = ModelCheckpoint(filepath=weights_file, monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(patience=self.parameters['patience'], restore_best_weights=True)

        t_start = time.time()

        # Fit Model
        self._model.fit(x=train_data,
                                  validation_data=valid_data,
                                  epochs=self.parameters['epochs'],
                                  callbacks=[save_best_model, early_stopping])

        print('Training took {0} sec'.format(str(time.time() - t_start)))

    def evaluate(self, test_data):

        def unpad(x, y_true, y_pred):
            y_true_unpad = []
            y_pred_unpad = []
            for i, x_i in enumerate(x):
                for j, x_ij in enumerate(x_i):
                    if x_ij == 0:
                        y_true_unpad.append(y_true[i][:j])
                        y_pred_unpad.append(y_pred[i][:j])
                        break
            return np.asarray(y_true_unpad), np.asarray(y_pred_unpad)

        # Generate samples
        x, y_true_forward, y_true_backward = [], [], []
        for i in range(len(test_data)):
            test_batch = test_data[i][0]
            x.extend(test_batch[0])
            y_true_forward.extend(test_batch[1])
            y_true_backward.extend(test_batch[2])
        x = np.asarray(x)
        y_true_forward = np.asarray(y_true_forward)
        y_true_backward = np.asarray(y_true_backward)

        # Predict outputs
        y_pred_forward, y_pred_backward = self._model.predict([x, y_true_forward, y_true_backward])

        # Unpad sequences
        y_true_forward, y_pred_forward = unpad(x, y_true_forward, y_pred_forward)
        y_true_backward, y_pred_backward = unpad(x, y_true_backward, y_pred_backward)

        # Compute and print perplexity
        print('Forward Langauge Model Perplexity: {}'.format(ELMo.perplexity(y_pred_forward, y_true_forward)))
        print('Backward Langauge Model Perplexity: {}'.format(ELMo.perplexity(y_pred_backward, y_true_backward)))

    def save(self, sampled_softmax=True):
        """
        Persist model in disk
        :param sampled_softmax: reload model using the full softmax function
        :return: None
        """
        if not sampled_softmax:
            self.parameters['num_sampled'] = self.parameters['vocab_size']
        self.compile_elmo()
        self._model.load_weights(os.path.join(MODELS_DIR, 'ss_model/elmo/elmo_best_weights.hdf5'))
        self._model.save(os.path.join(MODELS_DIR, 'ss_model/elmo/ELMo_LM.h5'))
        print('ELMo Language Model saved successfully')

    def load(self):
        self._model = load_model(os.path.join(MODELS_DIR, 'ss_model/elmo/ELMo_LM.h5'),
                                 custom_objects={'SampledSoftmax': SampledSoftmax})

    @staticmethod
    def reverse(inputs, axes=1):
        return K.reverse(inputs, axes=axes)

    @staticmethod
    def perplexity(y_pred, y_true):

        cross_entropies = []
        for y_pred_seq, y_true_seq in zip(y_pred, y_true):
            # Reshape targets to one-hot vectors
            y_true_seq = to_categorical(y_true_seq, y_pred_seq.shape[-1])
            # Compute cross_entropy for sentence words
            cross_entropy = K.categorical_crossentropy(K.tf.convert_to_tensor(y_true_seq, dtype=K.tf.float32),
                                                       K.tf.convert_to_tensor(y_pred_seq, dtype=K.tf.float32))
            cross_entropies.extend(cross_entropy.eval(session=K.get_session()))

        # Compute mean cross_entropy and perplexity
        cross_entropy = np.mean(np.asarray(cross_entropies), axis=-1)

        return pow(2.0, cross_entropy)
