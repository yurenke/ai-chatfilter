from tensorflow.keras.layers import Layer, Concatenate
import tensorflow as tf


class ScalarMix(Layer):
    """
    Computes a parameterised scalar mixture of N tensors, `mixture = gamma * sum(s_k * tensor_k)`
    where `s = softmax(w)`, with `w` and `gamma` scalar parameters.
    """
    def __init__(self, **kwargs):
        super(ScalarMix, self).__init__(**kwargs)
        
    def build(self, input_shape):
        
        self.W = self.add_weight(
                    shape=(1,1,len(input_shape)),
                    name='{}_W'.format(self.name),
                    initializer='uniform',
                    dtype=tf.float32,
                    trainable=True)

        self._gamma = tf.Variable(1.0, name='{}_gamma'.format(self.name))
        
    def call(self, inputs):

        # inputs is a list of tensor of shape [(n_batch, n_feat), ..., (n_batch, n_feat)]
        # expand last dim of each input passed [(n_batch, n_feat, 1), ..., (n_batch, n_feat, 1)]
        # inputs = [tf.expand_dims(i, -1) for i in inputs]
        # inputs = tf.convert_to_tensor(inputs)
        # inputs = tf.map_fn(lambda x: tf.expand_dims(x, -1), inputs)
        for i in range(len(inputs)):
            inputs[i] = tf.expand_dims(inputs[i], -1)

        inputs = Concatenate(axis=-1)(inputs) # (n_batch, n_feat, n_inputs)
        weights = tf.nn.softmax(self.W, axis=-1) # (1,1,n_inputs)
        # weights sum up to one on last dim

        return self._gamma * tf.reduce_sum(weights*inputs, axis=-1) # (n_batch, n_feat) 
    
    def get_config(self):
        base_config = super(ScalarMix, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0]
