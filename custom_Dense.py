from profiler_utils import update_profile
import tensorflow as tf
import time
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class CustomDense(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        start_time = time.perf_counter()

        output = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            output = self.activation(output)

        end_time = time.perf_counter()
        update_profile("CustomDense.call", end_time - start_time)

        return output
