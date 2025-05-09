from profiler_utils import update_profile
import tensorflow as tf
import time
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

# ------------------ Bulk Convolve Function ------------------
def sw_convolve(patches, filters):
    print(f"calling sw_convolve with patches: {patches.shape} and filters: {filters.shape}")
    start_time = time.perf_counter()

    output = tf.matmul(patches, filters)

    end_time = time.perf_counter()
    update_profile("sw_convolve", end_time - start_time)

    return output

# ------------------ Custom Conv2D Layer ------------------

@register_keras_serializable()
class CustomConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, activation=None, **kwargs):
        super(CustomConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        kernel_shape = (self.kernel_size, self.kernel_size, input_shape[-1], self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, inputs):
        start_time = time.perf_counter()

        batch_size = tf.shape(inputs)[0]
        in_height = tf.shape(inputs)[1]
        in_width = tf.shape(inputs)[2]
        k = self.kernel_size

        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, k, k, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        out_h = in_height - k + 1
        out_w = in_width - k + 1

        patches_flat = tf.reshape(patches, (batch_size * out_h * out_w, -1))
        filters_flat = tf.reshape(self.kernel, (-1, self.filters))

        outputs_flat = sw_convolve(patches_flat, filters_flat)

        outputs = tf.reshape(outputs_flat, (batch_size, out_h, out_w, self.filters))

        if self.activation is not None:
            outputs = self.activation(outputs)

        end_time = time.perf_counter()
        update_profile("CustomConv2D.call", end_time - start_time)

        return outputs
