from tensorflow.keras import layers
import tensorflow as tf


class Padding2D(layers.Layer):
    """ 2D padding layer.
    """

    def __init__(self, padding=(1, 1), pad_type='constant', **kwargs):
        assert pad_type in ['constant', 'reflect', 'symmetric']
        super(Padding2D, self).__init__(**kwargs)
        self.padding = (padding, padding) if type(padding) is int else tuple(padding)
        self.pad_type = pad_type

    def call(self, inputs, training=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(inputs, padding_tensor, mode=self.pad_type)


class InstanceNorm(layers.Layer):
    def __init__(self, epsilon=1e-5, affine=False, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.affine = affine

    def build(self, input_shape):
        if self.affine:
            self.gamma = self.add_weight(name='gamma',
                                         shape=(input_shape[-1],),
                                         initializer=tf.random_normal_initializer(0, 0.02),
                                         trainable=True)
            self.beta = self.add_weight(name='beta',
                                        shape=(input_shape[-1],),
                                        initializer=tf.zeros_initializer(),
                                        trainable=True)

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        x = tf.divide(tf.subtract(inputs, mean), tf.math.sqrt(tf.add(var, self.epsilon)))
        if self.affine:
            return self.gamma * x + self.beta
        return x


class LayerInstanceNorm(layers.Layer):
    def __init__(self, adaptive=False):
        super().__init__()
        self.inst = InstanceNorm(affine=False)
        self.ln = layers.LayerNormalization()

        self.rho = tf.Variable(tf.constant(1.0),
                               constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0))

        self.adaptive = adaptive

    def build(self, shape):
        if self.adaptive:
            dim = shape[0][-1]
            self.gamma = layers.Dense(dim)
            self.beta = layers.Dense(dim)

    def call(self, inputs):
        if self.adaptive:
            x, w = inputs
        else:
            x = inputs
        x_in = self.inst (x)
        x_ln = self.ln(x)
        rho = tf.clip_by_value(self.rho - 0.1, 0.0, 1.0)
        x = rho * x_in + (1 - rho) * x_ln
        if self.adaptive:
            x = self.gamma(w) * x + self.beta(w)
        return x


class ConvBlock(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 norm_layer=None,
                 activation='linear',
                 use_bias=True,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.conv2d = layers.Conv2D(filters,
                                    kernel_size,
                                    strides,
                                    padding,
                                    use_bias=use_bias,
                                    kernel_initializer=initializer)
        if activation != 'none':
            self.activation = layers.Activation(activation)
        else:
            self.activation = tf.identity
        if norm_layer == 'batch':
            self.normalization = layers.BatchNormalization()
        elif norm_layer == 'inst':
            self.normalization = InstanceNorm(affine=False)
        elif norm_layer == 'adain':
            self.normalization = LayerInstanceNorm(True)
        else:
            self.normalization = tf.identity

        self.norm_layer = norm_layer

    def call(self, inputs, training=None):
        if self.norm_layer == 'adain':
            x, w = inputs
        else:
            x = inputs
        x = self.conv2d(x)

        if self.norm_layer == 'adain':
            x = self.normalization([x, w])
        else:
            x = self.normalization(x)
        x = self.activation(x)
        return x


class ConvTransposeBlock(layers.Layer):
    """ ConvTransposeBlock layer consists of Conv2DTranspose + Normalization + Activation.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 norm_layer=None,
                 activation='linear',
                 use_bias=True,
                 **kwargs):
        super(ConvTransposeBlock, self).__init__(**kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.convT2d = layers.Conv2DTranspose(filters,
                                              kernel_size,
                                              strides,
                                              padding,
                                              use_bias=use_bias,
                                              kernel_initializer=initializer)
        self.activation = layers.Activation(activation)
        if norm_layer == 'batch':
            self.normalization = layers.BatchNormalization()
        elif norm_layer == 'instance':
            self.normalization = InstanceNorm(affine=False)
        elif norm_layer == 'layerinst':
            self.normalization = LayerInstanceNorm(False)
        elif norm_layer == 'layer':
            self.normalization = layers.LayerNormalization()
        else:
            self.normalization = tf.identity

    def call(self, inputs, training=None):
        x = self.convT2d(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class ResBlock(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 norm_layer,
                 act,
                 use_bias=True,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)

        self.reflect_pad1 = Padding2D(1, pad_type='reflect')
        self.conv_block1 = ConvBlock(filters,
                                     kernel_size,
                                     padding='valid',
                                     use_bias=use_bias,
                                     norm_layer=norm_layer,
                                     activation=act)

        self.reflect_pad2 = Padding2D(1, pad_type='reflect')
        self.conv_block2 = ConvBlock(filters,
                                     kernel_size,
                                     padding='valid',
                                     use_bias=use_bias,
                                     norm_layer=norm_layer,
                                     activation=act
                                    )
        self.filters = filters
        self.norm_layer = norm_layer

    def build(self, shape):
        dim = shape[-1]
        if dim != self.filters:
            self.skip = ConvBlock(self.filters,
                                  1,
                                  padding='valid',
                                  use_bias=False,
                                  norm_layer='none')
        else:
            self.skip = tf.identity

    def call(self, inputs, training=None):
        if self.norm_layer == 'adain':
            x, w = inputs
            skip = self.skip(x)
            x = self.reflect_pad1(x)
            x = self.conv_block1([x, w])
            x = self.reflect_pad2(x)
            x = self.conv_block2([x, w])
        else:
            x = inputs
            skip = self.skip(x)
            x = self.reflect_pad1(x)
            x = self.conv_block1(x)
            x = self.reflect_pad2(x)
            x = self.conv_block2(x)
        return x + skip
