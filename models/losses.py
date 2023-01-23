from modules import *
import tensorflow as tf
from tensorflow.keras import layers


class Discriminator(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        disc_type = config['disc_type']

        if disc_type == 'classic':
            self.disc = None

        elif disc_type == 'patch':
            self.disc = Patch_Discriminator(config)

        elif disc_type == 'cam':
            self.disc = CAM_Discriminator(config)

        elif disc_type == 'Multi_scale':
            self.disc = None

    def call(self, x):
        return self.disc(x)


class Patch_Discriminator(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.act = config['act']
        self.use_bias = config['use_bias']
        self.norm = config['norm']
        self.num_downsampls = config['num_downsamples']
        self.num_resblocks = config['num_resblocks']
        dim = config['base']

        self.blocks = tf.keras.Sequential([
            ConvBlock(dim, 4, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm,
                      activation=tf.nn.leaky_relu)
        ])

        for _ in range(self.num_downsampls):
            dim = dim * 2
            self.blocks.add(ConvBlock(dim, 4, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm,
                                      activation=tf.nn.leaky_relu))

        self.blocks.add(Padding2D(1, pad_type='constant'))
        self.blocks.add(ConvBlock(512, 4, padding='valid', use_bias=self.use_bias, norm_layer=self.norm,
                                  activation=tf.nn.leaky_relu))
        self.blocks.add(Padding2D(1, pad_type='constant'))
        self.blocks.add(ConvBlock(1, 4, padding='valid'))

    def call(self, x):
        return self.blocks(x)


class CAM_Discriminator(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.act = config['act']
        self.use_bias = config['use_bias']
        self.norm = config['norm']
        self.num_downsampls = config['num_downsamples']
        self.num_resblocks = config['num_resblocks']
        dim = config['base']

        # discriminator
        self.blocks = tf.keras.Sequential([
            ConvBlock(dim, 4, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm,
                      activation=tf.nn.leaky_relu)
        ])
        for _ in range(self.num_downsampls):
            dim = dim * 2
            self.blocks.add(ConvBlock(dim, 4, strides=2, padding='same', use_bias=self.use_bias, norm_layer=self.norm,
                                      activation=tf.nn.leaky_relu))
        self.blocks.add(Padding2D(1, pad_type='constant'))
        self.blocks.add(ConvBlock(512, 4, padding='valid', use_bias=self.use_bias, norm_layer=self.norm,
                                  activation=tf.nn.leaky_relu))

        # class activation map
        self.gap = layers.GlobalAveragePooling2D()
        self.gmp = layers.GlobalMaxPool2D()
        self.gap_fc = layers.Dense(1, use_bias=False)
        self.gmp_fc = layers.Dense(1, use_bias=False)
        self.fuse = ConvBlock(dim, 1, activation=tf.nn.leaky_relu)
        self.pad = Padding2D(1, pad_type='constant')
        self.conv = ConvBlock(1, 4, padding='valid')

    def call(self, x):
        x = self.blocks(x)

        # average
        gap = self.gap(x)
        gap_logits = self.gap_fc(gap)
        gap_weights = self.gap_fc.trainable_weights[0]
        gap_weights = tf.gather(tf.transpose(gap_weights), 0)
        x_gap = x * gap_weights  # (b, h, w, c)

        # max
        gmp = self.gap(x)
        gmp_logits = self.gmp_fc(gmp)
        gmp_weights = self.gmp_fc.trainable_weights[0]
        gmp_weights = tf.gather(tf.transpose(gmp_weights), 0)
        x_gmp = x * gmp_weights  # (b, h, w, c)

        # cam
        cam_logits = tf.concat([gap_logits, gmp_logits], axis=-1)
        x = tf.concat([x_gap, x_gmp], axis=-1)
        x = self.fuse(x)
        x = self.pad(x)
        x = self.conv(x)
        return x, cam_logits
