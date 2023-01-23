import tensorflow as tf
import math
from tensorflow.keras import losses


def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def l2_loss(x, y):
    return tf.reduce_mean((x - y) ** 2)
  
  
def crossentropy(y_true, y_pred):
  return losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)

def nnl_loss(z, mu, logvar):
  log2pi = tf.math.log(2. * math.pi)
  nnl = -tf.reduce_sum(-.5 * ((z - mu) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=-1)
  return nnl

def gan_loss(critic_real, critic_fake, gan_mode):
    if gan_mode == 'lsgan':
        d_loss = tf.reduce_mean((1 - critic_real) ** 2 + critic_fake ** 2)
        g_loss = tf.reduce_mean((1 - critic_fake) ** 2)

    elif gan_mode == 'nonsaturate':
        d_loss = tf.reduce_mean(tf.math.softplus(-critic_real) + tf.math.softplus(critic_fake))
        g_loss = tf.reduce_mean(tf.math.softplus(-critic_fake))

    elif gan_mode == 'wgangp':
        d_loss = tf.reduce_mean(-critic_real + critic_fake)
        g_loss = tf.reduce_mean(-critic_fake)
    return 0.5 * d_loss, g_loss
