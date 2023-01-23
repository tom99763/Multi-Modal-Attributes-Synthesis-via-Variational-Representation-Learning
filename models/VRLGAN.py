import sys
sys.path.append('./models')
import tensorflow as tf
from modules import *
from losses import *
from discriminators import *


class Encoder(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    dim = config['base']
    latent_dim = config['latent_dim']
    num_downsamples = config['num_downsamples']
    num_resblocks = config['num_resblocks']
    norm = config['norm']
    act = config['act']
    
    self.blocks = tf.keras.Sequential([
      ConvBlock(dim, 7, 1, 'same', norm, act)
    ])
    
    for i in range(num_downsamples):
      self.blocks.add(ConvBlock(dim, 3, 2, 'same', norm, act))
      dim = dim *2
      
    self.Ec = tf.keras.Sequential([
      ResBlock(dim, norm, act) for _ in range(num_resblocks//2)
    ])
    
    self.Ef = tf.keras.Sequential([
      ConvBlock(dim, 3, 2, 'same', 'none', act),
      ConvBlock(dim, 3, 2, 'same', 'none', act),
      layers.GlobalAveragePooling2D(),
      layers.Dense(latent_dim)
    ])
    
  def call(self, x):
    x = self.blocks(x)
    c, f = self.Ec(x), self.Ef(x)
    return c, f
    

class Decoder(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    dim = config['base']
    num_downsamples = config['num_downsamples']
    num_resblocks = config['num_resblocks']
    act = config['act']
    
    dim = dim * 2 ** num_downsamples
    self.blocks = [
      ResBlock(dim, 'adain', act) for _ in range(num_resblocks//2)
    ]
    
    self.upsample = tf.keras.Sequential([
      TConvBlock(int(dim * 2 ** (-i)), 3, 2, 'same', 'layer', act) for i in range(1, num_downsamples + 1)
    ])
    
    self.output = ConvBlock(3, 7, 1, 'same', activation = 'tanh')
    
  def call(self, inputs):
    c, f = inputs
    
    for block in self.blocks:
      c = block([c, f])
    
    c = self.upsample(c)
    c = self.output(c)
    return c


class Generator(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    self.E=Encoder(config)
    self.D=Decoder(config)
    self.emb_mu = layers.Embedding(config['num_classes'], config['latent_dim'])
    self.emb_logvar = layers.Embedding(config['num_classes'], config['latent_dim'])
    
  def call(self, x):
    c, f = self.encode(x)
    x = self.decode(c, f)
    return x
  
  def encode(self, x):
    c, f = self.E(x)
    return c, f
  
  def decode(self, c, f):
    x = self.D([c, f])
    return x
  
  def reparameterize(self, mu, logvar, eps=None):
    if eps is None:
      eps = tf.random.normal(mu.shape)
    z = mu + tf.exp(0.5 * logvar) * eps
    return z
  
  def encode_emb(self, y):
    mu = self.emb_mu(y)
    logvar = self.emb_logvar(y)
    return mu, logvar
  
  def sample(self, y, eps=None):
    mu, logvar = self.encode_emb(y)
    z = self.reparameterize(mu, logvar, eps)
    return z

class VRLGAN(tf.keras.Model):
  def __init__(self, config):
    super().__init__()
    
    self.G = Generator(config)
    self.D = Discriminator(config)
    self.config = config
    
  @tf.function
  def train_step(self, inputs):
    x, y = inputs
    xa, xb = tf.split(x, 2, axis=0)
    ya, yb = tf.split(y, 2, axis=0)
    with tf.GradientTape(persistent=True):
      ###forward
      ca, fa = self.G.encode(xa)
      cb, fb = self.G.encode(xb)
      c = tf.concat([ca, cb], axis=0)
      f = tf.concat([fa, fb], axis=0)
      
      #vae
      mua, logvara = self.G.encode_emb(ya)
      mub, logvarb = self.G.encode_emb(yb)
      za = self.G.reparameterize(mua, logvara)
      zb = self.G.reparameterize(mub, logvarb)
      z = tf.concat([za, zb], axis=0)
      mu = tf.concat([mua, mub], axis=0)
      logvar = tf.concat([logvara, logvarb], axis=0)
      
      #reconstruction
      xr = self.G(x)
      
      #translation
      xab = self.G.decode(ca, zb)
      xba = self.G.decode(cb, za)
      xt = tf.concat([xba, xab], axis=0)
      
      #discrimination
      critic_real, logits_real = self.D(x)
      critic_fake, logits_fake = self.D(xt)
      
      ###compute loss
      lr = l1_loss(x, xr)
      lg, ld = gan_loss(critic_real, critic_fake, self.config['gan_loss'])
      
      if self.config['cluster_loss'] == 'nnl_self_stopgrad':
        lkld = tf.reduce_mean(nll_loss(z, 0. ,0.) - nll_loss(z, mu, logvar) + nll_loss(f, 0. ,0.) -\
                              nll_loss(f, tf.stop_gradient(mu), tf.stop_gradient(logvar)))
      elif self.config['cluster_loss'] == 'nnl_self':
        lkld = tf.reduce_mean(nll_loss(z, 0. ,0.) - nll_loss(z, mu, logvar) + nll_loss(f, 0. ,0.) -\
                              nll_loss(f, mu, logvar))
      elif self.config['cluster_loss'] == 'nnl':
        lkld = tf.reduce_mean(nll_loss(z, 0. ,0.) - nll_loss(z, mu, logvar))
        
      lcls_g, lcls_d = crossentropy(y, logits_fake), crossentropy(y, logits_real)
      g_loss = self.config['lambda_lr'] * lr + self.config['lambda_kld'] * lkld +\
               self.config['lambda_cls'] * lcls_g + lg
      d_loss = self.config['lambda_cls'] * lcls_d + ld
    Ggrads = tape.gradient(g_loss, self.G.trainable_weights)
    Dgrads = tape.gradient(d_loss, self.D.trainable_weights)
    self.optimizer[0].apply_gradients(zip(Ggrads, self.G.trainable_weights))
    self.optimizer[1].apply_gradients(zip(Dgrads, self.D.trainable_weights))
    return {'lr':lr, 'lkld':lkld, 'lcls_g':lcls_g, 'lcls_d':lcls_d, 'lg':lg, 'ld':ld}
