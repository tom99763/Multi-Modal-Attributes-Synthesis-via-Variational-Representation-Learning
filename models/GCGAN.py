

class GCGAN(tf.keras.Model):
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
      f = tf.concat([fa, fb], axis=0)
      
      #vae
      mua, logvara = self.G.encode_emb(ya)
      mub, logvarb = self.G.encode_emb(yb)
      za = self.G.reparameterize(mua, logvara)
      zb = self.G.reparameterize(mub, logvarb)
      mu = tf.concat([mua, mub], axis=0)
      logvar = tf.concat([logvara, logvarb], axis=0)
      
      #reconstruction
      xaa = self.G.decode(ca, fa)
      xbb = self.G.decode(cb, fb)
      xr = tf.concat([xaa, xbb], axis=0)
      
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
      lkld = kl_div(f, mu, logvar)
      lcls_g, lcls_d = crossentropy(y, logits_fake), crossentropy(y, logits_real)
      g_loss = self.config['lambda_lr'] * lr + self.config['lambda_kld'] * lkld +\
               self.config['lambda_cls'] * lcls_g + lg
      d_loss = self.config['lambda_cls'] * lcls_d + ld
    Ggrads = tape.gradient(g_loss, self.G.trainable_weights)
    Dgrads = tape.gradient(d_loss, self.D.trainable_weights)
    self.optimizer[0].apply_gradients(zip(Ggrads, self.G.trainable_weights))
    self.optimizer[1].apply_gradients(zip(Dgrads, self.D.trainable_weights))
    return {'lr':lr, 'lkld':lkld, 'lcls_g':lcls_g, 'lcls_d':lcls_d, 'lg':lg, 'ld':ld}
