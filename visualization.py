import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
import os

def random(params, opt, ds_val, model):
  for x, y in ds_val.take(1):
    x = x[:opt.num_samples]
    y = y[:opt.num_samples]
    
  c, f = model.G.encode(x)
  z = model.G.sample(y)
  
  fig, ax = plt.subplots(ncols = opt.num_samples, nrows = opt.num_samples + 1, figsize=(8, 8))
  
  for j, xj in enumerate(x):
    ax[0, j].imshow(xi[0] * 0.5 + 0.5)
    ax[0, j].axis('off')
  
  for i, ci in enumerate(c):
    ci = tf.repeat(ci[None, ...], opt.num_samples, axis=0)
    xti = model.G.decode(ci, z)
    for j, xtij in enumerate(xti):
      ax[i + 1, j].imshow(xtij[0] * 0.5 + 0.5)
      ax[i+1, j].axis('off')
      
  plt.tight_layout()
  dir = f'{self.opt.output_dir}/{self.opt.model}/{self.params_}'
  if not os.path.exists(dir):
    os.makedirs(dir)
  plt.savefig(f'{dir}/{epoch}.png')

  
  
def reference(parmas, opt, ds_val, model):
  for x, y in ds_val.take(1):
    x = x[:opt.num_samples]
    y = y[:opt.num_samples]
    
  src, ref = x, x
    
  b, h, w, c = ref.shape
  xa_repeat = tf.repeat(src, b, axis=0)
  xb_repeat = tf.reshape(tf.stack([ref for _ in range(b)], axis=0), (b ** 2, h, w, c))
  
  ca, fa = model.G.encode(xa_repeat)
  cb, fb = model.G.encode(xb_repeat)
  
  xab = model.G.decode(ca, fb)
  
  fig, ax = plt.subplots(ncols=b + 1, nrows=b + 1, figsize=(8, 8))

  for k in range(b + 1):
    if k == 0:
      ax[0, k].imshow(tf.ones(source[0].shape))
      ax[0, k].axis('off')
    else:
      ax[0, k].imshow(source[k - 1])
      ax[0, k].axis('off')

  for k in range(1, b + 1):
    ax[k, 0].imshow(ref[k - 1])
    ax[k, 0].axis('off')

  k = 0
  for j in range(b):
      for i in range(b):
        ax[i + 1, j + 1].imshow(xa2b[k])
        ax[i + 1, j + 1].axis('off')
        k += 1
  plt.tight_layout()

  dir = f'{opt.output_dir}/{opt.model}/image_{key_var}'
  if not os.path.exists(dir):
    os.makedirs(dir)
  plt.savefig(f'{dir}/{epoch}.png')


def feature(params, opt, ds_val, model):
  pass

class VizCallback(callbacks.Callback):
    def __init__(self, params, opt, ds_val):
        super().__init__()
        self.ds = ds_val
        self.opt = opt
        self.params_ = params

    def on_epoch_end(self, epoch, logs=None):
        random(self.params, self.opt, self.ds_val)
        reference(self.params, self.opt, self.ds_val)
        feature(self.params, self.opt, self.ds_val)
