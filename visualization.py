import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
import os

def random(params, opt, ds_val, model):
  for x, y in ds_val.take(1):
    x = x[:opt.num_samples]
    y = y[:opt.num_samples]
    
  c, f = model.encode(x)
  z = model.sample(y)
  
  fig, ax = plt.subplots(ncols = opt.num_samples, nrows = opt.num_samples + 1, figsize=(8, 8))
  
  for j, xj in enumerate(x):
    ax[0, j].imshow(xi[0] * 0.5 + 0.5)
    ax[0, j].axis('off')
  
  for i, ci in enumerate(c):
    ci = tf.repeat(ci[None, ...], opt.num_samples, axis=0)
    xti = model.decode(ci, z)
    for j, xtij in enumerate(xti):
      ax[i + 1, j].imshow(xtij[0] * 0.5 + 0.5)
      ax[i+1, j].axis('off')
      
  plt.tight_layout()
  dir = f'{self.opt.output_dir}/{self.opt.model}/{self.params_}'
  if not os.path.exists(dir):
    os.makedirs(dir)
  plt.savefig(f'{dir}/{epoch}.png')

def reference(params, opt, ds_val, model):
  pass

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
