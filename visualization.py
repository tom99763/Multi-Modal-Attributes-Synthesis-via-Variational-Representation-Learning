import tensorflow as tf
import matplotlib.pyplot as plt

def random():
  pass

def reference():
  pass

def feature():
  pass

class VisualizeCallback(callbacks.Callback):
    def __init__(self, params, opt, ds_val):
        super().__init__()
        self.ds = ds_val
        self.opt = opt
        self.params_ = params

    def on_epoch_end(self, epoch, logs=None):
        random(params, opt, ds_val)
        reference(params, opt, ds_val)
        feature(params, opt, ds_val)
