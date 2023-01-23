import os
import tensorflow as tf
from models import VRLGAN
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import yaml
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_model(opt):
    config = get_config(f'./configs/{opt.model}.yaml')
    if opt.model == 'VRLGAN':
        model = VRLGAN.VRLGAN(config)
        params = f"{config['cluster_loss']}_{config['lambda_kld']}_{config['lambda_cls']}_{config['lambda_lr']}"

    return model, params


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def set_callbacks(params, opt, val_ds = None):
    ckpt_dir = f"{opt.ckpt_dir}/{opt.model}"
    output_dir = f"{opt.output_dir}/{opt.model}"

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint_callback = callbacks.ModelCheckpoint(filepath=f"{ckpt_dir}/{params}/{opt.model}", save_weights_only=True)
    history_callback = callbacks.CSVLogger(f"{output_dir}/{params}.csv", separator=",", append=False)
    visualize_callback = VisualizeCallback(params, opt, val_ds)
    return [checkpoint_callback, history_callback, visualize_callback]
