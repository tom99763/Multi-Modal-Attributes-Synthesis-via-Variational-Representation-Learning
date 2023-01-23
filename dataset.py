import tensorflow as tf
import os
from sklearn.model_selection import train_step_split as ttp
AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_image(pth, label, opt):
    image = tf.image.decode_jpeg(tf.io.read_file(pth), channels=opt.num_channels)
    image = tf.cast(tf.image.resize(image, (opt.image_size, opt.image_size)), 'float32')
    return (image-127.5)/127.5, label

  
def build_dataset(opt):
  classes = sorted(os.listdir(opt.img_dir))
  paths = []
  labels = []
  for i, class_name in enumerate(classes):
    img_pth = list(map(lambda name: f'{opt.img_dir}/{class_name}/{name}',os.listdir(f'{opt.img_dir}/{class_name}')))
    paths += img_pth
    labels += [i] * len(img_pth)
  
  x_train, x_test, y_train, y_test = ttp(paths, labels, test_size = opt.val_size, random_state = 999)
  
  ds_train = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)). \
            map(lambda path, label: get_image(path, label, opt), num_parallel_calls=AUTOTUNE). \
            batch(opt.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
  
  ds_test = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)). \
            map(lambda path, label: get_image(path, label, opt), num_parallel_calls=AUTOTUNE). \
            batch(opt.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
  
  return ds_train, ds_test
