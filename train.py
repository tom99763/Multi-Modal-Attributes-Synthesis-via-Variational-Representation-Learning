import argparse
from utils import *
from dataset import *
from tensorflow.keras import optimizers


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--model', type=str, default='VRLGAN')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--val_size', type=int, default=0.4)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_samples', type=int, default=4)
    opt, _ = parser.parse_known_args()
    return opt
  
  
def main():
    opt = parse_opt()
    model, params = load_model(opt)
    model.compile(optimizer = [optimizers.Adam(learning_rate = opt.lr, beta_1 = 0.5, beta_2 = 0.999),
                               optimizers.Adam(learning_rate = opt.lr, beta_1 = 0.5, beta_2 = 0.999)])
    ds_train, ds_val = build_dataset(opt)
    _callbacks = set_callbacks(params, opt, ds_val)
    model.fit(
        x=ds_train,
        validation_data=ds_val,
        epochs=opt.num_epochs,
        callbacks=_callbacks
    )
    
if __name__ == '__main__':
  main()
