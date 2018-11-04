from torch import optim
from functions import trainer
from config import *
import argparse


parser = argparse.ArgumentParser(description='Texture and Structure Awareness Network')
parser.add_argument('-e', '--epochs', type=int, default=default_epochs, metavar='E',
                    help='the required total training epochs(default: {})'.format(default_epochs))
parser.add_argument('--mode', type=str, default=default_mode, choices=['train', 'test', 'unknown'], metavar='Mode',
                    help='train or test (default: {})'.format(default_mode))
parser.add_argument('-m', '--model', type=str, choices=['tpn', 'tsafn', 'spn', 'all'], default=default_model, metavar='M',
                    help='specify which model to use. They are tpn, spn, tsafn and all. (default: {})'.format(default_model))
parser.add_argument('-p', '--pretrained', type=int, choices=[1, 0], default=default_pretrained, metavar='NL',
                    help='specific if use pre-trained models. The pre-trained_model should be under folder '
                         '\'pretrained\'. If not, it will be set to False automatically in the program.'
                         'Pass 1 (True) or 0 (False) to the it. (default: {})'.format(default_pretrained))
parser.add_argument('-s', '--separate', type=int, choices=[1, 0], default=default_separate, metavar='S',
                    help=' determine if train combination model from three separate pre-trained network '
                         'or one combination network. Only works when -p is 1 and -m is all. (default: {})'.format(default_separate))
parser.add_argument('-l', '--lr', type=float, default=default_lr, metavar='L',
                    help='the learning rate (default: {})'.format(default_lr))
# parser.add_argument('-v', '--vgg', type=str, default='11', metavar='Vgg',
#                     help='the configuration of vgg (default: 11)')
parser.add_argument('-n', '--numloader', type=int, default=default_numloader, metavar='NL',
                    help='the num of CPU for data loading. 0 means only use one CPU. '
                         '(default: {})'.format(default_numloader))
parser.add_argument('-b', '--batch', type=tuple, default=default_batch_size, metavar='Ba',
                    help='the batch size of training (index zero) and validation (index one)'
                         '(default: {})'.format(default_batch_size))
parser.add_argument('--samples', type=int, default=default_samples, metavar='SA',
                    help='how many samples to be sampled by using random sampler'
                         '(default: {})'.format(default_samples))
parser.add_argument('-c', '--crop', type=tuple, default=default_random_crop_size, metavar='CR',
                    help='specifiy the random crop size on sampling.'
                         '(default: {})'.format(default_random_crop_size))
parser.add_argument('--path', type=str, default=default_path, metavar='P',
                    help='data path for unknown data set. Only needed when mode=unknown')

args = parser.parse_args()

epochs = args.epochs
mode = args.mode
model = args.model
lr = args.lr
pretrained = args.pretrained
numloader = args.numloader
separate = args.separate
path = args.path
batch = args.batch
samples = args.samples
random_crop_size = args.crop

if __name__ == "__main__":
    optimizer = optim.Adam
    t = trainer(model, lr, optimizer, pretrained=pretrained, separate=separate, epochs=epochs, numloader=numloader,
                batchsize=batch, sample_amount=samples, random_crop_size=random_crop_size)
    if mode == 'train':
        t.train(path)
    elif mode == 'test':
        t.test()  # currently not support test with specific data set on pre-trained models
    else:
        t.test_unknown(path)