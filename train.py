from torch import optim
from functions import trainer

import argparse


default_path = '../data/newmini_dataset'


parser = argparse.ArgumentParser(description='Texture and Structure Awareness Network')
parser.add_argument('-e', '--epochs', type=int, default=100, metavar='E',
                    help='the required total training epochs(default: 100)')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'unknown'], metavar='Mode',
                    help='train or test (default: train)')
parser.add_argument('-m', '--model', type=str, choices=['tpn', 'tsafn', 'spn', 'all'], default='tpn', metavar='M',
                    help='specify which model to use. They are tpn, spn, tsafn and all. (default: tpn)')
parser.add_argument('-p', '--pretrained', type=int, choices=[1, 0], default=0, metavar='NL',
                    help='specific if use pre-trained models. The pre-trained_model should be under folder '
                         '\'pretrained\'. If not, it will be set to False automatically in the program.'
                         'Pass 1 (True) or 0 (False) to the it. (default: 0)')
parser.add_argument('-s', '--separate', type=int, choices=[1, 0], default='1', metavar='S',
                    help=' determine if train combination model from three separate pre-trained network '
                         'or one combination network. Only works when -p is 1 and -m is all. (default: 1)')
# parser.add_argument('-v', '--vgg', type=str, default='11', metavar='Vgg',
#                     help='the configuration of vgg (default: 11)')
parser.add_argument('-n', '--numloader', type=int, default=0, metavar='NL',
                    help='the num of CPU for data loading. 0 means only use one CPU. '
                         '(default: 0)')
parser.add_argument('-b', '--batch', type=tuple, default=(2,1), metavar='NL',
                    help='the batch size of training (index zero) and validation (index one)'
                         '(default: (2, 1))')
parser.add_argument('--path', type=str, default=default_path, metavar='P',
                    help='data path for unknown data set. Only needed when mode=unknown')

args = parser.parse_args()
epochs = args.epochs
mode = args.mode
model = args.model
pretrained = args.pretrained
numloader = args.numloader
separate = args.separate
path = args.path
batch = args.batch

if __name__ == "__main__":
    optimizer = optim.Adam
    t = trainer(model, 1e-3, optimizer, pretrained=pretrained, separate=separate, epochs=epochs, numloader=numloader,
                batchsize=batch)
    if mode == 'train':
        t.train()
    elif mode == 'test':
        t.test()  # currently not support test with specific data set on pre-trained models
    else:
        t.test_unknown(path)