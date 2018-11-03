from torch.utils.data import DataLoader
from torch import optim

from dataloader import *
from old_functions import *
from models import *

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
parser.add_argument('-v', '--vgg', type=str, default='11', metavar='Vgg',
                    help='the configuration of vgg (default: 11)')
parser.add_argument('-n', '--numloader', type=int, default=0, metavar='NL',
                    help='the num of CPU for data loading. 0 means only use one CPU. '
                         '(default: 0)')
parser.add_argument('--path', type=str, default=default_path, metavar='P',
                    help='data path for unknown data set. Only needed when mode=unknown')

args = parser.parse_args()
epochs = args.epochs
mode = args.mode
model = args.model
pretrained = args.pretrained
numloader = args.numloader
separate = args.separate
vgg = 'vgg'+args.vgg
path = args.path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if mode == 'unknown':
    data = SplitData(path, unknown=True)
else:
    data = SplitData(path)

if mode == 'train':
    train_loader = DataLoader(data('train'), batch_size=2, num_workers=numloader)
    val_loader = DataLoader(data('val'), batch_size=1, num_workers=numloader)

    print('Total amount of training set is {}'.format(len(train_loader)))
    print('Total amount of validation set is {}'.format(len(val_loader)))

    if model == 'all':
        model = nn.DataParallel(Combination(pretrained, vgg, separate)).to(device)

        optimizer = optim.Adam(model.parameters(), 1e-3)
        loss, val_losses = train(model, combination_one_iter, train_loader, val_loader, optimizer=optimizer, epochs=epochs)

    elif model == 'spn':
        model = nn.DataParallel(SPN(pretrained, vgg)).to(device)

        optimizer = optim.Adam(model.parameters(), 1e-3)
        loss, val_losses = train(model, spn_one_iter, train_loader, val_loader, optimizer=optimizer, epochs=epochs)

    elif model == 'tpn':
        model = nn.DataParallel(TPN(pretrained)).to(device)

        optimizer = optim.Adam(model.parameters(), 1e-3)
        loss, val_losses = train(model, tpn_one_iter, train_loader, val_loader, optimizer=optimizer, epochs=epochs)

    elif model == 'tsafn':
        model = nn.DataParallel(TSAFN(pretrained)).to(device)

        optimizer = optim.Adam(model.parameters(), 1e-3)
        loss, val_losses = train(model, tsafn_one_iter, train_loader, val_loader, optimizer=optimizer, epochs=epochs)

    else:
        raise Exception('Not valid model name, please refer to help')

elif mode == 'test':
    test_loader = DataLoader(data('test'), batch_size=1, num_workers=0)  # batch_size here has to be 1
    print('Total amount of test set is {}'.format(len(test_loader)))
    if model == 'all':
        model = nn.DataParallel(Combination(vgg, pretrained, separate)).to(device)
        loss, output = test(model, combination_one_iter, test_loader)

    elif model == 'spn':
        model = nn.DataParallel(SPN(vgg, pretrained)).to(device)
        loss, output = test(model, spn_one_iter, test_loader)

    elif model == 'tpn':
        model = nn.DataParallel(TPN(pretrained)).to(device)
        loss, output = test(model, tpn_one_iter, test_loader)

    elif model == 'tsafn':
        model = nn.DataParallel(TSAFN(pretrained)).to(device)
        loss, output = test(model, tsafn_one_iter, test_loader)

    else:
        raise Exception('Not valid model name, please refer to help')

elif mode == 'unknown':
    unknown_loader = DataLoader(data('unknown'), batch_size=1, num_workers=0)

    if model == 'all':
        model = nn.DataParallel(Combination(vgg, pretrained, separate)).to(device)

    elif model == 'spn':
        model = nn.DataParallel(SPN(vgg, pretrained)).to(device)

    elif model == 'tpn':
        model = nn.DataParallel(TPN(pretrained)).to(device)

    elif model == 'tsafn':
        model = nn.DataParallel(TSAFN(pretrained)).to(device)

    else:
        raise Exception('Not valid model name, please refer to help')

    loss, output = test_unknown(model, unknown_loader)
