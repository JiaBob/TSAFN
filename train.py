import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
from torchvision import utils

from dataloader import *
from functions import *
from models import *

import argparse, os

parser = argparse.ArgumentParser(description='Texture and Structure Awareness Network')
parser.add_argument('-e', '--epochs', type=int, default=100, metavar='E',
                    help='the required total training epochs(default: 100)')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], metavar='Mode',
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

args = parser.parse_args()
epochs = args.epochs
mode = args.mode
model = args.model
pretrained = args.pretrained
numloader = args.numloader
separate = args.separate
vgg = 'vgg'+args.vgg


# already imported from functions
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([transforms.Resize((200, 200)),
                                      transforms.ToTensor()])

data = SmoothData('../verysmall_dataset', transform=transform)


train_loader = DataLoader(data('train'), batch_size=2, num_workers=numloader)
val_loader = DataLoader(data('val'), batch_size=10, num_workers=numloader)
test_loader = DataLoader(data('test'), batch_size=1, num_workers=numloader)  # batch_size here has to be 1

if mode == 'train':
    if model == 'all':
        model = Combination(vgg, pretrained, separate)

        optimizer = optim.Adam(model.parameters(), 1e-3)
        loss, val_losses = train(model, three_to_one, train_loader, val_loader, optimizer=optimizer, epochs=epochs)

    elif model == 'spn':
        model = SPN(vgg, pretrained)

        optimizer = optim.Adam(model.parameters(), 1e-3)
        loss, val_losses = train(model, SPN_one_iter, train_loader, val_loader, optimizer=optimizer, epochs=epochs)

    elif model == 'tpn':
        model = TPN(pretrained)

        optimizer = optim.Adam(model.parameters(), 1e-3)
        loss, val_losses = train(model, TPN_one_iter, train_loader, val_loader, optimizer=optimizer, epochs=epochs)

    elif model == 'tsafn':
        model = TSAFN(pretrained)

        optimizer = optim.Adam(model.parameters(), 1e-3)
        loss, val_losses = train(model, TSAFN_one_iter, train_loader, val_loader, optimizer=optimizer, epochs=epochs)

    else:
        raise Exception('Not valid model name, please refer to help')

elif mode == 'test':
    if model == 'all':
        model = Combination(vgg, pretrained, separate)
        loss, output = test(model, three_to_one, test_loader)

    elif model == 'spn':
        model = SPN(vgg, pretrained)
        loss, output = test(model, SPN_one_iter, test_loader)

    elif model == 'tpn':
        model = TPN(pretrained)
        loss, output = test(model, TPN_one_iter, test_loader)

    elif model == 'tsafn':
        model = TSAFN(pretrained)
        loss, output = test(model, TSAFN_one_iter, test_loader)

    else:
        raise Exception('Not valid model name, please refer to help')

    utils.save_image(output, 'result.jpg', nrow=2, padding=20)