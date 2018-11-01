import torch
from torchvision.models.vgg import VGG
from torchvision import models
import torch.nn.functional as F
from torch import nn

import os
from functions import device

class TSAFN(nn.Module):
    def __init__(self, pretrained=False):
        super(TSAFN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 3, 5, 1, 2),
            nn.BatchNorm2d(3)
        )

        if pretrained and os.path.exists('./pretrained/TSAFN.pth'):
            self.load_state_dict(torch.load('./pretrained/TSAFN.pth')['state_dict'])
            current_val_loss = torch.load('./pretrained/TSAFN.pth')['val_loss']
            print('Finish loading pre-trained data, current validation loss is {:1.5f}'.format(current_val_loss))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x.sigmoid()


class TPN(nn.Module):
    def __init__(self, pretrained=False):
        super(TPN, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.conv2_4 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(True)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(True)
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(True)
        )
        self.conv3_4 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        if pretrained and os.path.exists('./pretrained/TPN.pth'):
            self.load_state_dict(torch.load('./pretrained/TPN.pth')['state_dict'])
            current_val_loss = torch.load('./pretrained/TPN.pth')['val_loss']
            print('Finish loading pre-trained data, current validation loss is {:1.5f}'.format(current_val_loss))

    def forward(self, x):
        N, C, h, w = x.size()

        size1 = x
        size2 = F.interpolate(x, size=(h // 2, w // 2))
        size4 = F.interpolate(x, size=(h // 4, w // 4))
        size8 = F.interpolate(x, size=(h // 8, w // 8))

        size1 = self.conv1_1(size1)
        size2 = self.conv1_2(size2)
        size4 = self.conv1_3(size4)
        size8 = self.conv1_4(size8)

        size1 = self.conv2_1(size1)
        size2 = self.conv2_2(size2)
        size4 = self.conv2_3(size4)
        size8 = self.conv2_4(size8)

        size1 = self.conv3_1(size1)
        size2 = self.conv3_2(size2)
        size4 = self.conv3_3(size4)
        size8 = self.conv3_4(size8)

        size2 = F.interpolate(size2, size=(h, w))
        size4 = F.interpolate(size4, size=(h, w))
        size8 = F.interpolate(size8, size=(h, w))

        concat = torch.cat((size1, size2, size4, size8), 1)

        return self.conv4(concat)


class SPN(nn.Module):
    def __init__(self, vgg='vgg11', pretrained=False):
        super(SPN, self).__init__()

        self.vgg = vgg
        self.body = VGGNet(pretrained=True, model=vgg)

        self.side_output_layer1 = nn.Conv2d(64, 1, 1)
        self.side_output_layer2 = nn.Conv2d(128, 1, 1)
        self.side_output_layer3 = nn.Conv2d(256, 1, 1)
        self.side_output_layer4 = nn.Conv2d(512, 1, 1)
        self.side_output_layer5 = nn.Conv2d(512, 1, 1)

        self.fusion = nn.Conv2d(5, 1, 1)

        if pretrained and os.path.exists('./pretrained/SPN.pth'):
            vgg = torch.load('./pretrained/SPN.pth')['vgg']
            self.body = VGGNet(pretrained=True, model=vgg)
            self.load_state_dict(torch.load('./pretrained/SPN.pth')['state_dict'])
            current_val_loss = torch.load('./pretrained/SPN.pth')['val_loss']
            print('Finish loading pre-trained data, current validation loss is {:1.5f}'.format(current_val_loss))

    def forward(self, x):
        N, C, h, w = x.size()

        # load vgg
        body = self.body(x)

        # side-output layers
        side_output1 = body['side_output1']
        side_output1 = self.side_output_layer1(side_output1)
        side_output1 = F.interpolate(side_output1, size=(h, w))

        side_output2 = body['side_output2']
        side_output2 = self.side_output_layer2(side_output2)
        side_output2 = F.interpolate(side_output2, size=(h, w))

        side_output3 = body['side_output3']
        side_output3 = self.side_output_layer3(side_output3)
        side_output3 = F.interpolate(side_output3, size=(h, w))

        side_output4 = body['side_output4']
        side_output4 = self.side_output_layer4(side_output4)
        side_output4 = F.interpolate(side_output4, size=(h, w))

        side_output5 = body['side_output5']
        side_output5 = self.side_output_layer5(side_output5)
        side_output5 = F.interpolate(side_output5, size=(h, w))

        # fusion layer
        fuse = torch.cat((side_output1, side_output2, side_output3, side_output4, side_output5), 1)
        fusion = self.fusion(fuse)

        # will use cross_entropy_with_logit loss, so no need apply sigmoid on the final output.
        return side_output1, side_output2, side_output3, side_output4, side_output5, fusion


# the strategy to both get the structure and pretrained weight of in-built vgg
class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg11', requires_grad=True):

        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.{}_bn(pretrained=True).state_dict())".format(model))

        # for fix weight transfor learning
        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        # first reconstructure same vgg structure as in-bulit to utilize pretraining, but del the redundent fc later
        del self.classifier

    def forward(self, x):
        output = {}

        # catch the middle outputs
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)

            output["side_output%d" % (idx + 1)] = x

        return output


# this range only for batchnorm version
ranges = {
    'vgg11': ((0, 3), (3, 7), (7, 14), (14, 21), (21, 28)),
    'vgg13': ((0, 6), (6, 13), (13, 20), (20, 27), (27, 34)),
    'vgg16': ((0, 6), (6, 13), (13, 23), (23, 32), (32, 43)),
    'vgg19': ((0, 6), (6, 13), (13, 26), (26, 39), (39, 52))
}


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Combination(nn.Module):
    def __init__(self, vgg, pretrained=True, separate=True):
        if pretrained:
            if separate and os.path.exists('./pretrained/SPN.pth') and os.path.exists('./pretrained/TPN.pth') \
                    and os.path.exists('./pretrained/TSAFN.pth'):

                vgg = torch.load('./pretrained/SPN.pth')['vgg']
                self.spn = SPN(vgg)
                self.spn.load_state_dict(torch.load('./pretrained/SPN.pth')['state_dict']).to(device)

                self.tpn = TPN()
                self.tpn.load_state_dict(torch.load('./pretrained/TPN.pth')['state_dict']).to(device)

                self.tsafn = TSAFN()
                self.tsafn.load_state_dict(torch.load('./pretrained/TSAFN.pth')['state_dict']).to(device)

                current_val_loss = torch.load('./pretrained/TSAFN.pth')['val_loss']
                print('Finish loading pre-trained data, current validation loss is {:1.5f}'.format(current_val_loss))
            if not separate and os.path.exists('./pretrained/Combination.pth'):
                vgg = torch.load('./pretrained/Combination.pth')['vgg']
        else:
            self.spn = SPN(vgg)
            self.tpn = TPN()
            self.tsafn = TSAFN()

    def forward(self, x):
        output_tpn = self.tpn(x)
        output_spn = self.spn(x)
        concat = torch.cat((x, output_tpn, output_spn[-1]), 1)  # what really counts to SPN is only the fusion output
        output_tsafn = self.tsafn(concat)

        return output_spn, output_tpn, output_tsafn
