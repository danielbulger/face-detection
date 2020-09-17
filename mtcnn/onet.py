from collections import OrderedDict

import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Softmax, Linear


class ONet(Module):

    def __init__(self):
        super(ONet, self).__init__()

        self.features = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)),
            ('relu1', ReLU()),
            ('pool1', MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ('conv2', Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)),
            ('relu2', ReLU()),
            ('pool2', MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ('conv3', Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            ('relu3', ReLU()),
            ('pool3', MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)),

            ('conv4', Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1)),
            ('relu4', ReLU())
        ]))

        self.flatten = Sequential(OrderedDict([
            ('dense', Linear(1152, 256)),
            ('relu', ReLU())
        ]))

        self.classification = Sequential(OrderedDict([
            ('dense', Linear(256, 2)),
            ('softmax', Softmax(dim=1))
        ]))

        self.regression = Linear(256, 4)

        self.landmarks = Linear(256, 10)

    def forward(self, x):

        x = torch.flatten(self.features(x), start_dim=1)

        x = self.flatten(x)

        return {
            'classification': self.classification(x),
            'regression': self.regression(x),
            'landmarks': self.landmarks(x)
        }
