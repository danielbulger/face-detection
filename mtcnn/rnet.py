from collections import OrderedDict
import torch

from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Linear, Softmax


class RNet(Module):

    def __init__(self):
        super(RNet, self).__init__()
        self.features = Sequential(OrderedDict([
            ('conv1', Conv2d(3, 28, kernel_size=3, stride=1)),
            ('relu1', ReLU()),
            ('pool1', MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ('conv2', Conv2d(28, 64, kernel_size=2, stride=1)),
            ('relu2', ReLU()),
            ('pool2', MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))

        self.flat = Sequential(OrderedDict([
            ('dense', Linear(1600, 128)),
            ('relu', ReLU())
        ]))

        self.classification = Sequential(OrderedDict([
            ('dense', Linear(128, 2)),
            ('softmax', Softmax(dim=1)),
        ]))

        self.regression = Linear(128, 4)

    def forward(self, x):
        output = torch.flatten(self.features(x), start_dim=1)

        flat = self.flat(output)

        return {
            'classification': self.classification(flat),
            'regression': self.regression(flat)
        }
