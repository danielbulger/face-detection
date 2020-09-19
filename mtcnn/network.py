from collections import OrderedDict

from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Softmax, Linear


class Flatten(Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class PNet(Module):

    def __init__(self):
        super(PNet, self).__init__()
        self.features = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1)),
            ('relu1', ReLU()),
            ('pool1', MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)),

            ('conv2', Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1)),
            ('relu2', ReLU()),

            ('conv3', Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)),
            ('relu3', ReLU())
        ]))

        self.classification = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1)),
            ('softmax', Softmax(dim=1))
        ]))

        self.regression = Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.features(x)

        return {
            'classification': self.classification(x),
            'regression': self.regression(x)
        }


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

            ('flatten', Flatten()),
            ('dense', Linear(1600, 128)),
            ('relu', ReLU())
        ]))

        self.classification = Sequential(OrderedDict([
            ('dense', Linear(128, 2)),
            ('softmax', Softmax(dim=1)),
        ]))

        self.regression = Linear(128, 4)

    def forward(self, x):
        output = self.features(x)

        return {
            'classification': self.classification(output),
            'regression': self.regression(output)
        }


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
            ('relu4', ReLU()),

            ('flatten', Flatten()),
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
        x = self.features(x)

        return {
            'classification': self.classification(x),
            'regression': self.regression(x),
            'landmarks': self.landmarks(x)
        }
