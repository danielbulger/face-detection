from collections import OrderedDict

from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Softmax


class PNet(Module):

    def __init__(self):
        super(PNet, self).__init__()
        self.model = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1)),
            ('relu1', ReLU()),
            ('pool1', MaxPool2d(kernel_size=2, stride=2)),

            ('conv2', Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1)),
            ('relu2', ReLU()),

            ('conv3', Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)),
            ('relu3', ReLU())
        ]))

        self.classification = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1)),
            ('softmax', Softmax(dim=3))
        ]))

        self.regression = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1))
        ]))

    def forward(self, x):
        x = self.model(x)

        return {
            'output': x,
            'classification': self.classification(x),
            'regression': self.regression(x)
        }
