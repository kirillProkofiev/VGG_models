''' Implementation of VGG class Neural Network for cifar10'''
import torch
import numpy as np
import torch.nn as nn

# architecture is looked like: image -> 64*64, M, 128*128, M, 256*256*256, M, 512*512*512, M, FC, FC, FC
VGG_types = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512],
}


class VGG(torch.nn.Module):
    def __init__(self, VGG_type='A', in_channels=3, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.average_pool = nn.AvgPool2d(kernel_size=2)
        self.conv_layers = self.create_conv_layers(VGG_types[VGG_type])
        # FC layers
        self.fcs = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.average_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)

        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                                    kernel_size=3, stride=(1,1), padding=(1,1)),
                                    nn.BatchNorm2d(x), 
                                    nn.ReLU(inplace=True)]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VGG(VGG_type='A', in_channels=3, num_classes=10).to(device)
    x = torch.randn(1, 3, 32, 32).to(device)
    print(model(x).shape, device)