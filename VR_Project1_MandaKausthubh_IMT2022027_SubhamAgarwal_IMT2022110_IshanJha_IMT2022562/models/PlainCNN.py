import torch
import torch.nn as nn


class ConvNN(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.ConvLayer1 = nn.Conv2d(in_channels, 64, 3, 1, padding=1)
        self.Relu1 = nn.ReLU()
        self.ConvLayer2 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.Relu2 = nn.ReLU()
        self.Pool = nn.AdaptiveMaxPool2d((1,1))
        self.hidden = nn.Linear(128, 1024)
        self.Gelu = nn.GELU()
        self.Final = nn.Linear(1024, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.Relu1(self.ConvLayer1(x))
        x = self.Relu2(self.ConvLayer2(x))
        x = self.Pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.Gelu(self.hidden(x))
        return self.sigmoid(self.Final(x))
