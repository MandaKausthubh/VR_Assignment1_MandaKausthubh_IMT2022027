import torch 
import torchvision
import torch.nn as nn

class ContractionConvBlock(nn.Module):

    def __init__(self, channels) -> None:
        super().__init__()
        in_channels = out_channels = channels
        padding, kernel_size, stride = 0, 3, 1
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.Conv2 = nn.Conv2d(in_channels, 2*out_channels, kernel_size, stride, padding)
        self.ReLU = nn.ReLU()
        self.MaxPool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        return self.MaxPool(self.ReLU(self.Conv2(self.ReLU(self.Conv1(x)))))



class ExpansionConvBlock(nn.Module):

    def __init__(self, channels) -> None:
        super().__init__()
        in_channels = out_channels = channels
        padding, kernel_size, stride = 2, 3, 1
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.Conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, skip_feature):
        return (self.Conv2(self.Conv1(x)))



