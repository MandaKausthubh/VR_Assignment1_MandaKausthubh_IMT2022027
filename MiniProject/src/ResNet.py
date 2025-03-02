import torch
import torch.nn as nn

class ConvLayerStack(nn.Module):

    def __init__(self, in_c, out_c, kernel, stride, padding):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_c, in_c//4, kernel, stride, padding)
        # self.Conv2 = nn.Conv2D(in_c/4, in_c/4, kernel, stride, padding)
        self.Conv3 = nn.Conv2d(in_c//4, out_c, kernel, stride, padding)
        self.BNorm1 = nn.BatchNorm2d(in_c//4)
        self.BNorm2 = nn.BatchNorm2d(out_c)
        self.Relu = nn.ReLU()

    def forward(self, x):
        f = self.Conv3( self.Relu( self.BNorm1(self.Conv1(x)) ))
        # print(f.shape)
        return self.Relu(self.BNorm2(f + x))


class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv2d(2,64,2,2,0)
        self.Pooling1 = nn.AvgPool2d(1,1)
        self.Res1 = ConvLayerStack(64,64,3,1,1)
        self.Res2 = ConvLayerStack(64,64,3,1,1)
        self.Res3 = ConvLayerStack(64,64,3,1,1)
        self.Res4 = ConvLayerStack(64,64,3,1,1)
        self.Res5 = ConvLayerStack(64,64,3,1,1)
        self.Res6 = ConvLayerStack(64,64,3,1,1)
        self.Pooling2 = nn.AvgPool2d(2,2,0)
        self.Lin = nn.Linear(64*8*8, 2)

    def forward(self,x):
        # print(x.shape)
        f = self.Conv1(x)
        # print(f.shape)
        f = self.Pooling1(f)
        # print(f.shape)
        f = self.Res1(f)
        f = self.Res2(f)
        f = self.Res3(f)
        f = self.Res4(f)
        f = self.Res5(f)
        f = self.Res6(f)
        # print(f.shape)
        f = self.Pooling2(f)
        # print(f.shape)
        f = torch.flatten(f,1)
        # print(f.shape)
        return self.Lin(f)



