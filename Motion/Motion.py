import torch
import torch.nn as nn
import math
from MSRN.MSRNV2 import MSRN



class _ResBLockDB(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBLockDB, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, stride, 1, bias=True)
        )
        for i in self.modules(): # 初始化卷积核的权重
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = torch.add(residual, out)
        return out


class Motion(nn.Module):
    def __init__(self):
        super(Motion, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        )
        layers = []
        for i in range(9):
            layers.append(_ResBLockDB(128, 128))
        self.body = nn.Sequential(*layers)
        self.tail = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(0.2),
            nn.ConvTranspose2d(128, 128, kernel_size=5, stride=1, padding=2)
        )
        self.out = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        )

        self.deblur = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.head(x)
        x2 = self.body(x1)
        x3 = self.tail(x2)
        deblur_feature = self.out(x1+x3)
        deblur_out = self.deblur(deblur_feature)
        return deblur_feature, deblur_out







class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.deblurmodel = Motion()
        self.srmodel = MSRN()

    def forward(self, x):
        deblurFeature, deblur_out = self.deblurmodel(x)
        sr = self.srmodel(torch.cat((x, deblurFeature), 1))
        return deblur_out, sr



