import torch
import torch.nn as nn
import math
import torch.nn.init as init
import os
from MSRN.MSRNV2 import MSRN
from SRN.network64 import SRNDeblurNet


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

class _ResBlockSR(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBlockSR, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, stride, 1, bias=True)
        )
        # for i in self.modules():
        #     if isinstance(i, nn.Conv2d):
        #         j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
        #         i.weight.data.normal_(0, math.sqrt(2 / j))
        #         if i.bias is not None:
        #             i.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = torch.add(residual, out)
        return out


class _ReconstructMoudle(nn.Module):
    def __init__(self):
        super(_ReconstructMoudle, self).__init__()
        self.head = nn.Conv2d(131, 64, kernel_size=3, stride=1, padding=1)
        self.resBlock = self._makelayers(64, 64, 8)
        self.conv1 = nn.Conv2d(64, 256, (3, 3), 1, 1)
        self.pixelShuffle1 = nn.PixelShuffle(2)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(64, 256, (3, 3), 1, 1)
        self.pixelShuffle2 = nn.PixelShuffle(2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1, 1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(64, 3, (3, 3), 1, 1)

        # for i in self.modules():
        #     if isinstance(i, nn.Conv2d):
        #         j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
        #         i.weight.data.normal_(0, math.sqrt(2 / j))
        #         if i.bias is not None:
        #             i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockDB(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)
        res1 = self.resBlock(x)
        con1 = self.conv1(res1)
        pixelshuffle1 = self.relu1(self.pixelShuffle1(con1))
        con2 = self.conv2(pixelshuffle1)
        pixelshuffle2 = self.relu2(self.pixelShuffle2(con2))
        con3 = self.relu3(self.conv3(pixelshuffle2))
        sr_deblur = self.conv4(con3)
        return sr_deblur


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.deblurMoudle      = SRNDeblurNet()
        self.srMoudle          = MSRN()
        self.reconstructMoudle = self._make_net(_ReconstructMoudle)

    def forward(self, x, y, z, gated, isTest):
        if isTest == True:
            origin_size = x.size()
            input_size  = (math.ceil(origin_size[2]/4)*4, math.ceil(origin_size[3]/4)*4)
            out_size    = (origin_size[2]*4, origin_size[3]*4)
            x           = nn.functional.upsample(x, size=input_size, mode='bilinear')

        deblurx32, deblurx16, deblurx8, deblur_feature = self.deblurMoudle(x, y, z)
        sr_feature = self.srMoudle(torch.cat((deblur_feature, x), 1))
        recon = self.reconstructMoudle(torch.cat((deblur_feature, x, sr_feature), 1))


        if isTest == True:
            recon = nn.functional.upsample(recon, size=out_size, mode='bilinear')

        return deblurx32, deblurx16, deblurx8, recon




    def _make_net(self, net):
        nets = []
        nets.append(net())
        return nn.Sequential(*nets)






