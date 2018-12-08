import torch
import torch.nn as nn


class GyroscopeNet(nn.Module):
    def __init__(self):
        super(GyroscopeNet, self).__init__()

        self.layer1_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1))
        self.layer1_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.layer1_3 = nn.Sequential(nn.MaxPool2d(2))

        self.layer2_1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        self.layer2_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.layer2_3 = nn.Sequential(nn.MaxPool2d(2))

        self.layer3_1 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        self.layer3_2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.layer3_3 = nn.Sequential(nn.MaxPool2d(2))

        self.layer4_1 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1))
        self.layer4_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.layer4_3 = nn.Sequential(nn.MaxPool2d(2))

        self.layer5_1 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1))
        self.layer5_2 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=3, padding=1))

        self.layer6_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'))
        self.layer6_2 = nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1))
        self.layer6_3 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1))

        self.layer7_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'))
        self.layer7_2 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1))
        self.layer7_3 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1))

        self.layer8_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'))
        self.layer8_2 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1))
        self.layer8_3 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1))

        self.layer9_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'))
        self.layer9_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1))
        self.layer9_3 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1))

        self.conv = nn.Sequential(nn.Conv2d(64, 3, kernel_size=1))
        self.layer10 = nn.Sequential(nn.Tanh())





    def forward(self, x):
        x1 = self.layer1_1(x) # 64*128*128
        x2 = self.layer1_2(x1) # 64*128*128
        x3 = self.layer1_3(x2) # 64*64*64
        x4 = self.layer2_1(x3) # 128*64*64
        x5 = self.layer2_2(x4) # 128*64*64
        x6 = self.layer2_3(x5) # 128*32*32
        x7 = self.layer3_1(x6) # 256*32*32
        x8 = self.layer3_2(x7) # 256*32*32
        x9 = self.layer3_3(x8) # 256*16*16
        x10 = self.layer4_1(x9) # 512*16*16
        x11 = self.layer4_2(x10) # 512*16*16
        x12 = self.layer4_3(x11) # 512*8*8
        x13 = self.layer5_1(x12) # 1024*8*8
        x14 = self.layer5_2(x13) # 1024*8*8

        x15 = self.layer6_1(x14) # 1024*16*16
        x16 = self.layer6_2(x15) # 512*16*16
        x17 = self.layer6_3(x16+x10) # 512*16*16
        x18 = self.layer7_1(x17) # 512*32*32
        x19 = self.layer7_2(x18) # 256*32*32
        x20 = self.layer7_3(x19+x7) # 256*32*32
        x21 = self.layer8_1(x20) # 256*64*64
        x22 = self.layer8_2(x21) # 128*64*64
        x23 = self.layer8_3(x22+x4) # 128*64*64
        x24 = self.layer9_1(x23) # 128*128*128
        x25 = self.layer9_2(x24) # 64*128*128
        x26 = self.layer9_3(x25+x1) # 64*128*128

        out = self.conv(x26)
        out = self.layer10(out)
        return out




