import torch.nn as nn
from spectral import SpectralNorm


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2)
#         )
#
#         self.linear = nn.Sequential(
#             nn.Linear(8192, 100),
#             nn.LeakyReLU(0.2),
#             nn.Linear(100, 1)
#         )
#
#     def forward(self, x):
#         out = self.net(x)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1 = nn.Conv2d(64, 64, 4, 2, 1)
        # 64, 64
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 128, 4, 2, 1)
        # 32, 128
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 256, 4, 2, 1)
        # 16, 256
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 512, 4, 2, 1)
        # 8, 512
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv9 = nn.Conv2d(512, 512, 4, 2, 1)
        # 4, 512

        # classifier
        self.linear0 = nn.Linear(512 * 4 * 4, 100)
        self.linear1 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x
