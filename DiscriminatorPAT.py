import torch.nn as nn
import torch.nn.functional as F
from spectral import SpectralNorm

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=2),  # 128*128*32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.linear = nn.Sequential(
            nn.Linear(8192, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 1)
        )
    def forward(self, x):
        out1 = self.conv(x)
        out1 = out1.view(out1.size(0), -1)
        out = self.linear(out1)
        # out = F.sigmoid(out)
        return out1
