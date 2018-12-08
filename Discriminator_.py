import torch.nn as nn
import torch.nn.functional as F
from spectral import SpectralNorm

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(512, 3, kernel_size=4, stride=1, padding=0))

        )

    def forward(self, x):
        out = self.net(x)
        return out
