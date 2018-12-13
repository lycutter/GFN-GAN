import torch.nn as nn
import torch.nn.functional as F
from spectral import SpectralNorm

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),  # CR1
            nn.ReLU(0.2),

            SpectralNorm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)), # CR2
            # nn.BatchNorm2d(64),
            nn.ReLU(0.2),

            nn.MaxPool2d(2),                                                     # M3

            SpectralNorm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)), # CR4
            # nn.BatchNorm2d(64),
            nn.ReLU(0.2),

            nn.MaxPool2d(2),                                                     # M5

            SpectralNorm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)), # CR6
            # nn.BatchNorm2d(64),
            nn.ReLU(0.2),

            nn.MaxPool2d(2),                                                     # M7

            SpectralNorm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)), # CR8
            nn.ReLU(0.2),

            SpectralNorm(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)),  # CR9


            nn.AdaptiveAvgPool2d(16)

        )

    def forward(self, x):
        out = self.net(x)
        return out
