import torch.nn as nn
import torch.nn.functional as F
from spectral import SpectralNorm

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(256, 512, kernel_size=3, padding=1)),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(512, 1, kernel_size=1)),

        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.net(x)
        out = out.view(batch_size, -1)
        # return F.sigmoid(self.net(x).view(batch_size))
        return out
