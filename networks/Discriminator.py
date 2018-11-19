import torch
import torch.nn as nn
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        kw = 4
        padw = int(np.ceil(kw-1)/2)
        sequence = [
            nn.Conv2d(3, 64, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=True),
                nn.BatchNorm2d(64 * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** 3, 8)
        sequence += [
            nn.Conv2d(64 * nf_mult_prev, 64 * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=True),
            nn.BatchNorm2d(64 * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(64 * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        out = self.model(input)
        return out
