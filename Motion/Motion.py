import torch
import torch.nn as nn
import math



class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # 1*1卷积，用来减少通道数
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # 1*1卷积，用来减少通道数
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

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
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
            # nn.Tanh()
        )

    def forward(self, x):
        x1 = self.head(x)
        x2 = self.body(x1)
        x3 = self.tail(x2)
        out = self.out(x1+x3)
        return out



class _ReconstructMoudle(nn.Module):
    def __init__(self):
        super(_ReconstructMoudle, self).__init__()
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
        self.attn = Self_Attn(64, 'relu')

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockDB(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        res1 = self.resBlock(x)
        con1 = self.conv1(res1 + x)
        pixelshuffle1 = self.relu1(self.pixelShuffle1(con1))
        # pixelshuffle1, p1 = self.attn(pixelshuffle1)
        con2 = self.conv2(pixelshuffle1)
        pixelshuffle2 = self.relu2(self.pixelShuffle2(con2))
        con3 = self.relu3(self.conv3(pixelshuffle2))
        sr_deblur = self.conv4(con3)
        return sr_deblur

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.deblurmodel = Motion()
        self.reconstructMoudle = _ReconstructMoudle()

    def forward(self, x):
        deblurFeature = self.deblurmodel(x)
        sr = self.reconstructMoudle(deblurFeature)
        return sr



