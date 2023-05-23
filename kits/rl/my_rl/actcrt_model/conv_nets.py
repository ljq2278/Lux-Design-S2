""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1, bias=True),
            # nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1, bias=True),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    # @staticmethod
    # def init(m):
    #     if isinstance(m, nn.Conv2d):
    #         torch.nn.init.normal_(m.weight)
    #         torch.nn.init.normal_(m.bias)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class BaseNet(nn.Module):
    def __init__(self, n_channels, base_channel=8):
        super(BaseNet, self).__init__()
        # self.n_channels = n_channels
        self.inc = DoubleConv(n_channels, base_channel)
        self.down1 = Down(base_channel, base_channel * 2)
        self.down2 = Down(base_channel * 2, base_channel * 4)
        self.down3 = Down(base_channel * 4, base_channel * 8)
        self.down4 = Down(base_channel * 8, base_channel * 16)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(torch.argwhere(x1 != 0))
        return x1, x2, x3, x4, x5


class UActNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_net, base_channel=8):
        super(UActNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.base_net = base_net
        self.up1 = Up(base_channel * 16 + base_channel * 8, base_channel * 8)
        self.up2 = Up(base_channel * 8 + base_channel * 4, base_channel * 4)
        self.up3 = Up(base_channel * 4 + base_channel * 2, base_channel * 2)
        self.up4 = Up(base_channel * 2 + base_channel * 1, base_channel * 1)
        self.outc = OutConv(base_channel, n_classes)

    def forward(self, x):
        x1, x2, x3, x4, x = self.base_net(x)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class FActNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_net, base_channel=8):
        super(FActNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.base_net = base_net
        self.up1 = Up(base_channel * 16 + base_channel * 8, base_channel * 8)
        self.up2 = Up(base_channel * 8 + base_channel * 4, base_channel * 4)
        self.up3 = Up(base_channel * 4 + base_channel * 2, base_channel * 2)
        self.up4 = Up(base_channel * 2 + base_channel * 1, base_channel * 1)
        self.outc = OutConv(base_channel, n_classes)

    def forward(self, x):
        x1, x2, x3, x4, x = self.base_net(x)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ValueNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_net, fc_dims, base_channel=8):
        super(ValueNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.fc_dims = fc_dims
        self.base_net = base_net
        self.fc = nn.Linear(fc_dims, 1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.base_net(x)
        ret = self.fc(x5.reshape([-1, self.fc_dims]))
        return ret
