from unet_utils import InConv, DownSamp, UpSamp, OutConv
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        self.inp_conv = InConv(n_channels, 64)
        self.down1 = DownSamp(64, 128)
        self.down2 = DownSamp(128, 256)
        self.down3 = DownSamp(256, 512)
        self.down4 = DownSamp(512, 1024)
        self.up1 = UpSamp(1024, 512)
        self.up2 = UpSamp(512, 256)
        self.up3 = UpSamp(256, 128)
        self.up4 = UpSamp(128, 64)
        self.out_conv = OutConv(64, n_classes)
    
    def forward(self, x):
        x1 = self.inp_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        outputs = self.up1(x5, x4)
        outputs = self.up2(outputs, x3)
        outputs = self.up3(outputs, x2)
        outputs = self.up4(outputs, x1)

        outputs = self.out_conv(outputs)

        return outputs