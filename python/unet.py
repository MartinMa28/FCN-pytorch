from unet_utils import InConv, DownSamp, UpSamp, OutConv
import torch
import torch.nn as nn
import torch.optim as optim

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
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

class UNetWithBilinear(nn.Module):
    # Considering the limited memory, bilinear interpolation substitutes
    # for transposed-conv in the last 2 upsample layers
    def __init__(self, n_channels, n_classes):
        super(UNetWithBilinear, self).__init__()
        self.inp_conv = InConv(n_channels, 64)
        self.down1 = DownSamp(64, 128)
        self.down2 = DownSamp(128, 256)
        self.down3 = DownSamp(256, 512)
        self.down4 = DownSamp(512, 1024)
        self.up1 = UpSamp(1024, 512)
        self.up2 = UpSamp(512, 256)
        self.up3 = UpSamp(256, 128, bilinear=True)
        self.up4 = UpSamp(128, 64, bilinear=True)
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

if __name__ == "__main__":
    # dimension-match tests
    batch_size, n_classes, height, width = 4, 21, 224, 224
    inputs = torch.autograd.Variable(torch.randn(batch_size, 3, height, width))
    unet = UNet(3, n_classes)
    outputs = unet(inputs)
    assert outputs.size() == torch.Size([batch_size, n_classes, height, width])
    # inputs = torch.autograd.Variable(torch.randn(batch_size, 3, height, width))
    # unet = UNetWithBilinear(3, n_classes)
    # outputs = unet(inputs)
    # assert outputs.size() == torch.Size([batch_size, n_classes, height, width])
    print('pass the dimension check')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(unet.parameters(), lr=1e-4)
    inputs = torch.autograd.Variable(torch.randn(batch_size, 3, height, width))
    targets = torch.autograd.Variable(torch.randint(low=0, high=21, size=(batch_size, height, width)), requires_grad=False)

    for iter in range(10):
        optimizer.zero_grad()
        outputs = unet(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        print('iter{}, loss = {}'.format(iter, loss.data.item()))
        optimizer.step()


