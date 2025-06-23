import torch
import torch.nn as nn

class TinyUNet(nn.Module):
    def __init__(self):
        super(TinyUNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(3, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(32, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = conv_block(64, 32)

        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder2 = conv_block(32, 16)

        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d1 = self.up1(b)
        d1 = self.decoder1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.decoder2(torch.cat([d2, e1], dim=1))
        return self.output(d2)
