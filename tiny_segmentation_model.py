import torch.nn as nn

class TinyUNet(nn.Module):
    def __init__(self):
        super(TinyUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
