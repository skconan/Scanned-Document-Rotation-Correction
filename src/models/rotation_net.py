from torch import nn


class Conv(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            Conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class RotationNet(nn.Module):
    def __init__(self, out_channels, n_fc=1, drop_rate=.5, n_channels=1):
        super(RotationNet, self).__init__()
        self.n_channels = n_channels
        self.n_out = out_channels
        self.n_fc = n_fc
        # 128
        self.inc = Conv(n_channels, self.n_out)
        self.down1 = Down(self.n_out, self.n_out*2)
        # 64
        self.down2 = Down(self.n_out*2, self.n_out*4)
        # 32
        self.down3 = Down(self.n_out*4, self.n_out*4)
        # 16
        self.down4 = Down(self.n_out*4, self.n_out*8)
        # 8
        self.down5 = Down(self.n_out*8, self.n_out*16)
        # 4
        self.down6 = Down(self.n_out*16, self.n_out*16)
        # 2
        self.down7 = Down(self.n_out*16, self.n_out*16)
        # 1
        if self.n_fc == 1:
            self.fc_1 = nn.Linear(1024, 1)
        else:
            self.fc_1 = nn.Linear(1024, 512)
            self.bn_1 = nn.BatchNorm1d(512)
            self.fc_2 = nn.Linear(512, 1)

        self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.down6(x)
        x = self.down7(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_1(x)

        if self.n_fc == 2:
            x = self.bn_1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc_2(x)

        return x
