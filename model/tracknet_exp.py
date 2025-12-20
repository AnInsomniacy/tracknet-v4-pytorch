import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = torch.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        for _ in range(num_convs - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        self.conv_block = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv_block(x) + self.shortcut(x))


class TrackNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = ResidualBlock(9, 96, 2)
        self.cbam1 = CBAM(96)
        self.encoder2 = ResidualBlock(96, 192, 2)
        self.cbam2 = CBAM(192)
        self.encoder3 = ResidualBlock(192, 384, 3)
        self.cbam3 = CBAM(384)
        self.encoder4 = ResidualBlock(384, 768, 3)
        self.cbam4 = CBAM(768)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.2)

        self.decoder1 = ResidualBlock(768 + 384, 384, 3)
        self.decoder2 = ResidualBlock(384 + 192, 192, 2)
        self.decoder3 = ResidualBlock(192 + 96, 96, 2)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.output = nn.Conv2d(96, 3, 1)

    def forward(self, x):
        enc1 = self.cbam1(self.encoder1(x))
        enc1_pool = self.pool(enc1)

        enc2 = self.cbam2(self.encoder2(enc1_pool))
        enc2_pool = self.pool(enc2)

        enc3 = self.cbam3(self.encoder3(enc2_pool))
        enc3_pool = self.pool(enc3)

        bottleneck = self.cbam4(self.encoder4(enc3_pool))
        bottleneck = self.dropout(bottleneck)

        dec1 = self.upsample(bottleneck)
        dec1 = torch.cat([dec1, enc3], dim=1)
        dec1 = self.decoder1(dec1)

        dec2 = self.upsample(dec1)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)

        dec3 = self.upsample(dec2)
        dec3 = torch.cat([dec3, enc1], dim=1)
        dec3 = self.decoder3(dec3)

        return torch.sigmoid(self.output(dec3))


if __name__ == "__main__":
    model = TrackNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    x = torch.randn(2, 9, 288, 512)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
