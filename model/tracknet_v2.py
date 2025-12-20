import torch
import torch.nn as nn


class TrackNet(nn.Module):
    """
    TrackNet MIMO - Badminton tracking network
    Input: [B, 9, 288, 512] -> Output: [B, 3, 288, 512]
    """

    def __init__(self, dropout=0.0):
        super(TrackNet, self).__init__()

        # Encoder - VGG16 style
        self.encoder_block1 = self._make_encoder_block(9, 64, 2)  # 288x512
        self.encoder_block2 = self._make_encoder_block(64, 128, 2)  # 144x256
        self.encoder_block3 = self._make_encoder_block(128, 256, 3)  # 72x128
        self.encoder_block4 = self._make_encoder_block(256, 512, 3)  # 36x64 (bottleneck)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout)

        # Decoder with skip connections
        self.decoder_block1 = self._make_decoder_block(768, 256, 3)  # 512+256
        self.decoder_block2 = self._make_decoder_block(384, 128, 2)  # 256+128
        self.decoder_block3 = self._make_decoder_block(192, 64, 2)  # 128+64

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # MIMO output: 3 heatmaps
        self.output_conv = nn.Conv2d(64, 3, 1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def _make_encoder_block(self, in_channels, out_channels, num_convs):
        """Create encoder block with convolution + batch norm + relu"""
        layers = []

        # First conv
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

        # Additional convs
        for _ in range(num_convs - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])

        return nn.Sequential(*layers)

    def _make_decoder_block(self, in_channels, out_channels, num_convs):
        """Create decoder block with convolution + batch norm + relu"""
        layers = []

        # First conv
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

        # Additional convs
        for _ in range(num_convs - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: [B, 9, 288, 512] - 3 frames × 3 channels
        Returns:
            [B, 3, 288, 512] - 3 heatmaps
        """
        # Encoder - progressive downsampling
        enc1 = self.encoder_block1(x)  # [B, 64, 288, 512]
        enc1_pool = self.pool(enc1)  # [B, 64, 144, 256]

        enc2 = self.encoder_block2(enc1_pool)  # [B, 128, 144, 256]
        enc2_pool = self.pool(enc2)  # [B, 128, 72, 128]

        enc3 = self.encoder_block3(enc2_pool)  # [B, 256, 72, 128]
        enc3_pool = self.pool(enc3)  # [B, 256, 36, 64]

        # Bottleneck
        bottleneck = self.encoder_block4(enc3_pool)  # [B, 512, 36, 64]
        bottleneck = self.dropout(bottleneck)

        # Decoder - progressive upsampling with skip connections
        dec1 = self.upsample(bottleneck)  # [B, 512, 72, 128]
        dec1 = torch.cat([dec1, enc3], dim=1)  # [B, 768, 72, 128]
        dec1 = self.decoder_block1(dec1)  # [B, 256, 72, 128]

        dec2 = self.upsample(dec1)  # [B, 256, 144, 256]
        dec2 = torch.cat([dec2, enc2], dim=1)  # [B, 384, 144, 256]
        dec2 = self.decoder_block2(dec2)  # [B, 128, 144, 256]

        dec3 = self.upsample(dec2)  # [B, 128, 288, 512]
        dec3 = torch.cat([dec3, enc1], dim=1)  # [B, 192, 288, 512]
        dec3 = self.decoder_block3(dec3)  # [B, 64, 288, 512]

        # MIMO output: 3 heatmaps
        output = self.output_conv(dec3)  # [B, 3, 288, 512]
        output = self.sigmoid(output)  # Probability values [0,1]

        return output  # [:, 0]=frame1, [:, 1]=frame2, [:, 2]=frame3


def generate_heatmap(size, center, sigma=5):
    """
    Generate 2D Gaussian heatmap as ground truth
    Args:
        size: (H, W) heatmap dimensions
        center: (x, y) center coordinates
        sigma: Gaussian standard deviation
    Returns:
        heatmap: [H, W] heatmap tensor
    """
    H, W = size
    x, y = center

    # Create coordinate grids
    X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    X = X.float()
    Y = Y.float()

    # Calculate Gaussian distribution
    heatmap = torch.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))

    return heatmap


if __name__ == "__main__":
    # Create model
    model = TrackNet()

    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Test forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, 9, 288, 512)  # 3 frames × 3 channels
    output = model(test_input)

    print(f"\nInput shape: {test_input.shape} (3 frames × 3 channels)")
    print(f"Output shape: {output.shape} (3 heatmaps)")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Create test ground truth heatmaps
    gt_heatmaps = torch.zeros(batch_size, 3, 288, 512)

    # Generate random ball positions for each batch and frame
    for b in range(batch_size):
        for f in range(3):
            # Random ball position
            ball_x = torch.randint(50, 462, (1,)).item()
            ball_y = torch.randint(50, 238, (1,)).item()

            # Generate Gaussian heatmap
            heatmap = generate_heatmap((288, 512), (ball_x, ball_y))
            gt_heatmaps[b, f] = heatmap

    print(f"\nGround truth shape: {gt_heatmaps.shape}")
    print(f"GT range: [{gt_heatmaps.min():.3f}, {gt_heatmaps.max():.3f}]")
