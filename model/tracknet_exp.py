import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch, height, width, device):
        return (torch.zeros(batch, self.hidden_dim, height, width, device=device),
                torch.zeros(batch, self.hidden_dim, height, width, device=device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        B, T, C, H, W = x.shape
        state = self.cell.init_hidden(B, H, W, x.device)
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], state)
            state = (h, c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


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
        return x * torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class TrackNet(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.enc1 = self._block(3, 96, 2)
        self.enc2 = self._block(96, 192, 2)
        self.enc3 = self._block(192, 384, 3)
        self.enc4 = self._block(384, 768, 3)

        self.cbam1 = CBAM(96)
        self.cbam2 = CBAM(192)
        self.cbam3 = CBAM(384)
        self.cbam4 = CBAM(768)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout)
        self.convlstm = ConvLSTM(768, 384)

        self.dec1 = self._block(384 + 384, 384, 3)
        self.dec2 = self._block(384 + 192, 192, 2)
        self.dec3 = self._block(192 + 96, 96, 2)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.output = nn.Conv2d(96, 1, 1)

    def _block(self, in_ch, out_ch, n):
        layers = []
        for i in range(n):
            layers.extend([
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def forward(self, x):
        B = x.size(0)
        frames = x.view(B, 3, 3, 288, 512)

        enc1_list, enc2_list, enc3_list, bot_list = [], [], [], []
        for t in range(3):
            frame = frames[:, t]
            e1 = self.cbam1(self.enc1(frame))
            e2 = self.cbam2(self.enc2(self.pool(e1)))
            e3 = self.cbam3(self.enc3(self.pool(e2)))
            e4 = self.cbam4(self.enc4(self.pool(e3)))
            enc1_list.append(e1)
            enc2_list.append(e2)
            enc3_list.append(e3)
            bot_list.append(e4)

        bot_seq = torch.stack(bot_list, dim=1)
        bot_seq = self.dropout(bot_seq.view(B * 3, 768, -1, bot_seq.size(-1))).view(B, 3, 768, -1, bot_seq.size(-1))
        lstm_out = self.convlstm(bot_seq)

        outputs = []
        for t in range(3):
            d1 = self.upsample(lstm_out[:, t])
            d1 = self.dec1(torch.cat([d1, enc3_list[t]], dim=1))
            d2 = self.upsample(d1)
            d2 = self.dec2(torch.cat([d2, enc2_list[t]], dim=1))
            d3 = self.upsample(d2)
            d3 = self.dec3(torch.cat([d3, enc1_list[t]], dim=1))
            out = torch.sigmoid(self.output(d3))
            outputs.append(out)

        return torch.cat(outputs, dim=1)


if __name__ == "__main__":
    model = TrackNet(dropout=0.2)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    x = torch.randn(2, 9, 288, 512)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
