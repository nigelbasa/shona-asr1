"""QuartzNet-style 1D CNN acoustic model with CTC, optional BiLSTM head."""
import torch
import torch.nn as nn


class SeparableConv(nn.Module):
    """Depthwise-separable 1D conv -- the cheap building block that lets a pure
    CNN see a wide context without an RNN's sequential cost."""

    def __init__(self, cin, cout, k, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.dw = nn.Conv1d(cin, cin, k, stride=stride, padding=pad,
                            dilation=dilation, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, act=True):
        x = self.bn(self.pw(self.dw(x)))
        return self.drop(torch.relu(x)) if act else x


class Block(nn.Module):
    """R repeated separable convs with a residual 1x1 projection."""

    def __init__(self, cin, cout, k, repeat=3, dropout=0.1):
        super().__init__()
        layers = [SeparableConv(cin if i == 0 else cout, cout, k, dropout=dropout)
                  for i in range(repeat)]
        self.convs = nn.ModuleList(layers)
        self.res = nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False),
                                 nn.BatchNorm1d(cout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = x
        for i, c in enumerate(self.convs):
            y = c(y, act=(i < len(self.convs) - 1))
        return self.drop(torch.relu(y + self.res(x)))


class ShonaCNN(nn.Module):
    """log-mel -> strided conv (4x subsample) -> CNN trunk -> [BiLSTM] -> CTC."""

    def __init__(self, n_mels=80, n_classes=30, width=256, dropout=0.1,
                 lstm_layers=0, lstm_hidden=320):
        super().__init__()
        self.subsample = nn.Sequential(
            SeparableConv(n_mels, width, 11, stride=2, dropout=dropout),
            SeparableConv(width, width, 11, stride=2, dropout=dropout),
        )
        self.trunk = nn.Sequential(
            Block(width, width, 13, dropout=dropout),
            Block(width, width, 15, dropout=dropout),
            Block(width, width, 17, dropout=dropout),
            Block(width, width * 2, 21, dropout=dropout),
            Block(width * 2, width * 2, 25, dropout=dropout),
        )
        self.head_conv = SeparableConv(width * 2, width * 2, 29, dilation=2,
                                       dropout=dropout)
        self.lstm = None
        feat = width * 2
        if lstm_layers:
            self.lstm = nn.LSTM(feat, lstm_hidden, lstm_layers, batch_first=True,
                                bidirectional=True, dropout=dropout if lstm_layers > 1 else 0)
            feat = lstm_hidden * 2
        self.out = nn.Conv1d(feat, n_classes, 1)
        self.subsample_factor = 4

    def output_lengths(self, in_lengths):
        return torch.div(in_lengths + 3, 4, rounding_mode="floor")

    def forward(self, x, lengths):
        """x: (B, n_mels, T) -> log-probs (T', B, C) for nn.CTCLoss."""
        x = self.head_conv(self.trunk(self.subsample(x)))
        if self.lstm is not None:
            x, _ = self.lstm(x.transpose(1, 2))
            x = x.transpose(1, 2)
        x = self.out(x)                                  # (B, C, T')
        logp = x.log_softmax(1).permute(2, 0, 1)         # (T', B, C)
        return logp, self.output_lengths(lengths).clamp(max=x.shape[2])
