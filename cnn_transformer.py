import math
import torch
import torch.nn as nn


class EMGConvBlock(nn.Module):
    """
    Multi-scale temporal convolutional block for EMG preprocessing.

    This block applies three parallel depthwise 1D convolutions with different
    kernel sizes to capture temporal patterns at multiple time scales.
    The filtered signals are averaged, rectified (absolute value),
    and smoothed with a low-pass convolution to compute the EMG envelope.

    Input:
        x : torch.Tensor of shape (B, C, T)
            B - batch size (number of trials)
            C - number of EMG channels
            T - number of time samples

    Output:
        feat : torch.Tensor of shape (B, C, T)
            Smoothed EMG envelope after multi-scale temporal filtering.

    Processing steps:
        1. Depthwise temporal convolutions (per-channel, no cross-channel mixing)
           with kernel sizes k1, k2, k3.
        2. Averaging of the three filtered signals (multi-scale fusion).
        3. Rectification via absolute value.
        4. Low-pass filtering to obtain the amplitude envelope.

    Notes:
        - Convolutions use padding = kernel_size // 2 to preserve time length.
        - groups=C ensures independent filtering of each channel.
        - Designed for EMG feature extraction prior to classification.
    """

    def __init__(
        self,
        n_channels: int,
        k1: int = 21,
        k2: int = 51,
        k3: int = 71,
        env_size: int = 101,
    ):

        super().__init__()

        assert k1 % 2 == 1 and k2 % 2 == 1 and k3 % 2 == 1

        C = n_channels

        self.conv1 = nn.Conv1d(
            C, C, kernel_size=k1, padding=k1 // 2, stride=1, groups=C, bias=False
        )
        self.conv2 = nn.Conv1d(
            C, C, kernel_size=k2, padding=k2 // 2, stride=1, groups=C, bias=False
        )
        self.conv3 = nn.Conv1d(
            C, C, kernel_size=k3, padding=k3 // 2, stride=1, groups=C, bias=False
        )
        self.env_lp = nn.Conv1d(
            C, C, kernel_size=env_size, padding=env_size // 2, groups=C, bias=False
        )

        # optional: careful initialization
        for m in (self.conv1, self.conv2, self.conv3, self.env_lp):
            nn.init.normal_(m.weight, mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)

        y = (y1 + y2 + y3) / 3  # (B, C, T)

        env = torch.abs(y)
        feat = self.env_lp(env)

        return feat  # (B, C, T)


class ChannelCNN(nn.Module):
    """Pointwise mixing (C->D) + depthwise temporal conv (optional downsampling)."""

    def __init__(
        self,
        in_ch: int,
        d_model: int,
        k: int = 7,
        stride: int = 2,
        dropout: float = 0.0,
    ):

        super().__init__()

        assert k % 2 == 1

        self.pw = nn.Conv1d(in_ch, d_model, kernel_size=1, bias=False)  # mix channels
        self.dw = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=k,
            padding=k // 2,  # сохраняем края
            stride=stride,  # stride=2 -> T примерно пополам
            groups=d_model,
            bias=False,
        )
        self.norm = nn.GroupNorm(1, d_model)
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        z = self.pw(x)  # (B, D, T)
        z = self.dw(z)  # (B, D, T') where T' ~ T/stride
        z = self.drop(self.act(self.norm(z)))
        return z


class EMGAugment(nn.Module):
    """Simple EMG augmentation for x: (B,C,T). Applied only during training."""

    def __init__(
        self,
        p_noise=0.5,
        noise_std=0.02,
        p_gain=0.5,
        gain_std=0.05,
        p_shift=0.3,
        max_shift=50,
        p_ch_drop=0.2,
        ch_drop_rate=0.125,
        p_tmask=0.2,
        tmask=60,
    ):

        super().__init__()

        self.p_noise = p_noise
        self.noise_std = noise_std
        self.p_gain = p_gain
        self.gain_std = gain_std
        self.p_shift = p_shift
        self.max_shift = int(max_shift)
        self.p_ch_drop = p_ch_drop
        self.ch_drop_rate = ch_drop_rate
        self.p_tmask = p_tmask
        self.tmask = int(tmask)

    def forward(self, x):
        # x: (B,C,T)
        if not self.training:
            return x
        B, C, T = x.shape
        out = x

        if self.p_noise and torch.rand(()) < self.p_noise:
            out = out + self.noise_std * torch.randn_like(out)

        if self.p_gain and torch.rand(()) < self.p_gain:
            gain = 1.0 + self.gain_std * torch.randn(
                B, C, 1, device=out.device, dtype=out.dtype
            )
            out = out * gain

        if self.p_shift and self.max_shift > 0 and torch.rand(()) < self.p_shift:
            shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
            out = torch.roll(out, shifts=shift, dims=-1)

        if self.p_ch_drop and torch.rand(()) < self.p_ch_drop:
            drop = torch.rand(B, C, 1, device=out.device) < self.ch_drop_rate
            out = out.masked_fill(drop, 0.0)

        if (
            self.p_tmask
            and self.tmask > 0
            and self.tmask < T
            and torch.rand(()) < self.p_tmask
        ):
            t0 = int(torch.randint(0, T - self.tmask, (1,)).item())
            out = out.clone()
            out[..., t0 : t0 + self.tmask] = 0.0

        return out


class SinusoidalPosEnc(nn.Module):
    """Sinusoidal positional encoding for x: (B, T, D). Adds pe and applies dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):

        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        pe = torch.zeros(max_len, d_model)  # (T, D)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(0)  # (1, T, D) for broadcasting
        self.register_buffer("pe", pe)
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        pe = self.pe[:, :T, :]
        x = x + pe
        return self.dropout(x)


class CNNTransformerClassifier(nn.Module):
    def __init__(
        self,
        in_ch,
        n_classes,
        d_model=64,
        n_heads=4,
        ff_dim=256,
        num_layers=2,
        dropout=0.2,
        k_cnn=21,
        stride_cnn=8,
        k1=21,
        k2=51,
        k3=71,
        env_size=101,
    ):

        super().__init__()
        self.augment = EMGAugment()
        self.pre = EMGConvBlock(in_ch, k1=k1, k2=k2, k3=k3, env_size=env_size)
        self.cnn = ChannelCNN(
            in_ch, d_model, k=k_cnn, stride=stride_cnn, dropout=dropout
        )
        self.pos = SinusoidalPosEnc(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.augment(x)  # (B,C,2500)
        x = self.pre(x)  # (B,C,2500)
        z = self.cnn(x)  # (B,D,~312) if stride=8
        z = z.transpose(1, 2)  # (B,~312,D)
        z = self.pos(z)
        z = self.tr(z)
        z = self.norm(z).mean(dim=1)
        return self.cls(z)
