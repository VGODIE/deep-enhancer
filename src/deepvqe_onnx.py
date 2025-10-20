"""
ONNX-compatible version of DeepVQE models

This module provides ONNX-exportable versions of DeepVQE and DeepVQE_S models.
All einops operations have been replaced with native PyTorch operations for
full ONNX compatibility.

Key differences from deepvqe.py:
- All einops.rearrange calls replaced with torch operations (view, permute, reshape)
- Otherwise identical architecture and behavior
- Can load weights from regular DeepVQE checkpoints
"""
import torch
import torch.nn as nn
import numpy as np


class FE(nn.Module):
    """Feature extraction - ONNX compatible"""
    def __init__(self, c=0.3):
        super().__init__()
        self.c = c

    def forward(self, x):
        """x: (B,F,T,2)"""
        x_mag = torch.sqrt(x[..., [0]]**2 + x[..., [1]]**2 + 1e-12)
        x_c = torch.div(x, x_mag.pow(1-self.c) + 1e-12)
        return x_c.permute(0, 3, 2, 1).contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])
        self.conv = nn.Conv2d(channels, channels, kernel_size=(4, 3))
        self.bn = nn.BatchNorm2d(channels)
        self.elu = nn.ELU()

    def forward(self, x):
        """x: (B,C,T,F)"""
        y = self.elu(self.bn(self.conv(self.pad(x))))
        return y + x


class AlignBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, delay=100, in_channels_ref=None):
        super().__init__()
        if in_channels_ref is None:
            in_channels_ref = in_channels

        self.pconv_mic = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pconv_ref = nn.Conv2d(in_channels_ref, hidden_channels, 1)
        self.unfold = nn.Sequential(
            nn.ZeroPad2d([0, 0, delay-1, 0]),
            nn.Unfold((delay, 1))
        )
        self.conv = nn.Sequential(
            nn.ZeroPad2d([1, 1, 4, 0]),
            nn.Conv2d(hidden_channels, 1, (5, 3))
        )
        self.in_channels_ref = in_channels_ref
        self.delay = delay

    def forward(self, x_mic, x_ref):
        """
        x_mic: (B,C_mic,T,F)
        x_ref: (B,C_ref,T,F)
        """
        Q = self.pconv_mic(x_mic)  # (B,H,T,F)
        K = self.pconv_ref(x_ref)  # (B,H,T,F)

        Ku = self.unfold(K)  # (B, H*D, T*F)
        # Reshape: (B, H*D, T*F) -> (B, H, D, T, F)
        B, _, TF = Ku.shape
        T, F = K.shape[2], K.shape[3]
        Ku = Ku.view(B, K.shape[1], self.delay, T, F)
        # Permute to (B, H, T, D, F)
        Ku = Ku.permute(0, 1, 3, 2, 4).contiguous()

        V = torch.sum(Q.unsqueeze(-2) * Ku, dim=-1)  # (B,H,T,D)
        V = self.conv(V)  # (B,1,T,D)
        A = torch.softmax(V, dim=-1).unsqueeze(-1)  # (B,1,T,D,1)

        # Unfold x_ref
        y = self.unfold(x_ref)  # (B, C*D, T*F)
        # Reshape: (B, C*D, T*F) -> (B, C, D, T, F)
        y = y.view(B, x_ref.shape[1], self.delay, T, F)
        # Permute to (B, C, T, D, F)
        y = y.permute(0, 1, 3, 2, 4).contiguous()

        y = torch.sum(y * A, dim=-2)  # (B,C,T,F)
        return y


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 3), stride=(1, 2), use_residual=True):
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.use_residual = use_residual
        if use_residual:
            self.resblock = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.elu(self.bn(self.conv(self.pad(x))))
        if self.use_residual:
            x = self.resblock(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """x : (B,C,T,F)"""
        B, C, T, F = x.shape
        # Reshape: (B,C,T,F) -> (B,T,C*F)
        y = x.permute(0, 2, 1, 3).contiguous()
        y = y.view(B, T, C * F)

        y = self.gru(y)[0]
        y = self.fc(y)

        # Reshape back: (B,T,C*F) -> (B,C,T,F)
        y = y.view(B, T, C, F)
        y = y.permute(0, 2, 1, 3).contiguous()
        return y


class SubpixelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 3)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size)
        self.out_channels = out_channels

    def forward(self, x):
        y = self.conv(self.pad(x))
        B, C, T, F = y.shape
        # Reshape: (B, r*c, T, F) -> (B, c, T, r*F) where r=2
        y = y.view(B, 2, self.out_channels, T, F)
        y = y.permute(0, 2, 3, 1, 4).contiguous()
        y = y.view(B, self.out_channels, T, 2 * F)
        return y


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 3), is_last=False, use_residual=True):
        super().__init__()
        self.skip_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.use_residual = use_residual
        if use_residual:
            self.resblock = ResidualBlock(in_channels)
        self.deconv = SubpixelConv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.is_last = is_last

    def forward(self, x, x_en):
        y = x + self.skip_conv(x_en)
        if self.use_residual:
            y = self.resblock(y)
        y = self.deconv(y)
        if not self.is_last:
            y = self.elu(self.bn(y))
        return y


class CCM(nn.Module):
    """Complex convolving mask block - ONNX compatible"""
    def __init__(self):
        super().__init__()
        v = torch.tensor([[1,        -1/2,           -1/2],
                          [0, np.sqrt(3)/2, -np.sqrt(3)/2]], dtype=torch.float32)
        self.register_buffer('v', v)
        self.unfold = nn.Sequential(
            nn.ZeroPad2d([1, 1, 2, 0]),
            nn.Unfold(kernel_size=(3, 3))
        )

    def forward(self, m, x):
        """
        m: (B,27,T,F)
        x: (B,F,T,2)
        """
        B, C, T, F = m.shape
        # Reshape: (B, r*c, T, F) -> (B, r, c, T, F) where r=3
        m = m.view(B, 3, C // 3, T, F)

        # Apply v transformation
        H_real = torch.sum(self.v[0].view(1, 3, 1, 1, 1) * m, dim=1)  # (B, C/3, T, F)
        H_imag = torch.sum(self.v[1].view(1, 3, 1, 1, 1) * m, dim=1)  # (B, C/3, T, F)

        # Reshape H_real and H_imag: (B, m*n, T, F) -> (B, m, n, T, F) where m=3, n=3
        M_real = H_real.view(B, 3, 3, T, F)
        M_imag = H_imag.view(B, 3, 3, T, F)

        # Process input x
        x = x.permute(0, 3, 2, 1).contiguous()  # (B,2,T,F)
        x_unfold = self.unfold(x)  # (B, 2*3*3, T*F)

        # Reshape: (B, c*m*n, T*F) -> (B, c, m, n, T, F)
        x_unfold = x_unfold.view(B, 2, 3, 3, T, F)

        # Complex multiplication
        x_enh_real = torch.sum(M_real * x_unfold[:, 0] - M_imag * x_unfold[:, 1], dim=(1, 2))  # (B,T,F)
        x_enh_imag = torch.sum(M_real * x_unfold[:, 1] + M_imag * x_unfold[:, 0], dim=(1, 2))  # (B,T,F)

        # Stack and rearrange
        x_enh = torch.stack([x_enh_real, x_enh_imag], dim=3)  # (B,T,F,2)
        x_enh = x_enh.transpose(1, 2).contiguous()  # (B,F,T,2)
        return x_enh


class DeepVQE_S_ONNX(nn.Module):
    """
    ONNX-compatible version of DeepVQE-S

    All einops operations replaced with native PyTorch operations.
    Can load weights from standard DeepVQE_S checkpoints.
    """
    def __init__(self):
        super().__init__()
        # Feature extraction
        self.fe = FE()

        # Microphone encoder: 16 -> 40 -> 56 -> 24 (NO residual blocks)
        self.enblock1 = EncoderBlock(2, 16, use_residual=False)
        self.enblock2 = EncoderBlock(16, 40, use_residual=False)
        self.enblock3 = EncoderBlock(40, 56, use_residual=False)
        self.enblock4 = EncoderBlock(56, 24, use_residual=False)

        # System/Far-end encoder: 8 -> 24 (NO residual blocks)
        self.system_enblock1 = EncoderBlock(2, 8, use_residual=False)
        self.system_enblock2 = EncoderBlock(8, 24, use_residual=False)

        # Alignment block
        self.align = AlignBlock(
            in_channels=40,
            hidden_channels=32,
            in_channels_ref=24
        )
        self.enblock3_actual = EncoderBlock(64, 56, use_residual=False)
        self.bottle = Bottleneck(24 * 17, 24 * 8)

        # Decoder
        self.deblock4 = DecoderBlock(24, 56, use_residual=False)
        self.deblock3 = DecoderBlock(56, 40, use_residual=True)
        self.deblock2 = DecoderBlock(40, 16, use_residual=True)
        self.deblock1 = DecoderBlock(16, 27, use_residual=False)

        # Complex Convolving Mask
        self.ccm = CCM()

    def forward(self, x_mic, x_system):
        """
        Args:
            x_mic: (B,F,T,2) - noisy microphone signal (complex STFT)
            x_system: (B,F,T,2) - system reference signal (complex STFT)

        Returns:
            x_enh: (B,F,T,2) - enhanced signal (complex STFT)
        """
        x_mic_orig = x_mic

        x_mic = self.fe(x_mic)
        x_system = self.fe(x_system)

        # Microphone path
        en_x_mic1 = self.enblock1(x_mic)
        en_x_mic2 = self.enblock2(en_x_mic1)

        # System path
        en_x_system1 = self.system_enblock1(x_system)
        en_x_system2 = self.system_enblock2(en_x_system1)

        # Align and concatenate
        aligned = self.align(en_x_mic2, en_x_system2)
        en_x_mic_n_system_aligned = torch.cat([en_x_mic2, aligned], dim=1)

        # Continue encoding
        en_x3 = self.enblock3_actual(en_x_mic_n_system_aligned)
        en_x4 = self.enblock4(en_x3)

        # Bottleneck
        en_xr = self.bottle(en_x4)

        # Decoder with skip connections
        de_x4 = self.deblock4(en_xr, en_x4)[..., :en_x3.shape[-1]]
        de_x3 = self.deblock3(de_x4, en_x3)[..., :en_x_mic2.shape[-1]]
        de_x2 = self.deblock2(de_x3, en_x_mic2)[..., :en_x_mic1.shape[-1]]
        de_x1 = self.deblock1(de_x2, en_x_mic1)[..., :x_mic.shape[-1]]

        # Complex convolving mask
        x_enh = self.ccm(de_x1, x_mic_orig)

        return x_enh


class DeepVQE_ONNX(nn.Module):
    """
    ONNX-compatible version of DeepVQE (full model)

    All einops operations replaced with native PyTorch operations.
    Can load weights from standard DeepVQE checkpoints.
    """
    def __init__(self):
        super().__init__()
        self.fe = FE()
        self.enblock1 = EncoderBlock(2, 64)
        self.enblock2 = EncoderBlock(64, 128)
        self.enblock3 = EncoderBlock(256, 128)
        self.system_enblock1 = EncoderBlock(2, 32)
        self.system_enblock2 = EncoderBlock(32, 128)
        self.align = AlignBlock(128, 128)
        self.enblock4 = EncoderBlock(128, 128)
        self.enblock5 = EncoderBlock(128, 128)

        self.bottle = Bottleneck(128 * 9, 64 * 9)

        self.deblock5 = DecoderBlock(128, 128)
        self.deblock4 = DecoderBlock(128, 128)
        self.deblock3 = DecoderBlock(128, 128)
        self.deblock2 = DecoderBlock(128, 64)
        self.deblock1 = DecoderBlock(64, 27)
        self.ccm = CCM()

    def forward(self, x_mic, x_system):
        """
        Args:
            x_mic: (B,F,T,2) - noisy microphone signal (complex STFT)
            x_system: (B,F,T,2) - system reference signal (complex STFT)

        Returns:
            x_enh: (B,F,T,2) - enhanced signal (complex STFT)
        """
        x_mic_orig = x_mic

        x_mic = self.fe(x_mic)
        x_system = self.fe(x_system)

        en_x_mic1 = self.enblock1(x_mic)
        en_x_mic2 = self.enblock2(en_x_mic1)
        en_x_system1 = self.system_enblock1(x_system)
        en_x_system2 = self.system_enblock2(en_x_system1)
        aligned = self.align(en_x_mic2, en_x_system2)
        en_x_mic_n_system_aligned = torch.cat([en_x_mic2, aligned], dim=1)

        en_x3 = self.enblock3(en_x_mic_n_system_aligned)
        en_x4 = self.enblock4(en_x3)
        en_x5 = self.enblock5(en_x4)

        en_xr = self.bottle(en_x5)

        de_x5 = self.deblock5(en_xr, en_x5)[..., :en_x4.shape[-1]]
        de_x4 = self.deblock4(de_x5, en_x4)[..., :en_x3.shape[-1]]
        de_x3 = self.deblock3(de_x4, en_x3)[..., :en_x_mic2.shape[-1]]
        de_x2 = self.deblock2(de_x3, en_x_mic2)[..., :en_x_mic1.shape[-1]]
        de_x1 = self.deblock1(de_x2, en_x_mic1)[..., :x_mic.shape[-1]]

        x_enh = self.ccm(de_x1, x_mic_orig)

        return x_enh


if __name__ == "__main__":
    print("=" * 80)
    print("Testing DeepVQE_S_ONNX")
    print("=" * 80)

    device = torch.device("cpu")
    model = DeepVQE_S_ONNX().eval().to(device)

    # Test with typical input shape for 1 second of audio at 24kHz
    # n_fft=512, hop_length=128 -> ~188 frames per second
    x_mic = torch.randn(1, 257, 63, 2).to(device)
    x_system = torch.randn(1, 257, 63, 2).to(device)

    y = model(x_mic, x_system)
    print(f"Input shape: {x_mic.shape}, Output shape: {y.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\n" + "=" * 80)
    print("ONNX Export Test")
    print("=" * 80)

    try:
        import io
        # Test ONNX export
        torch.onnx.export(
            model,
            (x_mic, x_system),
            io.BytesIO(),
            input_names=['noisy_mic', 'farend_ref'],
            output_names=['enhanced'],
            dynamic_axes={
                'noisy_mic': {2: 'time'},
                'farend_ref': {2: 'time'},
                'enhanced': {2: 'time'}
            },
            opset_version=17,
            do_constant_folding=True
        )
        print("✓ ONNX export successful!")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
