import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class FE(nn.Module):
    """Feature extraction"""
    def __init__(self, c=0.3):
        super().__init__()
        self.c = c
    def forward(self, x):
        """x: (B,F,T,2)"""
        x_mag = torch.sqrt(x[...,[0]]**2 + x[...,[1]]**2 + 1e-12)
        x_c = torch.div(x, x_mag.pow(1-self.c) + 1e-12)
        return x_c.permute(0,3,2,1).contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
        self.conv = nn.Conv2d(channels, channels, kernel_size=(4,3))
        self.bn = nn.BatchNorm2d(channels)
        self.elu = nn.ELU()
    def forward(self, x):
        """x: (B,C,T,F)"""
        y = self.elu(self.bn(self.conv(self.pad(x))))
        return y + x
    
        
class AlignBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, delay=100, in_channels_ref=None):
        super().__init__()
        # Support different channel counts for mic and ref
        if in_channels_ref is None:
            in_channels_ref = in_channels

        self.pconv_mic = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pconv_ref = nn.Conv2d(in_channels_ref, hidden_channels, 1)
        self.unfold = nn.Sequential(nn.ZeroPad2d([0,0,delay-1,0]),
                                    nn.Unfold((delay, 1)))
        self.conv = nn.Sequential(nn.ZeroPad2d([1,1,4,0]),
                                  nn.Conv2d(hidden_channels, 1, (5,3)))
        self.in_channels_ref = in_channels_ref


    def forward(self, x_mic, x_ref):
        """
        x_mic: (B,C_mic,T,F)
        x_ref: (B,C_ref,T,F) - can have different channel count
        """
        Q = self.pconv_mic(x_mic)  # (B,H,T,F)
        K = self.pconv_ref(x_ref)  # (B,H,T,F)
        Ku = self.unfold(K)        # (B, H*D, T*F)
        Ku = Ku.view(K.shape[0], K.shape[1], -1, K.shape[2], K.shape[3])\
            .permute(0,1,3,2,4).contiguous()  # (B,H,T,D,F)
        V = torch.sum(Q.unsqueeze(-2) * Ku, dim=-1)      # (B,H,T,D)
        V = self.conv(V)           # (B,1,T,D)
        A = torch.softmax(V, dim=-1)[..., None]  # (B,1,T,D,1)

        # Use x_ref dimensions for unfold, not K dimensions (K is projected to hidden_channels)
        y = self.unfold(x_ref).view(x_ref.shape[0], x_ref.shape[1], -1, x_ref.shape[2], x_ref.shape[3])\
                .permute(0,1,3,2,4).contiguous()  # (B,C,T,D,F)
        y = torch.sum(y * A, dim=-2)  # (B,C,T,F)
        return y


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3), stride=(1,2), use_residual=True):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
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
        y = rearrange(x, 'b c t f -> b t (c f)')
        y = self.gru(y)[0]
        y = self.fc(y)
        y = rearrange(y, 'b t (c f) -> b c t f', c=x.shape[1])
        return y
    

class SubpixelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
        self.conv = nn.Conv2d(in_channels, out_channels*2, kernel_size)
        
    def forward(self, x):
        y = self.conv(self.pad(x))
        y = rearrange(y, 'b (r c) t f -> b c t (r f)', r=2)
        return y
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3), is_last=False, use_residual=True):
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
    """Complex convolving mask block"""
    def __init__(self):
        super().__init__()
        # Register as buffer so it moves with model to correct device
        v = torch.tensor([[1,        -1/2,           -1/2],
                          [0, np.sqrt(3)/2, -np.sqrt(3)/2]], dtype=torch.float32)  # (2,3)
        self.register_buffer('v', v)
        self.unfold = nn.Sequential(nn.ZeroPad2d([1,1,2,0]),
                                    nn.Unfold(kernel_size=(3,3)))
    
    def forward(self, m, x):
        """
        m: (B,27,T,F)
        x: (B,F,T,2)"""
        m = rearrange(m, 'b (r c) t f -> b r c t f', r=3)
        H_real = torch.sum(self.v[0][None,:,None,None,None] * m, dim=1)  # (B,C/3,T,F)
        H_imag = torch.sum(self.v[1][None,:,None,None,None] * m, dim=1)  # (B,C/3,T,F)

        M_real = rearrange(H_real, 'b (m n) t f -> b m n t f', m=3)  # (B,3,3,T,F)
        M_imag = rearrange(H_imag, 'b (m n) t f -> b m n t f', m=3)  # (B,3,3,T,F)
        
        x = x.permute(0,3,2,1).contiguous()  # (B,2,T,F)
        x_unfold = self.unfold(x)
        x_unfold = rearrange(x_unfold, 'b (c m n) (t f) -> b c m n t f', m=3,n=3,f=x.shape[-1])

        x_enh_real = torch.sum(M_real * x_unfold[:,0] - M_imag * x_unfold[:,1], dim=(1,2))  # (B,T,F)
        x_enh_imag = torch.sum(M_real * x_unfold[:,1] + M_imag * x_unfold[:,0], dim=(1,2))  # (B,T,F)
        x_enh = torch.stack([x_enh_real, x_enh_imag], dim=3).transpose(1,2).contiguous()
        return x_enh


class DeepVQE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = FE()
        self.enblock1 = EncoderBlock(2, 64)
        self.enblock2 = EncoderBlock(64, 128)
        self.enblock3 = EncoderBlock(256, 128)  # 256 = 128 (mic) + 128 (aligned system)
        self.system_enblock1 = EncoderBlock(2, 32)
        self.system_enblock2 = EncoderBlock(32, 128)
        self.align = AlignBlock(128, 128)
        self.enblock4 = EncoderBlock(128, 128)
        self.enblock5 = EncoderBlock(128, 128)
        
        self.bottle = Bottleneck(128*9, 64*9)
        
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
        # Store original input for CCM
        x_mic_orig = x_mic

        x_mic = self.fe(x_mic)            # ; print(en_x0.shape)
        x_system = self.fe(x_system)

        en_x_mic1 = self.enblock1(x_mic)  # (B, 64, T, F//2)
        en_x_mic2 = self.enblock2(en_x_mic1)  # (B, 128, T, F//4)
        en_x_system1 = self.system_enblock1(x_system)  # (B, 32, T, F//2)
        en_x_system2 = self.system_enblock2(en_x_system1)  # (B, 128, T, F//4)
        aligned = self.align(en_x_mic2, en_x_system2)  # (B, 128, T, F//4) - aligned system ref
        en_x_mic_n_system_aligned = torch.cat([en_x_mic2, aligned], dim=1)  # (B, 256, T, F//4) - concat channels

        en_x3 = self.enblock3(en_x_mic_n_system_aligned)  # ; print(en_x3.shape)
        en_x4 = self.enblock4(en_x3)  # ; print(en_x4.shape)
        en_x5 = self.enblock5(en_x4)  # ; print(en_x5.shape)

        en_xr = self.bottle(en_x5)    # ; print(en_xr.shape)

        de_x5 = self.deblock5(en_xr, en_x5)[..., :en_x4.shape[-1]]  # ; print(de_x5.shape)
        de_x4 = self.deblock4(de_x5, en_x4)[..., :en_x3.shape[-1]]  # ; print(de_x4.shape)
        de_x3 = self.deblock3(de_x4, en_x3)[..., :en_x_mic2.shape[-1]]  # ; print(de_x3.shape)
        de_x2 = self.deblock2(de_x3, en_x_mic2)[..., :en_x_mic1.shape[-1]]  # ; print(de_x2.shape)
        de_x1 = self.deblock1(de_x2, en_x_mic1)[..., :x_mic.shape[-1]]  # ; print(de_x1.shape)

        x_enh = self.ccm(de_x1, x_mic_orig)  # (B,F,T,2)

        return x_enh


class DeepVQE_S(nn.Module):
    """
    DeepVQE-S: Small/Production-sized version of DeepVQE

    From paper ablation studies:
    - Microphone branch: 4 blocks with 16, 40, 56, 24 filters
    - Far-end branch: 8, 24 filters
    - Decoding branch: 4 blocks with 40, 32, 32, 27 filters
    - Residual blocks omitted in ALL encoder blocks
    - Residual blocks omitted in FIRST and LAST decoder blocks only

    This saves computation while maintaining reasonable quality.
    """
    def __init__(self):
        super().__init__()
        # Feature extraction (same as full model)
        self.fe = FE()

        # Microphone encoder: 16 -> 40 -> 56 -> 24 (NO residual blocks)
        self.enblock1 = EncoderBlock(2, 16, use_residual=False)
        self.enblock2 = EncoderBlock(16, 40, use_residual=False)
        self.enblock3 = EncoderBlock(40, 56, use_residual=False)
        self.enblock4 = EncoderBlock(56, 24, use_residual=False)

        # System/Far-end encoder: 8 -> 24 (NO residual blocks)
        self.system_enblock1 = EncoderBlock(2, 8, use_residual=False)
        self.system_enblock2 = EncoderBlock(8, 24, use_residual=False)

        # Alignment block - needs to handle both mic (40 ch) and farend (24 ch) inputs
        # The AlignBlock's pconv layers will project to hidden dimension
        self.align = AlignBlock(
            in_channels=40,        # mic channels
            hidden_channels=32,    # projection dimension
            in_channels_ref=24     # farend channels
        )
        # Update: enblock3 should take 64 input channels (40 mic + 24 aligned farend)
        self.enblock3_actual = EncoderBlock(64, 56, use_residual=False)  # Takes concatenated input
        self.bottle = Bottleneck(24*17, 24*8)  # Hidden size reduced

        # Decoder: Must match encoder skip connection channels
        # Skip connections: en_x4(24), en_x3(56), en_x_mic2(40), en_x_mic1(16)
        # First decoder (no residual), middle two (with residual), last (no residual)
        self.deblock4 = DecoderBlock(24, 56, use_residual=False)   # First: no residual, skip from en_x4(24)
        self.deblock3 = DecoderBlock(56, 40, use_residual=True)    # Middle: with residual, skip from en_x3(56)
        self.deblock2 = DecoderBlock(40, 16, use_residual=True)    # Middle: with residual, skip from en_x_mic2(40)
        self.deblock1 = DecoderBlock(16, 27, use_residual=False)   # Last: no residual, skip from en_x_mic1(16)

        # Complex Convolving Mask (same as full model)
        self.ccm = CCM()

    def forward(self, x_mic, x_system):
        """
        Args:
            x_mic: (B,F,T,2) - noisy microphone signal (complex STFT)
            x_system: (B,F,T,2) - system reference signal (complex STFT)

        Returns:
            x_enh: (B,F,T,2) - enhanced signal (complex STFT)
        """
        # Store original input for CCM
        x_mic_orig = x_mic

        x_mic = self.fe(x_mic)            # (B, 2, T, F)
        x_system = self.fe(x_system)      # (B, 2, T, F)

        # Microphone path
        en_x_mic1 = self.enblock1(x_mic)  # (B, 16, T, F//2)
        en_x_mic2 = self.enblock2(en_x_mic1)  # (B, 40, T, F//4)

        # System path
        en_x_system1 = self.system_enblock1(x_system)  # (B, 8, T, F//2)
        en_x_system2 = self.system_enblock2(en_x_system1)  # (B, 24, T, F//4)

        # Align and concatenate
        aligned = self.align(en_x_mic2, en_x_system2)  # (B, 24, T, F//4) - aligned system ref
        en_x_mic_n_system_aligned = torch.cat([en_x_mic2, aligned], dim=1)  # (B, 64, T, F//4) - concat channels

        # Continue encoding
        en_x3 = self.enblock3_actual(en_x_mic_n_system_aligned)  # (B, 56, T, F//8)
        en_x4 = self.enblock4(en_x3)  # (B, 24, T, F//16)

        # Bottleneck
        en_xr = self.bottle(en_x4)    # (B, 24, T, F//16)

        # Decoder with skip connections (U-Net style)
        de_x4 = self.deblock4(en_xr, en_x4)[..., :en_x3.shape[-1]]  # (B, 40, T, F//8)
        de_x3 = self.deblock3(de_x4, en_x3)[..., :en_x_mic2.shape[-1]]  # (B, 32, T, F//4)
        de_x2 = self.deblock2(de_x3, en_x_mic2)[..., :en_x_mic1.shape[-1]]  # (B, 32, T, F//2)
        de_x1 = self.deblock1(de_x2, en_x_mic1)[..., :x_mic.shape[-1]]  # (B, 27, T, F)

        # Complex convolving mask
        x_enh = self.ccm(de_x1, x_mic_orig)  # (B, F, T, 2)

        return x_enh


if __name__ == "__main__":
    print("=" * 80)
    print("Testing DeepVQE (Full Model)")
    print("=" * 80)
    device = torch.device("mps")
    model = DeepVQE().eval().to(device)
    x_mic = torch.randn(1, 257, 63, 2).to(device)
    x_system = torch.randn(1, 257, 63, 2).to(device)
    y = model(x_mic, x_system)
    print(f"Input shape: {x_mic.shape}, Output shape: {y.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\n" + "=" * 80)
    print("Testing DeepVQE-S (Small Model)")
    print("=" * 80)
    model_s = DeepVQE_S().eval().to(device)
    y_s = model_s(x_mic, x_system)
    print(f"Input shape: {x_mic.shape}, Output shape: {y_s.shape}")

    # Count parameters
    total_params_s = sum(p.numel() for p in model_s.parameters())
    print(f"Total parameters: {total_params_s:,}")
    print(f"Parameter reduction: {(1 - total_params_s/total_params)*100:.1f}%")

    """causality check - verify model doesn't look into the future"""
    a_mic = torch.randn(1, 257, 100, 2, device=device)
    b_mic = torch.randn(1, 257, 100, 2, device=device)
    c_mic = torch.randn(1, 257, 100, 2, device=device)
    a_sys = torch.randn(1, 257, 100, 2, device=device)
    b_sys = torch.randn(1, 257, 100, 2, device=device)
    c_sys = torch.randn(1, 257, 100, 2, device=device)

    x1_mic = torch.cat([a_mic, b_mic], dim=2)
    x2_mic = torch.cat([a_mic, c_mic], dim=2)
    x1_sys = torch.cat([a_sys, b_sys], dim=2)
    x2_sys = torch.cat([a_sys, c_sys], dim=2)

    y1 = model(x1_mic, x1_sys)
    y2 = model(x2_mic, x2_sys)

    print(f"Causality check (should be ~0.0): {(y1[:,:,:100,:] - y2[:,:,:100,:]).abs().max().item()}")
    print(f"Different future (should be >0.0): {(y1[:,:,100:,:] - y2[:,:,100:,:]).abs().max().item()}")
        