import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------
# Simplified Channel Attention
# -----------------------------
class SCA(nn.Module):
    """Simplified Channel Attention (SCA) used in MW-Enc block"""
    def __init__(self, channels):
        super(SCA, self).__init__()
        # global average pooling + 1x1 conv (parameter-efficient)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels//2, channels//2, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(self.avg_pool(x)))
        return x * attn

# -----------------------------
# MWEncBlock
# -----------------------------
class MWEncBlock(nn.Module):
    """
    MW-Enc block: Activation-free convolutional encoder block
    """
    def __init__(self, channels):
        super(MWEncBlock, self).__init__()

        # ----- First Sub-stage -----
        # channels = channels*2
        self.ln1 = nn.LayerNorm(channels)
        self.conv1_1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
        self.dwconv3_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        self.sca = SCA(channels)
        self.conv1_2 = nn.Conv2d(channels//2, channels, kernel_size=1, padding=0, bias=True)
        self.beta = nn.Parameter(torch.zeros(1))

        # ----- Second Sub-stage -----
        self.ln2 = nn.LayerNorm(channels)
        self.conv2_1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
        self.dwconv3_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        self.conv2_2 = nn.Conv2d(channels//2, channels, kernel_size=1, padding=0, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        # ---- Sub-stage 1 ----
        y = self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)   # LayerNorm over channel dim
        y = self.conv1_1(y)
        y = self.dwconv3_1(y)

        # Split channel into two halves and element-wise multiply (NAFNet-style gating)
        y1, y2 = torch.chunk(y, 2, dim=1)
        y = y1 * y2

        y_sca = self.sca(y)                       # Simplified channel attention
        Y = y_sca*y
        Y = self.conv1_2(Y)
        y = x + self.beta * Y                 # Residual connection

        # ---- Sub-stage 2 ----
        out = self.ln2(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.conv2_1(out)
        out = self.dwconv3_2(out)

        y1, y2 = torch.chunk(out, 2, dim=1)
        out = y1 * y2                         # gating again
        out = self.conv2_2(out)
        out = y + self.gamma * out            # Residual connection

        return out


class WaveletPooling(nn.Module):
    """
    Wavelet Pooling Block (Section III‑C, MW‑ConvNet)
    - Performs Haar wavelet transform–based downsampling.
    - Outputs low-frequency features and combined high-frequency features.
    """

    def __init__(self, out_channels):
        super(WaveletPooling, self).__init__()

        # ----- Haar basis filters (2×2) -----
        # √2 분모는 논문 Eqn.(10)에 따라 정규화
        h = 1 / math.sqrt(2.0)
        self.register_buffer('KLL', torch.tensor([[h, h],
                                                  [h, h]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer('KLH', torch.tensor([[-h, -h],
                                                  [h,  h]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer('KHL', torch.tensor([[-h,  h],
                                                  [-h,  h]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer('KHH', torch.tensor([[ h, -h],
                                                  [-h,  h]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.conv = nn.Conv2d(out_channels*3, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # wavelet filter (grouped conv)
        filters = torch.cat([self.KLL, self.KLH, self.KHL, self.KHH], dim=0)  # [4,1,2,2]
        filters = filters.repeat(C, 1, 1, 1)  # [4*C,1,2,2]

        # group conv로 4*C
        out = F.conv2d(x, filters, stride=2, padding=0, groups=C)  # [B,4C,H/2,W/2]

        # 결과를 4개의 주파수 밴드로 분리
        out = out.view(B, C, 4, H // 2, W // 2)
        F_LL, F_LH, F_HL, F_HH = out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]

        # 저주파(LL): 다음 encoder 단계로
        low_freq = F_LL
        # 고주파(LH, HL, HH): skip connetion to corresponding feature map
        high_freq = torch.cat([F_LH, F_HL, F_HH], dim=1)  # [B,3C,H/2,W/2]
        high_freq = self.conv(high_freq) 

        return low_freq, high_freq

class PromptModule(nn.Module):
    """
    Prompt Generation Module (Section III-D, MW-ConvNet)
    - Extracts weather-aware prompt from encoder's low-frequency feature
    - Also outputs logits for classification loss (during training)
    """
    def __init__(self, in_channels, prompt_dim=128, num_classes=3):
        super(PromptModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.conv1x1 = nn.Conv2d(in_channels, prompt_dim, kernel_size=1, bias=True)
        self.fc = nn.Linear(prompt_dim, num_classes)  # For weather classification

    def forward(self, x):
        """
        Args:
            x: Low-frequency feature map from encoder [B, C, H, W]
        Returns:
            prompt: weather prompt vector [B, prompt_dim]
            logits: classification prediction [B, num_classes]
        """
        x = self.pool(x)                       # [B, C, 1, 1]
        x = self.conv1x1(x)                    # [B, prompt_dim, 1, 1]
        prompt = x.view(x.size(0), -1)         # [B, prompt_dim]
        logits = self.fc(prompt)               # [B, num_classes]
        return prompt, logits
    
class MWDecBlock(nn.Module):
    def __init__(self, channels, prompt_dim):
        super(MWDecBlock, self).__init__()
        self.channels = channels
        self.prompt_dim = prompt_dim

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        self.conv2 = nn.Conv2d(channels//2, channels, kernel_size=1, padding=0, bias=True)

        self.affine_mu = nn.Linear(prompt_dim, channels)
        self.affine_std = nn.Linear(prompt_dim, channels)
        self.gamma = nn.Parameter(torch.zeros(1))  # residual scaling

    def forward(self, x, prompt):
        B, C, H, W = x.shape

        # (1) Feature map 통계 계산 (spatial mean, std)
        mu_f = x.mean(dim=(2, 3), keepdim=True)           # [B, C, 1, 1]
        std_f = x.std(dim=(2, 3), keepdim=True) + 1e-6     # [B, C, 1, 1]

        # (2) Prompt → affine 계수 (μ_WE, σ_WE)
        mu_we = self.affine_mu(prompt).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        std_we = self.affine_std(prompt).unsqueeze(-1).unsqueeze(-1)

        # (3) Weather-Adaptive Normalization (논문 수식 (11))
        x_norm = std_we * (x - mu_f) / std_f + mu_we

        # (4) Convolution stack (NAFNet-style)
        y = self.conv1(x_norm)
        y = self.dwconv(y)
        y1, y2 = torch.chunk(y, 2, dim=1)
        y = y1 * y2
        y = self.conv2(y)

        return x_norm + self.gamma * y  # residual connection

class UpSampleFusionBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpSampleFusionBlock, self).__init__()
        self.fuse_conv = nn.Conv2d(in_channels, in_channels*2, kernel_size=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, dec_feat, skip_feat):
        """
        dec_feat: Decoder feature (B, C, H, W)
        skip_feat: Skip connection feature from encoder (B, C, H, W)
        """
        # (1) Feature fusion (pixel-wise addition)
        fused = dec_feat + skip_feat

        # (2) Channel expansion for pixel shuffle
        fused = self.fuse_conv(fused)  # (B, 2C, H, W)

        # (3) PixelShuffle → (B, C//2, 2H, 2W)
        upsampled = self.pixel_shuffle(fused)

        return upsampled
    
class UpchannelBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpchannelBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels*2, kernel_size=1, bias=True)
        
    def forward(self, x):
        x = self.conv(x)
        return x

class MWConvNet(nn.Module):
    def __init__(self, in_channels=3, prompt_dim=128, num_stages=5, num_classes=4):
        super(MWConvNet, self).__init__()

        # 채널 설정: [32, 64, 128, 256]
        self.chan_dims = [32 * (2 ** i) for i in range(num_stages)]  # [32, 64, 128, 256]
        self.num_classes = num_classes
        mw_enc_repeat = [2,2,4,8,5]
        mw_dec_repeat = [5,2,2,2,2]

        # Shallow feature extract
        self.input_proj = nn.Conv2d(in_channels, self.chan_dims[0], kernel_size=3, padding=1)

        # -----------------------------
        # stage 1
        # -----------------------------
        self.stage_1_enc = nn.Sequential(
            *[MWEncBlock(self.chan_dims[0]) for _ in range(mw_enc_repeat[0])],
            WaveletPooling(self.chan_dims[0]),
        )
        self.stage_1_1x1 = nn.Conv2d(self.chan_dims[0], self.chan_dims[1], kernel_size=1, bias=True) # skip connection 용도(high frequency가 input)
        
        # -----------------------------
        # stage 2
        # -----------------------------
        self.stage_2_enc = nn.Sequential(
            nn.Conv2d(self.chan_dims[0], self.chan_dims[1], kernel_size=1, bias=True), # low frequency가 input
            *[MWEncBlock(self.chan_dims[1]) for _ in range(mw_enc_repeat[1])],
            WaveletPooling(self.chan_dims[1])
        )
        self.stage_2_1x1 = nn.Conv2d(self.chan_dims[1], self.chan_dims[2], kernel_size=1, bias=True) # skip connection 용도(high frequency가 input)

        # -----------------------------
        # stage 3
        # -----------------------------
        self.stage_3_enc = nn.Sequential(
            nn.Conv2d(self.chan_dims[1], self.chan_dims[2], kernel_size=1, bias=True), # low frequency가 input
            *[MWEncBlock(self.chan_dims[2]) for _ in range(mw_enc_repeat[2])],
            WaveletPooling(self.chan_dims[2])
        )
        self.stage_3_1x1 = nn.Conv2d(self.chan_dims[2], self.chan_dims[3], kernel_size=1, bias=True) # skip connection 용도(high frequency가 input)

        # -----------------------------
        # stage 4
        # -----------------------------
        self.stage_4_enc = nn.Sequential(
            nn.Conv2d(self.chan_dims[2], self.chan_dims[3], kernel_size=1, bias=True), # low frequency가 input
            *[MWEncBlock(self.chan_dims[3]) for _ in range(mw_enc_repeat[3])],
            WaveletPooling(self.chan_dims[3])
        )
        self.stage_4_1x1 = nn.Conv2d(self.chan_dims[3], self.chan_dims[4], kernel_size=1, bias=True) # skip connection 용도(high frequency가 input)   

        # -----------------------------
        # stage 5
        # -----------------------------
        self.stage_5_enc = nn.Sequential(
            nn.Conv2d(self.chan_dims[3], self.chan_dims[4], kernel_size=1, bias=True), # low frequency가 input
            *[MWEncBlock(self.chan_dims[4]) for _ in range(mw_enc_repeat[4])],
        )             

        # 3️⃣ Prompt Module: 최종 encoder 출력에서 prompt 추출
        self.prompt_module = PromptModule(
            in_channels=self.chan_dims[-1], prompt_dim=prompt_dim, num_classes=self.num_classes
        )

        self.decoders = nn.ModuleList([
            nn.ModuleList([MWDecBlock(ch, prompt_dim) for _ in range(n)])
            for ch, n in zip(reversed(self.chan_dims), mw_dec_repeat)
        ])

        # 업샘플링 단계 구성
        self.upsamples = nn.ModuleList([
            UpSampleFusionBlock(ch) for ch in self.chan_dims[::-1][:len(self.chan_dims)-1]
        ])        

        self.final = nn.Conv2d(self.chan_dims[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_shallow = self.input_proj(x)
        skips = []

        lf, hf = self.stage_1_enc(x_shallow)
        skips.append(self.stage_1_1x1(hf))
        lf, hf = self.stage_2_enc(lf)
        skips.append(self.stage_2_1x1(hf))
        lf, hf = self.stage_3_enc(lf)
        skips.append(self.stage_3_1x1(hf))
        lf, hf = self.stage_4_enc(lf)
        skips.append(self.stage_4_1x1(hf))

        weather_prompt = self.stage_5_enc(lf)

        prompt, logits = self.prompt_module(weather_prompt)

        skips = skips[::-1]

        x_feat = weather_prompt

        for i in range(len(self.decoders)):
            for block in self.decoders[i]:
                x_feat = block(x_feat, prompt)

            if i < len(self.upsamples):
                x_feat = self.upsamples[i](x_feat, skips[i])
        
        out = x_feat + x_shallow
        
        out = self.final(out) + x

        return out, logits
    
if __name__ == "__main__":
    # x = torch.randn(1, 64, 64, 64)
    # block = MWEncBlock(64)
    # y = block(x)
    # print(y.shape)  # → (1, 64, 64, 64)

    # x = torch.randn(1, 32, 128, 128)
    # wavelet = WaveletPooling(in_channels=32)
    # lf, hf = wavelet(x)

    # x = torch.randn(8, 512, 8, 8)  # low-frequency feature
    # module = PromptModule(in_channels=512, prompt_dim=128, num_classes=3)
    # prompt, logits = module(x)
    # print(prompt.shape)  # torch.Size([8, 128])
    # print(logits.shape)  # torch.Size([8, 3])    

    # x = torch.randn(4, 128, 32, 32)           # decoder feature
    # prompt = torch.randn(4, 64)               # prompt vector
    # block = MWDecBlock(channels=128, prompt_dim=64)
    # out = block(x, prompt)
    # print(out.shape)  # → torch.Size([4, 128, 32, 32])

    # x = torch.randn(4, 128, 32, 32)
    # skip = torch.randn(4, 128, 32, 32)
    # block = UpSampleFusionBlock(in_channels=128)
    # out = block(x, skip)
    # print(out.shape)  # → torch.Size([4, 64, 64, 64])

    model = MWConvNet()
    x = torch.randn(1, 3, 256, 256)
    out, logits = model(x)
    print(out.shape)    # torch.Size([1, 3, 256, 256])
    print(logits.shape) # torch.Size([1, 3])
