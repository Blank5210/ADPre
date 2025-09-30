import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from lightning_resgistry.models.builder import MODELS


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model).to('cuda')

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.time_embedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv3d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch, up_conv=True, kernel_size=2):
        super().__init__()
        self.up_conv = up_conv
        self.kernel_size = kernel_size
        if self.up_conv:
            self.main = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=self.kernel_size, stride=2)
        else:
            self.main = nn.Conv3d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        if self.up_conv:
            x = self.main(x)
        else:
            _, _, D, H, W = x.shape
            x = F.interpolate(
                x, scale_factor=2, mode='nearest')
            x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        h = self.group_norm(x)

        # Q, K, V projections
        q = self.proj_q(h)   # (B, C, D, H, W)
        k = self.proj_k(h)
        v = self.proj_v(h)

        # reshape for attention: flatten (D*H*W)
        q = q.permute(0, 2, 3, 4, 1).reshape(B, D*H*W, C)   # (B, N, C)
        k = k.reshape(B, C, D*H*W)                         # (B, C, N)
        v = v.permute(0, 2, 3, 4, 1).reshape(B, D*H*W, C)   # (B, N, C)

        # compute attention weights
        w = torch.bmm(q, k) * (C ** -0.5)                  # (B, N, N)
        w = F.softmax(w, dim=-1)

        # apply attention to V
        h = torch.bmm(w, v)                                # (B, N, C)
        h = h.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)   # (B, C, D, H, W)

        # final projection
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv3d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h = h + self.temb_proj(temb)[:, :, None, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class ClassierEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass


@MODELS.register_module("unet-backbone")
class UNet(nn.Module):
    def __init__(self,
                 T,
                 ch,
                 ch_mult,
                 attn,
                 num_res_blocks,
                 dropout,
                 num_classes=3,
                 cond_dim=16,      # 条件 embedding 维度
                 head_stride=1,
                 up_sample_conv=True,
                 up_sample_kernel_size=2):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.cond_embedding = nn.Embedding(num_classes, cond_dim)  # 标签 embedding
        self.cond_proj = nn.Linear(cond_dim, tdim)                  # 条件映射到 temb 维度

        self.head = nn.Conv3d(1, ch, kernel_size=3, stride=head_stride, padding=1)
        self.unet_downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.unet_downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.unet_downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.unet_middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.unet_upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.unet_upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.unet_upblocks.append(UpSample(now_ch, up_conv=up_sample_conv, kernel_size=up_sample_kernel_size))
        assert len(chs) == 0

        self.classier_encoder = ClassierEncoder()

    def forward(self, x, t, label=None):
        # 时间步 embedding
        temb = self.time_embedding(t)

        # 条件 embedding
        if label is not None:
            cond_emb = self.cond_proj(self.cond_embedding(label))
            temb = temb + cond_emb

        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                skip = hs.pop()
                _, _, D, H, W = h.shape
                _, _, D_s, H_s, W_s = skip.shape
                if (D > D_s) or (H > H_s) or (W > W_s):
                    h = h[:, :, :D_s, :H_s, :W_s]
                h = torch.cat([h, skip], dim=1)
            h = layer(h, temb)

        h = self.tail(h)
        return h  # 输出噪声预测 epsilon



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    model = UNet(
        T=1000, ch=64, ch_mult=[1, 2, 4, 4], attn=[-1],
        num_res_blocks=1, dropout=0.1)
    model.to(device)
    x = torch.randn(batch_size, 1, 76, 84, 76)
    x = x.to(device)
    t = torch.randint(1000, (batch_size, ))
    t = t.to(device)
    y = model(x, t)