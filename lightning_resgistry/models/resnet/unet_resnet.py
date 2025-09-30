import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from lightning_resgistry.models.builder import MODELS


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv3d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x):
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
        # self.initialize()

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
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            # nn.GroupNorm(32, inchannel),
            # Swish(),
            nn.BatchNorm3d(inchannel),
            nn.ReLU(),
            # nn.Conv3d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.Conv3d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(outchannel),
            # nn.GroupNorm(32, outchannel),
            nn.ReLU(),
            # Swish(),
            nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(outchannel)

        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                # nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=stride),
                nn.BatchNorm3d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

# class ResBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
#         super().__init__()
#         self.block1 = nn.Sequential(
#             nn.GroupNorm(32, in_ch),
#             Swish(),
#             nn.Conv3d(in_ch, out_ch, 3, stride=1, padding=1),
#         )
#
#         self.block2 = nn.Sequential(
#             nn.GroupNorm(32, out_ch),
#             Swish(),
#             nn.Dropout(dropout),
#             nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1),
#         )
#         if in_ch != out_ch:
#             self.shortcut = nn.Conv3d(in_ch, out_ch, 1, stride=1, padding=0)
#         else:
#             self.shortcut = nn.Identity()
#         if attn:
#             self.attn = AttnBlock(out_ch)
#         else:
#             self.attn = nn.Identity()
#         # self.initialize()
#
#     def initialize(self):
#         for module in self.modules():
#             if isinstance(module, (nn.Conv2d, nn.Linear)):
#                 init.xavier_uniform_(module.weight)
#                 init.zeros_(module.bias)
#         init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)
#
#     def forward(self, x):
#         h = self.block1(x)
#         h = self.block2(h)
#
#         h = h + self.shortcut(x)
#         h = self.attn(h)
#         return h


@MODELS.register_module("unet-resnet")
class UNetResNet(nn.Module):
    def __init__(self,
                 T,
                 ch,
                 ch_mult,
                 attn,
                 num_res_blocks,
                 dropout,
                 head_stride=1,
                 up_sample_conv=True,
                 up_sample_kernel_size=2,
                 ):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4

        self.head = nn.Conv3d(1, ch, kernel_size=3, stride=head_stride, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool3d(3, 2)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    inchannel=now_ch, outchannel=out_ch))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch),
            ResBlock(now_ch, now_ch),
        ])
        self.conv = nn.Conv3d(kernel_size=3, stride=2, in_channels=now_ch, out_channels=now_ch)
        self.classier_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        # self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)

    def forward(self, x):
        # Timestep embedding
        # Downsampling
        h = self.head(x)
        # h = self.conv1(x)
        # h = self.maxpool(h)
        for layer in self.downblocks:
            h = layer(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h)
        h = self.conv(h)
        h = F.avg_pool3d(h, 3)
        h = h.view(h.size(0), -1)

        h = self.classier_head(h)

        return h


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    model = UNetResNet(
        T=1000, ch=64, ch_mult=[1, 2, 4, 8], attn=[-1],
        num_res_blocks=1, dropout=0.1)
    model.to(device)
    x = torch.randn(batch_size, 1, 70, 90, 70)
    x = x.to(device)
    y = model(x)