import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from lightning_resgistry.models.builder import MODELS


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid_(x)


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
    def __init__(self, in_ch, out_ch, dropout, attn=False, stride=1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.ReLU(),
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1, stride=stride, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        # self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


@MODELS.register_module("resnet_simple")
class ResNet(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.head = nn.Conv3d(1, 64, kernel_size=3, stride=2)
        self.layer1 = ResBlock(64, 64, dropout, stride=2)
        self.layer2 = ResBlock(64, 128, dropout, stride=2)
        self.layer3 = ResBlock(128, 256, dropout, stride=2)
        self.layer4 = ResBlock(256, 512, dropout, stride=2)
        self.tail = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        x = x[:, 0:1, :, :, :]
        x = self.head(x)
        # x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean(dim=[2, 3, 4])
        out = self.tail(x)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    model = ResNet()
    model.to(device)
    x = torch.randn(batch_size, 2, 192, 218, 192)
    x = x.to(device)
    y = model(x)