from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.image import math_img, load_img
from nilearn import plotting
import numpy as np
import nibabel as nib
import torch

#from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.utils.deprecate_utils import deprecated_arg

from lightning_resgistry.models.builder import MODELS


# # 加载图集图像
# # 这三行应该不变
# atlas_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm')
# # 模板路径
# atlas_cort_filename = atlas_cort.filename
# # 脑区标签
# labels = atlas_cort.labels
rearrange, _ = optional_import("einops", name="rearrange")

# 输出不同大小的遮盖模板
def template_multiscale(template_name, area_label):
    atlas_cort = datasets.fetch_atlas_harvard_oxford(template_name)
    # 模板路径
    atlas_cort_filename = atlas_cort.filename
    # 脑区标签
    labels = atlas_cort.labels
    atlas_data = nib.load(atlas_cort_filename).get_fdata()
    atlas_data = torch.tensor(atlas_data)
    # 提取想要的区域
    img_occ = torch.zeros_like(atlas_data, dtype=int)

    for index, label in enumerate(labels):
        if label == area_label:
            img_occ = torch.where(atlas_data == index, torch.tensor(index, dtype=int), img_occ)
    nonzero_count = torch.count_nonzero(img_occ)
    img_occ = atlas_data.unsqueeze(0).unsqueeze(0)
    # for index, label in enumerate(labels):
    #     if label == area_label:
    #         img_occ = (1 if atlas_data == index else 0)
    # 跟随数据一起做卷积
    max_pool1 = nn.MaxPool3d(3, 2, padding=1)
    max_pool2 = nn.MaxPool3d(3, 2)
    max_pool3 = nn.MaxPool3d(3, 2, padding=1)
    max_pool4 = nn.MaxPool3d(3, 1, padding=1)
    img_occ1 = max_pool1(img_occ)
    img_occ1 = max_pool2(img_occ1)
    img_occ2 = max_pool3(img_occ1)
    img_occ2 = max_pool4(img_occ2)
    # img_occ1, img_occ2 = img_occ1.squeeze(), img_occ2.squeeze()
    nonzero_count1 = torch.count_nonzero(img_occ1)
    nonzero_count2 = torch.count_nonzero(img_occ2)
    for index, label in enumerate(labels):
        if label == area_label:
            img_occ1 = img_occ1 == index
            img_occ2 = img_occ2 == index
    # 输出分割模板
    return img_occ1, img_occ2


# 根据模板和对应的标签分割出img相应的脑部区域
def segment_area(img, img_occ):
    b, c, d, h, w = img.shape
    # 加载数据
    img_area = img * img_occ
    # 分割出包含脑区的最小区域
    nonzero_coords = torch.nonzero(img_occ.squeeze())  # 获取非零像素的体素坐标

    # 如果没有非零像素，返回空图像
    if nonzero_coords.shape[0] == 0:
        return None

    # 获取最小和最大坐标来定义最小边界框
    min_coords = nonzero_coords.min(dim=0).values
    max_coords = nonzero_coords.max(dim=0).values

    # 提取包含非零像素的最小立方体区域
    sliced_data = img_area[:, :, min_coords[0]:max_coords[0] + 1, min_coords[1]:max_coords[1] + 1, min_coords[2]:max_coords[2] + 1]
    # 提取图像区域内的坐标
    # 使用 meshgrid 创建每个坐标
    coords_d = torch.arange(min_coords[0], max_coords[0] + 1)  # d轴范围
    coords_h = torch.arange(min_coords[1], max_coords[1] + 1)  # h轴范围
    coords_w = torch.arange(min_coords[2], max_coords[2] + 1)  # w轴范围

    # 创建网格坐标
    grid = torch.cartesian_prod(coords_d, coords_h, coords_w)
    return sliced_data, sliced_data.shape, min_coords, max_coords


def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.size()  # length 4 or 5 only
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    else:  # if len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)

    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention2(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n//2, :n//2].reshape(-1)
        ].reshape(n//2, n//2, -1)
        relative_position_bias = relative_position_bias.repeat(2, 2, 1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock2(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention2(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x1, x2, mask_matrix):
        x_shape = x1.size()
        x1 = self.norm1(x1)
        x2 = self.norm1(x2)
        if len(x_shape) == 5:
            b, d, h, w, c = x1.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x1 = F.pad(x1, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            x2 = F.pad(x2, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x1.shape
            dims = [b, dp, hp, wp]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x1 = torch.roll(x1, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
                shifted_x2 = torch.roll(x2, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x1, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x1 = x1
            shifted_x2 = x2
            attn_mask = None


        x1_windows = window_partition(shifted_x1, window_size)
        x2_windows = window_partition(shifted_x2, window_size)
        x_windows = torch.cat((x1_windows, x2_windows), dim=1)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        # 自己添加
        half_n = attn_windows.shape[1] // 2  # 计算第二维的一半
        attn_windows = attn_windows[:, :half_n, :]

        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def load_from(self, weights, n_block, layer):
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])

    def forward(self, x1, x2, mask_matrix):
        shortcut = x1
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x1, x2, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x1, x2, mask_matrix)
        # 此处可以更改为以mri还是以pet为主
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


# 用于计算MRI和PET相对应脑区的的自注意力
class BasicLayer2(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.use_checkpoint = use_checkpoint
        self.block = SwinTransformerBlock2(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=0.0,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )

    def forward(self, x1, x2):
        x_shape = x1.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            x1 = rearrange(x1, "b c d h w -> b d h w c")
            x2 = rearrange(x2, "b c d h w -> b d h w c")
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x1.device)
            x = self.block(x1, x2, attn_mask)
            x = x.view(b, d, h, w, -1)
            x = rearrange(x, "b d h w c -> b c d h w")

        return x


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNetFeature(nn.Module):
    def __init__(self, ResBlock):
        super(ResNetFeature, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool3d(3, 2)
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=2)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.layer_norm = nn.LayerNorm(512)
        # self.fc = nn.Linear(512, num_classes)

    # 用来重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out_list = []
        out = self.conv1(x)
        out_list.append(out)
        out = self.maxpool(out)
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        # 填充操作
        padding = (1, 0, 0, 0, 1, 0)  # (pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2)
        # 应用填充
        out = F.pad(out, padding, mode='constant', value=0)
        out = F.avg_pool3d(out, 4)
        out = out.view(out.size(0), -1)
        out_list.append(out)

        return out_list


class ResnetFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = ResNetFeature(ResBlock)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.fc(features[5])
        return features, out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, mri_feature, pet_feature):
        mri_feature = mri_feature[5].unsqueeze(1)
        pet_feature = pet_feature[5].unsqueeze(1)
        mri_pet_feature = torch.cat((mri_feature, pet_feature), dim=1)
        #
        x = self.transformer(mri_pet_feature)
        x = x[:, 0, :]
        x = self.fc(x)
        return x


class LowFeatureBlock(nn.Module):
    def __init__(self,
                 window_size,
                 num_heads,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.resnet_feature = ResnetFeature()
        self.layer0_1 = BasicLayer2(
                    dim=64,
                    num_heads=num_heads,
                    window_size=window_size,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop)
        self.layer1_2 = BasicLayer2(
                    dim=64,
                    num_heads=num_heads,
                    window_size=window_size,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop)

    def forward(self, mri_feature, pet_feature):
        mri_low_feature = self.layer0_1(mri_feature[1], pet_feature[1])
        low_feature_out = self.resnet_feature.feature_extractor.layer1(mri_feature[1])
        low_feature_out = self.layer1_2(low_feature_out, pet_feature[2])
        low_feature_out = self.resnet_feature.feature_extractor.layer2(low_feature_out)
        low_feature_out = self.resnet_feature.feature_extractor.layer3(low_feature_out)
        low_feature_out = self.resnet_feature.feature_extractor.layer4(low_feature_out)
        # 填充操作
        padding = (1, 0, 0, 0, 1, 0)  # (pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2)
        # 应用填充
        low_feature_out = F.pad(low_feature_out, padding, mode='constant', value=0)
        low_feature_out = F.avg_pool3d(low_feature_out, 4)
        low_feature_out = low_feature_out.view(low_feature_out.size(0), -1)
        low_feature_out = self.resnet_feature.fc(low_feature_out)
        return low_feature_out


class HighLowFeatureBlock(nn.Module):
    def __init__(self,
                 window_size,
                 num_heads,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.resnet_feature = ResnetFeature()
        self.layer0_1 = BasicLayer2(
                    dim=64,
                    num_heads=num_heads,
                    window_size=window_size,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop)
        self.layer1_2 = BasicLayer2(
                    dim=64,
                    num_heads=num_heads,
                    window_size=window_size,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.1,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, mri_feature, pet_feature):
        mri_low_feature = self.layer0_1(mri_feature[1], pet_feature[1])
        low_feature_out = self.resnet_feature.feature_extractor.layer1(mri_low_feature)
        low_feature_out = self.layer1_2(low_feature_out, pet_feature[2])
        low_feature_out = self.resnet_feature.feature_extractor.layer2(low_feature_out)
        low_feature_out = self.resnet_feature.feature_extractor.layer3(low_feature_out)
        low_feature_out = self.resnet_feature.feature_extractor.layer4(low_feature_out)
        # 填充操作
        padding = (1, 0, 0, 0, 1, 0)  # (pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2)
        # 应用填充
        low_feature_out = F.pad(low_feature_out, padding, mode='constant', value=0)
        low_feature_out = F.avg_pool3d(low_feature_out, 4)
        low_feature_out = low_feature_out.view(low_feature_out.size(0), -1)
        low_feature_out = low_feature_out.unsqueeze(1)
        pet_feature = pet_feature[5].unsqueeze(1)
        mri_pet_feature = torch.cat((low_feature_out, pet_feature), dim=1)
        #
        x = self.transformer(mri_pet_feature)
        x = x[:, 0, :]
        x = self.fc(x)
        return x


@MODELS.register_module("brain_transformer")
class BrainResNetTransformer(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            # window_size: Sequence[int],
            # shift_size: Sequence[int],
            # mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            # drop_path: float = 0.0,
            # act_layer: str = "GELU",
            # norm_layer: type[LayerNorm] = nn.LayerNorm,
            # use_checkpoint: bool = False,
    ) -> None:
        super(BrainResNetTransformer, self).__init__()
        # 定义mri和pet的resnet
        self.mri_resnet = ResnetFeature()
        self.pet_resnet = ResnetFeature()

        # 高级特征融合
        # self.high_feature_block = TransformerBlock(embed_dim=512, num_heads=8, num_layers=2, dropout=0.1)

        # 低级特征融合
        # self.low_feature_block = LowFeatureBlock(window_size=[7, 7, 7], num_heads=1)

        self.high_low_feature_block = HighLowFeatureBlock(window_size=[7, 7, 7], num_heads=1)

    def forward(self, x):
        mri_img = x[:, 0:1, :, :, :]
        pet_img = x[:, 1:2, :, :, :]
        mri_feature, mri_out = self.mri_resnet(mri_img)
        pet_feature, pet_out = self.pet_resnet(pet_img)
        # out = self.high_feature_block(mri_feature, pet_feature)
        # 低级特征融合
        # out = self.low_feature_block(mri_feature, pet_feature)

        out = self.high_low_feature_block(mri_feature, pet_feature)

        # out = 0.8 * out + 0.1 * mri_out + 0.1 * pet_out
        # out = mri_out + 0.15 * pet_out
        return out



# 测试
# patch_size = ensure_tuple_rep(4, 3)
# window_size = [21, 6, 15]
# model = BrainResNetTransformer(
#             dim=1,
#             num_heads=1
#             )
# model.to('cuda')
#
# x1 = np.random.randint(0, 256, size=(1, 2, 182, 218, 182), dtype=np.uint8)
# x1 = torch.from_numpy(x1).float()
# x1 = x1.to('cuda')
# #
# # # x2 = np.random.randint(0, 256, size=(1, 96, 23, 28, 23), dtype=np.uint8)
# # # x2 = torch.from_numpy(x2).float()
# # # x2 = x2.to('cuda')
# #
# output = model(x1)
# print(output.size())
