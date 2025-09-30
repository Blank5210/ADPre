import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from .builder import AUGMENT


@AUGMENT.register_module()
class Augment3D:
    def __init__(self,
                 apply_rotate=False,
                 rotate_max_deg=10,  # 旋转限制角度
                 p_rotate=0.9,  # 旋转概率

                 apply_crop=False,
                 crop_size=(140, 180, 180),

                 apply_elastic=False,
                 elastic_alpha=10,
                 elastic_sigma=3,
                 p_elastic=0.8,  # 形变概率

                 apply_downsample=False,

                 apply_normalize=False,

                 device='cuda:0',
                 seed=None):
        """
        GPU 版本 3D 医学图像增强类
        """
        self.apply_rotate = apply_rotate
        self.apply_crop = apply_crop
        self.apply_elastic = apply_elastic
        self.apply_downsample = apply_downsample
        self.apply_normalize = apply_normalize
        self.crop_size = crop_size
        self.rotate_max_deg = rotate_max_deg
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.p_rotate = p_rotate
        self.p_elastic = p_elastic
        self.device = device
        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(seed)

    # ---------------- 内部方法 ---------------- #
    def _random_rotate(self, vol):
        """
        三轴随机旋转 (C,D,H,W)
        """
        C,D,H,W = vol.shape
        device = vol.device

        angles = (torch.rand(3, generator=self.rng, device=device) - 0.5) * 2 * self.rotate_max_deg
        angles = angles * torch.pi / 180
        alpha, beta, gamma = angles

        # 构建旋转矩阵
        Rx = torch.tensor([[1,0,0],[0,torch.cos(alpha),-torch.sin(alpha)],[0,torch.sin(alpha),torch.cos(alpha)]], device=device)
        Ry = torch.tensor([[torch.cos(beta),0,torch.sin(beta)],[0,1,0],[-torch.sin(beta),0,torch.cos(beta)]], device=device)
        Rz = torch.tensor([[torch.cos(gamma),-torch.sin(gamma),0],[torch.sin(gamma),torch.cos(gamma),0],[0,0,1]], device=device)
        R = Rz @ Ry @ Rx

        theta = torch.zeros(3,4, device=device)
        theta[:3,:3] = R
        theta = theta.unsqueeze(0)  # batch=1

        vol = vol.unsqueeze(0)  # batch=1
        grid = F.affine_grid(theta, vol.size(), align_corners=True)
        vol_rot = F.grid_sample(vol, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return vol_rot.squeeze(0)

    def _crop(self, vol):
        """
        随机裁剪 (C,D,H,W)
        """
        C, D, H, W = vol.shape
        x_size, y_size, z_size = self.crop_size[0], self.crop_size[1], self.crop_size[2]
        x_start, y_start, z_start = (D - x_size) // 2, (H - y_size) // 2, (W - z_size) // 2

        x_end = x_start + x_size
        y_end = y_start + y_size
        z_end = z_start + z_size

        cropped = vol[:, x_start:x_end, y_start:y_end, z_start:z_end]
        return cropped

    def _elastic_deform(self, vol):
        """
        GPU 弹性形变，vol: (C,D,H,W)
        """
        C, D, H, W = vol.shape
        vol = vol.unsqueeze(0)  # batch=1

        # 构造网格坐标 [-1,1]
        z = torch.linspace(-1, 1, D, device=self.device)
        y = torch.linspace(-1, 1, H, device=self.device)
        x = torch.linspace(-1, 1, W, device=self.device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        grid = torch.stack((xx, yy, zz), dim=-1)  # (D,H,W,3)
        grid = grid.unsqueeze(0)  # (1,D,H,W,3)

        # 随机位移
        displacement = (torch.rand(grid.shape, generator=self.rng, device=grid.device) * 2 - 1) * (self.elastic_alpha / max(D, H, W))
        grid = grid + displacement
        grid = grid.clamp(-1, 1)

        vol_warp = F.grid_sample(vol, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return vol_warp.squeeze(0)  # (C,D,H,W)

    # def _elastic_deform(self, vol):
    #     """
    #     弹性形变 (C,D,H,W)
    #     注意：用 numpy 做计算，再转回 GPU
    #     """
    #     vol_np = vol.cpu().numpy()
    #     C,D,H,W = vol_np.shape
    #     vols_out = []
    #     for c in range(C):
    #         shape = vol_np[c].shape
    #         dx = gaussian_filter(torch.randn(*shape).numpy(), self.elastic_sigma) * self.elastic_alpha
    #         dy = gaussian_filter(torch.randn(*shape).numpy(), self.elastic_sigma) * self.elastic_alpha
    #         dz = gaussian_filter(torch.randn(*shape).numpy(), self.elastic_sigma) * self.elastic_alpha
    #
    #         z, y, x = torch.meshgrid(
    #             torch.arange(shape[0]), torch.arange(shape[1]), torch.arange(shape[2]), indexing='ij'
    #         )
    #         coords = np.vstack([(z.numpy()+dz).ravel(),
    #                             (y.numpy()+dy).ravel(),
    #                             (x.numpy()+dx).ravel()])
    #         warped = map_coordinates(vol_np[c], coords, order=3, mode='reflect').reshape(shape)
    #         vols_out.append(torch.tensor(warped, device=self.device))
    #     return torch.stack(vols_out, dim=0)

    # ---------------- 主接口 ---------------- #

    def _downsample(self, x, method="interpolate", scale_factor=0.5):
        """
        将 3D 图像缩小一半
        参数:
            x: torch.Tensor, shape=(B, C, D, H, W)
            method: str, 下采样方式:
                - "interpolate": 使用 trilinear 插值
                - "nearest": 使用最近邻插值
                - "avgpool": 使用平均池化
        返回:
            torch.Tensor, shape=(B, C, D/2, H/2, W/2)
        """
        if method == "interpolate":
            x = x.unsqueeze(0)  # (1, C, D, H, W)
            x = F.interpolate(x, scale_factor=scale_factor, mode="trilinear", align_corners=False)
            x = x.squeeze(0)
            return x
        elif method == "nearest":
            return F.interpolate(x, scale_factor=scale_factor, mode="nearest")
        elif method == "avgpool":
            pool = nn.AvgPool3d(kernel_size=2, stride=2)
            return pool(x)

    def _normalize_image(self, img: torch.Tensor, scale="[0,1]"):
        """
        图像归一化 (纯 PyTorch 实现，支持 GPU)

        参数:
            img   : torch.Tensor, 输入图像 (任意维度)
            scale : str, "[-1,1]" 或 "[0,1]"

        返回:
            norm_img: torch.Tensor, 归一化后的图像
        """
        # img = img.to(torch.float32)

        # 1. 计算 0.5% 和 99.5% 分位数
        min_q = torch.quantile(img, 0.005)
        max_q = torch.quantile(img, 0.995)

        # 2. clip
        img = torch.clamp(img, min_q, max_q)

        # 3. Min-Max 归一化到 [0,1]
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min + 1e-8)

        # 4. 可选缩放到 [-1,1]
        if scale == "[-1,1]":
            img = img * 2 - 1

        return img

    def __call__(self, vol):
        """
        vol: (C,D,H,W) Tensor
        """
        vol = vol.to(self.device)
        with torch.no_grad():
            # 1. 随机旋转
            if self.apply_rotate and torch.rand(1, generator=self.rng, device=self.device).item() < self.p_rotate:
                vol = self._random_rotate(vol)

            # 2. 弹性形变
            if self.apply_elastic and torch.rand(1, generator=self.rng, device=self.device).item() < self.p_elastic:
                vol = self._elastic_deform(vol)

            # 3. 裁剪
            if self.apply_crop and self.crop_size != vol.shape[1:]:
                vol = self._crop(vol)

            if self.apply_normalize:
                vol = self._normalize_image(vol)

            # 下采样
            if self.apply_downsample:
                vol = self._downsample(vol)
        vol = vol.cpu()

        return vol


if __name__ == '__main__':

    B, C, D, H, W = 4, 2, 192, 218, 192
    batch_vol = torch.randn(B, C, D, H, W).cuda()
    augment = Augment3D(
        apply_rotate=False,
        apply_elastic=False,
        apply_crop=True,
        apply_downsample=True,
        apply_normalize=True,
        crop_size=(140,180,140),   # 随机裁剪成 48^3
        rotate_max_deg=20,      # 最大旋转角度 ±20°
        elastic_alpha=5,        # 弹性形变强度
        elastic_sigma=2,        # 弹性形变平滑
        device='cuda',
        seed=None
    )
    augmented_batch = []
    for b in range(B):
        augmented_batch.append(augment(batch_vol[b]))
    augmented_batch = torch.stack(augmented_batch)
    print("增强后批量尺寸:", augmented_batch.shape)
    # torch.Size([4, 2, 48, 48, 48])
