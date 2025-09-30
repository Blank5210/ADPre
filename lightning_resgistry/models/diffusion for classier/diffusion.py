import math
import torch
import torch.nn as nn
import numpy as np
from inspect import signature

from lightning_resgistry.models.builder import build_model, MODELS
from lightning_resgistry.models.losses import build_criteria


class DiffusionClassier(nn.Module):
    '''
    Classifier with classifier-free (free) guidance support.

    Assumptions:
      - backbone(x, t, y=None) -> (logits, noise_pred)
        If your backbone doesn't accept `y`, see notes below.
      - self.criteria expects ([logits, noise_pred], [label, noise])
    '''

    def __init__(
            self,
            backbone=None,
            criteria=None,
            num_classes=3,
            T=1000,
            beta_start=0.0001,
            beta_end=0.02,
            noise_schedule="linear",
            T_dim=128,
            dm=True,
            dm_min_snr=None,

            # ---- classifier-free guidance params ----
            p_uncond=0.1,  # training: prob to drop conditioning (uncond sample)
            uncond_label=None,  # what to pass as label for 'unconditioned' (None or -1)
            guidance_w=2.0,  # inference guidance weight w
    ):
        super().__init__()

        self.backbone = build_model(backbone)
        self.criteria = build_criteria(cfg=criteria)

        self.num_classes = num_classes
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_schedule = noise_schedule
        self.dm = dm

        # guidance hyperparams
        self.p_uncond = p_uncond
        self.uncond_label = uncond_label
        self.guidance_w = guidance_w

        if self.dm:
            # ---- diffusion params ----
            self.eps = 1e-6
            self.Beta, self.Alpha, self.Alpha_bar, self.Sigma, self.SNR = self.get_diffusion_hyperparams(
                noise_schedule=noise_schedule,
                T=self.T,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
            )
            # ---- diffusion params ----

            self.Beta = self.Beta.float().cuda()
            self.Alpha = self.Alpha.float().cuda()
            self.Alpha_bar = self.Alpha_bar.float().cuda()
            self.Sigma = self.Sigma.float().cuda()
            self.SNR = self.SNR.float().cuda() if dm_min_snr is None else torch.clamp(self.SNR.float().cuda(),
                                                                                      max=dm_min_snr)

    # --- diffusion helper functions (unchanged) ---
    def get_diffusion_hyperparams(self, noise_schedule, beta_start, beta_end, T):
        Beta = self.get_diffusion_betas(
            type=noise_schedule,
            start=beta_start,
            stop=beta_end,
            T=T
        )
        Alpha = 1 - Beta
        Alpha_bar = Alpha + 0
        Beta_tilde = Beta + 0
        for t in range(1, T):
            Alpha_bar[t] *= Alpha_bar[t - 1]
            Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])
        Sigma = torch.sqrt(Beta_tilde)
        Sigma[0] = 0.0
        SNR = Alpha_bar / (1 - Alpha_bar)
        return Beta, Alpha, Alpha_bar, Sigma, SNR

    def get_diffusion_betas(self, type='linear', start=0.0001, stop=0.02, T=1000):
        if type == 'linear':
            scale = 1000 / T
            beta_start = scale * start
            beta_end = scale * stop
            return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)

        elif type == 'cosine':
            steps = T + 1
            s = 0.008
            t = torch.linspace(start, stop, steps, dtype=torch.float64) / T
            alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)

        elif type == 'sigmoid':
            start = -3
            end = 3
            tau = 1
            steps = T + 1
            t = torch.linspace(0, T, steps, dtype=torch.float64) / T
            v_start = torch.tensor(start / tau).sigmoid()
            v_end = torch.tensor(end / tau).sigmoid()
            alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)

        elif type == "laplace":
            mu = 0.0
            b = 0.5
            lmb = lambda t: mu - b * torch.sign(0.5 - t) * torch.log(1 - 2 * torch.abs(0.5 - t))
            snr_func = lambda t: torch.exp(lmb(t))
            alpha_func = lambda t: torch.sqrt(snr_func(t) / (1 + snr_func(t)))
            timesteps = torch.linspace(0, 1, 1002)[1:-1]
            alphas_cumprod = []
            for t in timesteps:
                a = alpha_func(t) ** 2
                alphas_cumprod.append(a)
            alphas_cumprod = torch.cat(alphas_cumprod, dim=0)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
        else:
            raise NotImplementedError(type)

    def continuous_q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0, device=x_0.device)
        alpha_t = self.Alpha_bar[t]  # (B,)
        # make sure shape broadcast: (B,1,1,1,1) or (B,1,1,1) depending on x dims
        view_shape = [x_0.shape[0]] + [1] * (x_0.ndim - 1)
        alpha_t = alpha_t.view(*view_shape)
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        return x_t

    # --- guidance helpers ---
    def is_backbone_accepts_label(self):
        # heuristic: if backbone signature contains 'y' or 'label' arg
        try:
            sig = signature(self.backbone.__call__)
            params = sig.parameters
            for name in params:
                if name in ('y', 'label', 'cond', 'class_label'):
                    return True
            return False
        except Exception:
            # fallback: assume True and let runtime error inform user
            return True

    # --- training forward (with classifier-free mix) ---
    def forward(self, img, label):
        """
        During training:
          - sample timesteps t
          - sample noise
          - create mixed conditioned/unconditioned minibatch (via p_uncond)
          - run backbone on mixed batch: returns logits and noise_pred for each sample
          - compute loss via self.criteria

        Returns:
          logits_cond: logits for the original conditioned examples (for metrics)
          loss: training loss scalar or tensor
        """
        if not self.dm:
            raise RuntimeError("Diffusion module (dm) must be True for this forward implementation.")

        B = img.shape[0]
        device = img.device

        # sample timesteps and noise
        ts = torch.randint(0, self.T, size=(B,), device=device)
        noise = torch.randn_like(img, device=device)

        # create x_t
        img_t = self.continuous_q_sample(img, ts, noise)

        # --- create mask for which samples are unconditioned in this batch ---
        uncond_mask = (torch.rand(B, device=device) < self.p_uncond)  # boolean mask

        labels_for_backbone = label.clone()
        # if uncond_label is None we pass None for those entries (need to handle per-sample)
        # but many backbones expect a tensor; so we set to special index if provided
        if self.uncond_label is None:
            # create a tensor with same dtype but mark unconditioned entries with -1
            labels_for_backbone = labels_for_backbone.masked_fill(uncond_mask, -1)
        else:
            labels_for_backbone = labels_for_backbone.masked_fill(uncond_mask, self.uncond_label)

        # call backbone: support both signatures backbone(x, t, y) or backbone(x, t)
        if labels_for_backbone is None:
            logits, noise_pre = self.backbone(img_t, ts)
        else:
            logits, noise_pre = self.backbone(img_t, ts, y=labels_for_backbone)

        # compute loss: note that for unconditioned samples, labels_for_loss should be uncond_label or None
        labels_for_loss = label.clone()

        # call criteria: keep same API; criteria should be robust to -1 or None labels for uncond entries
        loss = self.criteria([logits[~uncond_mask], noise_pre], [labels_for_loss[~uncond_mask], noise])

        # return logits for full batch (may include unconditioned entries); user can pick conditioned subset for metrics
        return logits, loss

    # --- inference with classifier-free guidance ---
    def inference(self, img, t=None, guidance_w=None, ensemble=False):
        """
        Inference with optional free-guidance.

        If guidance_w is provided (float > 0), we compute both conditioned and unconditioned predictions
        and mix them:
            pred = pred_uncond + w * (pred_cond - pred_uncond)

        Note:
          - we assume backbone accepts y arg; unconditioned label will be self.uncond_label (or -1).
          - if backbone doesn't accept label, we fallback to single forward pass.
        """
        if t is None:
            t = torch.tensor([self.T // 2], device=img.device).expand(img.shape[0])

        if guidance_w is None:
            guidance_w = self.guidance_w

        # simple ensemble over a few t positions if requested
        if ensemble:
            ts = [self.T // 4, self.T // 2, 3 * self.T // 4]
            logits_list = []
            for step in ts:
                t_step = torch.tensor([step], device=img.device).expand(img.shape[0])
                logits_list.append(self._inference_single_t(img, t_step, guidance_w))
            # average logits across ensemble
            return torch.stack(logits_list).mean(0)
        else:
            return self._inference_single_t(img, t, guidance_w)

    def _inference_single_t(self, img, t, guidance_w):
        """
        helper: do guided/unconditioned forward for a single t.
        returns logits (guided) for the batch.
        """
        device = img.device
        B = img.shape[0]

        noise = torch.randn_like(img, device=device)
        img_t = self.continuous_q_sample(img, t, noise)

        # if backbone can't accept labels, fallback to single prediction
        if not self.is_backbone_accepts_label():
            logits, _ = self.backbone(img_t, t)
            return logits

        # conditioned prediction
        # NOTE: for inference we don't have ground-truth labels to pass as condition
        # typical classification usage: we want class-conditioned sampling for a target class.
        # Here we assume user passes desired `y` via some external mechanism. For simple scoring,
        # we can run conditioned on each class and choose best class (classifier guidance used differently).
        raise RuntimeError(
            "Inference with guidance requires providing a target label/class to condition on. "
            "Use `inference_conditional(img, t, target_label, guidance_w)` or call backbone directly."
        )

    def inference_conditional(self, img, target_label, t=None, guidance_w=None):
        """
        Run inference conditioned on `target_label` using free-guidance mixing with unconditioned prediction.

        Parameters:
          - img: input batch
          - target_label: tensor shape (B,) or scalar int (will be expanded)
          - guidance_w: float guidance weight (overrides self.guidance_w)
        Returns:
          - guided_logits: logits after guidance mixing
        """
        if t is None:
            t = torch.tensor([self.T // 2], device=img.device).expand(img.shape[0])

        if isinstance(target_label, int):
            target_label = torch.tensor([target_label], device=img.device).expand(img.shape[0])
        elif isinstance(target_label, torch.Tensor) and target_label.ndim == 0:
            target_label = target_label.expand(img.shape[0])

        if guidance_w is None:
            guidance_w = self.guidance_w

        noise = torch.randn_like(img, device=img.device)
        img_t = self.continuous_q_sample(img, t, noise)

        # conditioned forward
        try:
            logits_cond, noise_cond = self.backbone(img_t, t, y=target_label)
        except TypeError:
            logits_cond, noise_cond = self.backbone(img_t, t, target_label)

        # unconditioned forward: pass uncond_label (or -1)
        if self.uncond_label is None:
            uncond_label_tensor = torch.full_like(target_label, -1)
        else:
            uncond_label_tensor = torch.full_like(target_label, self.uncond_label)

        try:
            logits_uncond, noise_uncond = self.backbone(img_t, t, y=uncond_label_tensor)
        except TypeError:
            logits_uncond, noise_uncond = self.backbone(img_t, t, uncond_label_tensor)

        # guidance mixing on noise prediction (preferred) OR on logits (alternate)
        # prefer mixing noise predictions then re-computing x_{t-1} etc. But here we only have logits+noise_pred interface.
        # We'll mix noise_pred and logits linearly (simple and commonly used in practice for classifier-free guidance).
        guided_noise = noise_uncond + guidance_w * (noise_cond - noise_uncond)
        guided_logits = logits_uncond + guidance_w * (logits_cond - logits_uncond)

        # return guided logits (and optionally guided_noise)
        return guided_logits
