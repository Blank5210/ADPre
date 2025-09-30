import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from lightning_resgistry.models.builder import build_model, MODELS
from lightning_resgistry.models.losses import build_criteria
from lightning_resgistry.models.ddpm.unet import UNet


@MODELS.register_module("diffusion_3d")
class DiffusionClassier(nn.Module):
    '''
        use diffusion in training, add noise when interface
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
            self.SNR = self.SNR.float().cuda() if dm_min_snr is None else torch.clamp(self.SNR.float().cuda(),max=dm_min_snr)

    def get_diffusion_hyperparams(
            self,
            noise_schedule,
            beta_start,
            beta_end,
            T
    ):
        """
        Compute diffusion process hyperparameters

        Parameters:
        T (int):                    number of diffusion steps
        beta_0 and beta_T (float):  beta schedule start/end value,
                                    where any beta_t in the middle is linearly interpolated

        Returns:
        a dictionary of diffusion hyperparameters including:
            T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
            These cpu tensors are changed to cuda tensors on each individual gpu
        """

        # Beta = torch.linspace(noise_schedule,beta_start, beta_end, T)
        Beta = self.get_diffusion_betas(
            type=noise_schedule,
            start=beta_start,
            stop=beta_end,
            T=T
        )
        # at = 1 - bt
        Alpha = 1 - Beta
        # at_
        Alpha_bar = Alpha + 0
        # 方差
        Beta_tilde = Beta + 0
        for t in range(1, T):
            # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
            Alpha_bar[t] *= Alpha_bar[t - 1]
            # \tilde{\beta}_t = (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t) * \beta_t
            Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])
        # 标准差
        Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t
        Sigma[0] = 0.0

        '''
            SNR = at ** 2 / sigma ** 2
            at = sqrt(at_), sigma = sqrt(1 - at_)
            q(xt|x0) = sqrt(at_) * x0 + sqrt(1 - at_) * noise
        '''
        SNR = Alpha_bar / (1 - Alpha_bar)

        return Beta, Alpha, Alpha_bar, Sigma, SNR

    def get_diffusion_betas(self, type='linear', start=0.0001, stop=0.02, T=1000):
        """Get betas from the hyperparameters."""
        if type == 'linear':
            # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
            # To be used with Gaussian diffusion models in continuous and discrete
            # state spaces.
            # To be used with transition_mat_type = 'gaussian'
            scale = 1000 / T
            beta_start = scale * start
            beta_end = scale * stop
            return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)

        elif type == 'cosine':
            # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
            # To be used with transition_mat_type = 'uniform'.
            steps = T + 1
            s = 0.008
            # t = torch.linspace(0, T, steps, dtype=torch.float64) / T
            t = torch.linspace(start, stop, steps, dtype=torch.float64) / T
            alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)


        elif type == 'sigmoid':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
            # To be used with absorbing state models.
            # ensures that the probability of decaying to the absorbing state
            # increases linearly over time, and is 1 for t = T-1 (the final time).
            # To be used with transition_mat_type = 'absorbing'
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
            # sigma_func = lambda t: torch.sqrt(1 / (1 + snr_func(t)))

            timesteps = torch.linspace(0, 1, 1002)[1:-1]
            alphas_cumprod = []
            for t in timesteps:
                a = alpha_func(t) ** 2
                alphas_cumprod.append(a)
            alphas_cumprod = torch.cat(alphas_cumprod,dim=0)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
        else:
            raise NotImplementedError(type)


    def continuous_p_ddim_sample(self, x_t, t, noise):

        if(self.dm_target == "noise"):
            # x0 = (xt - sqrt(1-at_) * noise) / sqrt(at_)
            c_x0 = (x_t - torch.sqrt(1 - self.Alpha_bar[t]) * noise) / torch.sqrt(self.Alpha_bar[t])
        elif(self.dm_target == "x0"):
            c_x0 = noise
            # noise = (xt - sqrt(1-at_) * x0) / sqrt(1-at_)
            noise = (x_t - torch.sqrt(self.Alpha_bar[t]) * c_x0) / torch.sqrt(1 - self.Alpha_bar[t])

        if(t[0] == 0):
            return c_x0

        # sqrt(at-1_) * (xt - sqrt(1-at_) * noise) / sqrt(at_)
        c_xt_1_1 = torch.sqrt(self.Alpha_bar[t-1]) * c_x0

        # sqrt(1 - at-1_) * noise
        c_xt_1_2 = torch.sqrt(1 - self.Alpha_bar[t-1]) * noise

        # xt-1 = sqrt(at-1_) * (xt - sqrt(1-at_) * noise) / sqrt(at_) + sqrt(1 - at-1_) * noise
        c_xt_1 = c_xt_1_1 + c_xt_1_2

        return c_xt_1

    def continuous_q_sample(self,x_0, t, noise=None):
        if noise is None:
            # sampling from Gaussian distribution
            noise = torch.normal(0, 1, size=x_0.shape, dtype=torch.float32).cuda()
        # xt = sqrt(at_) * x0 + sqrt(1-at_) * noise
        alpha_t = self.Alpha_bar[t]  # (B,)
        alpha_t = alpha_t.view(-1, 1, 1, 1, 1)  # (B, 1, 1, 1, 1)
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        return x_t

    def get_time_schedule(self, T=1000, step=5):
        times = np.linspace(-1, T - 1, num = step + 1, dtype=int)[::-1]
        return times

    def add_gaussian_noise(self, pts, sigma=0.1, clamp=0.03):
        # input: (b, 3, n)

        assert (clamp > 0)
        # jittered_data = torch.clamp(sigma * torch.randn_like(pts), -1 * clamp, clamp)
        jittered_data = sigma * torch.randn_like(pts).cuda()
        jittered_data = jittered_data + pts

        return jittered_data

    def add_random_noise(self, pts, sigma=0.1, clamp=0.03):
        # input: (b, 3, n)

        assert (clamp > 0)
        #         jittered_data = torch.clamp(sigma * torch.rand_like(pts), -1 * clamp, clamp).cuda()
        jittered_data = sigma * torch.rand_like(pts).cuda()
        jittered_data = jittered_data + pts

        return jittered_data


    def add_laplace_noise(self, pts, sigma=0.1, clamp=0.03, loc=0.0, scale=1.0):
        # input: (b, 3, n)

        assert (clamp > 0)
        laplace_distribution = torch.distributions.Laplace(loc=loc, scale=scale)
        jittered_data = sigma * laplace_distribution.sample(pts.shape).cuda()
        # jittered_data = torch.clamp(sigma * laplace_distribution.sample(pts.shape), -1 * clamp, clamp).cuda()
        jittered_data = jittered_data + pts

        return jittered_data

    def add_possion_noise(self, pts, sigma=0.1, clamp=0.03, rate=3.0):
        # input: (b, 3, n)

        assert (clamp > 0)
        poisson_distribution = torch.distributions.Poisson(rate)
        jittered_data = sigma * poisson_distribution.sample(pts.shape).cuda()
        # jittered_data = torch.clamp(sigma * poisson_distribution.sample(pts.shape), -1 * clamp, clamp).cuda()
        jittered_data = jittered_data + pts

        return jittered_data

    def inference(self, img, t=None, ensemble=False):
        if t is None:
            # 默认用 T//2
            t = torch.tensor([self.T // 2], device=img.device).expand(img.shape[0], 1)

        if ensemble:
            ts = [self.T // 4, self.T // 2, 3 * self.T // 4]
            logits_list = []
            for step in ts:
                t_step = torch.tensor([step], device=img.device).expand(img.shape[0], 1)
                img_t = self.continuous_q_sample(img, t_step)
                logits, _ = self.backbone(img_t, t_step)
                logits_list.append(logits)
            return torch.stack(logits_list).mean(0)
        else:
            img_t = self.continuous_q_sample(img, t)
            logits, _ = self.backbone(img_t, t)
            return logits

    def forward(self, img, label):
        if self.dm:
            ts = torch.randint(0, self.T, size=(img.shape[0], ), device=img.device)
            noise = torch.randn_like(img, device=img.device, requires_grad=False)  # 直接生成和 img 一样 shape & device 的标准正态噪声

            img_t = self.continuous_q_sample(img, ts, noise)
            logits, noise_pre = self.backbone(img_t, ts)

            loss = self.criteria([logits, noise_pre], [label, noise])
            return logits, loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    cfg = dict(
        type="diffusion_3d",
        backbone=dict(
            type="unet-3D",
            T=1000,
            ch=64,
            ch_mult=[1, 2, 4, 4],
            attn=[-1],
            num_res_blocks=1,
            dropout=0.1
            ),
        criteria=[
            dict(type="CrossEntropyLoss"),
            dict(type="MSELoss")
        ],
    )
    model = build_model(cfg)
    model.to(device)
    x = torch.randn(batch_size, 1, 70, 90, 70)
    x = x.to(device)
    labels = torch.randint(0, 3, (batch_size,)).to(device)
    y = model(x, labels)


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)