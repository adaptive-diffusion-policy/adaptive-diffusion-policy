import math
from functools import wraps

import torch
import torch.nn.functional as F
from einops import reduce, repeat
from torch import nn, sqrt
from torch.amp import autocast
from torch.special import expm1
from tqdm import tqdm

# helpers


def exists(val):
    return val is not None


def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"


def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d


# diffusion helpers


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.reshape(*t.shape, *((1,) * padding_dims))


# logsnr schedules and shifting / interpolating decorators
# only cosine for now


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def logsnr_schedule_cosine(t, logsnr_min=-15, logsnr_max=15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))


def beta_linear_log_snr(t):
    return -log(expm1(1e-4 + 10 * (t**2)))


def alpha_cosine_log_snr(t, s=0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


def logsnr_schedule_shifted(fn, image_d, noise_d):
    shift = 2 * math.log(noise_d / image_d)

    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal shift
        return fn(*args, **kwargs) + shift

    return inner


def logsnr_schedule_interpolated(fn, image_d, noise_d_low, noise_d_high):
    logsnr_low_fn = logsnr_schedule_shifted(fn, image_d, noise_d_low)
    logsnr_high_fn = logsnr_schedule_shifted(fn, image_d, noise_d_high)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        nonlocal logsnr_low_fn
        nonlocal logsnr_high_fn
        return t * logsnr_low_fn(t, *args, **kwargs) + (1 - t) * logsnr_high_fn(
            t, *args, **kwargs
        )

    return inner


# main gaussian diffusion class


class RollingDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        num_features,
        window_size,
        num_clean,
        num_steps,
        init_prob=0.1,
        pred_objective="v",
        noise_schedule=alpha_cosine_log_snr,
        noise_d=None,
        noise_d_low=None,
        noise_d_high=None,
        clip_sample_denoised=True,
        min_snr_loss_weight=True,
        min_snr_gamma=5,
    ):
        super().__init__()
        assert pred_objective in {
            "v",
            "eps",
        }, "whether to predict v-space (progressive distillation paper) or noise"

        self.model = model

        self.window_size = window_size
        self.num_clean = num_clean
        self.num_steps = num_steps

        self.init_prob = init_prob

        # feature dimensions

        self.num_features = num_features

        # training objective

        self.pred_objective = pred_objective

        # noise schedule

        assert not all([*map(exists, (noise_d, noise_d_low, noise_d_high))]), (
            "you must either set noise_d for shifted schedule, or noise_d_low and noise_d_high for shifted and interpolated schedule"
        )

        # determine shifting or interpolated schedules

        self.log_snr = noise_schedule

        if exists(noise_d):
            self.log_snr = logsnr_schedule_shifted(
                self.log_snr, self.num_features, noise_d
            )

        if exists(noise_d_low) or exists(noise_d_high):
            assert exists(noise_d_low) and exists(noise_d_high), (
                "both noise_d_low and noise_d_high must be set"
            )

            self.log_snr = logsnr_schedule_interpolated(
                self.log_snr, self.num_features, noise_d_low, noise_d_high
            )

        # sampling

        self.clip_sample_denoised = clip_sample_denoised

        # loss weight

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, cond, time, time_next):
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)[None, :, None]

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (
            (-log_snr).sigmoid(),
            (-log_snr_next).sigmoid(),
        )

        alpha, sigma, alpha_next = map(
            sqrt, (squared_alpha, squared_sigma, squared_alpha_next)
        )

        alpha = alpha[None, :, None]
        sigma = sigma[None, :, None]
        alpha_next = alpha_next[None, :, None]

        batch_log_snr = repeat(log_snr, "t -> b t", b=x.shape[0])
        pred = self.model(x, batch_log_snr[:, 0], cond)

        if self.pred_objective == "v":
            x_start = alpha * x - sigma * pred

        elif self.pred_objective == "eps":
            x_start = (x - sigma * pred) / alpha

        x_start.clamp_(-1.0, 1.0)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next[None, :, None] * c

        return model_mean, posterior_variance

    # sampling related functions

    @torch.no_grad()
    def p_sample(self, x, cond, time, time_next):
        model_mean, model_variance = self.p_mean_variance(x, cond, time, time_next)

        if time_next[0] == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape):
        noise = torch.randn(shape, device=self.device)

        steps = torch.linspace(1.0, 0.0, self.num_steps, device=self.device)
        steps = self.init_t(steps)

        action = noise
        for i in tqdm(
            range(self.num_steps - 1),
            desc="sampling loop time step",
            total=self.num_steps - 1,
            leave=False,
        ):
            time = steps[i]
            time_next = steps[i + 1]
            action = self.p_sample(action, cond, time, time_next)

        action.clamp_(-1.0, 1.0)
        return action

    @torch.no_grad()
    def sample_init(self, cond):
        return self.p_sample_loop(
            cond,
            (cond.shape[0], self.window_size, self.num_features),
        )

    @torch.no_grad()
    def sample(self, window, cond):
        steps = torch.linspace(1.0, 0.0, self.window_size, device=self.device)
        steps = self.init_t(steps)
        return self.p_sample(window, cond, steps[-2], steps[-1])

    # training related functions - noise prediction

    @autocast("cuda", enabled=False)
    def q_sample(self, x_start, times, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised = x_start * alpha + noise * sigma

        return x_noised, log_snr

    def init_t(self, t):
        w = torch.arange(self.window_size, device=self.device)
        t_k = w[..., None, :] - (self.num_clean - 1)
        t_k = t_k.clip(0, self.window_size)
        t_k = t_k / (self.window_size - self.num_clean)
        t_k = t_k + t[..., None]
        t_k = t_k.clip(0, 1)
        return t_k

    def p_losses(self, x_start, cond, times, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        init_times = self.init_t(times)

        steps = torch.linspace(1.0, 0.0, self.num_steps, device=self.device)
        window_times = self.init_t(steps[-2:-1])

        mask = (
            torch.zeros((x_start.shape[0],), device=self.device).float().uniform_(0, 1)
        )
        mask = (mask < self.init_prob)[:, None].float()

        times = init_times * mask + window_times * (1 - mask)

        x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)

        model_out = self.model(x, log_snr[:, 0], cond)

        if self.pred_objective == "v":
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = (
                padded_log_snr.sigmoid().sqrt(),
                (-padded_log_snr).sigmoid().sqrt(),
            )
            target = alpha * noise - sigma * x_start

        elif self.pred_objective == "eps":
            target = noise

        loss = F.mse_loss(model_out, target, reduction="none")

        loss = reduce(loss, "b t c -> b t", "mean")

        snr = log_snr.exp()

        maybe_clip_snr = snr.clone()
        if self.min_snr_loss_weight:
            maybe_clip_snr.clamp_(max=self.min_snr_gamma)

        if self.pred_objective == "v":
            loss_weight = maybe_clip_snr / (snr + 1)

        elif self.pred_objective == "eps":
            loss_weight = maybe_clip_snr / snr

        return (loss * loss_weight).mean()

    def forward(self, action, cond):
        times = (
            torch.zeros((action.shape[0],), device=self.device).float().uniform_(0, 1)
        )

        return self.p_losses(action, cond, times)
